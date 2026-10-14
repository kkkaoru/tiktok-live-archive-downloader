import AVFoundation
import Foundation

/// Half-open decoded PCM ordinals, never packet or presentation-time offsets.
struct PCMCheckpoint: Codable {
  let startSample: Int
  let sampleCount: Int
}

struct ContinuousClockResult: Codable {
  let origin: ProbeTime
  let buffers: [ClockProbeBuffer]
  let checkpoints: [PCMCheckpoint]
  let decodedSamples: Int
  let elapsedSeconds: Double
  let writerTemporaryDirectories: [String]
}

/// One reader from asset zero; only checkpoint payloads are written. No seeking or restart.
enum ContinuousClockProbe {
  static func validate(_ checkpoints: [PCMCheckpoint], request: ClockProbeRequest) throws {
    guard request.startValue == 0, request.compressedOnly != true,
      !checkpoints.isEmpty, checkpoints.count <= 2
    else { throw ClockProbeError.invalidRequest }
    var end = 0
    for checkpoint in checkpoints {
      guard checkpoint.startSample >= end, checkpoint.startSample <= 960_000,
        checkpoint.sampleCount > 0, checkpoint.sampleCount <= 4800
      else { throw ClockProbeError.invalidRequest }
      end = checkpoint.startSample + checkpoint.sampleCount
      guard Int64(end) * Int64(request.timescale) <= request.durationValue * 48000 else {
        throw ClockProbeError.invalidRequest
      }
    }
  }

  static func intersection(_ checkpoint: PCMCheckpoint, ordinal: Int, samples: Int) -> Range<Int> {
    let lower = max(ordinal, checkpoint.startSample)
    let upper = min(ordinal + samples, checkpoint.startSample + checkpoint.sampleCount)
    return lower..<max(lower, upper)
  }

  static func slice(_ sample: CMSampleBuffer, range: Range<Int>) throws -> CMSampleBuffer {
    var result: CMSampleBuffer?
    let status = CMSampleBufferCopySampleBufferForRange(
      allocator: kCFAllocatorDefault, sampleBuffer: sample,
      sampleRange: CFRange(location: range.lowerBound, length: range.count),
      sampleBufferOut: &result)
    guard status == noErr, let result else { throw ClockProbeError.readerFailed }
    return result
  }

  static func read(
    _ request: ClockProbeRequest, checkpoints: [PCMCheckpoint], directory: URL, root: URL
  ) async throws -> ContinuousClockResult {
    try validate(checkpoints, request: request)
    let begin = ContinuousClock.now
    let deadline = begin.advanced(by: .seconds(30))
    let asset = AVURLAsset(url: URL(fileURLWithPath: request.source))
    let tracks = try await asset.loadTracks(withMediaType: .audio)
    guard tracks.count == 1, let track = tracks.first else {
      throw ClockProbeError.unsupportedAudio
    }
    let reader = try AVAssetReader(asset: asset)
    reader.timeRange = CMTimeRange(
      start: .zero, duration: CMTime(value: request.durationValue, timescale: request.timescale))
    let settings: [String: Int] = [
      AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: 48000,
      AVNumberOfChannelsKey: 2, AVLinearPCMBitDepthKey: 16,
      AVLinearPCMIsFloatKey: 0, AVLinearPCMIsBigEndianKey: 0, AVLinearPCMIsNonInterleaved: 0,
    ]
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
    guard reader.canAdd(output) else { throw ClockProbeError.unsupportedAudio }
    reader.add(output)
    var writers: [AVAssetWriter] = []
    var inputs: [AVAssetWriterInput] = []
    var scratchPaths: [String] = []
    defer {
      if reader.status == .reading { reader.cancelReading() }
      for writer in writers where writer.status == .writing { writer.cancelWriting() }
    }
    for index in checkpoints.indices {
      let writer = try AVAssetWriter(
        outputURL: directory.appendingPathComponent("checkpoint-\(index).mov"), fileType: .mov)
      writer.movieTimeScale = 48000
      let scratch = directory.appendingPathComponent("checkpoint-\(index)-scratch")
      try FileManager.default.createDirectory(
        at: scratch, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
      scratchPaths.append(
        try AudioClockProbe.configureScratch(writer, directory: scratch, root: root).path)
      let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil)
      guard writer.canAdd(input) else { throw ClockProbeError.unsupportedAudio }
      writer.add(input)
      writers.append(writer)
      inputs.append(input)
      guard writer.startWriting() else { throw ClockProbeError.writerFailed }
    }
    guard reader.startReading() else { throw ClockProbeError.readerFailed }
    var ordinal = 0
    var retained = Array(repeating: 0, count: checkpoints.count)
    var buffers: [ClockProbeBuffer] = []
    while let sample = output.copyNextSampleBuffer() {
      try Task.checkCancellation()
      guard buffers.count < 10000, ContinuousClock.now < deadline else {
        throw ClockProbeError.budgetExceeded
      }
      let pts = CMSampleBufferGetPresentationTimeStamp(sample)
      let duration = CMSampleBufferGetDuration(sample)
      let count = CMSampleBufferGetNumSamples(sample)
      // Clock discontinuities remain in the ledger; unsupported sample durations never get stretched.
      guard pts.isNumeric, count > 0,
        duration == CMTime(value: Int64(count), timescale: 48000)
      else { throw ClockProbeError.unsupportedAudio }
      buffers.append(
        ClockProbeBuffer(
          value: pts.value, timescale: pts.timescale, durationValue: duration.value,
          durationTimescale: duration.timescale, samples: count))
      for index in checkpoints.indices {
        let selected = intersection(checkpoints[index], ordinal: ordinal, samples: count)
        if selected.isEmpty { continue }
        let part = try slice(
          sample, range: (selected.lowerBound - ordinal)..<(selected.upperBound - ordinal))
        let writer = writers[index]
        let input = inputs[index]
        if retained[index] == 0 {
          writer.startSession(atSourceTime: CMSampleBufferGetPresentationTimeStamp(part))
        }
        while !input.isReadyForMoreMediaData {
          guard writer.status == .writing, ContinuousClock.now < deadline else {
            throw ClockProbeError.writerFailed
          }
          try await Task.sleep(for: .milliseconds(1))
        }
        guard input.append(part) else { throw ClockProbeError.writerFailed }
        retained[index] += selected.count
      }
      ordinal += count
    }
    guard reader.status == .completed, !buffers.isEmpty,
      retained == checkpoints.map(\.sampleCount)
    else { throw ClockProbeError.readerFailed }
    for index in writers.indices {
      inputs[index].markAsFinished()
      await writers[index].finishWriting()
      guard writers[index].status == .completed else { throw ClockProbeError.writerFailed }
    }
    let elapsed = begin.duration(to: .now).components
    return ContinuousClockResult(
      origin: ProbeTime(.zero), buffers: buffers, checkpoints: checkpoints, decodedSamples: ordinal,
      elapsedSeconds: Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18,
      writerTemporaryDirectories: scratchPaths)
  }
}
