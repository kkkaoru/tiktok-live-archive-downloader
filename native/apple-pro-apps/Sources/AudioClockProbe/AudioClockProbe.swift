import AVFoundation
import Foundation

struct ClockProbeRequest: Codable {
  let source: String
  let outputDirectory: String
  let startValue: Int64
  let durationValue: Int64
  let timescale: Int32
  var compressedOnly: Bool? = nil
  var continuousCheckpoints: [PCMCheckpoint]? = nil
  var statefulSelection: StatefulSelection? = nil
}

struct ClockProbeBuffer: Codable {
  let value: Int64
  let timescale: Int32
  let durationValue: Int64
  let durationTimescale: Int32
  let samples: Int
}

struct ClockProbeResult: Codable {
  let foundationTemporaryDirectory: String
  let explicitOutputDirectory: String
  let writerTemporaryDirectory: String
  let mode: String
  let buffers: [ClockProbeBuffer]
}

enum ClockProbeError: Error {
  case invalidRequest
  case unsupportedAudio
  case readerFailed
  case writerFailed
  case budgetExceeded
}

/// Diagnostic only: retain the actual reader buffers and PTS in a new PCM MOV.
/// Track and AudioMix reads are independent; neither certifies missing speech.
enum AudioClockProbe {
  static func validate(_ request: ClockProbeRequest, root: URL) throws -> URL {
    let directory = URL(fileURLWithPath: request.outputDirectory).standardizedFileURL
    let source = URL(fileURLWithPath: request.source)
    guard request.timescale > 0, request.startValue >= 0,
      request.durationValue > 0,
      request.statefulSelection != nil
        || request.durationValue <= Int64(request.timescale)
          * (request.continuousCheckpoints == nil ? 10 : 20),
      request.source.hasPrefix("/"), request.outputDirectory.hasPrefix("/"),
      FileManager.default.fileExists(atPath: source.path),
      directory.path.hasPrefix(root.path + "/edits/work/"),
      directory.resolvingSymlinksInPath() == directory,
      !FileManager.default.fileExists(atPath: directory.path)
    else { throw ClockProbeError.invalidRequest }
    if let checkpoints = request.continuousCheckpoints {
      try ContinuousClockProbe.validate(checkpoints, request: request)
    }
    if let selection = request.statefulSelection {
      try StatefulClockProbe.validate(selection, request: request)
    }
    return directory
  }

  static func validateScratch(_ directory: URL, root: URL) throws -> URL {
    let canonical = directory.standardizedFileURL
    let canonicalRoot = root.standardizedFileURL
    guard canonicalRoot.resolvingSymlinksInPath() == canonicalRoot,
      canonical.path.hasPrefix(canonicalRoot.path + "/edits/work/"),
      canonical.resolvingSymlinksInPath() == canonical
    else { throw ClockProbeError.invalidRequest }
    let values = try canonical.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
    guard values.isDirectory == true, values.isSymbolicLink == false else {
      throw ClockProbeError.invalidRequest
    }
    return canonical
  }

  static func configureScratch(_ writer: AVAssetWriter, directory: URL, root: URL) throws -> URL {
    let validated = try validateScratch(directory, root: root)
    writer.directoryForTemporaryFiles = validated
    guard let actual = writer.directoryForTemporaryFiles,
      try validateScratch(actual, root: root).path == validated.path
    else { throw ClockProbeError.writerFailed }
    return validated
  }

  static func run(
    _ request: ClockProbeRequest, root: URL,
    writerBoundary: (AVAssetWriter, AVAssetWriterInput) -> any StatefulWriting = {
      NativeStatefulWriter(writer: $0, input: $1)
    },
    progress: @Sendable (Int) async throws -> Void = { _ in }
  ) async throws {
    let directory = try validate(request, root: root)
    try FileManager.default.createDirectory(
      at: directory, withIntermediateDirectories: false,
      attributes: [.posixPermissions: 0o700])
    if let selection = request.statefulSelection {
      let result = try await StatefulClockProbe.read(
        request, selection: selection, directory: directory, root: root,
        writerBoundary: writerBoundary, progress: progress)
      try Task.checkCancellation()
      let data = try JSONEncoder().encode(result)
      guard
        try StatefulClockProbe.savedBytes(directory) + data.count <= selection.budget.maxSavedBytes
      else { throw ClockProbeError.budgetExceeded }
      try data.write(
        to: directory.appendingPathComponent("stateful.json"), options: .withoutOverwriting)
      return
    }
    if let checkpoints = request.continuousCheckpoints {
      let result = try await ContinuousClockProbe.read(
        request, checkpoints: checkpoints, directory: directory, root: root)
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(result).write(
        to: directory.appendingPathComponent("continuous.json"), options: .withoutOverwriting)
      return
    }
    if request.compressedOnly == true {
      let result = try await CompressedClockProbe.read(request)
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(result).write(
        to: directory.appendingPathComponent("compressed.json"), options: .withoutOverwriting)
      return
    }
    for mixed in [false, true] {
      try Task.checkCancellation()
      let result = try await read(request, directory: directory, root: root, mixed: mixed)
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(result).write(
        to: directory.appendingPathComponent(result.mode + ".json"), options: .withoutOverwriting)
    }
  }

  private static func read(
    _ request: ClockProbeRequest, directory: URL, root: URL, mixed: Bool
  ) async throws -> ClockProbeResult {
    let mode = mixed ? "audio-mix" : "track"
    let asset = AVURLAsset(url: URL(fileURLWithPath: request.source))
    let tracks = try await asset.loadTracks(withMediaType: .audio)
    guard tracks.count == 1, let track = tracks.first else {
      throw ClockProbeError.unsupportedAudio
    }
    let reader = try AVAssetReader(asset: asset)
    let start = CMTime(value: request.startValue, timescale: request.timescale)
    reader.timeRange = CMTimeRange(
      start: start,
      duration: CMTime(value: request.durationValue, timescale: request.timescale))
    let settings: [String: Int] = [
      AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: 48000,
      AVNumberOfChannelsKey: 2, AVLinearPCMBitDepthKey: 16,
      AVLinearPCMIsFloatKey: 0, AVLinearPCMIsBigEndianKey: 0,
      AVLinearPCMIsNonInterleaved: 0,
    ]
    let output: AVAssetReaderOutput
    if mixed {
      output = AVAssetReaderAudioMixOutput(audioTracks: tracks, audioSettings: settings)
    } else {
      output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
    }
    guard reader.canAdd(output) else { throw ClockProbeError.unsupportedAudio }
    reader.add(output)
    let writer = try AVAssetWriter(
      outputURL: directory.appendingPathComponent(mode + ".mov"), fileType: .mov)
    let scratch = directory.appendingPathComponent(mode + "-scratch", isDirectory: true)
    _ = try validateScratch(directory, root: root)
    try FileManager.default.createDirectory(
      at: scratch, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    let writerScratch = try configureScratch(writer, directory: scratch, root: root)
    let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil)
    guard writer.canAdd(input) else { throw ClockProbeError.unsupportedAudio }
    writer.add(input)
    guard writer.startWriting(), reader.startReading() else {
      throw ClockProbeError.readerFailed
    }
    defer {
      if reader.status == .reading { reader.cancelReading() }
      if writer.status == .writing { writer.cancelWriting() }
    }
    writer.startSession(atSourceTime: start)
    var buffers: [ClockProbeBuffer] = []
    let deadline = ContinuousClock.now.advanced(by: .seconds(30))
    while let sample = output.copyNextSampleBuffer() {
      try Task.checkCancellation()
      guard buffers.count < 10000, ContinuousClock.now < deadline else {
        throw ClockProbeError.budgetExceeded
      }
      let pts = CMSampleBufferGetPresentationTimeStamp(sample)
      let duration = CMSampleBufferGetDuration(sample)
      buffers.append(
        ClockProbeBuffer(
          value: pts.value, timescale: pts.timescale,
          durationValue: duration.value, durationTimescale: duration.timescale,
          samples: CMSampleBufferGetNumSamples(sample)))
      while !input.isReadyForMoreMediaData {
        guard writer.status == .writing, ContinuousClock.now < deadline else {
          throw ClockProbeError.writerFailed
        }
        try await Task.sleep(for: .milliseconds(1))
      }
      guard input.append(sample) else { throw ClockProbeError.writerFailed }
    }
    guard reader.status == .completed, !buffers.isEmpty else {
      throw ClockProbeError.readerFailed
    }
    input.markAsFinished()
    await writer.finishWriting()
    guard writer.status == .completed else { throw ClockProbeError.writerFailed }
    return ClockProbeResult(
      foundationTemporaryDirectory: FileManager.default.temporaryDirectory.path,
      explicitOutputDirectory: directory.path,
      writerTemporaryDirectory: writerScratch.path, mode: mode, buffers: buffers)
  }
}
