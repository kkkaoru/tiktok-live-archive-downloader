import AVFoundation
import CryptoKit
import Foundation

struct SourceCheckpoint: Codable {
  let startValue: Int64
  let timescale: Int32
  let sampleCount: Int

  var start: CMTime { CMTime(value: startValue, timescale: timescale) }
  var end: CMTime { start + CMTime(value: Int64(sampleCount), timescale: 48000) }
}

struct StatefulBudget: Codable {
  let maxRecords: Int
  let maxLedgerBytes: Int
  let maxSavedBytes: Int
  let maxDecodedSamples: Int
  let maxWallSeconds: Double
  let minimumFreeBytes: Int64
}

struct StatefulSelection: Codable {
  let checkpoints: [SourceCheckpoint]
  let budget: StatefulBudget
  var productionMaterial: Bool? = nil
}

struct StatefulRecord: Codable {
  let bufferOrdinal: Int
  let sampleOrdinal: Int
  let sourcePTS: ProbeTime
  let duration: ProbeTime
  let samples: Int
  let discontinuity: ProbeTime
}

struct StatefulResult: Codable {
  let version: Int
  let origin: ProbeTime
  let selection: StatefulSelection
  let decodedSamples: Int
  let records: Int
  let ledgerBytes: Int
  let ledgerSHA256: String
  let initializationSeconds: Double
  let decoderSeconds: Double
  let sampleCopySeconds: Double
  let finalizeSeconds: Double
  let writerTemporaryDirectories: [String]
  var supportedEnd: ProbeTime? = nil
  var assetDuration: ProbeTime? = nil
}

/// State is bounded independently of source duration. A record is persisted before selection.
struct SourcePTSSelector {
  let checkpoints: [SourceCheckpoint]
  var retained: [Int]
  var previousEnd = CMTime.zero
  var ordinal = 0
  var records = 0

  init(_ checkpoints: [SourceCheckpoint]) {
    self.checkpoints = checkpoints
    retained = Array(repeating: 0, count: checkpoints.count)
  }

  mutating func accept(pts: CMTime, duration: CMTime, samples: Int) throws -> [Range<Int>] {
    guard pts.isNumeric, pts >= .zero, samples > 0,
      duration == CMTime(value: Int64(samples), timescale: 48000)
    else { throw ClockProbeError.unsupportedAudio }
    let end = pts + duration
    var ranges: [Range<Int>] = []
    for (index, checkpoint) in checkpoints.enumerated() {
      // Reject gaps and repeated source support even if an earlier buffer filled the checkpoint.
      if pts != previousEnd,
        min(pts, previousEnd) < checkpoint.end, max(pts, previousEnd) > checkpoint.start
      {
        throw ClockProbeError.unsupportedAudio
      }
      let lower = max(pts, checkpoint.start)
      let upper = min(end, checkpoint.end)
      if upper <= lower {
        ranges.append(0..<0)
        continue
      }
      let offset = CMTimeConvertScale(lower - pts, timescale: 48000, method: .default)
      let length = CMTimeConvertScale(upper - lower, timescale: 48000, method: .default)
      guard !offset.hasBeenRounded, !length.hasBeenRounded,
        lower == checkpoint.start + CMTime(value: Int64(retained[index]), timescale: 48000)
      else { throw ClockProbeError.unsupportedAudio }
      ranges.append(Int(offset.value)..<Int(offset.value + length.value))
      retained[index] += Int(length.value)
    }
    previousEnd = end
    ordinal += samples
    records += 1
    return ranges
  }

  func finish() throws {
    guard records > 0, retained == checkpoints.map(\.sampleCount) else {
      throw ClockProbeError.readerFailed
    }
  }
}

/// Only the writer boundary is replaceable; the reader remains single-owner.
protocol StatefulWriting {
  var status: AVAssetWriter.Status { get }
  var isReady: Bool { get }
  func append(_ sample: CMSampleBuffer) -> Bool
  func finish() async
  func cancel()
}

struct NativeStatefulWriter: StatefulWriting {
  let writer: AVAssetWriter
  let input: AVAssetWriterInput
  var status: AVAssetWriter.Status { writer.status }
  var isReady: Bool { input.isReadyForMoreMediaData }
  func append(_ sample: CMSampleBuffer) -> Bool { input.append(sample) }
  func finish() async {
    input.markAsFinished()
    await writer.finishWriting()
  }
  func cancel() { writer.cancelWriting() }
}

enum StatefulClockProbe {
  static let maximumCheckpointSamples = 60 * 48000

  static func validate(_ selection: StatefulSelection, request: ClockProbeRequest) throws {
    let budget = selection.budget
    let material = selection.productionMaterial == true
    guard request.startValue == 0, request.compressedOnly != true,
      request.continuousCheckpoints == nil, request.timescale > 0,
      request.durationValue > 0, request.durationValue <= Int64(request.timescale) * 21600,
      (1...3).contains(selection.checkpoints.count),
      (1...2_000_000).contains(budget.maxRecords),
      (1...67_108_864).contains(budget.maxLedgerBytes),
      (1...(material ? 8_589_934_592 : 268_435_456)).contains(budget.maxSavedBytes),
      (1...1_036_800_000).contains(budget.maxDecodedSamples),
      budget.maxWallSeconds.isFinite, budget.maxWallSeconds > 0, budget.maxWallSeconds <= 3600,
      budget.minimumFreeBytes >= 0
    else { throw ClockProbeError.invalidRequest }
    if material {
      guard selection.checkpoints.count == 1, let checkpoint = selection.checkpoints.first,
        checkpoint.start == .zero,
        checkpoint.end == CMTime(value: request.durationValue, timescale: request.timescale),
        checkpoint.sampleCount == budget.maxDecodedSamples
      else { throw ClockProbeError.invalidRequest }
    }
    var end = CMTime.zero
    for checkpoint in selection.checkpoints {
      guard checkpoint.timescale > 0, checkpoint.startValue >= 0,
        (1...(material ? budget.maxDecodedSamples : maximumCheckpointSamples)).contains(
          checkpoint.sampleCount), checkpoint.start >= end,
        checkpoint.end <= CMTime(value: request.durationValue, timescale: request.timescale)
      else { throw ClockProbeError.invalidRequest }
      end = checkpoint.end
    }
  }

  static func seconds(_ start: ContinuousClock.Instant) -> Double {
    let elapsed = start.duration(to: .now).components
    return Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
  }

  static func checkStorage(_ directory: URL, budget: StatefulBudget) throws {
    let attributes = try FileManager.default.attributesOfFileSystem(forPath: directory.path)
    guard let free = attributes[.systemFreeSize] as? NSNumber,
      free.int64Value >= budget.minimumFreeBytes
    else { throw ClockProbeError.budgetExceeded }
    guard try savedBytes(directory) <= budget.maxSavedBytes else {
      throw ClockProbeError.budgetExceeded
    }
  }

  static func savedBytes(_ directory: URL) throws -> Int {
    let files = try FileManager.default.contentsOfDirectory(
      at: directory,
      includingPropertiesForKeys: [.fileSizeKey, .isSymbolicLinkKey, .isDirectoryKey])
    var bytes = 0
    for file in files {
      let values = try file.resourceValues(forKeys: [
        .fileSizeKey, .isSymbolicLinkKey, .isDirectoryKey,
      ])
      guard values.isSymbolicLink == false else { throw ClockProbeError.invalidRequest }
      if values.isDirectory == true {
        bytes += try savedBytes(file)
      } else {
        guard let size = values.fileSize else { throw ClockProbeError.invalidRequest }
        bytes += size
      }
    }
    return bytes
  }

  static func read(
    _ request: ClockProbeRequest, selection: StatefulSelection, directory: URL, root: URL,
    writerBoundary: (AVAssetWriter, AVAssetWriterInput) -> any StatefulWriting = {
      NativeStatefulWriter(writer: $0, input: $1)
    },
    progress: @Sendable (Int) async throws -> Void = { _ in }
  ) async throws -> StatefulResult {
    try validate(selection, request: request)
    let begin = ContinuousClock.now
    let deadline = begin.advanced(by: .seconds(selection.budget.maxWallSeconds))
    try Task.checkCancellation()
    try checkStorage(directory, budget: selection.budget)
    let asset = AVURLAsset(url: URL(fileURLWithPath: request.source))
    let tracks = try await asset.loadTracks(withMediaType: .audio)
    guard tracks.count == 1, let track = tracks.first else {
      throw ClockProbeError.unsupportedAudio
    }
    let reader = try AVAssetReader(asset: asset)
    let material = selection.productionMaterial == true
    let assetDuration = try await asset.load(.duration)
    if material {
      guard assetDuration.isNumeric, assetDuration > .zero,
        assetDuration <= CMTime(value: request.durationValue, timescale: request.timescale)
      else { throw ClockProbeError.invalidRequest }
      // Natural EOF, not a timed cancellation or a sequence of late seeks.
    } else {
      reader.timeRange = CMTimeRange(
        start: .zero, duration: CMTime(value: request.durationValue, timescale: request.timescale))
    }
    let settings: [String: Int] = [
      AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: 48000,
      AVNumberOfChannelsKey: 2, AVLinearPCMBitDepthKey: 16,
      AVLinearPCMIsFloatKey: 0, AVLinearPCMIsBigEndianKey: 0, AVLinearPCMIsNonInterleaved: 0,
    ]
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
    guard reader.canAdd(output) else { throw ClockProbeError.unsupportedAudio }
    reader.add(output)
    var writers: [AVAssetWriter] = []
    var boundaries: [any StatefulWriting] = []
    var scratchPaths: [String] = []
    defer {
      if reader.status == .reading { reader.cancelReading() }
      for boundary in boundaries where boundary.status == .writing { boundary.cancel() }
      for writer in writers where writer.status == .writing { writer.cancelWriting() }
    }
    for index in selection.checkpoints.indices {
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
      boundaries.append(writerBoundary(writer, input))
      guard writer.startWriting() else { throw ClockProbeError.writerFailed }
      writer.startSession(atSourceTime: selection.checkpoints[index].start)
    }
    let ledgerURL = directory.appendingPathComponent("ledger.jsonl")
    try Data().write(to: ledgerURL, options: .withoutOverwriting)
    let ledger = try FileHandle(forWritingTo: ledgerURL)
    defer { ledger.closeFile() }
    var digest = SHA256()
    var ledgerBytes = 0
    var selector = SourcePTSSelector(selection.checkpoints)
    guard reader.startReading() else { throw ClockProbeError.readerFailed }
    let initializationSeconds = seconds(begin)
    var decoderSeconds = 0.0
    var sampleCopySeconds = 0.0
    while true {
      try Task.checkCancellation()
      guard ContinuousClock.now < deadline else { throw ClockProbeError.budgetExceeded }
      let decodeStart = ContinuousClock.now
      let sample = output.copyNextSampleBuffer()
      decoderSeconds += seconds(decodeStart)
      guard let sample else { break }
      let pts = CMSampleBufferGetPresentationTimeStamp(sample)
      let duration = CMSampleBufferGetDuration(sample)
      let count = CMSampleBufferGetNumSamples(sample)
      let record = StatefulRecord(
        bufferOrdinal: selector.records, sampleOrdinal: selector.ordinal,
        sourcePTS: ProbeTime(pts),
        duration: ProbeTime(duration), samples: count,
        discontinuity: ProbeTime(pts - selector.previousEnd))
      var data = try JSONEncoder().encode(record)
      data.append(10)
      guard selector.records < selection.budget.maxRecords,
        count <= selection.budget.maxDecodedSamples - selector.ordinal,
        data.count <= selection.budget.maxLedgerBytes - ledgerBytes
      else { throw ClockProbeError.budgetExceeded }
      try ledger.write(contentsOf: data)
      digest.update(data: data)
      ledgerBytes += data.count
      let ranges = try selector.accept(pts: pts, duration: duration, samples: count)
      let copyStart = ContinuousClock.now
      for index in ranges.indices where !ranges[index].isEmpty {
        let part = try ContinuousClockProbe.slice(sample, range: ranges[index])
        while !boundaries[index].isReady {
          try Task.checkCancellation()
          guard boundaries[index].status == .writing, ContinuousClock.now < deadline else {
            throw ClockProbeError.budgetExceeded
          }
          try await Task.sleep(for: .milliseconds(1))
        }
        guard boundaries[index].append(part) else { throw ClockProbeError.writerFailed }
      }
      sampleCopySeconds += seconds(copyStart)
      try checkStorage(directory, budget: selection.budget)
      try await progress(selector.records)
      try Task.checkCancellation()
    }
    guard reader.status == .completed else { throw ClockProbeError.readerFailed }
    if material {
      guard selector.records > 0, selector.retained == [selector.ordinal],
        selector.previousEnd == CMTime(value: Int64(selector.ordinal), timescale: 48000)
      else { throw ClockProbeError.readerFailed }
    } else {
      try selector.finish()
    }
    let finalizeStart = ContinuousClock.now
    for index in writers.indices {
      try Task.checkCancellation()
      await boundaries[index].finish()
      guard boundaries[index].status == .completed else { throw ClockProbeError.writerFailed }
    }
    try Task.checkCancellation()
    guard ContinuousClock.now < deadline else { throw ClockProbeError.budgetExceeded }
    try ledger.synchronize()
    try checkStorage(directory, budget: selection.budget)
    return StatefulResult(
      version: 1, origin: ProbeTime(.zero), selection: selection,
      decodedSamples: selector.ordinal,
      records: selector.records, ledgerBytes: ledgerBytes,
      ledgerSHA256: digest.finalize().map { String(format: "%02x", $0) }.joined(),
      initializationSeconds: initializationSeconds, decoderSeconds: decoderSeconds,
      sampleCopySeconds: sampleCopySeconds, finalizeSeconds: seconds(finalizeStart),
      writerTemporaryDirectories: scratchPaths,
      supportedEnd: material ? ProbeTime(selector.previousEnd) : nil,
      assetDuration: material ? ProbeTime(assetDuration) : nil)
  }
}
