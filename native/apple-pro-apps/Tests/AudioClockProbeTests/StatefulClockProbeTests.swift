import AVFoundation
import Foundation
import Testing

@testable import audio_clock_probe

@Test func sourcePTSPartialBuffersAndDistinctOrdinals() throws {
  var selector = SourcePTSSelector([
    SourceCheckpoint(startValue: 3, timescale: 48000, sampleCount: 17)
  ])
  let first = try selector.accept(
    pts: .zero, duration: CMTime(value: 10, timescale: 48000), samples: 10)
  #expect(first == [3..<10])
  let second = try selector.accept(
    pts: CMTime(value: 10, timescale: 48000),
    duration: CMTime(value: 20, timescale: 48000), samples: 20)
  #expect(second == [0..<10])
  #expect(selector.ordinal == 30)
  #expect(selector.records == 2)
  #expect(selector.retained == [17])
  try selector.finish()
}

@Test func initialEditDoesNotBecomeOrdinalOffset() throws {
  var selector = SourcePTSSelector([
    SourceCheckpoint(startValue: 48001, timescale: 48000, sampleCount: 17)
  ])
  let selected = try selector.accept(
    pts: CMTime(value: 1, timescale: 1),
    duration: CMTime(value: 2048, timescale: 48000), samples: 2048)
  #expect(selected == [1..<18])
  #expect(selector.ordinal == 2048)
  try selector.finish()
}

@Test(arguments: [1, -1]) func gapAndOverlapRejectAmbiguousSupport(delta: Int64) throws {
  var selector = SourcePTSSelector([
    SourceCheckpoint(startValue: 0, timescale: 48000, sampleCount: 17)
  ])
  _ = try selector.accept(
    pts: .zero, duration: CMTime(value: 10, timescale: 48000), samples: 10)
  #expect(throws: ClockProbeError.self) {
    try selector.accept(
      pts: CMTime(value: 10 + delta, timescale: 48000),
      duration: CMTime(value: 10, timescale: 48000), samples: 10)
  }
}

@Test func unknownExtentOffGridEmptyAndEarlyEOFReject() throws {
  var selector = SourcePTSSelector([
    SourceCheckpoint(startValue: 1, timescale: 96000, sampleCount: 17)
  ])
  #expect(throws: ClockProbeError.self) { try selector.finish() }
  #expect(throws: ClockProbeError.self) {
    try selector.accept(pts: .zero, duration: .invalid, samples: 2048)
  }
  #expect(throws: ClockProbeError.self) {
    try selector.accept(pts: .zero, duration: CMTime(value: 2048, timescale: 48000), samples: 2048)
  }
  var early = SourcePTSSelector([
    SourceCheckpoint(startValue: 100, timescale: 48000, sampleCount: 17)
  ])
  #expect(
    try early.accept(pts: .zero, duration: CMTime(value: 10, timescale: 48000), samples: 10) == [
      0..<0
    ])
  #expect(throws: ClockProbeError.self) { try early.finish() }
}

@Test(arguments: [
  "success", "records", "ledger", "samples", "wall", "disk", "free", "eof", "writer", "cancel",
])
func statefulNativeLifecycle(mode: String) async throws {
  let root = try testRoot()
  let work = root.appendingPathComponent(
    "edits/work/audio-clock-repair/stateful-test-" + UUID().uuidString)
  try FileManager.default.createDirectory(at: work, withIntermediateDirectories: false)
  let source = work.appendingPathComponent("source.wav")
  try writeWave(source, frames: mode == "records" ? 48000 : 4800)
  let directory = work.appendingPathComponent("result")
  let budget = StatefulBudget(
    maxRecords: mode == "records" ? 1 : 100,
    maxLedgerBytes: mode == "ledger" ? 1 : 100000,
    maxSavedBytes: mode == "disk" ? 1 : 1_000_000,
    maxDecodedSamples: mode == "samples" ? 1 : 48000,
    maxWallSeconds: mode == "wall" ? 0.000001 : 10,
    minimumFreeBytes: mode == "free" ? Int64.max : 0)
  let selection = StatefulSelection(
    checkpoints: [
      SourceCheckpoint(startValue: 1, timescale: 48000, sampleCount: 17),
      SourceCheckpoint(startValue: 2345, timescale: 48000, sampleCount: 17),
      SourceCheckpoint(startValue: mode == "eof" ? 5000 : 4700, timescale: 48000, sampleCount: 17),
    ], budget: budget)
  var request = ClockProbeRequest(
    source: source.path, outputDirectory: directory.path,
    startValue: 0, durationValue: 1, timescale: 1)
  request.statefulSelection = selection
  if mode == "success" {
    try await AudioClockProbe.run(request, root: root)
    let result = try JSONDecoder().decode(
      StatefulResult.self, from: Data(contentsOf: directory.appendingPathComponent("stateful.json"))
    )
    #expect(result.decodedSamples == 4800)
    #expect(result.records == 1)
    #expect(result.ledgerSHA256.count == 64)
    #expect(result.writerTemporaryDirectories.count == 3)
    return
  }
  if mode == "writer" {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    try Data([1]).write(to: directory.appendingPathComponent("checkpoint-0.mov"))
    await #expect(throws: (any Error).self) {
      try await StatefulClockProbe.read(
        request, selection: selection, directory: directory, root: root)
    }
  } else if mode == "cancel" {
    let immutableRequest = request
    await #expect(throws: CancellationError.self) {
      try await withThrowingTaskGroup(of: Void.self) { group in
        group.cancelAll()
        group.addTask { try await AudioClockProbe.run(immutableRequest, root: root) }
        try await group.waitForAll()
      }
    }
  } else {
    await #expect(throws: ClockProbeError.self) {
      try await AudioClockProbe.run(request, root: root)
    }
  }
  #expect(
    !FileManager.default.fileExists(atPath: directory.appendingPathComponent("stateful.json").path))
}

@Test func representativeCheckpointPreservesWholeDuration() async throws {
  let root = try testRoot()
  let work = root.appendingPathComponent(
    "edits/work/audio-clock-repair/stateful-representative-" + UUID().uuidString)
  try FileManager.default.createDirectory(at: work, withIntermediateDirectories: false)
  let source = work.appendingPathComponent("source.wav")
  try writeWave(source, frames: 96000)
  let directory = work.appendingPathComponent("result")
  let budget = StatefulBudget(
    maxRecords: 100, maxLedgerBytes: 100000, maxSavedBytes: 1_000_000,
    maxDecodedSamples: 96000, maxWallSeconds: 10, minimumFreeBytes: 0)
  var request = ClockProbeRequest(
    source: source.path, outputDirectory: directory.path,
    startValue: 0, durationValue: 2, timescale: 1)
  request.statefulSelection = StatefulSelection(
    checkpoints: [SourceCheckpoint(startValue: 0, timescale: 1, sampleCount: 96000)],
    budget: budget)
  try await AudioClockProbe.run(request, root: root)
  let result = try JSONDecoder().decode(
    StatefulResult.self, from: Data(contentsOf: directory.appendingPathComponent("stateful.json")))
  #expect(result.decodedSamples == 96000)
  let movie = AVURLAsset(url: directory.appendingPathComponent("checkpoint-0.mov"))
  let duration = try await movie.load(.duration)
  #expect(duration == CMTime(value: 2, timescale: 1))
}

@Test func representativeCheckpointLimit() throws {
  let budget = StatefulBudget(
    maxRecords: 1000, maxLedgerBytes: 1_000_000, maxSavedBytes: 16_000_000,
    maxDecodedSamples: 2_928_000, maxWallSeconds: 60, minimumFreeBytes: 0)
  let request = ClockProbeRequest(
    source: "/unused", outputDirectory: "/unused", startValue: 0, durationValue: 61, timescale: 1)
  try StatefulClockProbe.validate(
    StatefulSelection(
      checkpoints: [SourceCheckpoint(startValue: 0, timescale: 1, sampleCount: 2_880_000)],
      budget: budget), request: request)
  #expect(throws: ClockProbeError.self) {
    try StatefulClockProbe.validate(
      StatefulSelection(
        checkpoints: [SourceCheckpoint(startValue: 0, timescale: 1, sampleCount: 2_880_001)],
        budget: budget), request: request)
  }
}

@Test func productionMaterialNaturalEOF() async throws {
  let root = try testRoot()
  let work = root.appendingPathComponent(
    "edits/work/audio-clock-repair/material-" + UUID().uuidString)
  try FileManager.default.createDirectory(at: work, withIntermediateDirectories: false)
  let source = work.appendingPathComponent("source.wav")
  try writeWave(source, frames: 96000)
  var request = ClockProbeRequest(
    source: source.path, outputDirectory: work.appendingPathComponent("output").path,
    startValue: 0, durationValue: 120, timescale: 1)
  let budget = StatefulBudget(
    maxRecords: 1000, maxLedgerBytes: 1_000_000, maxSavedBytes: 1_000_000_000,
    maxDecodedSamples: 5_760_000, maxWallSeconds: 30, minimumFreeBytes: 0)
  request.statefulSelection = StatefulSelection(
    checkpoints: [SourceCheckpoint(startValue: 0, timescale: 1, sampleCount: 5_760_000)],
    budget: budget, productionMaterial: true)
  try await AudioClockProbe.run(request, root: root)
  let directory = URL(fileURLWithPath: request.outputDirectory)
  let result = try JSONDecoder().decode(
    StatefulResult.self, from: Data(contentsOf: directory.appendingPathComponent("stateful.json")))
  #expect(result.decodedSamples == 96000)
  #expect(result.supportedEnd?.value == 96000)
  #expect(result.supportedEnd?.timescale == 48000)
  let movie = AVURLAsset(url: directory.appendingPathComponent("checkpoint-0.mov"))
  #expect(try await movie.load(.duration) == CMTime(value: 2, timescale: 1))
  let invalid = StatefulSelection(
    checkpoints: [SourceCheckpoint(startValue: 1, timescale: 1, sampleCount: 5_760_000)],
    budget: budget, productionMaterial: true)
  #expect(throws: ClockProbeError.self) {
    try StatefulClockProbe.validate(invalid, request: request)
  }
  let diagnostic = StatefulSelection(
    checkpoints: [SourceCheckpoint(startValue: 0, timescale: 1, sampleCount: 5_760_000)],
    budget: budget)
  #expect(throws: ClockProbeError.self) {
    try StatefulClockProbe.validate(diagnostic, request: request)
  }
}

@Test func statefulBudgetValidation() throws {
  let budget = StatefulBudget(
    maxRecords: 10, maxLedgerBytes: 10000, maxSavedBytes: 1_000_000,
    maxDecodedSamples: 48000, maxWallSeconds: 10, minimumFreeBytes: 0)
  let selection = StatefulSelection(
    checkpoints: [SourceCheckpoint(startValue: 1, timescale: 48000, sampleCount: 17)],
    budget: budget)
  var request = ClockProbeRequest(
    source: "/unused", outputDirectory: "/unused", startValue: 0, durationValue: 1, timescale: 1)
  try StatefulClockProbe.validate(selection, request: request)
  request.continuousCheckpoints = [PCMCheckpoint(startSample: 1, sampleCount: 17)]
  #expect(throws: ClockProbeError.self) {
    try StatefulClockProbe.validate(selection, request: request)
  }
  request.continuousCheckpoints = nil
  #expect(throws: ClockProbeError.self) {
    try StatefulClockProbe.validate(
      StatefulSelection(
        checkpoints: [SourceCheckpoint(startValue: 48000, timescale: 48000, sampleCount: 17)],
        budget: budget), request: request)
  }
}
