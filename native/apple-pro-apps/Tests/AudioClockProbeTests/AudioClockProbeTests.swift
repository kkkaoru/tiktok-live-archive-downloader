import AVFoundation
import Foundation
import Testing

@testable import audio_clock_probe

func testRoot() throws -> URL {
  let value = try #require(ProcessInfo.processInfo.environment["AUDIO_CLOCK_TEST_ROOT"])
  return URL(fileURLWithPath: value)
}

func writeWave(_ path: URL, frames: Int = 4800) throws {
  let samples: [Int16] = (0..<(frames * 2)).map { Int16(($0 % 97) * 100) }
  let payload = samples.flatMap { value in
    let bits = UInt16(bitPattern: value)
    return [UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)]
  }
  func little(_ value: UInt32) -> [UInt8] {
    [0, 8, 16, 24].map { UInt8(truncatingIfNeeded: value >> $0) }
  }
  var data = Data("RIFF".utf8)
  data.append(contentsOf: little(UInt32(payload.count + 36)))
  data.append(Data("WAVEfmt ".utf8))
  data.append(contentsOf: little(16))
  data.append(contentsOf: [1, 0, 2, 0])
  data.append(contentsOf: little(48000))
  data.append(contentsOf: little(192000))
  data.append(contentsOf: [4, 0, 16, 0])
  data.append(Data("data".utf8))
  data.append(contentsOf: little(UInt32(payload.count)))
  data.append(contentsOf: payload)
  try data.write(to: path, options: .withoutOverwriting)
}

@Test func actualReadersRetainPCMClockAndRefuseOverwrite() async throws {
  let root = try testRoot()
  let work = root.appendingPathComponent(
    "edits/work/audio-clock-repair/native-tests-" + UUID().uuidString)
  try FileManager.default.createDirectory(at: work, withIntermediateDirectories: false)
  let source = work.appendingPathComponent("source.wav")
  try writeWave(source)
  let request = ClockProbeRequest(
    source: source.path, outputDirectory: work.appendingPathComponent("result").path,
    startValue: 0, durationValue: 1, timescale: 10)
  let requestFile = work.appendingPathComponent("request.json")
  try JSONEncoder().encode(request).write(to: requestFile, options: .withoutOverwriting)
  try await AudioClockProbeCommand.execute(arguments: ["probe", requestFile.path], root: root)
  let data = try Data(contentsOf: work.appendingPathComponent("result/audio-mix.json"))
  let result = try JSONDecoder().decode(ClockProbeResult.self, from: data)
  #expect(result.mode == "audio-mix")
  #expect(
    result.writerTemporaryDirectory == work.appendingPathComponent("result/audio-mix-scratch").path)
  #expect(result.buffers.reduce(0) { $0 + $1.samples } == 4800)
  #expect(result.buffers.first?.value == 0)
  await #expect(throws: ClockProbeError.self) {
    try await AudioClockProbe.run(request, root: root)
  }
}

@Test func emptyRequestedRangeFailsRatherThanInventingSamples() async throws {
  let root = try testRoot()
  let work = root.appendingPathComponent(
    "edits/work/audio-clock-repair/empty-range-test-" + UUID().uuidString)
  try FileManager.default.createDirectory(at: work, withIntermediateDirectories: false)
  let source = work.appendingPathComponent("source.wav")
  try writeWave(source)
  let request = ClockProbeRequest(
    source: source.path, outputDirectory: work.appendingPathComponent("result").path,
    startValue: 1, durationValue: 1, timescale: 1)
  await #expect(throws: ClockProbeError.self) {
    try await AudioClockProbe.run(request, root: root)
  }
  #expect(
    !FileManager.default.fileExists(atPath: work.appendingPathComponent("result/track.json").path))
}

@Test func commandRejectsInvalidArgumentsAndOversizedJSON() async throws {
  let root = try testRoot()
  await #expect(throws: ClockProbeError.self) {
    try await AudioClockProbeCommand.execute(arguments: [], root: root)
  }
  let path = root.appendingPathComponent(
    "edits/work/audio-clock-repair/oversized-" + UUID().uuidString)
  try Data(repeating: 32, count: 65537).write(to: path, options: .withoutOverwriting)
  await #expect(throws: ClockProbeError.self) {
    try await AudioClockProbeCommand.execute(arguments: ["probe", path.path], root: root)
  }
}

@Test func rejectsOutOfRepositoryOutput() throws {
  let request = ClockProbeRequest(
    source: "/missing", outputDirectory: "/outside", startValue: 0,
    durationValue: 1, timescale: 1)
  #expect(throws: ClockProbeError.self) {
    try AudioClockProbe.validate(request, root: URL(fileURLWithPath: "/repository"))
  }
}

@Test func explicitWriterScratchReadbackAndEscapingSymlinks() throws {
  let root = try testRoot()
  let work = root.appendingPathComponent(
    "edits/work/audio-clock-repair/scratch-test-" + UUID().uuidString)
  try FileManager.default.createDirectory(at: work, withIntermediateDirectories: false)
  let writer = try AVAssetWriter(
    outputURL: work.appendingPathComponent("unused.mov"), fileType: .mov)
  let actual = try AudioClockProbe.configureScratch(writer, directory: work, root: root)
  #expect(actual.path == work.path)
  #expect(writer.directoryForTemporaryFiles?.path == work.path)
  let escape = work.appendingPathComponent("escape")
  // The target is inside the checkout but outside the permitted scratch subtree.
  try FileManager.default.createSymbolicLink(at: escape, withDestinationURL: root)
  #expect(throws: ClockProbeError.self) {
    try AudioClockProbe.configureScratch(writer, directory: escape, root: root)
  }
  #expect(throws: ClockProbeError.self) {
    try AudioClockProbe.validateScratch(escape.appendingPathComponent("edits/work"), root: root)
  }
  #expect(throws: ClockProbeError.self) {
    try AudioClockProbe.validateScratch(root, root: root)
  }
  let file = work.appendingPathComponent("not-directory")
  try Data([0]).write(to: file, options: .withoutOverwriting)
  #expect(throws: ClockProbeError.self) {
    try AudioClockProbe.validateScratch(file, root: root)
  }
  #expect(writer.directoryForTemporaryFiles?.path == work.path)
}

@Test func compressedIdentityAndNoAudioRejection() async throws {
  let root = try testRoot()
  let work = root.appendingPathComponent(
    "edits/work/audio-clock-repair/compressed-test-" + UUID().uuidString)
  try FileManager.default.createDirectory(at: work, withIntermediateDirectories: false)
  let source = work.appendingPathComponent("source.mov")
  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg")
  process.arguments = [
    "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=0.1",
    "-c:a", "aac", source.path,
  ]
  try process.run()
  process.waitUntilExit()
  #expect(process.terminationStatus == 0)
  var request = ClockProbeRequest(
    source: source.path, outputDirectory: work.appendingPathComponent("result").path,
    startValue: 0, durationValue: 1, timescale: 10)
  request.compressedOnly = true
  try await AudioClockProbe.run(request, root: root)
  let result = try JSONDecoder().decode(
    CompressedClockResult.self,
    from: Data(contentsOf: work.appendingPathComponent("result/compressed.json")))
  #expect(!result.edits.isEmpty)
  let packet = try #require(result.packets.first)
  #expect(packet.sha256.count == 64)
  #expect(packet.bytes > 0)
  #expect(
    !FileManager.default.fileExists(atPath: work.appendingPathComponent("result/track.mov").path))
  let handle = try FileHandle(forReadingFrom: source)
  defer { handle.closeFile() }
  #expect(throws: ClockProbeError.self) {
    try CompressedClockProbe.packetData(
      source: source, location: nil, range: AVSampleCursorStorageRange(offset: 0, length: 1),
      handle: handle)
  }
  #expect(throws: ClockProbeError.self) {
    try CompressedClockProbe.packetData(
      source: source, location: work, range: AVSampleCursorStorageRange(offset: 0, length: 1),
      handle: handle)
  }
  #expect(throws: ClockProbeError.self) {
    try CompressedClockProbe.packetData(
      source: source, location: source,
      range: AVSampleCursorStorageRange(offset: 100_000_000, length: 1), handle: handle)
  }
  let noAudio = work.appendingPathComponent("video.mov")
  let video = Process()
  video.executableURL = process.executableURL
  video.arguments = [
    "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i", "color=size=16x16:duration=0.1",
    "-an", "-c:v", "mpeg4", noAudio.path,
  ]
  try video.run()
  video.waitUntilExit()
  #expect(video.terminationStatus == 0)
  let noAudioRequest = ClockProbeRequest(
    source: noAudio.path, outputDirectory: work.appendingPathComponent("no-audio").path,
    startValue: 0, durationValue: 1, timescale: 10)
  await #expect(throws: ClockProbeError.self) {
    try await AudioClockProbe.run(noAudioRequest, root: root)
  }
  await #expect(throws: ClockProbeError.self) {
    try await CompressedClockProbe.read(noAudioRequest)
  }
}

@Test func continuousReaderKeepsTwoPartialCheckpointsAndExactDuration() async throws {
  let root = try testRoot()
  let work = root.appendingPathComponent(
    "edits/work/audio-clock-repair/continuous-test-" + UUID().uuidString)
  try FileManager.default.createDirectory(at: work, withIntermediateDirectories: false)
  let source = work.appendingPathComponent("source.wav")
  try writeWave(source)
  var request = ClockProbeRequest(
    source: source.path, outputDirectory: work.appendingPathComponent("result").path,
    startValue: 0, durationValue: 1, timescale: 10)
  request.continuousCheckpoints = [
    PCMCheckpoint(startSample: 1, sampleCount: 17),
    PCMCheckpoint(startSample: 4700, sampleCount: 17),
  ]
  try await AudioClockProbe.run(request, root: root)
  let result = try JSONDecoder().decode(
    ContinuousClockResult.self,
    from: Data(contentsOf: work.appendingPathComponent("result/continuous.json")))
  #expect(result.decodedSamples == 4800)
  #expect(result.origin.value == 0)
  #expect(result.checkpoints.count == 2)
  #expect(result.writerTemporaryDirectories.count == 2)
  #expect(result.elapsedSeconds > 0)
  for (index, start) in [1, 4700].enumerated() {
    let decoded = work.appendingPathComponent("decoded-\(index).pcm")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg")
    process.arguments = [
      "-v", "error", "-nostdin", "-n", "-i",
      work.appendingPathComponent("result/checkpoint-\(index).mov").path,
      "-f", "s16le", "-c:a", "pcm_s16le", decoded.path,
    ]
    try process.run()
    process.waitUntilExit()
    #expect(process.terminationStatus == 0)
    let original = try Data(contentsOf: source)
    let actual = try Data(contentsOf: decoded)
    #expect(actual == original.subdata(in: (44 + start * 4)..<(44 + (start + 17) * 4)))
  }
}

@Test func continuousContractRejectsRestartAndUnboundedSelection() throws {
  var request = ClockProbeRequest(
    source: "/unused", outputDirectory: "/unused", startValue: 0, durationValue: 1, timescale: 1)
  #expect(throws: ClockProbeError.self) { try ContinuousClockProbe.validate([], request: request) }
  #expect(throws: ClockProbeError.self) {
    try ContinuousClockProbe.validate(
      [PCMCheckpoint(startSample: 0, sampleCount: 4801)], request: request)
  }
  #expect(throws: ClockProbeError.self) {
    try ContinuousClockProbe.validate(
      [PCMCheckpoint(startSample: 47999, sampleCount: 2)], request: request)
  }
  #expect(throws: ClockProbeError.self) {
    try ContinuousClockProbe.validate(
      [
        PCMCheckpoint(startSample: 1, sampleCount: 17),
        PCMCheckpoint(startSample: 2, sampleCount: 17),
      ], request: request)
  }
  request.compressedOnly = true
  #expect(throws: ClockProbeError.self) {
    try ContinuousClockProbe.validate(
      [PCMCheckpoint(startSample: 0, sampleCount: 1)], request: request)
  }
  let restart = ClockProbeRequest(
    source: "/unused", outputDirectory: "/unused", startValue: 1, durationValue: 1, timescale: 1)
  #expect(throws: ClockProbeError.self) {
    try ContinuousClockProbe.validate(
      [PCMCheckpoint(startSample: 0, sampleCount: 1)], request: restart)
  }
  let checkpoint = PCMCheckpoint(startSample: 3, sampleCount: 5)
  #expect(ContinuousClockProbe.intersection(checkpoint, ordinal: 0, samples: 4) == 3..<4)
  #expect(ContinuousClockProbe.intersection(checkpoint, ordinal: 4, samples: 6) == 4..<8)
  #expect(ContinuousClockProbe.intersection(checkpoint, ordinal: 9, samples: 1).isEmpty)
}

@Test func retainsRationalBufferClock() throws {
  let buffer = ClockProbeBuffer(
    value: 1025, timescale: 48000, durationValue: 1024,
    durationTimescale: 48000, samples: 1024)
  let decoded = try JSONDecoder().decode(
    ClockProbeBuffer.self, from: JSONEncoder().encode(buffer))
  #expect(decoded.value == 1025)
  #expect(decoded.timescale == 48000)
  #expect(decoded.samples == 1024)
}
