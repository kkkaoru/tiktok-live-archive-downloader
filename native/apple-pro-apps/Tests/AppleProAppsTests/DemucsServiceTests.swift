import Foundation
import MCP
import Testing

@testable import AppleProApps
@testable import ProAppsCore

struct DemucsServiceTests {
  @Test func nativeSeparationWritesOnlyANewBoundedStereoOutput() async throws {
    let model = try #require(ProcessInfo.processInfo.environment["DEMUCS_TEST_MODEL"])
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "demucs-service-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let original = try CueSound(durationSeconds: 0.1, onsetSeconds: [0], gain: 0.1).wave()
    let wrongRate = root.appendingPathComponent("wrong-rate.wav")
    try original.write(to: wrongRate)
    var wave = original
    wave.replaceSubrange(24..<28, with: [68, 172, 0, 0])
    wave.replaceSubrange(28..<32, with: [136, 88, 1, 0])
    let source = root.appendingPathComponent("channel.wav")
    try wave.write(to: source)
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      #expect(name == "audio_separate_vocals")
      let text = try await NativeService().measureLocalFile(name: name, path: path)
      return try JSONDecoder().decode(Value.self, from: Data(text.utf8))
    }
    let args: [String: Value] = [
      "leftPath": .string(source.path), "rightPath": .string(source.path),
      "compiledModelPath": .string(model), "outputDirectory": .string(root.path),
      "outputName": .string("voice.wav"),
    ]
    let result = await NativeService(interfaces: interfaces).call(
      .init(name: "audio_separate_vocals", arguments: args))
    #expect(result.isError == false)
    let value = try #require(result.structuredContent?.objectValue?["measurement"])
    let output = try JSONDecoder().decode(DemucsResult.self, from: JSONEncoder().encode(value))
    #expect(output.sampleRate == 44100)
    #expect(output.sampleCount == 1600)
    #expect(output.peak.isFinite)
    #expect(!output.humanReviewed)
    let saved = try Data(contentsOf: URL(fileURLWithPath: output.outputPath))
    #expect(saved.count == 12_844)
    #expect(Array(saved[20..<24]) == [3, 0, 2, 0])
    #expect(try Data(contentsOf: source) == wave)
    let invalidExtension = DemucsRequest(
      leftPath: source.path, rightPath: source.path,
      compiledModelPath: model, outputDirectory: root.path, outputName: "bad.mp4")
    await #expect(throws: ProAppsError.self) {
      try await DemucsSeparator().separate(invalidExtension)
    }
    let invalidRate = DemucsRequest(
      leftPath: wrongRate.path, rightPath: source.path,
      compiledModelPath: model, outputDirectory: root.path, outputName: "bad.wav")
    await #expect(throws: ProAppsError.self) { try await DemucsSeparator().separate(invalidRate) }
    let failedRoot = root.appendingPathComponent("failed-publication")
    try FileManager.default.createDirectory(at: failedRoot, withIntermediateDirectories: false)
    let failing = DemucsSeparator { data, path in
      if URL(fileURLWithPath: path).pathExtension == "wav" {
        throw ProAppsError.unavailable("Synthetic WAV publication failure")
      }
      return try Files.writeNew(data, to: path, extensions: ["json"])
    }
    let failedRequest = DemucsRequest(
      leftPath: source.path, rightPath: source.path,
      compiledModelPath: model, outputDirectory: failedRoot.path, outputName: "failed.wav")
    await #expect(throws: ProAppsError.self) { try await failing.separate(failedRequest) }
    #expect(try FileManager.default.contentsOfDirectory(atPath: failedRoot.path).isEmpty)
  }
}
