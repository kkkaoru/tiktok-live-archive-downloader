import Foundation
import MCP
import Testing

@testable import AppleProApps
@testable import ProAppsCore

struct ReferenceAudioServiceTests {
  @Test func decodesLocalWAVAndUsesMCPAdapterWithoutSourceWrites() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "reference-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let wave = try CueSound(durationSeconds: 0.1, onsetSeconds: [0], gain: 0.2).wave()
    let file = root.appendingPathComponent("reference.wav")
    try wave.write(to: file)
    let decoded = try PCMMeasurement.decodedSamples(wave)
    #expect(decoded.sampleRate == 16000)
    #expect(decoded.samples.count == 1600)
    #expect(decoded.samples[0] == 0)
    #expect(decoded.samples[1599] == 0)
    let request: Value = .object([
      "sourcePath": .string(file.path), "referencePath": .string(file.path),
      "search": .object([
        "referenceStartSample": .int(0), "searchRadiusSamples": .int(0),
        "minimumCorrelation": .double(0.99),
      ]),
    ])
    let input = root.appendingPathComponent("request.json")
    try JSONEncoder().encode(request).write(to: input)
    let raw = try await NativeService().measureLocalFile(
      name: "audio_reference_analyze", path: input.path)
    let match = try JSONDecoder().decode(ReferenceAudioMatch.self, from: Data(raw.utf8))
    #expect(match.gain == 1)
    #expect(match.correlation == 1)
    #expect(match.residualRMS == 0)
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      #expect(name == "audio_reference_analyze")
      let text = try await NativeService().measureLocalFile(name: name, path: path)
      return try JSONDecoder().decode(Value.self, from: Data(text.utf8))
    }
    let args = try #require(request.objectValue)
    let response = await NativeService(interfaces: interfaces).call(
      .init(name: "audio_reference_analyze", arguments: args))
    #expect(response.isError == false)
    #expect(response.structuredContent?.objectValue?["sourceModified"] == .bool(false))
    #expect(try Data(contentsOf: file) == wave)
  }

  @Test func mismatchedRatesAndMalformedWAVAreRejected() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "reference-rates-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let wave = try CueSound(durationSeconds: 0.1, onsetSeconds: [0], gain: 0.2).wave()
    let first = root.appendingPathComponent("first.wav")
    let second = root.appendingPathComponent("second.wav")
    try wave.write(to: first)
    var otherRate = wave
    otherRate[24] = 64
    otherRate[25] = 31
    otherRate[28] = 128
    otherRate[29] = 62
    try otherRate.write(to: second)
    await #expect(throws: ProAppsError.self) {
      try await ReferenceAudioProbe().analyze(
        sourcePath: first.path, referencePath: second.path,
        search: .init(referenceStartSample: 0, searchRadiusSamples: 0, minimumCorrelation: 0.5))
    }
    #expect(throws: ProAppsError.self) { try PCMMeasurement.decodedSamples(Data([0, 1, 2])) }
  }
}
