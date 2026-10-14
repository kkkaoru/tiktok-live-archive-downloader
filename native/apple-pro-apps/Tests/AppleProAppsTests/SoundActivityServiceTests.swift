import Foundation
import MCP
import Testing

@testable import AppleProApps
@testable import ProAppsCore

struct SoundActivityServiceTests {
  @Test func routesSoundActivityThroughValidatedReadOnlyRequest() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "sound-service-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = root.appendingPathComponent("tone.wav")
    let wave = try CueSound(durationSeconds: 1, onsetSeconds: [0], gain: 0.2).wave()
    try wave.write(to: source)
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      #expect(name == "audio_sound_activity")
      let text = try await NativeService().measureLocalFile(name: name, path: path)
      return try JSONDecoder().decode(Value.self, from: Data(text.utf8))
    }
    let result = await NativeService(interfaces: interfaces).call(
      .init(name: "audio_sound_activity", arguments: ["path": .string(source.path)]))
    #expect(result.isError == false)
    #expect(result.structuredContent?.objectValue?["sourceModified"] == .bool(false))
    let report = try #require(result.structuredContent?.objectValue?["measurement"]?.objectValue)
    #expect(report["humanReviewed"] == .bool(false))
    #expect(report["windows"]?.arrayValue?.isEmpty == false)
    #expect(try Data(contentsOf: source) == wave)
    let invalid = await NativeService(interfaces: interfaces).call(
      .init(
        name: "audio_sound_activity",
        arguments: ["path": .string(source.path), "upload": .bool(true)]))
    #expect(invalid.isError == true)
  }
}
