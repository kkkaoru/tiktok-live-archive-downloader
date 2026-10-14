import Foundation
import Testing

@testable import AppleProApps
@testable import ProAppsCore

struct MoviePCMMuxServiceTests {
  @Test func typedCLIAppendsPCMAndRejectsUnexpectedFields() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent("mux-cli-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().appendingPathComponent(
        "ProAppsNativeTests/Fixtures/quadrants-30.mp4")
    let video = root.appendingPathComponent("source.mp4")
    try FileManager.default.copyItem(at: fixture, to: video)
    let audio = root.appendingPathComponent("audio.wav")
    try CueSound(durationSeconds: 1, onsetSeconds: [0.1], gain: 0.1).wave().write(to: audio)
    let request = MoviePCMMuxRequest(
      rootDirectory: root.path, videoPath: video.path, audioPath: audio.path,
      outputDirectory: root.path, outputName: "final.mov")
    let path = root.appendingPathComponent("request.json")
    try JSONEncoder().encode(request).write(to: path)
    let runtime = ProgramRuntime(
      binary: URL(fileURLWithPath: "/unused/native"), environment: [:], serve: {},
      inventory: { "[]" },
      ui: { _ in }, execute: { _, _, _ in "" })
    let result = try #require(
      try await Program(runtime: runtime).run(Command.parse(["mux-pcm", path.path])))
    struct Report: Decodable { let outputPath: String }
    let report = try JSONDecoder().decode(Report.self, from: Data(result.utf8))
    #expect(FileManager.default.fileExists(atPath: report.outputPath))
    let invalid = root.appendingPathComponent("invalid.json")
    try Data("{\"unexpected\":true}".utf8).write(to: invalid)
    await #expect(throws: (any Error).self) {
      try await NativeService().muxPCMLocalFile(invalid.path)
    }
    #expect(throws: (any Error).self) { try Command.parse(["mux-pcm"]) }
    #expect(throws: (any Error).self) { try Command.parse(["mux-pcm", "relative.json"]) }
    #expect(throws: (any Error).self) { try Command.parse(["mux-pcm", path.path, "extra"]) }
  }
}
