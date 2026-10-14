import Foundation
import Testing

@testable import AppleProApps
@testable import ProAppsCore

struct ReferenceServiceTests {
  @Test func localCLIValidatesAndWritesAReferenceNotARender() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "reference-cli-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().appendingPathComponent(
        "ProAppsNativeTests/Fixtures/quadrants-30.mp4")
    let source = root.appendingPathComponent("source.mp4")
    try FileManager.default.copyItem(at: fixture, to: source)
    let request = ReferenceMovieRequest(
      rootDirectory: root.path, outputPath: root.appendingPathComponent("linked.mov").path,
      timescale: 90000,
      segments: [
        ReferenceSegment(
          sourcePath: source.path, sourceStart: .init(value: 0, timescale: 30),
          duration: .init(value: 15, timescale: 30), outputStart: .init(value: 0, timescale: 30),
          layer: 0)
      ])
    let path = root.appendingPathComponent("request.json")
    try JSONEncoder().encode(request).write(to: path)
    let runtime = ProgramRuntime(
      binary: URL(fileURLWithPath: "/unused/native"), environment: [:], serve: {},
      inventory: { "[]" },
      ui: { _ in }, execute: { _, _, _ in "" })
    let result = try #require(try await Program(runtime: runtime).run(.referenceMedia(path.path)))
    struct Report: Decodable {
      let selfContained: Bool
      let rendered: Bool
      let videoTrackCount: Int
    }
    let report = try JSONDecoder().decode(Report.self, from: Data(result.utf8))
    #expect(!report.selfContained && !report.rendered)
    #expect(report.videoTrackCount == 1)
    #expect(FileManager.default.fileExists(atPath: request.outputPath))
  }

  @Test(arguments: ["{}", "{\"unknown\":true}", "[]", "{not json}"])
  func refusesInvalidJSONBeforeWriting(_ json: String) async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "reference-invalid-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let path = root.appendingPathComponent("request.json")
    try Data(json.utf8).write(to: path)
    await #expect(throws: (any Error).self) {
      try await NativeService().referenceLocalFile(path.path)
    }
    #expect(try FileManager.default.contentsOfDirectory(atPath: root.path) == ["request.json"])
  }
}
