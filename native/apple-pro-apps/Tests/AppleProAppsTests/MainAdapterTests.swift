import Foundation
import ProAppsCore
import Testing

struct MainAdapterTests {
  @Test(arguments: [
    "capabilities", "serve", "invalid", "setup", "ui-exec-failure", "inspect-media", "edit-media",
    "ui-native",
  ])
  func processAdaptersAreBoundedAndKeepStdioClean(_ mode: String) async throws {
    let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let products = Bundle(for: TestBundle.self).bundleURL.deletingLastPathComponent()
    let binary = products.appendingPathComponent("apple-pro-apps")
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "pro-apps-main-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: directory.appendingPathComponent("scripts"), withIntermediateDirectories: true)
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    // This synthetic wrapper is /usr/bin/true, never the real Executor.
    try FileManager.default.createSymbolicLink(
      at: directory.appendingPathComponent("scripts/executor"),
      withDestinationURL: URL(fileURLWithPath: "/usr/bin/true"))
    let noExecutableFormat = directory.appendingPathComponent("not-peekaboo")
    #expect(
      FileManager.default.createFile(
        atPath: noExecutableFormat.path, contents: Data(), attributes: [.posixPermissions: 0o700]))
    let arguments: [String]
    switch mode {
    case "edit-media":
      let source = package.appendingPathComponent("Tests/ProAppsNativeTests/Fixtures/black.mp4")
        .path
      let request = EditRequest(
        recipe: .init(
          clips: [
            .init(
              sourcePath: source, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
          ],
          video: .init(
            width: 320, height: 240, frameRate: 30, resizeMode: .fit,
            titles: [.init(text: "CLI", x: 16, y: 16, fontSize: 24)])),
        outputDirectory: directory.path, outputName: "edited.mp4")
      let config = try Files.writeNew(
        JSONEncoder().encode(request), to: directory.appendingPathComponent("request.json").path,
        extensions: ["json"])
      arguments = [mode, config.path]
    case "inspect-media":
      arguments = [
        mode, package.appendingPathComponent("Tests/ProAppsNativeTests/Fixtures/black.mp4").path,
      ]
    case "ui-native": arguments = [mode, #"{"operation":"inputSourceStatus"}"#]
    case "invalid": arguments = []
    case "setup": arguments = ["setup", "--repo", directory.path]
    case "ui-exec-failure":
      arguments = ["serve-ui", "--repo", directory.path, "--peekaboo", noExecutableFormat.path]
    default: arguments = [mode]
    }
    var environment = ProcessInfo.processInfo.environment
    environment["LLVM_PROFILE_FILE"] =
      products.appendingPathComponent("codecov/main-%p-%m.profraw").path
    let result = try await Runner.run(
      binary, arguments, environment: environment, timeout: .seconds(10))
    #expect(
      result.status
        == ((mode == "serve" || mode == "capabilities" || mode == "inspect-media"
          || mode == "edit-media" || mode == "ui-native") ? 0 : 1))
    if mode == "edit-media" {
      let summary = try JSONDecoder().decode(EditRenderResult.self, from: Data(result.stdout.utf8))
      #expect(summary.videoTrackCount == 1)
      #expect(abs(summary.actualDurationSeconds - 1) < 0.04)
      #expect(FileManager.default.fileExists(atPath: summary.projectPath))
      let samples = try await FrameProbe().measure(
        path: summary.outputPath,
        samples: [.init(timeSeconds: 0.25, region: .init(x: 8, y: 8, width: 96, height: 64))])
      let visible = try #require(samples.first)
      #expect(visible.meanRed > 0.02)
    } else if mode == "inspect-media" {
      let summary = try JSONDecoder().decode(MediaSummary.self, from: Data(result.stdout.utf8))
      #expect(summary.firstFrameDecoded)
      #expect(summary.width == 16)
      #expect(summary.height == 16)
    } else if mode == "ui-native" {
      let outcome = try JSONDecoder().decode(UIChildOutcome.self, from: Data(result.stdout.utf8))
      #expect(outcome.error == nil)
      #expect(outcome.response?.inputSource?.id.isEmpty == false)
    } else if mode == "capabilities" {
      let decoded = try JSONDecoder().decode([InstalledApp].self, from: Data(result.stdout.utf8))
      #expect(decoded.count <= 10)
    } else {
      #expect(result.stdout.isEmpty)
    }
  }
}
