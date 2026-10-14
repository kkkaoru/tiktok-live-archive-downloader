import Foundation
import Testing

@testable import ProAppsCore

struct CompressorPathTests {
  @Test(arguments: [
    "/tmp/Motion Creator Studio/Template Name.motn",
    "/tmp/動画 編集 #1 & 100%.motn",
    "/tmp/literal%20space ; $(not-a-command).mov",
  ])
  func submitsRegularFilesAsLiteralPaths(path: String) throws {
    let arguments = try Compressor.submissionArguments(
      source: URL(fileURLWithPath: path),
      preset: URL(fileURLWithPath: "/tmp/Apple Preset.setting"),
      output: URL(fileURLWithPath: "/tmp/新規 出力.mp4"), batchName: "native proof")
    #expect(
      arguments == [
        "-batchname", "native proof", "-jobpath", path,
        "-settingpath", "/tmp/Apple Preset.setting",
        "-locationpath", "/tmp/新規 出力.mp4", "-outputformat", "json",
      ])
  }

  @Test(arguments: [0, 1, 2])
  func rejectsNonFileURLsAtEverySubmissionBoundary(index: Int) throws {
    var inputs = [
      URL(fileURLWithPath: "/tmp/source.mov"),
      URL(fileURLWithPath: "/tmp/preset.setting"),
      URL(fileURLWithPath: "/tmp/output.mp4"),
    ]
    inputs[index] = try #require(URL(string: "https://example.invalid/audio.mov"))
    #expect(throws: (any Error).self) {
      try Compressor.submissionArguments(
        source: inputs[0], preset: inputs[1], output: inputs[2], batchName: "test")
    }
  }

  @Test func rejectsRemotePresetAuthority() throws {
    let preset = try #require(URL(string: "file://other-host/tmp/preset.setting"))
    #expect(throws: ProAppsError.self) {
      try Compressor.submissionArguments(
        source: URL(fileURLWithPath: "/tmp/source.mov"), preset: preset,
        output: URL(fileURLWithPath: "/tmp/out.mp4"), batchName: "test")
    }
  }

  @Test func rejectsRemoteFileAuthorities() throws {
    let remote = try #require(URL(string: "file://other-host/tmp/source.mov"))
    #expect(throws: (any Error).self) {
      try Compressor.submissionArguments(
        source: remote, preset: URL(fileURLWithPath: "/tmp/preset.setting"),
        output: URL(fileURLWithPath: "/tmp/out.mp4"), batchName: "test")
    }
  }

  @Test func localhostIsLocalButInspectionStillUsesItsDocumentedURLForm() throws {
    let source = try #require(URL(string: "file://localhost/tmp/source%20name.mov"))
    let arguments = try Compressor.submissionArguments(
      source: source, preset: URL(fileURLWithPath: "/tmp/preset.setting"),
      output: URL(fileURLWithPath: "/tmp/out.mp4"), batchName: "test")
    #expect(arguments[3] == "/tmp/source name.mov")
    #expect(
      Compressor.inspectionArguments(source: source) == [
        "-checkstream", "file://localhost/tmp/source%20name.mov",
      ])
  }
}
