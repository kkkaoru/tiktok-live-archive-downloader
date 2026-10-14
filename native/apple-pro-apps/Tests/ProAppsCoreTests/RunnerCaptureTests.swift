import Foundation
import Testing

@testable import ProAppsCore

struct RunnerCaptureTests {
  @Test func captureIsSingleLinkRegularFile() async throws {
    let result = try await Runner.run(
      URL(fileURLWithPath: "/usr/bin/stat"), ["-L", "-f", "%l:%HT", "/dev/fd/1"])
    #expect(result.status == 0)
    #expect(result.stdout.trimmingCharacters(in: .whitespacesAndNewlines) == "1:Regular File")
  }
}
