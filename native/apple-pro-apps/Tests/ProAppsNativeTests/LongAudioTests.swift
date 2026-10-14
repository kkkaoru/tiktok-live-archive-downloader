import Foundation
import Testing

@testable import ProAppsCore

struct LongAudioTests {
  @Test func fullMinuteOfPCMNeedsExplicitBudgetAndPreservesSource() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "long-audio-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = root.appendingPathComponent("minute.wav")
    let header: [UInt8] = [
      82, 73, 70, 70, 36, 76, 29, 0, 87, 65, 86, 69, 102, 109, 116, 32, 16, 0, 0, 0, 1, 0, 1, 0,
      128, 62, 0, 0, 0, 125, 0, 0, 2, 0, 16, 0, 100, 97, 116, 97, 0, 76, 29, 0,
    ]
    let data = Data(header) + Data(repeating: 0, count: 1_920_000)
    try data.write(to: source, options: .withoutOverwriting)
    await #expect(throws: ProAppsError.self) {
      try await AudioProbe().measure(path: source.path, windows: [])
    }
    let measured = try await AudioProbe().measure(
      path: source.path,
      windows: [
        .init(startSeconds: 19.8, durationSeconds: 0.4),
        .init(startSeconds: 39.8, durationSeconds: 0.4),
      ], maximumDurationSeconds: 60)
    #expect(measured.whole.frames == 960000)
    #expect(measured.whole.durationSeconds == 60)
    #expect(measured.whole.peak == 0)
    #expect(measured.windows.first?.frames == 6400)
    #expect(measured.windows.last?.frames == 6400)
    #expect(try Data(contentsOf: source) == data)
  }

  @Test(arguments: [0.0, -1.0, 120.001, Double.infinity, Double.nan])
  func invalidBudgetsFailBeforeOpeningFiles(_ seconds: Double) async {
    do {
      _ = try await AudioProbe().measure(
        path: "/tmp/not-audio", windows: [], maximumDurationSeconds: seconds)
      Issue.record("Invalid audio budget was accepted")
    } catch ProAppsError.invalid(let reason) {
      #expect(reason == "Audio duration budget must be >0–120 seconds")
    } catch { Issue.record(error) }
  }
}
