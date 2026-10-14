import Foundation
import ProAppsCore
import Testing

struct MediaProbeTests {
  @Test(.timeLimit(.minutes(1)))
  func decodesOneSyntheticBlackFrameWithoutPlayback() async throws {
    let fixture = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    let before = try Data(contentsOf: fixture)
    let summary = try await MediaProbe().inspect(path: fixture.path)
    #expect(summary.durationSeconds == 1)
    #expect(summary.width == 16)
    #expect(summary.height == 16)
    // AVAssetWriter's preset records a nominal rate of 15 even though this
    // fixture holds one decoded frame over a one-second presentation session.
    #expect(summary.frameRate == 15)
    #expect(summary.audioTrackCount == 0)
    #expect(summary.firstFrameDecoded)
    #expect(try Data(contentsOf: fixture) == before)
    let decoded = try JSONDecoder().decode(MediaSummary.self, from: JSONEncoder().encode(summary))
    #expect(decoded.width == 16)
  }

  @Test func refusesAnAudioOnlyAssetWithoutPlayingIt() async throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "media-audio-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let file = directory.appendingPathComponent("silence.wav")
    // PCM: mono, 8000 Hz, 16 bits, 800 silent samples (0.1 seconds).
    var wave = Data([
      82, 73, 70, 70, 100, 6, 0, 0, 87, 65, 86, 69, 102, 109, 116, 32, 16, 0, 0, 0, 1, 0, 1, 0, 64,
      31, 0, 0, 128, 62, 0, 0, 2, 0, 16, 0, 100, 97, 116, 97, 64, 6, 0, 0,
    ])
    wave.append(Data(count: 1600))
    try wave.write(to: file)
    do {
      _ = try await MediaProbe().inspect(path: file.path)
      Issue.record("An audio-only asset must not pass video verification")
    } catch let error as ProAppsError {
      #expect(error.description == "Invalid request: Expected a finite-duration video track")
    }
  }

  @Test func refusesNonMediaAndMissingFiles() async throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "media-invalid-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let file = directory.appendingPathComponent("invalid.mp4")
    try Data("not a movie".utf8).write(to: file)
    await #expect(throws: (any Error).self) { try await MediaProbe().inspect(path: file.path) }
    await #expect(throws: (any Error).self) {
      try await MediaProbe().inspect(path: directory.appendingPathComponent("missing.mp4").path)
    }
  }
}
