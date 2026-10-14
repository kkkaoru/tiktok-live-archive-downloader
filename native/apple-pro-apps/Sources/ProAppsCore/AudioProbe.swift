import AVFoundation
import Dispatch
import Foundation

/// Bounded offline mono PCM measurements. Uses the macOS converter with fixed
/// arguments, not playback, a shell, network services, or raw audio pointers.
public actor AudioProbe {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.audio-measure")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }

  public init() {}

  public func measure(
    path: String, windows: [AudioWindow],
    maximumDurationSeconds: Double = PCMMeasurement.defaultDurationSeconds
  ) async throws -> PCMMeasurement {
    try Task.checkCancellation()
    guard maximumDurationSeconds.isFinite, maximumDurationSeconds > 0,
      maximumDurationSeconds <= PCMMeasurement.maximumDurationSeconds
    else { throw ProAppsError.invalid("Audio duration budget must be >0–120 seconds") }
    guard windows.count <= PCMMeasurement.maximumWindows else {
      throw ProAppsError.invalid("Too many audio windows")
    }
    let source = try Files.existing(path)
    let asset = AVURLAsset(url: source)
    let duration = try await asset.load(.duration).seconds
    let audio = try await asset.loadTracks(withMediaType: .audio)
    guard !audio.isEmpty, duration.isFinite, duration > 0,
      duration <= maximumDurationSeconds
    else {
      throw ProAppsError.invalid(
        "Audio measurement requires an audio-bearing clip within the requested duration budget")
    }
    for window in windows {
      guard window.startSeconds.isFinite, window.durationSeconds.isFinite,
        window.startSeconds >= 0, window.durationSeconds > 0,
        window.startSeconds + window.durationSeconds <= duration
      else { throw ProAppsError.invalid("Audio window lies outside the source duration") }
    }
    try Task.checkCancellation()
    let pcm = try Files.reserveOutput(
      directory: Files.temporaryDirectory().path, name: "decoded.wav", kind: .edit)
    defer {
      Cleanup.perform { try FileManager.default.removeItem(at: pcm.deletingLastPathComponent()) }
    }
    let result = try await Runner.run(
      URL(fileURLWithPath: "/usr/bin/afconvert"),
      [
        "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", "--no-filler", source.path, pcm.path,
      ], timeout: .seconds(30))
    guard result.status == 0 else { throw ProAppsError.commandFailed(result.status) }
    try Task.checkCancellation()
    return try PCMMeasurement.analyze(
      Files.read(pcm), windows: windows, maximumDurationSeconds: maximumDurationSeconds)
  }
}
