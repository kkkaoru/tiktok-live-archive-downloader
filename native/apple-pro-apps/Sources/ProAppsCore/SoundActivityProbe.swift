import CoreMedia
import Dispatch
import Foundation
import SoundAnalysis
import Synchronization

public struct SoundActivityWindow: Codable, Sendable {
  public let startSeconds: Double
  public let durationSeconds: Double
  public let speechConfidence: Double
  public let musicConfidence: Double
}

public struct SoundActivityReport: Codable, Sendable {
  public let durationSeconds: Double
  public let windowDurationSeconds: Double
  public let overlapFactor: Double
  public let windows: [SoundActivityWindow]
  public let humanReviewed: Bool
}

/// Callback state is protected even if the framework uses a different thread.
/// The analyzer retains its observer weakly; the worker retains it until removal.
final class SoundActivityObserver: NSObject, SNResultsObserving {
  private struct State {
    var windows: [SoundActivityWindow] = []
    var error: (any Error)?
    var completed = false
  }
  private let state = Mutex(State())

  func request(_ request: any SNRequest, didProduce result: any SNResult) {
    guard let result = result as? SNClassificationResult else {
      fail(ProAppsError.invalid("Unexpected sound classification result type"))
      return
    }
    guard let speech = result.classification(forIdentifier: "speech"),
      let music = result.classification(forIdentifier: "music")
    else {
      fail(ProAppsError.unavailable("Sound classifier omitted required speech/music labels"))
      return
    }
    append(
      .init(
        startSeconds: result.timeRange.start.seconds,
        durationSeconds: result.timeRange.duration.seconds,
        speechConfidence: speech.confidence, musicConfidence: music.confidence))
  }

  func append(_ window: SoundActivityWindow) {
    state.withLock { state in
      guard state.error == nil, !state.completed else { return }
      guard state.windows.count < 512, window.startSeconds.isFinite,
        window.startSeconds >= 0, window.durationSeconds.isFinite, window.durationSeconds > 0,
        window.speechConfidence.isFinite, (0...1).contains(window.speechConfidence),
        window.musicConfidence.isFinite, (0...1).contains(window.musicConfidence)
      else {
        state.error = ProAppsError.invalid("Invalid or excessive sound classification windows")
        return
      }
      state.windows.append(window)
    }
  }

  func fail(_ error: any Error) {
    state.withLock { state in
      if state.error == nil { state.error = error }
    }
  }

  func request(_ request: any SNRequest, didFailWithError error: any Error) { fail(error) }

  func requestDidComplete(_ request: any SNRequest) {
    state.withLock { $0.completed = true }
  }

  func results() throws -> [SoundActivityWindow] {
    try state.withLock { state in
      if let error = state.error { throw error }
      guard state.completed, !state.windows.isEmpty else {
        throw ProAppsError.unavailable("Sound classification did not complete with results")
      }
      return state.windows
    }
  }
}

/// Synchronous native analysis runs on a dedicated serial executor, not MainActor
/// or the cooperative pool. The caller's disposable child supplies a hard deadline.
/// Cancellation is checked before and after analysis; no unsafe callback bridge.
public actor SoundActivityProbe {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.sound-activity")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }

  public init() {}

  public func analyze(path: String) throws -> SoundActivityReport {
    try Task.checkCancellation()
    let source = try Files.existing(path, extensions: ["wav"])
    let pcm = try PCMMeasurement.analyze(
      Files.read(source), windows: [], maximumDurationSeconds: 60)
    guard pcm.whole.durationSeconds >= 0.5 else {
      throw ProAppsError.invalid("Sound activity requires 0.5–60 seconds of mono PCM16 WAV")
    }
    let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
    request.windowDuration = CMTime(seconds: 0.5, preferredTimescale: 16000)
    request.overlapFactor = 0.5
    let observer = SoundActivityObserver()
    let analyzer = try SNAudioFileAnalyzer(url: source)
    try analyzer.add(request, withObserver: observer)
    defer { analyzer.removeAllRequests() }
    analyzer.analyze()
    try Task.checkCancellation()
    return SoundActivityReport(
      durationSeconds: pcm.whole.durationSeconds,
      windowDurationSeconds: request.windowDuration.seconds, overlapFactor: request.overlapFactor,
      windows: try observer.results(), humanReviewed: false)
  }
}
