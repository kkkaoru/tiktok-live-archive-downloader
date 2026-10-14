import Foundation
import Testing

@testable import ProAppsCore

struct SpeechTranscriptTests {
  @Test func transcriptPreservesTextAndDoesNotClaimHumanReview() throws {
    let segment = SpeechSegment(text: "今日は動画を編集します。", startSeconds: 1, durationSeconds: 2)
    let result = try SpeechTranscript(locale: "ja_JP", durationSeconds: 12, segments: [segment])
    #expect(result.onDevice)
    #expect(!result.humanReviewed)
    #expect(result.segments == [segment])
    let decoded = try JSONDecoder().decode(
      SpeechTranscript.self, from: JSONEncoder().encode(result))
    #expect(decoded.segments == [segment])
    #expect(
      try SpeechTranscript(locale: "ja_JP", durationSeconds: 1, segments: []).segments.isEmpty)
  }

  @Test(arguments: [0.0, 60.001, Double.infinity, Double.nan])
  func invalidDurationIsRefused(_ duration: Double) {
    #expect(throws: ProAppsError.self) {
      try SpeechTranscript(locale: "ja_JP", durationSeconds: duration, segments: [])
    }
  }

  @Test(arguments: ["", " ", "bad\0text", String(repeating: "a", count: 32769)])
  func invalidTextIsRefused(_ text: String) {
    #expect(throws: ProAppsError.self) {
      try SpeechTranscript(
        locale: "ja_JP", durationSeconds: 2,
        segments: [.init(text: text, startSeconds: 0, durationSeconds: 1)])
    }
  }

  @Test(arguments: [(-1.0, 1.0), (0.0, 0.0), (59.0, 2.0), (Double.nan, 1.0)])
  func invalidTimingIsRefused(_ start: Double, _ duration: Double) {
    #expect(throws: ProAppsError.self) {
      try SpeechTranscript(
        locale: "ja_JP", durationSeconds: 2,
        segments: [.init(text: "hello", startSeconds: start, durationSeconds: duration)])
    }
  }

  @Test func localeAndSegmentBudgetsAreEnforced() {
    #expect(throws: ProAppsError.self) {
      try SpeechTranscript(locale: "", durationSeconds: 1, segments: [])
    }
    #expect(throws: ProAppsError.self) {
      try SpeechTranscript(
        locale: "ja_JP", durationSeconds: 1,
        segments: Array(
          repeating: .init(text: "a", startSeconds: 0, durationSeconds: 1), count: 1001))
    }
  }

  private actor Work: SpeechWork {
    enum Mode: Sendable { case success, analysisFailure, resultFailure, stall }
    let mode: Mode
    var cancelled = false
    private var started = false
    private var starters: [CheckedContinuation<Void, Never>] = []
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(_ mode: Mode) { self.mode = mode }
    func began() {
      started = true
      let waiting = starters
      starters.removeAll()
      for waiter in waiting { waiter.resume() }
    }
    func waitUntilStarted() async {
      if started { return }
      await withCheckedContinuation { starters.append($0) }
    }
    func stall() async throws {
      if !cancelled { await withCheckedContinuation { waiters.append($0) } }
      throw CancellationError()
    }
    func analyze() async throws {
      began()
      if mode == .analysisFailure { throw ProAppsError.invalid("Injected analysis failure") }
      if mode == .stall { try await stall() }
    }
    func collect() async throws -> [SpeechSegment] {
      began()
      if mode == .resultFailure { throw ProAppsError.invalid("Injected result failure") }
      if mode == .stall { try await stall() }
      return [.init(text: "hello", startSeconds: 0, durationSeconds: 1)]
    }
    func cancel() {
      cancelled = true
      let waiting = waiters
      waiters.removeAll()
      for waiter in waiting { waiter.resume() }
    }
  }

  @Test func successfulSessionJoinsItsWatchdog() async throws {
    let work = Work(.success)
    let result = try await SpeechRun.collect(work)
    #expect(result == [.init(text: "hello", startSeconds: 0, durationSeconds: 1)])
    #expect(await work.cancelled)
  }

  @Test(arguments: [Work.Mode.analysisFailure, .resultFailure])
  private func failuresCancelAndJoinTheSession(_ mode: Work.Mode) async {
    let work = Work(mode)
    await #expect(throws: ProAppsError.self) { try await SpeechRun.collect(work) }
    #expect(await work.cancelled)
  }

  @Test func timeoutClosesOtherwiseWaitingNativeStreams() async {
    let work = Work(.stall)
    do {
      _ = try await SpeechRun.collect(work, timeout: .zero)
      Issue.record("Speech watchdog did not time out")
    } catch ProAppsError.timedOut { #expect(await work.cancelled) } catch { Issue.record(error) }
  }

  @Test func cancellationWakesAndJoinsTheNativeSession() async {
    let work = Work(.stall)
    await withTaskGroup(of: Void.self) { group in
      group.addTask {
        await #expect(throws: CancellationError.self) { try await SpeechRun.collect(work) }
      }
      await work.waitUntilStarted()
      group.cancelAll()
    }
    #expect(await work.cancelled)
  }
}
