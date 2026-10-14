import Foundation

/// Recognizer-provided timing is approximate, not human-reviewed alignment.
public struct SpeechSegment: Codable, Sendable, Equatable {
  public let text: String
  public let startSeconds: Double
  public let durationSeconds: Double
  /// Present only when a native sample-grid overshoot was clipped to the audio end.
  public let originalDurationSeconds: Double?

  public init(
    text: String, startSeconds: Double, durationSeconds: Double,
    originalDurationSeconds: Double? = nil
  ) {
    self.text = text
    self.startSeconds = startSeconds
    self.durationSeconds = durationSeconds
    self.originalDurationSeconds = originalDurationSeconds
  }

  /// Reconcile only the native result clock's single-tick rounding discrepancy.
  /// This is not permission to accept arbitrary out-of-range transcript input.
  func normalizingNativeEnd(audioDuration: Double, timeScale: Int32) throws -> SpeechSegment {
    guard timeScale > 0, audioDuration.isFinite, audioDuration > 0,
      startSeconds.isFinite, startSeconds >= 0, durationSeconds.isFinite, durationSeconds > 0
    else { throw ProAppsError.invalid("Invalid native speech clock or interval") }
    let end = startSeconds + durationSeconds
    guard end > audioDuration else { return self }
    let tick = 1 / Double(timeScale)
    guard end - audioDuration <= tick, startSeconds < audioDuration else {
      throw ProAppsError.invalid(
        "Native speech timing exceeds the audio by more than one native clock tick")
    }
    // Round toward the interior so floating-point re-addition cannot put the
    // normalized end back outside the strict transcript interval.
    let clipped = (audioDuration - startSeconds).nextDown
    guard clipped > 0 else {
      throw ProAppsError.invalid("No positive native speech interval remains")
    }
    return SpeechSegment(
      text: text, startSeconds: startSeconds, durationSeconds: clipped,
      originalDurationSeconds: durationSeconds)
  }
}

public struct SpeechTranscript: Codable, Sendable {
  public let locale: String
  public let durationSeconds: Double
  public let segments: [SpeechSegment]
  public let onDevice: Bool
  public let humanReviewed: Bool

  public static let maximumSegments = 1000
  public static let maximumTextBytes = 32768
  public static let maximumSeconds = 60.0

  public init(locale: String, durationSeconds: Double, segments: [SpeechSegment]) throws {
    guard durationSeconds.isFinite, durationSeconds > 0, durationSeconds <= Self.maximumSeconds,
      !locale.isEmpty, locale.utf8.count <= 64, segments.count <= Self.maximumSegments
    else { throw ProAppsError.invalid("Invalid transcript duration, locale or segment budget") }
    var bytes = 0
    for segment in segments {
      bytes += segment.text.utf8.count
      guard bytes <= Self.maximumTextBytes,
        !segment.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
        !segment.text.contains("\0")
      else { throw ProAppsError.invalid("Invalid or oversized recognized text") }
      guard segment.startSeconds.isFinite, segment.durationSeconds.isFinite,
        segment.startSeconds >= 0, segment.durationSeconds > 0,
        segment.startSeconds + segment.durationSeconds <= durationSeconds
      else {
        // Numeric timing only: never include recognized words or private paths.
        throw ProAppsError.invalid(
          "Recognized timing outside audio: start=\(segment.startSeconds), duration=\(segment.durationSeconds), audio=\(durationSeconds)"
        )
      }
    }
    self.locale = locale
    self.durationSeconds = durationSeconds
    self.segments = segments
    self.onDevice = true
    self.humanReviewed = false
  }
}

protocol SpeechWork: Sendable {
  func analyze() async throws
  func collect() async throws -> [SpeechSegment]
  func cancel() async
}

/// Owns analysis, result consumption and the watchdog as joined child tasks.
/// Cancellation wakes the watchdog, which terminates the native result stream.
enum SpeechRun {
  private enum Event: Sendable {
    case analyzed
    case collected([SpeechSegment])
    case watchdogStopped
  }

  static func collect(_ work: any SpeechWork, timeout: Duration = .seconds(45)) async throws
    -> [SpeechSegment]
  {
    try Task.checkCancellation()
    return try await withThrowingTaskGroup(of: Event.self) { group in
      group.addTask {
        try await work.analyze()
        return .analyzed
      }
      group.addTask { .collected(try await work.collect()) }
      group.addTask {
        do { try await Task.sleep(for: timeout) } catch is CancellationError {
          await work.cancel()
          return .watchdogStopped
        }
        throw ProAppsError.timedOut
      }
      do {
        var analyzed = false
        var segments: [SpeechSegment]?
        while let event = try await group.next() {
          switch event {
          case .analyzed: analyzed = true
          case .collected(let value): segments = value
          case .watchdogStopped: try Task.checkCancellation()
          }
          if analyzed, let segments {
            try Task.checkCancellation()
            group.cancelAll()
            return segments
          }
        }
        throw ProAppsError.unavailable("Speech analysis ended without complete results")
      } catch {
        group.cancelAll()
        await work.cancel()
        throw error
      }
    }
  }
}
