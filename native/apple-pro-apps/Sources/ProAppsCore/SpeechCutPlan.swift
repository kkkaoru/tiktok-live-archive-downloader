import Foundation

/// Explicit source-time span. This DTO is not a claim of detected speech.
public struct SpeechSpan: Codable, Sendable {
  public let startSeconds: Double
  public let endSeconds: Double

  public init(startSeconds: Double, endSeconds: Double) {
    self.startSeconds = startSeconds
    self.endSeconds = endSeconds
  }
}

public struct SpeechCutRequest: Codable, Sendable {
  public let sourceDurationSeconds: Double
  public let spans: [SpeechSpan]
  public let paddingSeconds: Double
  public let minimumRemovedGapSeconds: Double

  public init(
    sourceDurationSeconds: Double, spans: [SpeechSpan], paddingSeconds: Double,
    minimumRemovedGapSeconds: Double
  ) {
    self.sourceDurationSeconds = sourceDurationSeconds
    self.spans = spans
    self.paddingSeconds = paddingSeconds
    self.minimumRemovedGapSeconds = minimumRemovedGapSeconds
  }
}

public struct SpeechCutSegment: Codable, Sendable {
  public let sourceStartSeconds: Double
  public let sourceEndSeconds: Double
  public let outputStartSeconds: Double
  public let outputEndSeconds: Double
}

/// Pure cut-level mapping. No file access, inference, retiming or frame rounding.
/// Consumers must explicitly quantize to their output frame rate before rendering.
public struct SpeechCutPlan: Codable, Sendable {
  public let segments: [SpeechCutSegment]
  public let removedSpans: [SpeechSpan]
  public let outputDurationSeconds: Double
  public let sourceDurationSeconds: Double
  public let speechVerified: Bool

  public static func make(_ request: SpeechCutRequest) throws -> SpeechCutPlan {
    try Task.checkCancellation()
    guard request.sourceDurationSeconds.isFinite,
      (0.001...21_600).contains(request.sourceDurationSeconds),
      !request.spans.isEmpty, request.spans.count <= 30_000,
      request.paddingSeconds.isFinite, (0...2).contains(request.paddingSeconds),
      request.minimumRemovedGapSeconds.isFinite,
      (0...5).contains(request.minimumRemovedGapSeconds)
    else { throw ProAppsError.invalid("Invalid speech cut duration, spans or padding budgets") }
    var merged: [SpeechSpan] = []
    var previousStart = 0.0
    for span in request.spans {
      try Task.checkCancellation()
      guard span.startSeconds.isFinite, span.endSeconds.isFinite,
        span.startSeconds >= previousStart, span.endSeconds > span.startSeconds,
        span.endSeconds <= request.sourceDurationSeconds
      else { throw ProAppsError.invalid("Speech spans must be ordered and inside the source") }
      previousStart = span.startSeconds
      let start = max(0, span.startSeconds - request.paddingSeconds)
      let end = min(request.sourceDurationSeconds, span.endSeconds + request.paddingSeconds)
      if let previous = merged.last,
        start - previous.endSeconds <= request.minimumRemovedGapSeconds
      {
        merged[merged.count - 1] = SpeechSpan(
          startSeconds: previous.startSeconds, endSeconds: max(previous.endSeconds, end))
      } else {
        merged.append(SpeechSpan(startSeconds: start, endSeconds: end))
      }
    }
    var segments: [SpeechCutSegment] = []
    var removed: [SpeechSpan] = []
    var sourceCursor = 0.0
    var outputCursor = 0.0
    for span in merged {
      if span.startSeconds > sourceCursor {
        removed.append(SpeechSpan(startSeconds: sourceCursor, endSeconds: span.startSeconds))
      }
      let outputEnd = outputCursor + span.endSeconds - span.startSeconds
      segments.append(
        SpeechCutSegment(
          sourceStartSeconds: span.startSeconds, sourceEndSeconds: span.endSeconds,
          outputStartSeconds: outputCursor, outputEndSeconds: outputEnd))
      sourceCursor = span.endSeconds
      outputCursor = outputEnd
    }
    if sourceCursor < request.sourceDurationSeconds {
      removed.append(
        SpeechSpan(startSeconds: sourceCursor, endSeconds: request.sourceDurationSeconds))
    }
    return SpeechCutPlan(
      segments: segments, removedSpans: removed,
      outputDurationSeconds: outputCursor, sourceDurationSeconds: request.sourceDurationSeconds,
      speechVerified: false)
  }
}
