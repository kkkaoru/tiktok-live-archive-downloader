import Foundation

public struct CaptionProjectionRequest: Codable, Sendable {
  public let sourceCaptions: [EditCaption]
  public let retainedSpans: [SpeechSpan]
  public let maximumCharacters: Int

  public init(sourceCaptions: [EditCaption], retainedSpans: [SpeechSpan], maximumCharacters: Int) {
    self.sourceCaptions = sourceCaptions
    self.retainedSpans = retainedSpans
    self.maximumCharacters = maximumCharacters
  }
}

public struct CaptionProjectionResult: Codable, Sendable {
  public let captions: [EditCaption]
  public let outputDurationSeconds: Double
  public let overlapAdjustedCount: Int
  public let omittedCaptionCount: Int
  public let timingIsEstimated: Bool
  public let humanReviewed: Bool
}

/// Character-proportional phrase splitting and deterministic cut-time mapping.
/// Not forced word alignment or a correction of recognizer text. Source overlap
/// is explicitly clipped, counted and retained in the caller's raw evidence.
public enum CaptionProjection {
  public static func make(_ request: CaptionProjectionRequest) throws -> CaptionProjectionResult {
    guard !request.sourceCaptions.isEmpty, request.sourceCaptions.count <= 5000,
      !request.retainedSpans.isEmpty, request.retainedSpans.count <= 3000,
      (8...80).contains(request.maximumCharacters)
    else { throw ProAppsError.invalid("Invalid caption projection count or character budget") }
    let estimatedPieces = request.sourceCaptions.reduce(0) {
      $0 + 2 * ($1.text.count / request.maximumCharacters + 1)
    }
    guard estimatedPieces <= 30_000,
      estimatedPieces <= 50_000_000 / request.retainedSpans.count
    else { throw ProAppsError.invalid("Caption projection work budget exceeded") }
    let cuts = try mapCuts(request.retainedSpans)
    var captions: [EditCaption] = []
    var previousSourceStart = 0.0
    var previousSourceEnd = 0.0
    var adjusted = 0
    var omitted = 0
    for caption in request.sourceCaptions {
      try Task.checkCancellation()
      let text = caption.text.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !text.isEmpty, text.count <= 2048, !text.contains("\0"),
        caption.startSeconds.isFinite, caption.endSeconds.isFinite,
        caption.startSeconds >= previousSourceStart, caption.endSeconds > caption.startSeconds,
        caption.endSeconds <= 21_600
      else { throw ProAppsError.invalid("Invalid ordered source caption text or interval") }
      previousSourceStart = caption.startSeconds
      let start = max(caption.startSeconds, previousSourceEnd)
      if start > caption.startSeconds { adjusted += 1 }
      previousSourceEnd = max(previousSourceEnd, caption.endSeconds)
      guard start < caption.endSeconds else {
        omitted += 1
        continue
      }
      let source = EditCaption(text: text, startSeconds: start, endSeconds: caption.endSeconds)
      for piece in split(source, maximum: request.maximumCharacters) {
        if let mapped = project(piece, cuts: cuts.segments) {
          captions.append(mapped)
        } else {
          omitted += 1
        }
      }
    }
    guard captions.count <= 30_000 else {
      throw ProAppsError.invalid("Too many projected captions")
    }
    return CaptionProjectionResult(
      captions: captions,
      outputDurationSeconds: cuts.duration,
      overlapAdjustedCount: adjusted, omittedCaptionCount: omitted,
      timingIsEstimated: true, humanReviewed: false)
  }

  private static func mapCuts(_ spans: [SpeechSpan]) throws -> (
    segments: [SpeechCutSegment], duration: Double
  ) {
    var sourceEnd = 0.0
    var outputEnd = 0.0
    var segments: [SpeechCutSegment] = []
    for span in spans {
      guard span.startSeconds.isFinite, span.endSeconds.isFinite,
        span.startSeconds >= sourceEnd, span.endSeconds > span.startSeconds,
        span.endSeconds <= 21_600
      else { throw ProAppsError.invalid("Retained cuts must be ordered and nonoverlapping") }
      let start = outputEnd
      sourceEnd = span.endSeconds
      outputEnd += span.endSeconds - span.startSeconds
      segments.append(
        SpeechCutSegment(
          sourceStartSeconds: span.startSeconds, sourceEndSeconds: span.endSeconds,
          outputStartSeconds: start, outputEndSeconds: outputEnd))
    }
    return (segments, outputEnd)
  }

  private static func split(_ caption: EditCaption, maximum: Int) -> [EditCaption] {
    let characters = Array(caption.text)
    let secondsPerCharacter = (caption.endSeconds - caption.startSeconds) / Double(characters.count)
    var offset = 0
    var result: [EditCaption] = []
    while offset < characters.count {
      var end = min(characters.count, offset + maximum)
      if end < characters.count,
        let punctuation = (offset..<end).last(where: { "。！？!?、".contains(characters[$0]) }),
        punctuation - offset >= maximum / 2
      {
        end = punctuation + 1
      }
      let text = String(characters[offset..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
      if !text.isEmpty {
        result.append(
          EditCaption(
            text: text,
            startSeconds: caption.startSeconds + Double(offset) * secondsPerCharacter,
            endSeconds: caption.startSeconds + Double(end) * secondsPerCharacter))
      }
      offset = end
    }
    return result
  }

  private static func project(_ caption: EditCaption, cuts: [SpeechCutSegment]) -> EditCaption? {
    var first: Double?
    var last: Double?
    for cut in cuts {
      let start = max(caption.startSeconds, cut.sourceStartSeconds)
      let end = min(caption.endSeconds, cut.sourceEndSeconds)
      if start < end {
        if first == nil { first = cut.outputStartSeconds + start - cut.sourceStartSeconds }
        last = cut.outputStartSeconds + end - cut.sourceStartSeconds
      }
    }
    guard let first, let last, last - first >= EditPlan.minimumTimeSeconds else { return nil }
    return EditCaption(text: caption.text, startSeconds: first, endSeconds: last)
  }
}
