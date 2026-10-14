import CoreMedia
import Foundation
import NaturalLanguage
import Speech

/// Optional local recognizer hints. No dictionary/model installation is performed.
public struct SpeechRecognitionOptions: Sendable {
  public let wordTiming: Bool
  public let contextualStrings: [String]

  public init(wordTiming: Bool = false, contextualStrings: [String] = []) throws {
    guard contextualStrings.count <= 128,
      contextualStrings.reduce(0, { $0 + $1.utf8.count }) <= 8192,
      contextualStrings.allSatisfy({
        !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
          && $0.count <= 128 && !$0.contains("\0")
      })
    else { throw ProAppsError.invalid("Invalid speech contextual vocabulary budget") }
    self.wordTiming = wordTiming
    self.contextualStrings = contextualStrings
  }
}

/// Preserve native timed text runs, never interpolate time from character counts.
@available(macOS 26.0, *)
public enum SpeechTimedText {
  public static func segments(from text: AttributedString, audioDuration: Double) throws
    -> [SpeechSegment]
  {
    var segments: [SpeechSegment] = []
    let nonLexical = CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters)
    for run in text.runs {
      try Task.checkCancellation()
      let fragment = String(text[run.range].characters)
      guard !fragment.isEmpty else { continue }
      if fragment.unicodeScalars.allSatisfy({ nonLexical.contains($0) }) {
        if let previous = segments.popLast() {
          segments.append(
            SpeechSegment(
              text: previous.text + fragment, startSeconds: previous.startSeconds,
              durationSeconds: previous.durationSeconds,
              originalDurationSeconds: previous.originalDurationSeconds))
        }
        continue
      }
      guard let range = run.audioTimeRange else {
        throw ProAppsError.invalid("Native lexical speech text is missing its audio time range")
      }
      let segment = try SpeechSegment(
        text: fragment, startSeconds: range.start.seconds,
        durationSeconds: range.duration.seconds
      ).normalizingNativeEnd(audioDuration: audioDuration, timeScale: range.end.timescale)
      segments.append(segment)
      guard segments.count <= SpeechTranscript.maximumSegments else {
        throw ProAppsError.outputLimit
      }
    }
    return groupedWords(segments)
  }

  // Japanese audio-time attributes can be individual characters. Join complete
  // lexical units for display, but never divide one coarse native timed run.
  private static func groupedWords(_ segments: [SpeechSegment]) -> [SpeechSegment] {
    let text = segments.map(\.text).joined()
    let tokenizer = NLTokenizer(unit: .word)
    tokenizer.string = text
    let ranges = tokenizer.tokens(for: text.startIndex..<text.endIndex).map {
      (
        start: text.distance(from: text.startIndex, to: $0.lowerBound),
        end: text.distance(from: text.startIndex, to: $0.upperBound)
      )
    }
    var result: [SpeechSegment] = []
    var pending: SpeechSegment?
    var consumed = 0
    var wordIndex = 0
    for segment in segments {
      if let previous = pending,
        segment.startSeconds - previous.startSeconds - previous.durationSeconds > 0.25
      {
        result.append(previous)
        pending = nil
      }
      if let previous = pending {
        pending = SpeechSegment(
          text: previous.text + segment.text, startSeconds: previous.startSeconds,
          durationSeconds: segment.startSeconds + segment.durationSeconds - previous.startSeconds,
          originalDurationSeconds: segment.originalDurationSeconds.map {
            segment.startSeconds + $0 - previous.startSeconds
          })
      } else {
        pending = segment
      }
      consumed += segment.text.count
      while wordIndex < ranges.count && ranges[wordIndex].end <= consumed { wordIndex += 1 }
      if wordIndex == ranges.count || consumed <= ranges[wordIndex].start {
        if let completed = pending { result.append(completed) }
        pending = nil
      }
    }
    if let remaining = pending { result.append(remaining) }
    return result
  }
}
