import Foundation
import Testing
import WhisperKit

@testable import ProAppsCore

struct WhisperTimingTests {
  @Test func unicodeBytePiecesStayTogetherAndNoTokensDisappear() {
    let result = WhisperUnicodeWords.split([1, 2, 3]) { tokens in
      switch tokens {
      case [1]: "\u{fffd}"
      case [1, 2]: "語"
      case [3]: "。"
      default: ""
      }
    }
    #expect(result.words == ["語", "。"])
    #expect(result.wordTokens == [[1, 2], [3]])
    let incomplete = WhisperUnicodeWords.split([9]) { _ in "\u{fffd}" }
    #expect(incomplete.words == ["\u{fffd}"])
    #expect(incomplete.wordTokens == [[9]])
    #expect(WhisperUnicodeWords.split([], decode: { _ in "" }).words == [])
  }

  @Test func nativeWordsPreserveClockAndPunctuationWithoutProportionalTiming() throws {
    guard #available(macOS 26.0, *) else {
      Issue.record("Requires macOS 26")
      return
    }
    let result = try WhisperTimedText.segments(
      [
        TranscriptionSegment(
          text: "今日は、編集します。",
          words: [
            WordTiming(word: "今日は、", tokens: [], start: 0.2, end: 0.8, probability: 0.9),
            WordTiming(word: "編集します", tokens: [], start: 1.2, end: 2, probability: 0.8),
            WordTiming(word: "。", tokens: [], start: 2, end: 2, probability: 1),
          ])
      ], audioDuration: 3)
    #expect(
      result == [
        SpeechSegment(text: "今日は、", startSeconds: 0.2, durationSeconds: 0.6),
        SpeechSegment(text: "編集します。", startSeconds: 1.2, durationSeconds: 0.8),
      ])
    #expect(try WhisperTimedText.segments([TranscriptionSegment()], audioDuration: 3) == [])
  }

  @Test func floatClockDoesNotLoseAnAlignmentTick() throws {
    guard #available(macOS 26.0, *) else {
      Issue.record("Requires macOS 26")
      return
    }
    let result = try WhisperTimedText.segments(
      [
        TranscriptionSegment(
          text: "声",
          words: [
            WordTiming(word: "声", tokens: [], start: 0.9, end: 1.56, probability: 1)
          ])
      ], audioDuration: 2)
    #expect(result == [SpeechSegment(text: "声", startSeconds: 0.9, durationSeconds: 0.66)])
    #expect(throws: ProAppsError.self) {
      try WhisperTimedText.segments(
        [
          TranscriptionSegment(
            text: "不正",
            words: [
              WordTiming(
                word: "不正", tokens: [], start: 0, end: .greatestFiniteMagnitude, probability: 1)
            ])
        ], audioDuration: 2)
    }
  }

  @Test func missingTimingAndInvalidNativeValuesFailRatherThanGuess() throws {
    guard #available(macOS 26.0, *) else {
      Issue.record("Requires macOS 26")
      return
    }
    #expect(throws: ProAppsError.self) {
      try WhisperTimedText.segments([TranscriptionSegment(text: "未計測")], audioDuration: 3)
    }
    #expect(throws: ProAppsError.self) {
      try WhisperTimedText.segments(
        [
          TranscriptionSegment(
            text: "不正",
            words: [
              WordTiming(word: "不正", tokens: [], start: .nan, end: 1, probability: 1)
            ])
        ], audioDuration: 3)
    }
    #expect(throws: ProAppsError.self) {
      try WhisperTimedText.segments(
        [
          TranscriptionSegment(
            text: "上限",
            words: Array(
              repeating:
                WordTiming(word: "語", tokens: [], start: 0, end: 1, probability: 1), count: 1001))
        ], audioDuration: 3)
    }
  }

  @Test func cancelledConversionStopsBeforeBuildingCaptions() async {
    guard #available(macOS 26.0, *) else {
      Issue.record("Requires macOS 26")
      return
    }
    await withTaskGroup(of: Void.self) { group in
      group.cancelAll()
      group.addTask {
        #expect(throws: CancellationError.self) {
          try WhisperTimedText.segments([TranscriptionSegment()], audioDuration: 1)
        }
      }
    }
  }
}
