import CoreMedia
import Foundation
import Speech
import Testing

@testable import ProAppsCore

struct SpeechRecognitionTests {
  @Test func vocabularyIsExplicitAndBounded() throws {
    let defaults = try SpeechRecognitionOptions()
    #expect(defaults.wordTiming == false)
    #expect(defaults.contextualStrings == [])
    let options = try SpeechRecognitionOptions(wordTiming: true, contextualStrings: ["動画", "編集"])
    #expect(options.wordTiming == true)
    #expect(options.contextualStrings == ["動画", "編集"])
    #expect(throws: ProAppsError.self) { try SpeechRecognitionOptions(contextualStrings: [" "]) }
    #expect(throws: ProAppsError.self) { try SpeechRecognitionOptions(contextualStrings: ["語\0"]) }
    #expect(throws: ProAppsError.self) {
      try SpeechRecognitionOptions(contextualStrings: [String(repeating: "あ", count: 129)])
    }
    #expect(throws: ProAppsError.self) {
      try SpeechRecognitionOptions(contextualStrings: Array(repeating: "語", count: 129))
    }
    #expect(throws: ProAppsError.self) {
      try SpeechRecognitionOptions(
        contextualStrings: Array(repeating: String(repeating: "あ", count: 100), count: 30))
    }
  }

  @Test func nativeRunTimesDoNotSpreadAcrossSilence() throws {
    guard #available(macOS 26.0, *) else {
      Issue.record("Requires macOS 26")
      return
    }
    var first = AttributedString("今日は")
    first.audioTimeRange = CMTimeRange(
      start: CMTime(value: 1, timescale: 2), duration: CMTime(value: 1, timescale: 2))
    var second = AttributedString("編集します")
    second.audioTimeRange = CMTimeRange(
      start: CMTime(value: 3, timescale: 1), duration: CMTime(value: 1, timescale: 1))
    let text =
      AttributedString("。") + first + AttributedString("、") + second + AttributedString("。")
    let result = try SpeechTimedText.segments(from: text, audioDuration: 5)
    #expect(
      result == [
        SpeechSegment(text: "今日は、", startSeconds: 0.5, durationSeconds: 0.5),
        SpeechSegment(text: "編集します。", startSeconds: 3, durationSeconds: 1),
      ])
    #expect(try SpeechTimedText.segments(from: AttributedString("、。 "), audioDuration: 5) == [])
  }

  @Test func characterRunsBecomeWholeWordsWithoutInventedTiming() throws {
    guard #available(macOS 26.0, *) else {
      Issue.record("Requires macOS 26")
      return
    }
    let parts: [(String, Int64, Int64)] = [
      ("hel", 0, 1), ("lo, ", 1, 1), ("wor", 2, 1), ("ld.", 3, 1),
    ]
    let text = parts.map { fragment, start, duration in
      var value = AttributedString(fragment)
      value.audioTimeRange = CMTimeRange(
        start: CMTime(value: start, timescale: 10), duration: CMTime(value: duration, timescale: 10)
      )
      return value
    }.reduce(AttributedString(), +)
    let result = try SpeechTimedText.segments(from: text, audioDuration: 1)
    #expect(
      result == [
        SpeechSegment(text: "hello, ", startSeconds: 0, durationSeconds: 0.2),
        SpeechSegment(text: "world.", startSeconds: 0.2, durationSeconds: 0.2),
      ])
    var coarse = AttributedString("hello world")
    coarse.audioTimeRange = CMTimeRange(start: .zero, duration: CMTime(value: 1, timescale: 1))
    #expect(
      try SpeechTimedText.segments(from: coarse, audioDuration: 1) == [
        SpeechSegment(text: "hello world", startSeconds: 0, durationSeconds: 1)
      ])
  }

  @Test func wordGroupingPreservesGapsAndNativeEndNormalization() throws {
    guard #available(macOS 26.0, *) else {
      Issue.record("Requires macOS 26")
      return
    }
    var first = AttributedString("hel")
    first.audioTimeRange = CMTimeRange(start: .zero, duration: CMTime(value: 1, timescale: 10))
    var last = AttributedString("lo")
    last.audioTimeRange = CMTimeRange(
      start: CMTime(value: 9, timescale: 10), duration: CMTime(value: 1, timescale: 10))
    #expect(try SpeechTimedText.segments(from: first + last, audioDuration: 1).count == 2)
    first.audioTimeRange = CMTimeRange(start: .zero, duration: CMTime(value: 5, timescale: 10))
    last.audioTimeRange = CMTimeRange(
      start: CMTime(value: 5000, timescale: 10000), duration: CMTime(value: 5001, timescale: 10000))
    let normalized = try SpeechTimedText.segments(from: first + last, audioDuration: 1)
    #expect(normalized.count == 1)
    #expect(normalized.first?.durationSeconds == 1)
    #expect(normalized.first?.originalDurationSeconds == 1.0001)
  }

  @Test func untimedLexicalOrOutOfRangeTextIsNotGuessed() throws {
    guard #available(macOS 26.0, *) else {
      Issue.record("Requires macOS 26")
      return
    }
    #expect(throws: ProAppsError.self) {
      try SpeechTimedText.segments(from: AttributedString("未計測"), audioDuration: 5)
    }
    var text = AttributedString("範囲外")
    text.audioTimeRange = CMTimeRange(
      start: CMTime(value: 8, timescale: 1), duration: CMTime(value: 1, timescale: 1))
    #expect(throws: ProAppsError.self) {
      try SpeechTimedText.segments(from: text, audioDuration: 5)
    }
  }

  @Test func timedRunBudgetAndCancellationAreEnforced() async throws {
    guard #available(macOS 26.0, *) else {
      Issue.record("Requires macOS 26")
      return
    }
    let text = (0...1000).map { index in
      var value = AttributedString("語")
      value.audioTimeRange = CMTimeRange(
        start: CMTime(value: Int64(index), timescale: 100),
        duration: CMTime(value: 1, timescale: 100))
      return value
    }.reduce(AttributedString(), +)
    #expect(throws: ProAppsError.self) {
      try SpeechTimedText.segments(from: text, audioDuration: 20)
    }
    await withTaskGroup(of: Void.self) { group in
      group.cancelAll()
      group.addTask {
        #expect(throws: CancellationError.self) {
          try SpeechTimedText.segments(from: text, audioDuration: 20)
        }
      }
    }
  }
}
