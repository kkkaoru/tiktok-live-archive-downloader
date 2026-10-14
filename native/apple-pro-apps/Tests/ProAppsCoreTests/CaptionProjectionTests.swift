import Testing

@testable import ProAppsCore

struct CaptionProjectionTests {
  @Test func splitsTextAndProjectsTwoCutsWithEstimatedTiming() throws {
    let result = try CaptionProjection.make(
      .init(
        sourceCaptions: [.init(text: "abcdefghijklmnop", startSeconds: 0, endSeconds: 8)],
        retainedSpans: [
          .init(startSeconds: 1, endSeconds: 3), .init(startSeconds: 5, endSeconds: 7),
        ],
        maximumCharacters: 8))
    #expect(result.outputDurationSeconds == 4)
    #expect(result.captions.count == 2)
    #expect(result.captions[0].text == "abcdefgh")
    #expect(result.captions[0].startSeconds == 0)
    #expect(result.captions[0].endSeconds == 2)
    #expect(result.captions[1].text == "ijklmnop")
    #expect(result.captions[1].startSeconds == 2)
    #expect(result.captions[1].endSeconds == 4)
    #expect(result.timingIsEstimated)
    #expect(!result.humanReviewed)
  }

  @Test func oneCaptionAcrossRemovedGapDoesNotRepeatItsText() throws {
    let result = try CaptionProjection.make(
      .init(
        sourceCaptions: [.init(text: "一つの字幕", startSeconds: 0, endSeconds: 8)],
        retainedSpans: [
          .init(startSeconds: 1, endSeconds: 3), .init(startSeconds: 5, endSeconds: 7),
        ],
        maximumCharacters: 24))
    #expect(result.captions.count == 1)
    #expect(result.captions[0].startSeconds == 0)
    #expect(result.captions[0].endSeconds == 4)
  }

  @Test func sourceOverlapIsCountedAndContainedCaptionsAreOmitted() throws {
    let result = try CaptionProjection.make(
      .init(
        sourceCaptions: [
          .init(text: "first", startSeconds: 0, endSeconds: 2),
          .init(text: "second", startSeconds: 1, endSeconds: 3),
          .init(text: "contained", startSeconds: 1.5, endSeconds: 2),
          .init(text: "removed", startSeconds: 4, endSeconds: 5),
        ], retainedSpans: [.init(startSeconds: 0, endSeconds: 3)], maximumCharacters: 24))
    #expect(result.captions.count == 2)
    #expect(result.captions[1].startSeconds == 2)
    #expect(result.overlapAdjustedCount == 2)
    #expect(result.omittedCaptionCount == 2)
  }

  @Test func prefersPunctuationAndOmitsWhitespaceOnlyPieces() throws {
    let punctuation = try CaptionProjection.make(
      .init(
        sourceCaptions: [
          .init(text: "あいうえお。かきくけこさ", startSeconds: 0, endSeconds: 12)
        ], retainedSpans: [.init(startSeconds: 0, endSeconds: 12)], maximumCharacters: 8))
    #expect(punctuation.captions[0].text == "あいうえお。")
    #expect(punctuation.captions[0].endSeconds == 6)
    let whitespace = try CaptionProjection.make(
      .init(
        sourceCaptions: [
          .init(text: "abcdefgh        ijklmnop", startSeconds: 0, endSeconds: 24)
        ], retainedSpans: [.init(startSeconds: 0, endSeconds: 24)], maximumCharacters: 8))
    #expect(whitespace.captions.count == 2)
    #expect(whitespace.captions[1].startSeconds == 16)
  }

  @Test func rejectsInvalidInputRatherThanInventingATimeline() {
    #expect(throws: ProAppsError.self) {
      try CaptionProjection.make(
        .init(sourceCaptions: [], retainedSpans: [], maximumCharacters: 24))
    }
    #expect(throws: ProAppsError.self) {
      try CaptionProjection.make(
        .init(
          sourceCaptions: [.init(text: "x", startSeconds: 0, endSeconds: 1)],
          retainedSpans: [.init(startSeconds: -1, endSeconds: 1)], maximumCharacters: 24))
    }
    #expect(throws: ProAppsError.self) {
      try CaptionProjection.make(
        .init(
          sourceCaptions: [.init(text: " ", startSeconds: 0, endSeconds: 1)],
          retainedSpans: [.init(startSeconds: 0, endSeconds: 1)], maximumCharacters: 24))
    }
    #expect(throws: ProAppsError.self) {
      try CaptionProjection.make(
        .init(
          sourceCaptions: Array(
            repeating:
              .init(text: String(repeating: "x", count: 2048), startSeconds: 0, endSeconds: 1),
            count: 1000),
          retainedSpans: [.init(startSeconds: 0, endSeconds: 1)], maximumCharacters: 8))
    }
  }
}
