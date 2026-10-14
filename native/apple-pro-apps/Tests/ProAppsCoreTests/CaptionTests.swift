import Foundation
import Testing

@testable import ProAppsCore

struct CaptionTests {
  private func recipe(_ captions: [EditCaption]) -> EditRecipe {
    .init(
      clips: [
        .init(
          sourcePath: "/tmp/caption-fixture.mp4",
          selection: .init(startSeconds: 0, durationSeconds: 2, rate: 1))
      ], video: .init(width: 320, height: 240, frameRate: 30, resizeMode: .fit, captions: captions))
  }

  @Test func adjacentCuesAndUnicodeRoundTrip() throws {
    let value = recipe([
      .init(text: "最初の字幕", startSeconds: 0, endSeconds: 1),
      .init(text: "次の字幕\n二行目です", startSeconds: 1, endSeconds: 2),
    ])
    #expect(try EditPlan.build(value).durationSeconds == 2)
    let decoded = try JSONDecoder().decode(EditRecipe.self, from: JSONEncoder().encode(value))
    #expect(decoded.video?.captions?.count == 2)
    #expect(decoded.video?.captions?.last?.text == "次の字幕\n二行目です")
    #expect(try EditPlan.build(recipe([])).durationSeconds == 2)
  }

  @Test(arguments: [
    "", " ", "bad\0text", String(repeating: "a", count: 121), String(repeating: "👨‍👩‍👧‍👦", count: 50),
  ])
  func invalidTextFailsBeforeRendering(_ text: String) {
    #expect(throws: ProAppsError.self) {
      try EditPlan.build(recipe([.init(text: text, startSeconds: 0, endSeconds: 1)]))
    }
  }

  @Test(arguments: [
    (-1.0, 1.0), (0.0, 0.0), (0.0, 2.01), (Double.nan, 1.0), (0.0, Double.infinity),
  ])
  func invalidTimesFailBeforeRendering(_ start: Double, _ end: Double) {
    #expect(throws: ProAppsError.self) {
      try EditPlan.build(recipe([.init(text: "caption", startSeconds: start, endSeconds: end)]))
    }
  }

  @Test func tinyCaptionCanvasIsRefusedDuringPlanning() {
    let value = EditRecipe(
      clips: [
        .init(
          sourcePath: "/tmp/source.mp4",
          selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
      ],
      video: .init(
        width: 32, height: 32, frameRate: 30, resizeMode: .fit,
        captions: [.init(text: "caption", startSeconds: 0, endSeconds: 1)]))
    #expect(throws: ProAppsError.self) { try EditPlan.build(value) }
  }

  @Test func overlappingUnorderedAndExcessiveCuesAreRefused() {
    #expect(throws: ProAppsError.self) {
      try EditPlan.build(
        recipe([
          .init(text: "one", startSeconds: 0, endSeconds: 1.1),
          .init(text: "two", startSeconds: 1, endSeconds: 2),
        ]))
    }
    #expect(throws: ProAppsError.self) {
      try EditPlan.build(
        recipe(Array(repeating: .init(text: "caption", startSeconds: 0, endSeconds: 1), count: 121))
      )
    }
  }
}
