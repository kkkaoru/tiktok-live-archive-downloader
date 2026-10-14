import Foundation
import Testing

@testable import ProAppsCore

@MainActor
struct CaptionTextMeasurementTests {
  @Test func nativeFontMeasuresProportionalGlyphsAndExplicitLines() throws {
    let result = try CaptionTextMeasurement.measure(
      .init(
        fontName: "ArialMT", assFontSize: 66, border: 3, rim: 1,
        texts: ["iiii", "WWWW", "WWWW\niiii"]))
    #expect(result.font.postScriptName == "ArialMT")
    #expect(result.bounds.map(\.text) == ["iiii", "WWWW", "WWWW\niiii"])
    #expect(result.bounds[0].width < result.bounds[1].width)
    #expect(result.bounds[2].height - result.bounds[1].height == 66)
    #expect(result.bounds[2].lineWidths.count == 2)
    #expect(!result.motionRenderVerified)
    let encoded = try JSONEncoder().encode(result)
    #expect(!encoded.isEmpty)
  }

  @Test func sizeAndOutlineChangeMeasuredFootprint() throws {
    let small = try CaptionTextMeasurement.measure(
      .init(
        fontName: "ArialMT", assFontSize: 60, border: 0, rim: 0, texts: ["日本語字幕"]))
    let large = try CaptionTextMeasurement.measure(
      .init(
        fontName: "ArialMT", assFontSize: 84, border: 4, rim: 2, texts: ["日本語字幕"]))
    #expect(large.bounds[0].width > small.bounds[0].width)
    #expect(large.bounds[0].height > small.bounds[0].height)
  }

  @Test(arguments: [
    [], [""], ["a\n\nb"], ["a\r"], ["a\0"],
    [String(repeating: "a", count: 513)],
    [String(repeating: "字", count: 513)],
    [String(repeating: "😀", count: 256) + "a"],
    [String(repeating: "a", count: 119) + "\nb"],
    Array(repeating: "a", count: 257),
    Array(repeating: String(repeating: "a", count: 120), count: 35),
  ])
  func invalidTextBudgets(texts: [String]) {
    #expect(throws: ProAppsError.self) {
      try CaptionTextMeasurement.measure(
        .init(
          fontName: "ArialMT", assFontSize: 66, border: 0, rim: 0, texts: texts))
    }
  }

  @Test(arguments: [
    String(repeating: "a", count: 120),
    String(repeating: "字", count: 120),
    String(repeating: "😀", count: 60),
  ])
  func utf16BoundaryUsesUnitsRatherThanBytes(text: String) throws {
    let result = try CaptionTextMeasurement.measure(
      .init(fontName: "ArialMT", assFontSize: 66, border: 0, rim: 0, texts: [text])
    ) { _, _, _ in CGSize(width: 10, height: 20) }
    #expect(result.bounds.map(\.text) == [text])
  }

  @Test(arguments: [
    String(repeating: "a", count: 512),
    String(repeating: "字", count: 512),
    String(repeating: "😀", count: 256),
  ])
  func longSingleLineRetainsTheCompleteText(text: String) throws {
    let result = try CaptionTextMeasurement.measure(
      .init(fontName: "ArialMT", assFontSize: 66, border: 0, rim: 0, texts: [text])
    ) { actual, _, _ in
      #expect(actual == text)
      return CGSize(width: 12_000, height: 100)
    }
    #expect(result.bounds.map(\.text) == [text])
    #expect(result.bounds.map(\.width) == [12_000])
    #expect(result.bounds.map(\.lineWidths) == [[12_000]])
  }

  @Test func longJapaneseLineIsMeasuredByTheNativeRenderer() throws {
    let result = try CaptionTextMeasurement.measure(
      .init(
        fontName: "ArialMT", assFontSize: 24, border: 1, rim: 1,
        texts: [String(repeating: "字幕", count: 80)]))
    #expect(result.bounds[0].text.utf16.count == 160)
    #expect(result.bounds[0].width > 1000)
    #expect(result.bounds[0].lineWidths.count == 1)
  }

  @Test func invalidGeometryAndPixelBudgetsAreRejected() {
    let request = CaptionTextMeasurementRequest(
      fontName: "ArialMT", assFontSize: 66, border: 0, rim: 0, texts: ["text"])
    #expect(throws: ProAppsError.self) {
      try CaptionTextMeasurement.measure(request) { _, _, _ in CGSize(width: -1, height: 1) }
    }
    #expect(throws: ProAppsError.self) {
      try CaptionTextMeasurement.measure(request) { _, _, _ in CGSize(width: 20_000, height: 1000) }
    }
    #expect(throws: ProAppsError.self) {
      try CaptionTextMeasurement.measure(request) { _, _, _ in CGSize(width: 16_385, height: 1) }
    }
    #expect(throws: ProAppsError.self) {
      try CaptionTextMeasurement.measure(request) { _, _, _ in CGSize(width: 16_000, height: 1001) }
    }
  }

  @Test func missingFontDoesNotSilentlyFallback() {
    #expect(throws: ProAppsError.self) {
      try CaptionTextMeasurement.measure(
        .init(
          fontName: "Missing-Caption-Font", assFontSize: 66, border: 0, rim: 0, texts: ["text"]))
    }
  }

  @Test func knownMultilineBoundsAndRequestRoundtrip() throws {
    let request = CaptionTextMeasurementRequest(
      fontName: "ArialMT", assFontSize: 66, border: 0, rim: 0, texts: ["first\nsecond"])
    let data = try JSONEncoder().encode(request)
    let decoded = try JSONDecoder().decode(CaptionTextMeasurementRequest.self, from: data)
    let result = try CaptionTextMeasurement.measure(decoded) { _, _, _ in
      CGSize(width: 10, height: 20)
    }
    #expect(result.bounds.map(\.width) == [10])
    #expect(result.bounds.map(\.height) == [86])
    #expect(result.bounds.map(\.lineWidths) == [[10, 10]])
  }

  @Test func rendererErrorsArePropagated() {
    let request = CaptionTextMeasurementRequest(
      fontName: "ArialMT", assFontSize: 66, border: 0, rim: 0, texts: ["text"])
    #expect(throws: CancellationError.self) {
      try CaptionTextMeasurement.measure(request) { _, _, _ in throw CancellationError() }
    }
  }
}
