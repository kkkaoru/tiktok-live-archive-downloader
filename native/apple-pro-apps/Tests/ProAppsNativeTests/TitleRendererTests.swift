import CoreImage
import Foundation
import Testing

@testable import ProAppsCore

struct TitleRendererTests {
  @Test(arguments: ["短い字幕", "複数行でも字幕の中心位置を変えずに表示するための検証です。"])
  @MainActor func captionBlockCenterDoesNotDependOnLineCount(_ text: String) throws {
    let canvas = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      captionStyle: .init(outlineWidth: 3, backgroundOpacity: 0, fontSize: 16, centerY: 120))
    let image = try TitleRenderer.renderCaption(
      .init(text: text, startSeconds: 0, endSeconds: 1), canvas: canvas)
    let bottom = TitleRenderer.captionBottomMargin(canvas, imageHeight: image.height)
    #expect(bottom + Double(image.height) / 2 == 120)
    let decoded = try JSONDecoder().decode(
      EditVideoSettings.self, from: JSONEncoder().encode(canvas))
    #expect(decoded.captionStyle?.centerY == 120)
  }

  @Test(arguments: [0.0, 240, Double.nan])
  func rejectsInvalidCenter(_ center: Double) {
    let canvas = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      captionStyle: .init(outlineWidth: 3, backgroundOpacity: 0, centerY: center))
    #expect(throws: ProAppsError.self) { try EditPlan.validate(canvas) }
  }

  @Test func rejectsConflictingCaptionPositionModes() {
    let canvas = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      captionStyle: .init(outlineWidth: 3, backgroundOpacity: 0, bottomMargin: 10, centerY: 120))
    #expect(throws: ProAppsError.self) { try EditPlan.validate(canvas) }
  }

  @Test @MainActor func offscreenTextHasVisibleGlyphsWithoutTruncation() throws {
    let title = EditTitle(text: "TEST", x: 8, y: 8, fontSize: 32)
    let canvas = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit, titles: [title])
    let image = try TitleRenderer.render(title, canvas: canvas)
    #expect(image.width > 32 && image.width < 312)
    #expect(image.height > 16 && image.height < 232)
    let pixels = CIImage(cgImage: image)
    let opaque = pixels.composited(over: CIImage(color: .black).cropped(to: pixels.extent))
    let color = try FrameProbe.average(opaque, region: pixels.extent, context: CIContext())
    #expect(color.red > 0.02)
    #expect(abs(color.red - color.green) < 0.01)
    let saved = try JSONDecoder().decode(EditVideoSettings.self, from: JSONEncoder().encode(canvas))
    #expect(saved.titles?.first?.text == "TEST")
  }

  @Test @MainActor func captionWrapsAndRefusesInsufficientCanvas() throws {
    let caption = EditCaption(text: "これは折り返しを確認する字幕です。\n二行目の字幕です。", startSeconds: 0, endSeconds: 1)
    let image = try TitleRenderer.renderCaption(
      caption, canvas: .init(width: 320, height: 240, frameRate: 30, resizeMode: .fit))
    #expect(image.width <= 320 && image.width > 200)
    #expect(image.height > 32 && image.height <= 120)
    #expect(throws: ProAppsError.self) {
      try TitleRenderer.renderCaption(
        caption, canvas: .init(width: 32, height: 32, frameRate: 30, resizeMode: .fit))
    }
    #expect(throws: ProAppsError.self) {
      try TitleRenderer.renderCaption(
        .init(text: String(repeating: "字", count: 120), startSeconds: 0, endSeconds: 1),
        canvas: .init(width: 160, height: 90, frameRate: 30, resizeMode: .fit))
    }
  }

  @Test @MainActor func outlineExpandsGlyphsWithoutAddingAnOpaqueBox() throws {
    let caption = EditCaption(text: "袋文字 TEST", startSeconds: 0, endSeconds: 1)
    let plain = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      captionStyle: .init(outlineWidth: 0, backgroundOpacity: 0))
    let outlined = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      captionStyle: .init(outlineWidth: 3, backgroundOpacity: 0))
    let bitmap = try TitleRenderer.renderCaption(caption, canvas: plain)
    let base = try TitleRenderer.captionImage(bitmap, canvas: plain)
    let decorated = try TitleRenderer.captionImage(bitmap, canvas: outlined)
    let white = CIImage(color: .white).cropped(to: base.extent)
    let black = CIImage(color: .black).cropped(to: base.extent)
    let context = CIContext()
    let before = try FrameProbe.average(
      base.composited(over: white), region: base.extent, context: context)
    let after = try FrameProbe.average(
      decorated.composited(over: white), region: base.extent, context: context)
    #expect(before.red > 0.99)
    #expect(before.red - after.red > 0.01)
    let corner = try FrameProbe.average(
      decorated.composited(over: white), region: CGRect(x: 0, y: 0, width: 4, height: 4),
      context: context)
    #expect(corner.red > 0.99)
    let plainGlyphs = try FrameProbe.average(
      base.composited(over: black), region: base.extent, context: context)
    let outlinedGlyphs = try FrameProbe.average(
      decorated.composited(over: black), region: base.extent, context: context)
    #expect(abs(plainGlyphs.red - outlinedGlyphs.red) < 0.015)
  }

  @Test @MainActor func positionedCaptionMustFitAboveItsBottomMargin() throws {
    let cue = EditCaption(text: "TEST", startSeconds: 0, endSeconds: 1)
    let good = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      captionStyle: .init(outlineWidth: 3, backgroundOpacity: 0, fontSize: 32, bottomMargin: 40))
    let image = try TitleRenderer.renderCaption(cue, canvas: good)
    #expect(image.height > 32 && image.height < 100)
    #expect(TitleRenderer.captionBottomMargin(good) == 40)
    let restored = try JSONDecoder().decode(
      EditVideoSettings.self, from: JSONEncoder().encode(good))
    #expect(restored.captionStyle?.fontSize == 32)
    #expect(restored.captionStyle?.bottomMargin == 40)
    #expect(throws: ProAppsError.self) {
      try TitleRenderer.renderCaption(
        cue,
        canvas: .init(
          width: 320, height: 240, frameRate: 30, resizeMode: .fit,
          captionStyle: .init(
            outlineWidth: 3, backgroundOpacity: 0, fontSize: 32, bottomMargin: 230)))
    }
  }

  @Test func aggregateBitmapBudgetCannotOverflow() throws {
    #expect(try TitleRenderer.addingPixels(width: 320, height: 40, to: 0) == 12800)
    #expect(try TitleRenderer.addingPixels(width: 1, height: 1, to: 16_777_215) == 16_777_216)
    #expect(throws: ProAppsError.self) {
      try TitleRenderer.addingPixels(width: 1, height: 1, to: 16_777_216)
    }
    #expect(throws: ProAppsError.self) {
      try TitleRenderer.addingPixels(width: Int.max, height: Int.max, to: 0)
    }
    #expect(throws: ProAppsError.self) {
      try TitleRenderer.addingPixels(width: 1, height: 0, to: 0)
    }
  }

  @Test @MainActor func oversizedTextIsRefusedInsteadOfClipped() {
    #expect(throws: (any Error).self) {
      try TitleRenderer.render(
        .init(text: "Too wide", x: 0, y: 0, fontSize: 128),
        canvas: .init(width: 32, height: 32, frameRate: 30, resizeMode: .fit))
    }
  }
}
