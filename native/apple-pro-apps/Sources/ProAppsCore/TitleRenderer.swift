import CoreGraphics
import CoreImage.CIFilterBuiltins
import Foundation
import SwiftUI

/// SwiftUI's offscreen renderer requires MainActor. Only bounded text rendering
/// runs there; native video export and file I/O remain on their own executors.
/// No window, screenshot, attributed-string dictionary or pixel pointer is used.
enum TitleRenderer {
  static let maximumOverlayPixels = 16_777_216

  static func captionMargin(_ canvas: EditVideoSettings) -> Double {
    min(max(8, Double(canvas.height) / 30), Double(canvas.width) / 8)
  }

  static func captionBottomMargin(_ canvas: EditVideoSettings, imageHeight: Int = 0) -> Double {
    if let center = canvas.captionStyle?.centerY {
      return Double(canvas.height) - center - Double(imageHeight) / 2
    }
    return canvas.captionStyle?.bottomMargin ?? captionMargin(canvas)
  }

  static func addingPixels(width: Int, height: Int, to current: Int) throws -> Int {
    guard width > 0, height > 0, current >= 0, current <= maximumOverlayPixels,
      width <= (maximumOverlayPixels - current) / height
    else { throw ProAppsError.invalid("Combined text overlay pixel budget exceeded") }
    return current + width * height
  }

  @MainActor
  static func renderCaption(_ caption: EditCaption, canvas: EditVideoSettings) throws -> CGImage {
    try Task.checkCancellation()
    try EditPlan.validate(canvas)
    try EditPlan.validate([caption], duration: EditPlan.maximumDurationSeconds)
    guard canvas.width >= 160, canvas.height >= 90 else {
      throw ProAppsError.invalid("Caption canvas must be at least 160 by 90 pixels")
    }
    let padding = 8.0
    let margin = captionMargin(canvas)
    let minimumFontSize = 16.0
    let maximumFontSize = 48.0
    let fontSize =
      canvas.captionStyle?.fontSize
      ?? min(maximumFontSize, max(minimumFontSize, Double(canvas.height) / 24))
    let view = Text(verbatim: caption.text)
      .font(.system(size: fontSize, weight: .bold))
      .foregroundStyle(.white)
      .multilineTextAlignment(.center)
      .frame(width: Double(canvas.width) - 2 * margin - 2 * padding)
      .fixedSize(horizontal: false, vertical: true)
      .padding(padding)
      // Background is composited after outline dilation, so it cannot become
      // part of the glyph mask. The padding is wider than the maximum outline.
      .background(.clear)
    let renderer = ImageRenderer(content: view)
    renderer.scale = 1
    renderer.isOpaque = false
    guard let image = renderer.cgImage else {
      throw ProAppsError.unavailable("Offscreen caption rendering is unavailable")
    }
    guard image.width > 0, image.height > 0, image.width <= canvas.width,
      image.height <= canvas.height / 2,
      captionBottomMargin(canvas, imageHeight: image.height) >= 0,
      Double(image.height) + captionBottomMargin(canvas, imageHeight: image.height)
        <= Double(canvas.height)
    else { throw ProAppsError.invalid("Caption exceeds the available area; shorten the cue") }
    return image
  }

  static func captionImage(_ bitmap: CGImage, canvas: EditVideoSettings) throws -> CIImage {
    try EditPlan.validate(canvas)
    let glyphs = CIImage(cgImage: bitmap)
    let style = canvas.captionStyle ?? EditCaptionStyle(outlineWidth: 0, backgroundOpacity: 0.7)
    var image = glyphs
    if style.outlineWidth > 0 {
      let dilation = CIFilter.morphologyMaximum()
      dilation.inputImage = glyphs
      dilation.radius = Float(style.outlineWidth)
      let black = CIFilter.colorMatrix()
      black.inputImage = dilation.outputImage
      black.rVector = CIVector(x: 0, y: 0, z: 0, w: 0)
      black.gVector = CIVector(x: 0, y: 0, z: 0, w: 0)
      black.bVector = CIVector(x: 0, y: 0, z: 0, w: 0)
      black.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)
      guard let outline = black.outputImage else {
        throw ProAppsError.unavailable("Caption outline filter produced no image")
      }
      image = glyphs.composited(over: outline)
    }
    let background = CIImage(
      color: CIColor(red: 0, green: 0, blue: 0, alpha: style.backgroundOpacity)
    )
    .cropped(to: glyphs.extent)
    return image.composited(over: background).cropped(to: glyphs.extent)
  }

  @MainActor
  static func render(_ title: EditTitle, canvas: EditVideoSettings) throws -> CGImage {
    try Task.checkCancellation()
    try EditPlan.validate(title, canvas: canvas)
    let view = Text(verbatim: title.text)
      .font(.system(size: title.fontSize, weight: .bold))
      .foregroundStyle(.white)
      .fixedSize()
    let renderer = ImageRenderer(content: view)
    renderer.scale = 1
    renderer.isOpaque = false
    guard let image = renderer.cgImage else {
      throw ProAppsError.unavailable("Offscreen text rendering is unavailable")
    }
    guard image.width > 0, image.height > 0,
      Double(image.width) + title.x <= Double(canvas.width),
      Double(image.height) + title.y <= Double(canvas.height)
    else {
      throw ProAppsError.invalid("Rendered title exceeds the canvas; shorten it or reduce its size")
    }
    return image
  }
}
