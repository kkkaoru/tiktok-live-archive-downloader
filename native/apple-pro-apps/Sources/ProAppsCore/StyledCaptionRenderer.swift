import CoreImage.CIFilterBuiltins
import CoreText
import Foundation
import SwiftUI

/// The exact installed font behind a caption appearance, with libass sizing applied.
public struct ResolvedCaptionFont: Codable, Equatable, Sendable {
  public let postScriptName: String
  public let family: String
  /// Style name (for example W6); absent when the font reports none.
  public let face: String?
  /// Em size in output pixels. libass requests the "real" size, so winAscent +
  /// winDescent (OS/2, like GDI) at this em size equals `assFontSize`.
  public let pointSize: Double
  /// Distance from a libass line top to its baseline, in output pixels.
  public let lineAscent: Double
  /// libass line advance for `\N` (the ASS font size), in output pixels.
  public let lineAdvance: Double
  /// CoreText (hhea) ascent at `pointSize`; the baseline inside a rendered line.
  public let renderedAscent: Double
}

/// libass-compatible caption bitmaps: glyph fill over an inner border over an outer
/// rim, each grown outside the glyph by morphological dilation (an ASS `\bord`
/// stack). Lines are drawn separately by SwiftUI's offscreen renderer (no
/// attributed-string dictionaries) and stacked on libass line metrics. The font is
/// resolved exactly so no silent fallback is possible.
enum StyledCaptionRenderer {
  static let edgePadding = 2.0
  static let os2WinAscentOffset = 74
  static let os2MinimumLength = 78

  static func resolve(_ appearance: EditCaptionAppearance) throws -> ResolvedCaptionFont {
    try EditPlan.validate(appearance)
    let font = CTFontCreateWithName(appearance.fontName as CFString, 1, nil)
    guard CTFontCopyPostScriptName(font) as String == appearance.fontName else {
      throw ProAppsError.invalid("Caption font is not installed: \(appearance.fontName)")
    }
    let unitsPerEm = Double(CTFontGetUnitsPerEm(font))
    let (ascent, descent) = try verticalMetrics(font, unitsPerEm: unitsPerEm)
    let height = ascent + descent
    let pointSize = appearance.assFontSize / height
    let face = CTFontCopyName(font, kCTFontStyleNameKey).map { $0 as String }
    return ResolvedCaptionFont(
      postScriptName: appearance.fontName, family: CTFontCopyFamilyName(font) as String,
      face: face, pointSize: pointSize, lineAscent: ascent * pointSize,
      lineAdvance: appearance.assFontSize,
      renderedAscent: Double(CTFontGetAscent(font)) * pointSize)
  }

  /// Per-em ascent/descent as libass uses them: OS/2 usWinAscent/usWinDescent when
  /// present and nonzero, otherwise the font's hhea ascent/descent.
  static func verticalMetrics(_ font: CTFont, unitsPerEm: Double) throws -> (Double, Double) {
    guard unitsPerEm > 0 else {
      throw ProAppsError.unavailable("Caption font reports no units per em")
    }
    if let table = CTFontCopyTable(font, CTFontTableTag(kCTFontTableOS2), []) as Data?,
      let metrics = os2WinMetrics(table), metrics.0 + metrics.1 > 0
    {
      return (Double(metrics.0) / unitsPerEm, Double(metrics.1) / unitsPerEm)
    }
    let ascent = Double(CTFontGetAscent(font))
    let descent = Double(CTFontGetDescent(font))
    guard ascent + descent > 0 else {
      throw ProAppsError.unavailable("Caption font reports no vertical metrics")
    }
    return (ascent, descent)
  }

  /// Big-endian usWinAscent/usWinDescent from an OS/2 table, or nil when too short.
  static func os2WinMetrics(_ table: Data) -> (Int, Int)? {
    let bytes = [UInt8](table)
    guard bytes.count >= os2MinimumLength else { return nil }
    let offset = os2WinAscentOffset
    let ascent = Int(bytes[offset]) << 8 | Int(bytes[offset + 1])
    let descent = Int(bytes[offset + 2]) << 8 | Int(bytes[offset + 3])
    return (ascent, descent)
  }

  static func padding(_ appearance: EditCaptionAppearance) -> Double {
    (appearance.border + appearance.rim).rounded(.up) + edgePadding
  }

  /// Glyph coverage (white on clear) for one line, padded for the edges.
  @MainActor
  static func glyphs(_ line: String, font: ResolvedCaptionFont, padding: Double) throws -> CGImage {
    try Task.checkCancellation()
    let view = Text(verbatim: line)
      .font(.custom(font.postScriptName, fixedSize: font.pointSize))
      .foregroundStyle(.white)
      .lineLimit(1)
      .fixedSize()
      .padding(padding)
    let renderer = ImageRenderer(content: view)
    renderer.scale = 1
    renderer.isOpaque = false
    guard let image = renderer.cgImage, image.width > 0, image.height > 0 else {
      throw ProAppsError.unavailable("Offscreen caption rendering is unavailable")
    }
    return image
  }

  /// A finished caption positioned in Core Image (bottom-left origin) coordinates.
  /// Line i's baseline sits at blockTop + i * lineAdvance + lineAscent, where the
  /// block of n lines ends at `bottom`, matching `\an2\pos(x,bottom)`.
  static func image(
    lines: [CGImage], caption: EditStyledCaption, appearance: EditCaptionAppearance,
    font: ResolvedCaptionFont, canvas: EditVideoSettings, colorSpace: CGColorSpace?
  ) throws -> CIImage {
    let pad = padding(appearance)
    let canvasHeight = Double(canvas.height)
    let blockTop = caption.bottom - Double(lines.count) * font.lineAdvance
    var glyphs = CIImage.empty()
    for (index, bitmap) in lines.enumerated() {
      let width = Double(bitmap.width)
      guard width - 2 * pad <= Double(canvas.width) else {
        throw ProAppsError.invalid("Styled caption is wider than the canvas; wrap it into lines")
      }
      let baseline = blockTop + Double(index) * font.lineAdvance + font.lineAscent
      let top = baseline - pad - font.renderedAscent
      let placed = CIImage(cgImage: bitmap).transformed(
        by: CGAffineTransform(
          translationX: caption.x - width / 2, y: canvasHeight - top - Double(bitmap.height)))
      glyphs = placed.composited(over: glyphs)
    }
    let extent = glyphs.extent
    let rim = try tinted(
      dilated(glyphs, radius: appearance.border + appearance.rim, extent: extent),
      hex: appearance.rimColor, colorSpace: colorSpace)
    let border = try tinted(
      dilated(glyphs, radius: appearance.border, extent: extent), hex: appearance.borderColor,
      colorSpace: colorSpace)
    let fill = try tinted(glyphs, hex: appearance.fill, colorSpace: colorSpace)
    return fill.composited(over: border.composited(over: rim)).cropped(to: extent)
  }

  static func dilated(_ image: CIImage, radius: Double, extent: CGRect) throws -> CIImage {
    guard radius > 0 else { return image }
    let filter = CIFilter.morphologyMaximum()
    filter.inputImage = image
    filter.radius = Float(radius)
    return try MaskRenderer.requireOutput(filter.outputImage).cropped(to: extent)
  }

  /// Colors are expressed in the frame's own color space when known, so the hex
  /// value lands unchanged in the output pixels (as libass writes RGB directly).
  static func tinted(_ mask: CIImage, hex: String, colorSpace: CGColorSpace?) throws -> CIImage {
    guard let color = EditRGB(hex: hex) else {
      throw ProAppsError.invalid("Caption colors must be #RRGGBB")
    }
    let ciColor: CIColor
    if let colorSpace,
      let spaced = CIColor(
        red: color.red, green: color.green, blue: color.blue, alpha: 1, colorSpace: colorSpace)
    {
      ciColor = spaced
    } else {
      ciColor = CIColor(red: color.red, green: color.green, blue: color.blue)
    }
    let solid = CIImage(color: ciColor).cropped(to: mask.extent)
    let blend = CIFilter.blendWithAlphaMask()
    blend.inputImage = solid
    blend.backgroundImage = CIImage.empty()
    blend.maskImage = mask
    return try MaskRenderer.requireOutput(blend.outputImage).cropped(to: mask.extent)
  }
}

/// Renders styled captions lazily while frames advance, keeping only active ones.
/// Captions are visited in start order, so memory stays bounded by overlap.
@MainActor
final class StyledCaptionCache {
  private let captions: [(index: Int, caption: EditStyledCaption)]
  private let appearance: EditCaptionAppearance
  private let font: ResolvedCaptionFont
  private let canvas: EditVideoSettings
  private let colorSpace: CGColorSpace?
  private var rendered: [Int: CIImage] = [:]

  /// `colorSpaceName` is the composed frames' color space, used for caption colors.
  init(
    _ captions: [EditStyledCaption], appearance: EditCaptionAppearance,
    canvas: EditVideoSettings, colorSpaceName: String? = nil
  ) throws {
    self.captions = captions.enumerated().map { ($0.offset, $0.element) }
      .sorted { ($0.caption.startSeconds, $0.index) < ($1.caption.startSeconds, $1.index) }
    self.appearance = appearance
    self.font = try StyledCaptionRenderer.resolve(appearance)
    self.canvas = canvas
    self.colorSpace = colorSpaceName.flatMap { CGColorSpace(name: $0 as CFString) }
  }

  var resolvedFont: ResolvedCaptionFont { font }
  var cachedCount: Int { rendered.count }

  /// Active captions at `seconds`, in input order so later cues draw on top.
  func active(at seconds: Double) throws -> [CIImage] {
    rendered = rendered.filter { entry in
      captions.contains { $0.index == entry.key && $0.caption.isActive(at: seconds) }
    }
    var images: [(Int, CIImage)] = []
    for entry in captions {
      guard entry.caption.startSeconds <= seconds else { break }
      guard entry.caption.isActive(at: seconds) else { continue }
      if let image = rendered[entry.index] {
        images.append((entry.index, image))
        continue
      }
      let padding = StyledCaptionRenderer.padding(appearance)
      let lines = try entry.caption.lines.map {
        try StyledCaptionRenderer.glyphs($0, font: font, padding: padding)
      }
      let image = try StyledCaptionRenderer.image(
        lines: lines, caption: entry.caption, appearance: appearance, font: font, canvas: canvas,
        colorSpace: colorSpace)
      rendered[entry.index] = image
      images.append((entry.index, image))
    }
    return images.sorted { $0.0 < $1.0 }.map(\.1)
  }
}
