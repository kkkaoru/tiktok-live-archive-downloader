import Accelerate
import AppKit
import CoreGraphics
import CoreText
import Foundation

/// Caption style 20 as Motion renders its finished three-object titles, in output pixels.
/// Motion grows an outline by an axis-aligned square of the outline's width, not a disc: the
/// band is that wide beside horizontal and vertical edges and √2 times as wide across
/// diagonal ones, convex corners stay square and holes shrink into diamonds (measured on
/// Motion renders of 〇, 口, ◆, ＼ and ／).
public struct MotionCaptionStyle: Codable, Equatable, Sendable {
  public let fontName: String
  public let fontSize: Double
  /// Growth of the face beyond CoreText's outline.
  public let faceGrow: Double
  /// Black keyline, white band and outer black rim growth beyond the outline.
  public let keyline: Double
  public let whiteBorder: Double
  public let rim: Double
  /// The rim object's shadow is cast by the glyphs alone: Gaussian, then moved down.
  public let shadowOpacity: Double
  public let shadowOffsetY: Double
  public let shadowSigma: Double
  /// sRGB intent of the gradient's two colours (#RRGGBB).
  public let gradientTop: String
  public let gradientBottom: String
  /// Gradient stops as fractions of each line's text box (ascent + descent), from its top.
  public let topLocation: Double
  public let bottomLocation: Double
  /// Motion writes linear light with this pure power law into its BT.709-tagged output.
  public let outputGamma: Double

  public init(
    fontName: String, fontSize: Double, faceGrow: Double, keyline: Double, whiteBorder: Double,
    rim: Double, shadowOpacity: Double, shadowOffsetY: Double, shadowSigma: Double,
    gradientTop: String, gradientBottom: String, topLocation: Double, bottomLocation: Double,
    outputGamma: Double
  ) {
    self.fontName = fontName
    self.fontSize = fontSize
    self.faceGrow = faceGrow
    self.keyline = keyline
    self.whiteBorder = whiteBorder
    self.rim = rim
    self.shadowOpacity = shadowOpacity
    self.shadowOffsetY = shadowOffsetY
    self.shadowSigma = shadowSigma
    self.gradientTop = gradientTop
    self.gradientBottom = gradientBottom
    self.topLocation = topLocation
    self.bottomLocation = bottomLocation
    self.outputGamma = outputGamma
  }

  func validate() throws {
    let lengths = [
      fontSize, faceGrow, keyline, whiteBorder, rim, shadowOffsetY, shadowSigma, outputGamma,
    ]
    guard lengths.allSatisfy({ $0.isFinite && $0 >= 0 }), fontSize > 0, outputGamma > 0,
      faceGrow <= keyline, keyline <= whiteBorder, whiteBorder <= rim,
      (0...1).contains(shadowOpacity),
      (0...1).contains(topLocation), (topLocation...1).contains(bottomLocation),
      EditRGB(hex: gradientTop) != nil, EditRGB(hex: gradientBottom) != nil
    else { throw ProAppsError.invalid("Invalid Motion caption style") }
  }
}

/// One revealing title: its final text, frames [startFrame, endFrame) and the frame each
/// grapheme (line breaks included) appears on, as `replay.motion_reveal_stages` records them.
/// The position is Motion's: the first line's baseline centre, origin at the canvas centre, y up.
public struct MotionCaptionTitle: Codable, Equatable, Sendable {
  public let text: String
  public let startFrame: Int
  public let endFrame: Int
  public let x: Double
  public let y: Double
  public let graphemeFrames: [Int]
  /// This title's own style when it differs from the part's (a title edited in Motion).
  public let style: MotionCaptionStyle?

  public init(
    text: String, startFrame: Int, endFrame: Int, x: Double, y: Double, graphemeFrames: [Int],
    style: MotionCaptionStyle? = nil
  ) {
    self.text = text
    self.startFrame = startFrame
    self.endFrame = endFrame
    self.x = x
    self.y = y
    self.graphemeFrames = graphemeFrames
    self.style = style
  }

  /// Graphemes shown on `frame`: Type On End shows each from its onset frame onward.
  public func visible(at frame: Int) -> Int {
    guard (startFrame..<endFrame).contains(frame) else { return 0 }
    return graphemeFrames.count { $0 <= frame }
  }

  func validate() throws {
    guard !text.isEmpty, startFrame >= 0, startFrame < endFrame,
      graphemeFrames.count == text.count,
      graphemeFrames.allSatisfy({ (startFrame..<endFrame).contains($0) }),
      zip(graphemeFrames, graphemeFrames.dropFirst()).allSatisfy({ $0 <= $1 }),
      x.isFinite, y.isFinite
    else { throw ProAppsError.invalid("Invalid Motion caption title") }
  }
}

/// A rendered caption state: straight-alpha RGBA8 placed at `origin` (top-left output pixels).
public struct MotionCaptionBitmap: Sendable {
  public let originX: Int
  public let originY: Int
  public let width: Int
  public let height: Int
  public let rgba: [UInt8]
}

/// Lays a title out once with CoreText and draws any prefix of its graphemes in that fixed
/// layout, as Motion's Type On does (the line and its centring never reflow).
public struct MotionCaptionRenderer {
  struct Glyph {
    let grapheme: Int
    let path: CGPath
  }

  let style: MotionCaptionStyle
  let glyphs: [Glyph]
  /// Top of the first line's text box and the box height, in output pixels.
  let boxTop: Double
  let boxHeight: Double
  let originX: Int
  let originY: Int
  let width: Int
  let height: Int
  let top: (Double, Double, Double)
  let bottom: (Double, Double, Double)

  public init(
    _ title: MotionCaptionTitle, style: MotionCaptionStyle, canvasWidth: Int,
    canvasHeight: Int
  ) throws {
    try style.validate()
    try title.validate()
    let font = CTFontCreateWithName(style.fontName as CFString, style.fontSize, nil)
    guard CTFontCopyPostScriptName(font) as String == style.fontName else {
      throw ProAppsError.invalid("Caption font is not installed: \(style.fontName)")
    }
    let ascent = Double(CTFontGetAscent(font))
    let advance = ascent + Double(CTFontGetDescent(font))
    let centreX = Double(canvasWidth) / 2 + title.x
    let firstBaseline = Double(canvasHeight) / 2 - title.y
    var glyphs: [Glyph] = []
    var grapheme = 0
    for (index, line) in title.text.split(separator: "\n", omittingEmptySubsequences: false)
      .enumerated()
    {
      let baseline = firstBaseline + Double(index) * advance
      glyphs += Self.glyphs(
        String(line), font: font, centreX: centreX, baseline: baseline, firstGrapheme: grapheme)
      grapheme += line.count + 1
    }
    var bounds = CGRect.null
    for glyph in glyphs { bounds = bounds.union(glyph.path.boundingBoxOfPath) }
    guard !bounds.isNull else { throw ProAppsError.invalid("Caption title draws no glyphs") }
    let spread = style.rim + 2
    let shadow = style.shadowOffsetY + 3 * style.shadowSigma + 2
    let left = (bounds.minX - max(spread, 3 * style.shadowSigma + 2)).rounded(.down)
    let right = (bounds.maxX + max(spread, 3 * style.shadowSigma + 2)).rounded(.up)
    let upper = (bounds.minY - spread).rounded(.down)
    let lower = (bounds.maxY + max(spread, shadow)).rounded(.up)
    guard let topColour = EditRGB(hex: style.gradientTop),
      let bottomColour = EditRGB(hex: style.gradientBottom)
    else { throw ProAppsError.invalid("Caption colors must be #RRGGBB") }
    self.style = style
    self.glyphs = glyphs
    self.boxTop = firstBaseline - ascent
    self.boxHeight = advance
    self.originX = Int(left)
    self.originY = Int(upper)
    self.width = Int(right - left)
    self.height = Int(lower - upper)
    self.top = (topColour.red, topColour.green, topColour.blue)
    self.bottom = (bottomColour.red, bottomColour.green, bottomColour.blue)
  }

  /// Positioned outlines (top-left output pixels) of one line's glyphs, each tagged with
  /// the title-wide index of the grapheme it draws.
  static func glyphs(
    _ line: String, font: CTFont, centreX: Double, baseline: Double, firstGrapheme: Int
  ) -> [Glyph] {
    guard !line.isEmpty else { return [] }
    let attributed = NSAttributedString(
      string: line, attributes: [NSAttributedString.Key(kCTFontAttributeName as String): font])
    let ctLine = CTLineCreateWithAttributedString(attributed)
    let left = centreX - CTLineGetTypographicBounds(ctLine, nil, nil, nil) / 2
    // UTF-16 offset -> grapheme index within the title.
    var graphemeAt: [Int] = []
    for (index, character) in line.enumerated() {
      graphemeAt += Array(repeating: firstGrapheme + index, count: character.utf16.count)
    }
    var result: [Glyph] = []
    for run in CTLineGetGlyphRuns(ctLine) as? [CTRun] ?? [] {
      let count = CTRunGetGlyphCount(run)
      var ids = [CGGlyph](repeating: 0, count: count)
      var positions = [CGPoint](repeating: .zero, count: count)
      var indices = [CFIndex](repeating: 0, count: count)
      CTRunGetGlyphs(run, CFRange(), &ids)
      CTRunGetPositions(run, CFRange(), &positions)
      CTRunGetStringIndices(run, CFRange(), &indices)
      let runFont = Self.font(of: run) ?? font
      for index in 0..<count {
        var transform = CGAffineTransform(
          a: 1, b: 0, c: 0, d: -1, tx: left + positions[index].x,
          ty: baseline - positions[index].y)
        guard let path = CTFontCreatePathForGlyph(runFont, ids[index], &transform),
          !path.isEmpty
        else { continue }
        result.append(Glyph(grapheme: graphemeAt[indices[index]], path: path))
      }
    }
    return result
  }

  static func font(of run: CTRun) -> CTFont? {
    let attributes = CTRunGetAttributes(run) as NSDictionary
    return (attributes[kCTFontAttributeName] as? NSFont).map { $0 as CTFont }
  }

  /// The title with its first `visible` graphemes drawn: shadow, black rim, white band,
  /// black keyline and the gradient fill, composited back to front in linear light (Motion's
  /// working space) and then encoded for its output.
  public func bitmap(visible: Int) throws -> MotionCaptionBitmap {
    let shown = glyphs.filter { $0.grapheme < visible }
    let path = CGMutablePath()
    for glyph in shown { path.addPath(glyph.path) }
    let glyph = try coverage(path)
    let fill = grown(glyph, by: style.faceGrow)
    let keyline = grown(glyph, by: style.keyline)
    let white = grown(glyph, by: style.whiteBorder)
    let rim = grown(glyph, by: style.rim)
    let shadow = shadowed(glyph)
    var rgba = [UInt8](repeating: 0, count: width * height * 4)
    for row in 0..<height {
      let colour = gradient(at: Double(originY + row) + 0.5)
      for column in 0..<width {
        let pixel = row * width + column
        // Premultiplied "over", back to front; black layers only scale the colour.
        var alpha = shadow[pixel]
        var red: Float = 0
        var green: Float = 0
        var blue: Float = 0
        alpha = rim[pixel] + (1 - rim[pixel]) * alpha
        let band = white[pixel]
        red = band + (1 - band) * red
        green = band + (1 - band) * green
        blue = band + (1 - band) * blue
        alpha = band + (1 - band) * alpha
        let line = keyline[pixel]
        red *= 1 - line
        green *= 1 - line
        blue *= 1 - line
        alpha = line + (1 - line) * alpha
        let face = fill[pixel]
        red = face * colour.0 + (1 - face) * red
        green = face * colour.1 + (1 - face) * green
        blue = face * colour.2 + (1 - face) * blue
        alpha = face + (1 - face) * alpha
        let offset = pixel * 4
        rgba[offset + 3] = Self.byte(alpha)
        guard alpha > 0 else { continue }
        rgba[offset] = Self.byte(encode(red / alpha))
        rgba[offset + 1] = Self.byte(encode(green / alpha))
        rgba[offset + 2] = Self.byte(encode(blue / alpha))
      }
    }
    return MotionCaptionBitmap(
      originX: originX, originY: originY, width: width, height: height, rgba: rgba)
  }

  /// The linear fill colour of an output row: each line's text box maps the gradient, which
  /// blends the sRGB-encoded stop colours.
  func gradient(at y: Double) -> (Float, Float, Float) {
    let inBox = (y - boxTop).truncatingRemainder(dividingBy: boxHeight)
    let location = (inBox < 0 ? inBox + boxHeight : inBox) / boxHeight
    let span = style.bottomLocation - style.topLocation
    let t = span > 0 ? min(max((location - style.topLocation) / span, 0), 1) : 0
    // Motion eases between its stops with smootherstep (fitted on every line of its renders).
    let mix = t * t * t * (t * (6 * t - 15) + 10)
    return (
      Float(Self.linear(top.0 + (bottom.0 - top.0) * mix)),
      Float(Self.linear(top.1 + (bottom.1 - top.1) * mix)),
      Float(Self.linear(top.2 + (bottom.2 - top.2) * mix))
    )
  }

  static func linear(_ value: Double) -> Double {
    value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)
  }

  /// Linear light -> Motion's output code value.
  func encode(_ value: Float) -> Float { pow(value, Float(1 / style.outputGamma)) }

  static func byte(_ value: Float) -> UInt8 { UInt8(min(max(value, 0), 1) * 255 + 0.5) }

  /// Anti-aliased coverage of the glyphs, one Float per pixel.
  func coverage(_ path: CGPath) throws -> [Float] {
    guard
      let context = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
        space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
    else { throw ProAppsError.unavailable("Cannot allocate a caption mask") }
    // Rows of the bitmap run top-down: flip so the paths' top-left coordinates land as is.
    context.translateBy(x: 0, y: CGFloat(height))
    context.scaleBy(x: 1, y: -1)
    context.translateBy(x: CGFloat(-originX), y: CGFloat(-originY))
    context.setFillColor(gray: 1, alpha: 1)
    context.addPath(path)
    context.fillPath()
    guard let data = context.data else {
      throw ProAppsError.unavailable("Cannot read a caption mask")
    }
    let bytes = data.bindMemory(to: UInt8.self, capacity: width * height)
    return (0..<(width * height)).map { Float(bytes[$0]) / 255 }
  }

  /// Coverage dilated by an axis-aligned square of half-side `radius`: horizontal then vertical
  /// running maxima (vImage), a fractional radius blending the two nearest whole windows.
  func grown(_ coverage: [Float], by radius: Double) -> [Float] {
    guard radius > 0 else { return coverage }
    let whole = Int(radius.rounded(.down))
    let part = Float(radius - Double(whole))
    let rows = blended(coverage, whole: whole, part: part, horizontal: true)
    return blended(rows, whole: whole, part: part, horizontal: false)
  }

  func blended(_ values: [Float], whole: Int, part: Float, horizontal: Bool) -> [Float] {
    let inner = maximum(values, size: 2 * whole + 1, horizontal: horizontal)
    guard part > 0 else { return inner }
    let outer = maximum(values, size: 2 * whole + 3, horizontal: horizontal)
    return zip(inner, outer).map { (1 - part) * $0 + part * $1 }
  }

  /// Running maximum over `size` samples along rows or columns.
  func maximum(_ values: [Float], size: Int, horizontal: Bool) -> [Float] {
    var source = values
    var result = [Float](repeating: 0, count: values.count)
    source.withUnsafeMutableBufferPointer { input in
      result.withUnsafeMutableBufferPointer { output in
        var from = vImage_Buffer(
          data: input.baseAddress, height: vImagePixelCount(height),
          width: vImagePixelCount(width), rowBytes: width * MemoryLayout<Float>.stride)
        var to = vImage_Buffer(
          data: output.baseAddress, height: vImagePixelCount(height),
          width: vImagePixelCount(width), rowBytes: width * MemoryLayout<Float>.stride)
        _ = vImageMax_PlanarF(
          &from, &to, nil, 0, 0, vImagePixelCount(horizontal ? 1 : size),
          vImagePixelCount(horizontal ? size : 1), vImage_Flags(kvImageTruncateKernel))
      }
    }
    return result
  }

  /// Glyph coverage blurred by the shadow's Gaussian, moved down and scaled by its opacity.
  func shadowed(_ fill: [Float]) -> [Float] {
    let sigma = style.shadowSigma
    let radius = max(1, Int((3 * sigma).rounded(.up)))
    var kernel = (-radius...radius).map { Float(exp(-Double($0 * $0) / (2 * sigma * sigma))) }
    let total = kernel.reduce(0, +)
    kernel = kernel.map { $0 / total }
    var blurred = [Float](repeating: 0, count: fill.count)
    var source = fill
    source.withUnsafeMutableBufferPointer { input in
      blurred.withUnsafeMutableBufferPointer { output in
        var from = vImage_Buffer(
          data: input.baseAddress, height: vImagePixelCount(height),
          width: vImagePixelCount(width), rowBytes: width * MemoryLayout<Float>.stride)
        var to = vImage_Buffer(
          data: output.baseAddress, height: vImagePixelCount(height),
          width: vImagePixelCount(width), rowBytes: width * MemoryLayout<Float>.stride)
        _ = vImageSepConvolve_PlanarF(
          &from, &to, nil, 0, 0, kernel, UInt32(kernel.count), kernel, UInt32(kernel.count), 0,
          0, vImage_Flags(kvImageBackgroundColorFill))
      }
    }
    let shift = style.shadowOffsetY
    let whole = Int(shift.rounded(.down))
    let part = Float(shift - Double(whole))
    var result = [Float](repeating: 0, count: fill.count)
    let opacity = Float(style.shadowOpacity)
    for row in 0..<height {
      let upper = row - whole - 1
      let lower = row - whole
      for column in 0..<width {
        let above = upper >= 0 && upper < height ? blurred[upper * width + column] : 0
        let below = lower >= 0 && lower < height ? blurred[lower * width + column] : 0
        result[row * width + column] = opacity * (part * above + (1 - part) * below)
      }
    }
    return result
  }
}
