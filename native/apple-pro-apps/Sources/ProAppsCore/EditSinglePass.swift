import Foundation

/// Encoder choice for the single-pass, video-only reader/writer render. Supplying it
/// replaces the export-then-effects double encode with one hardware encode.
public struct EditEncoding: Codable, Equatable, Sendable {
  public enum Codec: String, Codable, Sendable { case h264, hevc }
  public static let bitRates = 500_000...80_000_000

  public let codec: Codec
  /// Average bits per second requested from the encoder (not a hard peak limit).
  public let averageBitRate: Int
  /// False disables B-frames (frame reordering), matching `-bf 0`.
  public let allowFrameReordering: Bool

  public init(codec: Codec, averageBitRate: Int, allowFrameReordering: Bool) {
    self.codec = codec
    self.averageBitRate = averageBitRate
    self.allowFrameReordering = allowFrameReordering
  }
}

/// An sRGB color parsed from `#RRGGBB`.
public struct EditRGB: Equatable, Sendable {
  public let red: Double
  public let green: Double
  public let blue: Double

  public init(red: Double, green: Double, blue: Double) {
    self.red = red
    self.green = green
    self.blue = blue
  }

  public init?(hex: String) {
    let digits = Array(hex.utf8)
    guard digits.count == 7, digits[0] == UInt8(ascii: "#"),
      let value = UInt32(String(hex.dropFirst()), radix: 16)
    else { return nil }
    red = Double((value >> 16) & 0xFF) / 255
    green = Double((value >> 8) & 0xFF) / 255
    blue = Double(value & 0xFF) / 255
  }
}

/// libass-compatible caption look: glyph fill, an inner border and an outer rim,
/// each drawn outside the glyph (an ASS `\bord` stack). `assFontSize` follows
/// libass: the font's ascent plus descent in output pixels, not the em size.
public struct EditCaptionAppearance: Codable, Equatable, Sendable {
  public let fontName: String
  public let assFontSize: Double
  public let fill: String
  public let border: Double
  public let borderColor: String
  public let rim: Double
  public let rimColor: String

  public init(
    fontName: String, assFontSize: Double, fill: String, border: Double, borderColor: String,
    rim: Double, rimColor: String
  ) {
    self.fontName = fontName
    self.assFontSize = assFontSize
    self.fill = fill
    self.border = border
    self.borderColor = borderColor
    self.rim = rim
    self.rimColor = rimColor
  }
}

/// One positioned caption. Lines are the supplied `\n`-separated lines (no
/// automatic wrapping); the block is centered on `x` with its bottom edge at
/// `bottom`, in top-left output pixels (an ASS `\an2\pos(x,bottom)`). Times are a
/// half-open output interval; captions may overlap in time.
public struct EditStyledCaption: Codable, Equatable, Sendable {
  public let text: String
  public let startSeconds: Double
  public let endSeconds: Double
  public let x: Double
  public let bottom: Double

  public init(text: String, startSeconds: Double, endSeconds: Double, x: Double, bottom: Double) {
    self.text = text
    self.startSeconds = startSeconds
    self.endSeconds = endSeconds
    self.x = x
    self.bottom = bottom
  }

  public var lines: [String] { text.components(separatedBy: "\n") }

  public func isActive(at seconds: Double) -> Bool {
    seconds >= startSeconds && seconds < endSeconds
  }
}

extension EditPlan {
  public static let maximumStyledCaptions = 400
  public static let maximumStyledCaptionLines = 6
  public static let maximumCaptionEdge = 16.0
  public static let assFontSizes = 8.0...200.0
  static let postScriptNameCharacters = CharacterSet(
    charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")

  /// Single-pass rules. The render is video-only like `ffmpeg -an`; audio requests
  /// are refused rather than silently dropped.
  static func validateSinglePass(_ recipe: EditRecipe, duration: Double) throws {
    guard let video = recipe.video else { return }
    let captions = video.styledCaptions ?? []
    if let path = video.foregroundVideoPath {
      _ = try Files.absolute(path)
      guard video.encoding != nil else {
        throw ProAppsError.invalid("foregroundVideoPath requires video.encoding")
      }
    }
    guard let encoding = video.encoding else {
      guard captions.isEmpty, video.captionAppearance == nil else {
        throw ProAppsError.invalid("styledCaptions require video.encoding (single-pass render)")
      }
      return
    }
    guard EditEncoding.bitRates.contains(encoding.averageBitRate) else {
      throw ProAppsError.invalid("Encoding averageBitRate must be 500000–80000000 bits/s")
    }
    guard (recipe.additionalAudio ?? []).isEmpty, recipe.clips.allSatisfy({ $0.audio == nil })
    else {
      throw ProAppsError.invalid(
        "Single-pass encoding is video-only; remove audio adjustments and additional audio")
    }
    guard captions.count <= maximumStyledCaptions else {
      throw ProAppsError.invalid("At most 400 styled captions are supported")
    }
    guard !captions.isEmpty else { return }
    guard let appearance = video.captionAppearance else {
      throw ProAppsError.invalid("styledCaptions require captionAppearance")
    }
    try validate(appearance)
    for caption in captions {
      let lines = caption.lines
      guard (1...maximumStyledCaptionLines).contains(lines.count),
        lines.allSatisfy({ !$0.trimmingCharacters(in: .whitespaces).isEmpty }),
        caption.text.count <= 200, caption.text.utf8.count <= 2048, !caption.text.contains("\0"),
        !caption.text.contains("\r"),
        caption.startSeconds.isFinite, caption.endSeconds.isFinite, caption.startSeconds >= 0,
        caption.endSeconds <= duration + minimumTimeSeconds,
        caption.endSeconds - caption.startSeconds >= minimumTimeSeconds,
        caption.x.isFinite, caption.bottom.isFinite, caption.x > 0,
        caption.x < Double(video.width), caption.bottom > 0,
        caption.bottom <= Double(video.height)
      else {
        throw ProAppsError.invalid(
          "Styled captions need 1–6 nonblank lines, bounded text, an in-canvas anchor and times inside the output"
        )
      }
    }
  }

  static func validate(_ appearance: EditCaptionAppearance) throws {
    guard !appearance.fontName.isEmpty, appearance.fontName.utf8.count <= 128,
      appearance.fontName.unicodeScalars.allSatisfy(Self.postScriptNameCharacters.contains),
      appearance.assFontSize.isFinite, assFontSizes.contains(appearance.assFontSize),
      EditRGB(hex: appearance.fill) != nil, EditRGB(hex: appearance.borderColor) != nil,
      EditRGB(hex: appearance.rimColor) != nil,
      appearance.border.isFinite, (0...maximumCaptionEdge).contains(appearance.border),
      appearance.rim.isFinite, (0...maximumCaptionEdge).contains(appearance.rim)
    else {
      throw ProAppsError.invalid(
        "Caption appearance needs a PostScript font name, assFontSize 8–200, #RRGGBB colors and edges 0–16 pixels"
      )
    }
  }
}
