import Foundation

/// Native offscreen font bounds for layout planning, not Motion render acceptance.
public struct CaptionTextMeasurementRequest: Codable, Sendable {
  public let fontName: String
  public let assFontSize: Double
  public let border: Double
  public let rim: Double
  public let texts: [String]

  public init(fontName: String, assFontSize: Double, border: Double, rim: Double, texts: [String]) {
    self.fontName = fontName
    self.assFontSize = assFontSize
    self.border = border
    self.rim = rim
    self.texts = texts
  }
}

public struct CaptionTextBounds: Codable, Equatable, Sendable {
  public let text: String
  public let width: Double
  public let height: Double
  public let lineWidths: [Double]
}

public struct CaptionTextMeasurementResult: Encodable, Sendable {
  public let font: ResolvedCaptionFont
  public let bounds: [CaptionTextBounds]
  public let motionRenderVerified = false
}

public enum CaptionTextMeasurement {
  public static let maximumTexts = 256
  public static let maximumCharacters = 512
  public static let maximumMultilineCharacters = 120
  public static let maximumTotalCharacters = 4096
  static let maximumDimension = 16_384.0
  static let maximumPixels = 16_000_000

  /// Uses the existing exact-font, libass-sized renderer without launching an app.
  /// Bounds include outline/rim padding. Motion mapping requires independent calibration.
  @MainActor
  public static func measure(_ request: CaptionTextMeasurementRequest) throws
    -> CaptionTextMeasurementResult
  {
    try measure(request) { text, font, padding in
      let image = try StyledCaptionRenderer.glyphs(text, font: font, padding: padding)
      return CGSize(width: image.width, height: image.height)
    }
  }

  @MainActor
  static func measure(
    _ request: CaptionTextMeasurementRequest,
    lineSize: (String, ResolvedCaptionFont, Double) throws -> CGSize
  ) throws -> CaptionTextMeasurementResult {
    guard (1...maximumTexts).contains(request.texts.count),
      request.texts.reduce(0, { $0 + $1.utf16.count }) <= maximumTotalCharacters,
      request.texts.allSatisfy({ text in
        let lines = text.components(separatedBy: "\n")
        let limit = lines.count == 1 ? maximumCharacters : maximumMultilineCharacters
        return (1...limit).contains(text.utf16.count)
          && (1...2).contains(lines.count) && lines.allSatisfy { !$0.isEmpty }
          && !text.contains("\r") && !text.contains("\0")
      })
    else { throw ProAppsError.invalid("Caption measurements exceed text or line budgets") }
    let appearance = EditCaptionAppearance(
      fontName: request.fontName, assFontSize: request.assFontSize, fill: "#FFFFFF",
      border: request.border, borderColor: "#FFFFFF", rim: request.rim, rimColor: "#000000")
    let font = try StyledCaptionRenderer.resolve(appearance)
    let padding = StyledCaptionRenderer.padding(appearance)
    var pixels = 0.0
    var bounds: [CaptionTextBounds] = []
    for text in request.texts {
      try Task.checkCancellation()
      let sizes = try text.components(separatedBy: "\n").map {
        try lineSize($0, font, padding)
      }
      guard
        sizes.allSatisfy({
          $0.width.isFinite && $0.height.isFinite
            && (0...maximumDimension).contains($0.width)
            && (0...maximumDimension).contains($0.height)
            && $0.width > 0 && $0.height > 0
        }), let width = sizes.map(\.width).max(), let height = sizes.map(\.height).max()
      else {
        throw ProAppsError.unavailable("Native caption renderer returned invalid bounds")
      }
      pixels += sizes.reduce(0.0, { $0 + Double($1.width * $1.height) })
      guard pixels <= Double(maximumPixels) else {
        throw ProAppsError.invalid("Caption measurement pixel budget exceeded")
      }
      bounds.append(
        CaptionTextBounds(
          text: text, width: Double(width),
          height: Double(height) + Double(sizes.count - 1) * font.lineAdvance,
          lineWidths: sizes.map { Double($0.width) }))
    }
    return CaptionTextMeasurementResult(font: font, bounds: bounds)
  }
}
