import AppKit
import CoreText
import Foundation

struct FontIdentityRequest: Codable {
  let fontName: String
  let pointSize: Double
  let texts: [String]
}

struct FontVariation: Codable, Equatable {
  let identifier: UInt32
  let value: Double
}

struct FontIdentity: Codable {
  let postScriptName: String
  let family: String
  let face: String?
  let fontURL: URL
  let pointSize: Double
  let variations: [FontVariation]
  let ascent: Double
  let descent: Double
}

struct FontGlyphRun: Codable {
  let utf16Location: Int
  let utf16Length: Int
  let glyphCount: Int
  let font: FontIdentity
}

struct FontTextRuns: Codable {
  let text: String
  let runs: [FontGlyphRun]
}

struct FontIdentityResult: Codable {
  let font: FontIdentity
  let requestedTexts: [String]
  let textRuns: [FontTextRuns]
  let fallbackStatus: String
  let motionRenderVerified: Bool
}

enum FontIdentityError: Error {
  case invalidRequest
  case unavailableFont
  case unavailableURL
  case invalidVariation
  case unavailableRuns
}

/// Read-only CoreText registration lookup. No registration, downloads, files or temporary artifacts.
enum FontIdentityProbe {
  static func validate(_ request: FontIdentityRequest) throws {
    guard (1...256).contains(request.fontName.utf8.count),
      !request.fontName.unicodeScalars.contains(where: {
        CharacterSet.controlCharacters.contains($0)
      }),
      request.pointSize.isFinite, (1...1024).contains(request.pointSize),
      (1...256).contains(request.texts.count),
      request.texts.reduce(0, { $0 + $1.utf16.count }) <= 4096,
      request.texts.allSatisfy({ text in
        (1...120).contains(text.utf16.count)
          && !text.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) })
      })
    else { throw FontIdentityError.invalidRequest }
  }

  static func variations(_ values: [NSNumber: NSNumber]) throws -> [FontVariation] {
    try values.map { key, value in
      guard key.doubleValue.isFinite, key.doubleValue >= 0,
        key.doubleValue <= Double(UInt32.max), key.doubleValue.rounded() == key.doubleValue,
        value.doubleValue.isFinite
      else { throw FontIdentityError.invalidVariation }
      return FontVariation(identifier: key.uint32Value, value: value.doubleValue)
    }.sorted { $0.identifier < $1.identifier }
  }

  static func identity(_ font: CTFont) throws -> FontIdentity {
    guard let url = CTFontCopyAttribute(font, kCTFontURLAttribute) as? URL,
      url.isFileURL, url.path.hasPrefix("/")
    else { throw FontIdentityError.unavailableURL }
    let axes: [FontVariation]
    if let raw = CTFontCopyVariation(font) {
      guard let values = raw as? [NSNumber: NSNumber] else {
        throw FontIdentityError.invalidVariation
      }
      axes = try variations(values)
    } else {
      axes = []
    }
    return FontIdentity(
      postScriptName: CTFontCopyPostScriptName(font) as String,
      family: CTFontCopyFamilyName(font) as String,
      face: CTFontCopyName(font, kCTFontStyleNameKey).map { $0 as String },
      fontURL: url, pointSize: Double(CTFontGetSize(font)), variations: axes,
      ascent: Double(CTFontGetAscent(font)), descent: Double(CTFontGetDescent(font)))
  }

  static func measure(_ request: FontIdentityRequest) throws -> FontIdentityResult {
    try validate(request)
    let font = CTFontCreateWithName(request.fontName as CFString, request.pointSize, nil)
    guard CTFontCopyPostScriptName(font) as String == request.fontName else {
      throw FontIdentityError.unavailableFont
    }
    let textRuns = try request.texts.map { text in
      let line = CTLineCreateWithAttributedString(
        NSAttributedString(string: text, attributes: [.font: font]))
      guard let runs = CTLineGetGlyphRuns(line) as? [CTRun] else {
        throw FontIdentityError.unavailableRuns
      }
      let identities = try runs.map { run in
        let attributes = CTRunGetAttributes(run) as NSDictionary
        guard let runFont = attributes[kCTFontAttributeName] as? NSFont else {
          throw FontIdentityError.unavailableRuns
        }
        let range = CTRunGetStringRange(run)
        return FontGlyphRun(
          utf16Location: range.location, utf16Length: range.length,
          glyphCount: CTRunGetGlyphCount(run), font: try identity(runFont as CTFont))
      }
      return FontTextRuns(text: text, runs: identities)
    }
    return FontIdentityResult(
      font: try identity(font), requestedTexts: request.texts, textRuns: textRuns,
      fallbackStatus: "measured-coretext-runs", motionRenderVerified: false)
  }

  static func response(_ data: Data) throws -> Data {
    guard data.count <= 64 * 1024 else { throw FontIdentityError.invalidRequest }
    let request = try JSONDecoder().decode(FontIdentityRequest.self, from: data)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
    return try encoder.encode(measure(request))
  }
}
