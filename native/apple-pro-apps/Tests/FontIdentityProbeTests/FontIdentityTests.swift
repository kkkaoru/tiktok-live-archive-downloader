import CoreText
import Foundation
import Testing

@testable import font_identity_probe

private final class FontTestBundle {}

struct FontIdentityTests {
  @Test func registeredFont() throws {
    let result = try FontIdentityProbe.measure(
      FontIdentityRequest(fontName: "Helvetica", pointSize: 24, texts: ["Hello", "字幕"]))
    #expect(result.font.postScriptName == "Helvetica")
    #expect(result.font.family == "Helvetica")
    #expect(result.font.face != nil)
    #expect(result.font.pointSize == 24)
    #expect(result.font.fontURL.isFileURL)
    #expect(FileManager.default.fileExists(atPath: result.font.fontURL.path))
    #expect(result.font.ascent > 0)
    #expect(result.font.descent > 0)
    #expect(result.fallbackStatus == "measured-coretext-runs")
    #expect(result.textRuns.count == 2)
    let latin = try #require(result.textRuns.first?.runs.first)
    #expect(latin.utf16Location == 0)
    #expect(latin.utf16Length == 5)
    #expect(latin.glyphCount == 5)
    #expect(latin.font.postScriptName == "Helvetica")
    let japanese = try #require(result.textRuns.last?.runs.first)
    #expect(japanese.font.postScriptName != "Helvetica")
    #expect(japanese.font.fontURL.isFileURL)
    #expect(!result.motionRenderVerified)
    #expect(result.requestedTexts == ["Hello", "字幕"])
  }

  @Test func responseRoundTrip() throws {
    let input = Data(#"{"fontName":"Helvetica","pointSize":24,"texts":["abc"]}"#.utf8)
    let result = try JSONDecoder().decode(
      FontIdentityResult.self, from: FontIdentityProbe.response(input))
    #expect(result.font.postScriptName == "Helvetica")
    #expect(result.requestedTexts == ["abc"])
  }

  @Test func invalidJSONAndByteLimit() {
    #expect(throws: (any Error).self) { try FontIdentityProbe.response(Data()) }
    #expect(throws: FontIdentityError.self) {
      try FontIdentityProbe.response(Data(repeating: 32, count: 65537))
    }
  }

  @Test func missingFont() {
    #expect(throws: FontIdentityError.self) {
      try FontIdentityProbe.measure(
        FontIdentityRequest(
          fontName: "NoSuchFont-IdentityProbe-Fixture", pointSize: 20, texts: ["x"]))
    }
  }

  @Test(arguments: ["", String(repeating: "a", count: 257), "Bad\nFont", "Bad\0Font"])
  func invalidName(_ name: String) {
    #expect(throws: FontIdentityError.self) {
      try FontIdentityProbe.validate(
        FontIdentityRequest(fontName: name, pointSize: 24, texts: ["x"]))
    }
  }

  @Test(arguments: [0.0, -1, 1025, Double.nan, Double.infinity])
  func invalidSize(_ size: Double) {
    #expect(throws: FontIdentityError.self) {
      try FontIdentityProbe.validate(
        FontIdentityRequest(fontName: "Helvetica", pointSize: size, texts: ["x"]))
    }
  }

  @Test(arguments: [
    [String](), [""], ["a\0b"], ["a\nb"], [String(repeating: "x", count: 121)],
    Array(repeating: "x", count: 257),
    Array(repeating: String(repeating: "x", count: 120), count: 35),
  ])
  func invalidTexts(_ texts: [String]) {
    #expect(throws: FontIdentityError.self) {
      try FontIdentityProbe.validate(
        FontIdentityRequest(fontName: "Helvetica", pointSize: 24, texts: texts))
    }
  }

  @Test func validationBoundaries() throws {
    try FontIdentityProbe.validate(
      FontIdentityRequest(fontName: "x", pointSize: 1, texts: [String(repeating: "x", count: 120)]))
    try FontIdentityProbe.validate(
      FontIdentityRequest(
        fontName: String(repeating: "x", count: 256), pointSize: 1024,
        texts: Array(repeating: "x", count: 256)))
    try FontIdentityProbe.validate(
      FontIdentityRequest(
        fontName: "x", pointSize: 24,
        texts: Array(repeating: String(repeating: "x", count: 16), count: 256)))
  }

  @Test func variationValues() throws {
    #expect(try FontIdentityProbe.variations([:]) == [])
    #expect(
      try FontIdentityProbe.variations([2: 700, 1: 20]) == [
        FontVariation(identifier: 1, value: 20), FontVariation(identifier: 2, value: 700),
      ])
  }

  @Test(arguments: [-1.0, 0.5, Double(UInt32.max) + 1, Double.infinity])
  func invalidAxis(_ value: Double) {
    #expect(throws: FontIdentityError.self) {
      try FontIdentityProbe.variations([NSNumber(value: value): 1])
    }
  }

  @Test func commandStdio() throws {
    let products = Bundle(for: FontTestBundle.self).bundleURL.deletingLastPathComponent()
    let process = Process()
    process.executableURL = products.appendingPathComponent("font-identity-probe")
    let input = Pipe()
    let output = Pipe()
    process.standardInput = input
    process.standardOutput = output
    var environment = ProcessInfo.processInfo.environment
    environment["LLVM_PROFILE_FILE"] =
      products.appendingPathComponent("codecov/font-cli-%p-%m.profraw").path
    process.environment = environment
    try input.fileHandleForWriting.write(
      contentsOf: Data(#"{"fontName":"Helvetica","pointSize":24,"texts":["abc"]}"#.utf8))
    try input.fileHandleForWriting.close()
    try process.run()
    let data = try #require(try output.fileHandleForReading.readToEnd())
    process.waitUntilExit()
    #expect(process.terminationStatus == 0)
    let result = try JSONDecoder().decode(FontIdentityResult.self, from: data)
    #expect(result.font.postScriptName == "Helvetica")
    try output.fileHandleForReading.close()
  }

  @Test func invalidAxisValue() {
    #expect(throws: FontIdentityError.self) {
      try FontIdentityProbe.variations([1: NSNumber(value: Double.nan)])
    }
  }
}
