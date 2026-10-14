import Foundation
import Testing

@testable import ProAppsCore

struct MotionTextTests {
  private let scene = """
    <scenenode name="Title" id="7"><style id="9"/><text>AB</text>
    <styleRun style="9" offset="0" length="2"/>
    <object value="65"><parameter name="Kerning" id="1" value="0"/></object>
    <object value="66"><parameter name="Kerning" id="2" value="0"/></object></scenenode>
    """

  private func xml(_ content: String) -> Data {
    Data("<ozml version=\"4.0\">\(content)</ozml>".utf8)
  }

  @Test func inventoryIsBoundedAndExplicitlyNotRenderProof() throws {
    let report = try MotionText.inspect(xml(scene))
    #expect(report.sourceSHA256.count == 64)
    #expect(report.formatVersion == "4.0")
    #expect(!report.renderVerified)
    #expect(report.layers.count == 1)
    let layer = try #require(report.layers.first)
    #expect(layer.id == 7 && layer.name == "Title" && layer.text == "AB" && layer.editable)
    #expect(try MotionText.inspect(Data("<ozml/>".utf8)).layers.isEmpty)
  }

  @Test func copyUpdatesJapaneseCharactersAndRangesWithoutChangingSource() throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = root.appendingPathComponent("source.motn")
    let output = root.appendingPathComponent("copy.motn")
    let data = xml(scene)
    try data.write(to: source)
    let report = try MotionText.inspect(data)
    let change = MotionTextChange(layerID: 7, expectedText: "AB", replacement: "字幕検証")
    let result = try MotionText.copy(
      input: source.path, output: output.path, expectedSHA256: report.sourceSHA256,
      changes: [change], allowUndocumented: true)
    #expect(result == output)
    #expect(try Data(contentsOf: source) == data)
    let updated = try Data(contentsOf: output)
    let document = try Interchange.parse(updated, kind: .motion)
    #expect(try document.nodes(forXPath: "//text").first?.stringValue == "字幕検証")
    #expect(try document.nodes(forXPath: "//styleRun/@length").first?.stringValue == "4")
    #expect(
      try document.nodes(forXPath: "//object/@value").map(\.stringValue) == [
        "23383", "24149", "26908", "35388",
      ])
    #expect(
      try document.nodes(forXPath: "//object/parameter/@id").map(\.stringValue) == [
        "1", "2", "3", "4",
      ])
    #expect(try MotionText.inspect(updated).layers.first?.editable == true)
    #expect(throws: (any Error).self) {
      try MotionText.copy(
        input: source.path, output: output.path, expectedSHA256: report.sourceSHA256,
        changes: [change], allowUndocumented: true)
    }
    #expect(throws: ProAppsError.self) {
      try MotionText.copy(
        input: source.path, output: root.appendingPathComponent("stale.motn").path,
        expectedSHA256: "stale", changes: [change], allowUndocumented: true)
    }
    #expect(throws: ProAppsError.self) {
      try MotionText.copy(
        input: source.path, output: root.appendingPathComponent("partial.motn").path,
        expectedSHA256: report.sourceSHA256,
        changes: [change, .init(layerID: 999, expectedText: "AB", replacement: "No")],
        allowUndocumented: true)
    }
    #expect(
      !FileManager.default.fileExists(atPath: root.appendingPathComponent("partial.motn").path))
  }

  @Test(arguments: ["", "two\nlines", "😀", String(repeating: "字", count: 121)])
  func invalidReplacementCannotPublish(_ replacement: String) throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = root.appendingPathComponent("source.motn")
    let output = root.appendingPathComponent("copy.motn")
    let data = xml(scene)
    try data.write(to: source)
    let hash = try MotionText.inspect(data).sourceSHA256
    #expect(throws: ProAppsError.self) {
      try MotionText.copy(
        input: source.path, output: output.path, expectedSHA256: hash,
        changes: [.init(layerID: 7, expectedText: "AB", replacement: replacement)],
        allowUndocumented: true)
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
    #expect(try Data(contentsOf: source) == data)
  }

  @Test func optInAndChangeBudgetsAreEnforcedBeforeIO() {
    #expect(throws: ProAppsError.self) {
      try MotionText.copy(
        input: "/missing.motn", output: "/unused.motn", expectedSHA256: "", changes: [],
        allowUndocumented: false)
    }
    #expect(throws: ProAppsError.self) {
      try MotionText.copy(
        input: "/missing.motn", output: "/unused.motn", expectedSHA256: "", changes: [],
        allowUndocumented: true)
    }
  }

  @Test func unsupportedKerningAndInvalidCharacterTablesAreNotFlattened() throws {
    let nonzero = scene.replacingOccurrences(of: "value=\"0\"", with: "value=\"1\"")
    #expect(try MotionText.inspect(xml(nonzero)).layers.first?.editable == false)
    let wrongCharacters = scene.replacingOccurrences(of: "value=\"65\"", with: "value=\"64\"")
    #expect(try MotionText.inspect(xml(wrongCharacters)).layers.first?.editable == false)
    let wrongRange = scene.replacingOccurrences(of: "length=\"2\"", with: "length=\"3\"")
    #expect(try MotionText.inspect(xml(wrongRange)).layers.first?.editable == false)
    let otherVersion = String(decoding: xml(scene), as: UTF8.self).replacingOccurrences(
      of: "4.0", with: "5.0")
    #expect(try MotionText.inspect(Data(otherVersion.utf8)).layers.first?.editable == false)
  }

  @Test func malformedDuplicateAndExcessiveLayersFail() throws {
    #expect(throws: ProAppsError.self) { try MotionText.inspect(xml(scene + scene)) }
    #expect(throws: ProAppsError.self) {
      try MotionText.inspect(xml(scene.replacingOccurrences(of: "id=\"7\"", with: "id=\"bad\"")))
    }
    let excessive = (1...129).map {
      scene.replacingOccurrences(of: "id=\"7\"", with: "id=\"\($0)\"")
    }.joined()
    #expect(throws: ProAppsError.self) { try MotionText.inspect(xml(excessive)) }
  }
}
