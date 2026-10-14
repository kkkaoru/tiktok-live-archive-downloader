import CryptoKit
import Foundation

public struct MotionTextLayer: Codable, Sendable {
  public let id: Int
  public let name: String
  public let text: String
  public let editable: Bool
}

public struct MotionTextInventory: Codable, Sendable {
  public let sourceSHA256: String
  public let formatVersion: String
  public let layers: [MotionTextLayer]
  public let renderVerified: Bool
}

public struct MotionTextChange: Codable, Sendable {
  public let layerID: Int
  public let expectedText: String
  public let replacement: String

  public init(layerID: Int, expectedText: String, replacement: String) {
    self.layerID = layerID
    self.expectedText = expectedText
    self.replacement = replacement
  }
}

/// A bounded adapter for observed single-style Motion text, not a general
/// schema generator. Editing also updates character objects and style ranges.
public enum MotionText {
  public static func inspect(_ data: Data) throws -> MotionTextInventory {
    _ = try Interchange.inspect(data, kind: .motion)
    let document = try Interchange.parse(data, kind: .motion)
    let version = document.rootElement()?.attribute(forName: "version")?.stringValue ?? ""
    let nodes = try document.nodes(forXPath: "//scenenode[text]")
    guard nodes.count <= 128 else { throw ProAppsError.outputLimit }
    var layers: [MotionTextLayer] = []
    var ids: Set<Int> = []
    for node in nodes {
      guard let element = node as? XMLElement,
        let rawID = element.attribute(forName: "id")?.stringValue,
        let id = Int(rawID), (1...Int(Int32.max)).contains(id), ids.insert(id).inserted,
        let text = element.elements(forName: "text").first?.stringValue,
        text.utf8.count <= 8192
      else { throw ProAppsError.invalid("Invalid or duplicate Motion text identity") }
      layers.append(
        MotionTextLayer(
          id: id, name: element.attribute(forName: "name")?.stringValue ?? "",
          text: text, editable: version == "4.0" && supported(element, text: text)))
    }
    return MotionTextInventory(
      sourceSHA256: fingerprint(data), formatVersion: version, layers: layers, renderVerified: false
    )
  }

  /// Write only a NEW copy of a fingerprint-matched document. Reject unsupported
  /// formatting rather than dropping tracking/kerning/animation information.
  public static func copy(
    input: String, output: String, expectedSHA256: String,
    changes: [MotionTextChange], allowUndocumented: Bool
  ) throws -> URL {
    guard allowUndocumented else {
      throw ProAppsError.invalid("Motion copy editing requires explicit undocumented-format opt-in")
    }
    guard (1...32).contains(changes.count), Set(changes.map(\.layerID)).count == changes.count
    else {
      throw ProAppsError.invalid("Supply 1–32 distinct Motion text-layer changes")
    }
    let source = try Files.existing(input, extensions: InterchangeKind.motion.extensions)
    let data = try Files.read(source)
    let inventory = try inspect(data)
    guard inventory.sourceSHA256 == expectedSHA256 else {
      throw ProAppsError.invalid("Motion source fingerprint changed; inspect again")
    }
    let document = try Interchange.parse(data, kind: .motion)
    for change in changes {
      try Task.checkCancellation()
      guard let layer = inventory.layers.first(where: { $0.id == change.layerID }),
        layer.editable, layer.text == change.expectedText,
        validText(change.replacement), change.replacement.utf16.count <= 120
      else { throw ProAppsError.invalid("Unsupported, stale or invalid Motion text change") }
      let nodes = try document.nodes(forXPath: "//scenenode[@id='\(change.layerID)']")
      guard nodes.count == 1, let element = nodes.first as? XMLElement else {
        throw ProAppsError.invalid("Motion scene identity is ambiguous")
      }
      try replace(element, with: change.replacement)
    }
    let bytes = document.xmlData(options: [.nodePrettyPrint])
    _ = try inspect(bytes)
    return try Files.writeNew(bytes, to: output, extensions: InterchangeKind.motion.extensions)
  }

  private static func fingerprint(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private static func validText(_ text: String) -> Bool {
    !text.isEmpty && text.unicodeScalars.allSatisfy { $0.value >= 32 && $0.value <= 0xffff }
  }

  private static func supported(_ element: XMLElement, text: String) -> Bool {
    let runs = element.elements(forName: "styleRun")
    let objects = element.elements(forName: "object")
    guard validText(text), element.elements(forName: "text").count == 1,
      runs.count == 1, let run = runs.first,
      run.attribute(forName: "offset")?.stringValue == "0",
      run.attribute(forName: "length")?.stringValue == String(text.utf16.count),
      let styleID = run.attribute(forName: "style")?.stringValue,
      element.elements(forName: "style").contains(where: {
        $0.attribute(forName: "id")?.stringValue == styleID
      }), objects.count == text.utf16.count
    else { return false }
    return zip(objects, text.utf16).enumerated().allSatisfy { index, pair in
      let (object, character) = pair
      let parameters = object.elements(forName: "parameter")
      guard object.attribute(forName: "value")?.stringValue == String(character),
        object.childCount == 1, parameters.count == 1, let kerning = parameters.first
      else { return false }
      return kerning.attribute(forName: "name")?.stringValue == "Kerning"
        && kerning.attribute(forName: "id")?.stringValue == String(index + 1)
        && kerning.attribute(forName: "value")?.stringValue == "0"
        && kerning.childCount == 0
    }
  }

  private static func replace(_ element: XMLElement, with replacement: String) throws {
    let objects = element.elements(forName: "object")
    guard let first = objects.first,
      let text = element.elements(forName: "text").first,
      let length = element.elements(forName: "styleRun").first?.attribute(forName: "length")
    else { throw ProAppsError.invalid("Motion text layout is incomplete") }
    let blueprint = first.xmlString
    let insertion = first.index
    for object in objects { object.detach() }
    for (index, character) in replacement.utf16.enumerated() {
      let object = try XMLElement(xmlString: blueprint)
      object.attribute(forName: "value")?.stringValue = String(character)
      object.elements(forName: "parameter").first?.attribute(forName: "id")?.stringValue = String(
        index + 1)
      element.insertChild(object, at: insertion + index)
    }
    text.stringValue = replacement
    length.stringValue = String(replacement.utf16.count)
  }
}
