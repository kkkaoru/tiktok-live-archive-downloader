import Foundation

public enum InterchangeKind: String, Codable, Sendable {
  case fcpxml, motion

  public var extensions: Set<String> {
    switch self {
    case .fcpxml: return ["fcpxml"]
    case .motion: return ProApp.motion.documentExtensions
    }
  }
  var root: String { self == .fcpxml ? "fcpxml" : "ozml" }
}

public struct XMLSummary: Codable, Sendable {
  public let root: String
  public let version: String?
  public let elementCount: Int
  public let validation: String
}

public struct XMLChange: Codable, Sendable {
  public let xpath: String
  public let value: String
  public init(xpath: String, value: String) {
    self.xpath = xpath
    self.value = value
  }
}

public enum Interchange {
  static func parse(_ data: Data, kind: InterchangeKind) throws -> XMLDocument {
    guard data.count <= Files.maximumBytes, var text = String(data: data, encoding: .utf8),
      !text.contains("\0")
    else {
      throw ProAppsError.invalid("Expected bounded UTF-8 XML")
    }
    // Accept only the exact empty declaration emitted by the selected app,
    // in the XML prolog. Never load a DTD or strip declarations inside content.
    removeEmptyDeclaration(from: &text, kind: kind)
    guard !text.localizedCaseInsensitiveContains("<!DOCTYPE"),
      !text.localizedCaseInsensitiveContains("<!ENTITY")
    else {
      throw ProAppsError.invalid(
        "DTD/entity declarations are forbidden; external entities are never loaded")
    }
    let document = try XMLDocument(data: Data(text.utf8), options: [.nodeLoadExternalEntitiesNever])
    guard document.rootElement()?.name == kind.root else {
      throw ProAppsError.invalid("Unexpected XML root")
    }
    return document
  }

  private static func removeEmptyDeclaration(from text: inout String, kind: InterchangeKind) {
    let declaration =
      kind == .fcpxml ? "<!DOCTYPE fcpxml>" : "<!DOCTYPE ozxmlscene>"
    let prolog = #"\A(?:\uFEFF)?\s*(?:<\?xml\s[^?]*\?>\s*)?"#
    let pattern = prolog + NSRegularExpression.escapedPattern(for: declaration)
    guard let match = text.range(of: pattern, options: .regularExpression) else { return }
    let start = text.index(match.upperBound, offsetBy: -declaration.count)
    text.removeSubrange(start..<match.upperBound)
  }

  public static func inspect(_ data: Data, kind: InterchangeKind) throws -> XMLSummary {
    let document = try parse(data, kind: kind)
    var pending: [XMLNode] = [document]
    var count = 0
    while let node = pending.popLast() {
      if node.kind == .element { count += 1 }
      guard count <= 50_000 else { throw ProAppsError.invalid("XML node limit exceeded") }
      pending.append(contentsOf: node.children ?? [])
    }
    return XMLSummary(
      root: kind.root, version: document.rootElement()?.attribute(forName: "version")?.stringValue,
      elementCount: count,
      validation: kind == .fcpxml
        ? "Well-formed XML and fcpxml root only; not DTD or application validation"
        : "Well-formed ozml only; Motion format is undocumented and version-sensitive")
  }

  public static func query(_ data: Data, kind: InterchangeKind, xpath: String, limit: Int) throws
    -> [String]
  {
    guard !xpath.isEmpty, xpath.utf8.count <= 1024, (1...10).contains(limit) else {
      throw ProAppsError.invalid("Invalid XPath or result limit")
    }
    let document = try parse(data, kind: kind)
    let nodes = try document.nodes(forXPath: xpath)
    return nodes.prefix(limit).map { String(($0.xmlString).prefix(2048)) }
  }

  public static func write(
    _ xml: String, kind: InterchangeKind, output: String, allowUndocumented: Bool
  ) throws -> URL {
    if kind == .motion && !allowUndocumented {
      throw ProAppsError.invalid(
        "Motion writes require allowUndocumentedFormat=true and a disposable project copy")
    }
    let data = Data(xml.utf8)
    _ = try inspect(data, kind: kind)
    return try Files.writeNew(data, to: output, extensions: kind.extensions)
  }

  /// Change only an existing attribute or text-only leaf selected uniquely by
  /// XPath. Never overwrite the input, add nodes, or guess missing parameters.
  public static func patch(
    input: String, output: String, kind: InterchangeKind, changes: [XMLChange],
    allowUndocumented: Bool
  ) throws -> URL {
    guard !changes.isEmpty, changes.count <= 64 else {
      throw ProAppsError.invalid("Supply 1–64 changes")
    }
    if kind == .motion && !allowUndocumented {
      throw ProAppsError.invalid("Motion writes require allowUndocumentedFormat=true")
    }
    let url = try Files.existing(input, extensions: kind.extensions)
    let document = try parse(Files.read(url), kind: kind)
    for change in changes {
      guard !change.xpath.isEmpty, change.xpath.utf8.count <= 1024, change.value.utf8.count <= 65536
      else {
        throw ProAppsError.invalid("XPath or value limit exceeded")
      }
      let nodes = try document.nodes(forXPath: change.xpath)
      guard nodes.count == 1, let node = nodes.first else {
        throw ProAppsError.invalid("Each XPath must select exactly one existing node")
      }
      // Empty elements are valid leaves. Validate each existing child instead
      // of assuming that a missing optional children array means an empty one.
      let textOnly =
        node.kind == .element
        && (0..<node.childCount).allSatisfy {
          node.child(at: $0)?.kind == .text
        }
      guard node.kind == .attribute || textOnly else {
        throw ProAppsError.invalid("Only attributes and text-only leaf elements may be changed")
      }
      node.stringValue = change.value
    }
    let bytes = document.xmlData(options: [.nodePrettyPrint])
    _ = try inspect(bytes, kind: kind)
    return try Files.writeNew(bytes, to: output, extensions: kind.extensions)
  }
}
