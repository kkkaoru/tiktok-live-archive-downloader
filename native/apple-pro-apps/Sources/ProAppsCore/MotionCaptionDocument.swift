import Foundation

/// A finished caption part as the native renderer draws it, read from its Motion document.
public struct MotionCaptionPart: Codable, Equatable, Sendable {
  public let width: Int
  public let height: Int
  public let rate: Int
  public let frames: Int
  public let style: MotionCaptionStyle
  public let titles: [MotionCaptionTitle]
}

/// Reads the `.moti` caption parts that `replay.caption_parts` writes for Motion/Compressor
/// (`motion_reveal_stages` titles finished by `motion_finish`) so the native renderer draws from
/// the same data.
///
/// Compatibility is enforced while loading. Every title is three synchronized text objects
/// (`Caption Keyline Face`, `Caption Face`, `Caption Outer Rim`). Each object, and the document
/// around the titles, is reduced to a skeleton: one line per element attribute and text, keyed by
/// its path of parameter IDs, with Motion's generated object IDs normalised and the values the
/// renderer reads (`honoured`) masked. The skeleton must equal the pinned one of the supported
/// template (`MotionCaptionSkeleton`); any other element, attribute, parameter ID, flag or value
/// (an enabled glow, another interpolation, a moved anchor, a new template version) refuses the
/// part and names the difference. The honoured values are then checked against what the
/// renderer models (pure black/white outlines, a vertical shadow, whole frames).
public enum MotionCaptionDocument {
  public static let roles = ["Caption Keyline Face", "Caption Face", "Caption Outer Rim"]
  static let generatedIDs = 999_000_000
  /// Attributes holding Motion's generated object IDs (they differ per document).
  static let idAttributes: Set<String> = ["id", "style"]
  static let maximumBytes = 32 * 1024 * 1024
  static let reportedDifferences = 3
  /// Motion behaviour measured on its renders (Compressor, 2026-10-24): the face is drawn this
  /// much bolder than CoreText's outline, the visible keyline is this fraction of its width, the
  /// shadow Gaussian's sigma is its blur over √2 and its effective opacity this fraction of the
  /// control; output is linear light encoded with a pure power law.
  static let faceGrow = 0.3
  static let keylineScale = 0.7215
  static let shadowOpacityScale = 0.875
  static let outputGamma = 1.961
  static let downward = 3 * Double.pi / 2
  static let keyParts: Set<String> = [
    "time", "value", "inputTangentTime", "inputTangentValue", "outputTangentTime",
    "outputTangentValue",
  ]

  static let gradient = "/style/parameter[14]/parameter[17]/parameter[1]"
  static let outline = "/style/parameter[30]"
  static let shadow = "/style/parameter[21]"
  static let typeOn = "/parameter[2]/parameter[319]/parameter[321]/curve"

  /// Paths (suffixes, relative to a title object) whose attributes or text are read.
  static let titleValues: [(String, Set<String>)] = [
    ("/timing", Set(["in", "out", "offset"])),
    ("/parameter[1]/parameter[100]/parameter[101]/parameter[1]", Set(["value"])),
    ("/parameter[1]/parameter[100]/parameter[101]/parameter[2]", Set(["value"])),
    ("/parameter[2]/parameter[319]/parameter[321]/curve", Set(["default", "value"])),
    ("/parameter[2]/parameter[369]/text", Set(["#text"])),
    ("/styleRun", Set(["length"])),
    ("/style/parameter[3]", Set(["value"])),
    ("/style/parameter[83]/font", Set(["#text"])),
    ("/style/parameter[30]/parameter[36]", Set(["value"])),
    ("/style/parameter[30]/parameter[32]/parameter[1]", Set(["value"])),
    ("/style/parameter[30]/parameter[32]/parameter[2]", Set(["value"])),
    ("/style/parameter[30]/parameter[32]/parameter[3]", Set(["value"])),
    ("/style/parameter[21]/parameter[26]", Set(["value"])),
    ("/style/parameter[21]/parameter[27]", Set(["value"])),
    ("/style/parameter[21]/parameter[29]", Set(["value"])),
    ("/style/parameter[21]/parameter[75]/parameter[1]", Set(["value"])),
    ("/style/parameter[21]/parameter[75]/parameter[2]", Set(["value"])),
  ]

  static let honouredInTitles: [(String, Set<String>)] = titleValues + gradientStops

  /// Each gradient stop's location and colour.
  static let gradientStops: [(String, Set<String>)] = {
    var result: [(String, Set<String>)] = []
    for stop in ["RGB1", "RGB2"] {
      for leaf in [
        "/parameter[1]", "/parameter[3]/parameter[1]", "/parameter[3]/parameter[2]",
        "/parameter[3]/parameter[3]",
      ] {
        let path: String = gradient + "/parameter[" + stop + "]" + leaf + "/curve"
        result.append((path, ["value"]))
      }
    }
    return result
  }()

  /// Paths of the document outside the titles whose values are read or vary with the part.
  static let honouredInDocument: [(String, Set<String>)] = [
    ("/scene/sceneSettings/width", ["#text"]),
    ("/scene/sceneSettings/height", ["#text"]),
    ("/scene/sceneSettings/duration", ["#text"]),
    ("/scene/sceneSettings/frameRate", ["#text"]),
    ("/scene/timeRange", ["duration"]),
    ("/scene/playRange", ["duration"]),
    ("/scene/layer/timing", ["out"]),
  ]

  public static func read(contentsOf url: URL) throws -> MotionCaptionPart {
    try read(Data(contentsOf: url))
  }

  public static func read(_ data: Data) throws -> MotionCaptionPart {
    let (root, nodes) = try parse(data)
    try require(
      lines(
        of: root, honoured: honouredInDocument, skipping: Set(nodes.map(ObjectIdentifier.init))),
      equal: MotionCaptionSkeleton.document(), part: "document")
    let settings = try sceneSettings(root)
    guard nodes.count % roles.count == 0, !nodes.isEmpty else {
      throw invalid("expected title objects in groups of \(roles.count)")
    }
    var styles: [MotionCaptionStyle] = []
    var titles: [MotionCaptionTitle] = []
    for start in stride(from: 0, to: nodes.count, by: roles.count) {
      let group = Array(nodes[start..<(start + roles.count)])
      for (role, node) in zip(roles, group) {
        guard node.attribute(forName: "name")?.stringValue == role else {
          throw invalid("title \(start / roles.count): expected the object \(role)")
        }
        try require(
          lines(of: node, honoured: honouredInTitles, skipping: []),
          equal: MotionCaptionSkeleton.title(role), part: "title \(start / roles.count) \(role)")
      }
      let (title, style) = try titleAndStyle(group, rate: settings.rate, index: start / roles.count)
      titles.append(title)
      styles.append(style)
    }
    // The part's style is its first title's; a title edited to another style carries its own.
    let style = styles[0]
    let styled = zip(titles, styles).map { title, own in
      own == style
        ? title
        : MotionCaptionTitle(
          text: title.text, startFrame: title.startFrame, endFrame: title.endFrame, x: title.x,
          y: title.y, graphemeFrames: title.graphemeFrames, style: own)
    }
    return MotionCaptionPart(
      width: settings.width, height: settings.height, rate: settings.rate,
      frames: settings.frames, style: style, titles: styled)
  }

  /// The skeleton text of a supported part, for pinning a new template version.
  public static func skeleton(_ data: Data) throws -> String {
    let (root, nodes) = try parse(data)
    var sections = ["[document]"]
    sections += lines(
      of: root, honoured: honouredInDocument, skipping: Set(nodes.map(ObjectIdentifier.init)))
    for (role, node) in zip(roles, nodes) {
      sections.append("[\(role)]")
      sections += lines(of: node, honoured: honouredInTitles, skipping: [])
    }
    return sections.joined(separator: "\n") + "\n"
  }

  static func parse(_ data: Data) throws -> (XMLElement, [XMLElement]) {
    guard data.count <= maximumBytes else { throw invalid("the document is too large") }
    let document: XMLDocument
    do {
      document = try XMLDocument(data: data, options: [.nodePreserveWhitespace])
    } catch {
      throw invalid("not an XML document (\(error.localizedDescription))")
    }
    guard let root = document.rootElement(), root.name == "ozml",
      root.attribute(forName: "version")?.stringValue == "5.13",
      let scene = child(root, "scene"), let layer = child(scene, "layer")
    else { throw invalid("expected an ozml 5.13 document with one scene layer") }
    let nodes = (layer.children ?? []).compactMap { $0 as? XMLElement }.filter {
      $0.name == "scenenode"
    }
    return (root, nodes)
  }

  static func child(_ element: XMLElement, _ name: String) -> XMLElement? {
    let matches = element.elements(forName: name)
    return matches.count == 1 ? matches[0] : nil
  }

  static func invalid(_ reason: String) -> ProAppsError {
    ProAppsError.invalid("Unsupported Motion caption part: \(reason)")
  }

  // MARK: Skeleton

  static func isGenerated(_ value: String) -> Bool {
    guard let number = Int(value) else { return false }
    return number >= generatedIDs
  }

  /// Skeleton lines of `element`'s subtree (the element itself excluded), sorted.
  static func lines(
    of element: XMLElement, honoured: [(String, Set<String>)], skipping: Set<ObjectIdentifier>
  ) -> [String] {
    var result: [String] = []
    walk(element, path: "", honoured: honoured, skipping: skipping, into: &result)
    return result.sorted()
  }

  static func walk(
    _ element: XMLElement, path: String, honoured: [(String, Set<String>)],
    skipping: Set<ObjectIdentifier>, into result: inout [String]
  ) {
    for case let child as XMLElement in element.children ?? [] {
      guard !skipping.contains(ObjectIdentifier(child)), let name = child.name else { continue }
      var key = child.name == "parameter" ? child.attribute(forName: "id")?.stringValue : nil
      if let id = key, isGenerated(id) { key = child.attribute(forName: "name")?.stringValue }
      let childPath = path + "/" + name + (key.map { "[\($0)]" } ?? "")
      // Keyframes are read and checked one by one in typeOn(), not pinned.
      if name == "keypoint" { continue }
      if childPath.hasSuffix(typeOn + "/numberOfKeypoints") { continue }
      let masked = honoured.first { childPath.hasSuffix($0.0) }?.1 ?? []
      for attribute in child.attributes ?? [] {
        guard let attributeName = attribute.name else { continue }
        let value = attribute.stringValue ?? ""
        let generated = idAttributes.contains(attributeName) && isGenerated(value)
        let shown = masked.contains(attributeName) ? "*" : generated ? "<id>" : value
        result.append("\(childPath)@\(attributeName)=\(shown)")
      }
      let text = ownText(child)
      if !text.isEmpty {
        result.append("\(childPath)#text=\(masked.contains("#text") ? "*" : text)")
      }
      walk(child, path: childPath, honoured: honoured, skipping: skipping, into: &result)
    }
  }

  static func ownText(_ element: XMLElement) -> String {
    (element.children ?? []).filter { $0.kind == .text }.compactMap(\.stringValue).joined()
      .trimmingCharacters(in: .whitespacesAndNewlines)
  }

  /// Refuses a skeleton that differs from the pinned one, naming the first differences.
  static func require(_ actual: [String], equal expected: [String], part: String) throws {
    guard actual != expected else { return }
    let missing = difference(expected, actual).prefix(reportedDifferences)
    let unexpected = difference(actual, expected).prefix(reportedDifferences)
    throw invalid(
      "\(part) differs from the supported template; unexpected: "
        + (unexpected.isEmpty ? "none" : unexpected.joined(separator: " | ")) + "; missing: "
        + (missing.isEmpty ? "none" : missing.joined(separator: " | ")))
  }

  /// Lines of sorted `first` not matched one-for-one in sorted `second`.
  static func difference(_ first: [String], _ second: [String]) -> [String] {
    var remaining = [String: Int]()
    for line in second { remaining[line, default: 0] += 1 }
    var result: [String] = []
    for line in first {
      if let count = remaining[line], count > 0 {
        remaining[line] = count - 1
      } else {
        result.append(line)
      }
    }
    return result
  }

  // MARK: Values

  struct Settings {
    let width: Int
    let height: Int
    let rate: Int
    let frames: Int
  }

  static func sceneSettings(_ root: XMLElement) throws -> Settings {
    func integer(_ name: String) throws -> Int {
      guard let scene = child(root, "scene"), let settings = child(scene, "sceneSettings"),
        let element = child(settings, name), let value = Int(ownText(element)), value > 0
      else { throw invalid("scene \(name) must be a positive integer") }
      return value
    }
    return try Settings(
      width: integer("width"), height: integer("height"), rate: integer("frameRate"),
      frames: integer("duration"))
  }

  /// The element at a skeleton path below `node` (generated parameter IDs addressed by name).
  static func element(_ node: XMLElement, _ path: String) throws -> XMLElement {
    var current = node
    for step in path.split(separator: "/") {
      let name = String(step.prefix { $0 != "[" })
      let key = step.contains("[") ? String(step.dropFirst(name.count + 1).dropLast()) : nil
      let matches = (current.children ?? []).compactMap { $0 as? XMLElement }.filter { child in
        guard child.name == name else { return false }
        guard let key else { return true }
        let id = child.attribute(forName: "id")?.stringValue ?? ""
        return id == key
          || (isGenerated(id) && child.attribute(forName: "name")?.stringValue == key)
      }
      guard matches.count == 1 else { throw invalid("expected one \(path)") }
      current = matches[0]
    }
    return current
  }

  static func number(_ node: XMLElement, _ path: String, _ attribute: String = "value") throws
    -> Double
  {
    let target = try element(node, path)
    guard let text = target.attribute(forName: attribute)?.stringValue, let value = Double(text),
      value.isFinite
    else { throw invalid("\(path) needs a finite \(attribute)") }
    return value
  }

  static func hex(_ node: XMLElement, stop: String) throws -> String {
    let channels = try (1...3).map {
      try number(node, gradient + "/parameter[\(stop)]/parameter[3]/parameter[\($0)]/curve")
    }
    return try "#"
      + channels.map { try String(format: "%02X", byte($0, what: "gradient colour")) }
      .joined()
  }

  /// A 0–1 control that is an exact 8-bit value (how the caption colours are written).
  static func byte(_ value: Double, what: String) throws -> Int {
    let scaled = value * 255
    guard (0...255).contains(scaled), abs(scaled - scaled.rounded()) < 1e-6 else {
      throw invalid("\(what) \(value) is not an 8-bit sRGB value")
    }
    return Int(scaled.rounded())
  }

  /// Outline colour as black (0) or white (1); other colours' gamma semantics are not modelled.
  static func outlineColour(_ node: XMLElement) throws -> Double {
    let channels = try (1...3).map { try number(node, outline + "/parameter[32]/parameter[\($0)]") }
    guard let first = channels.first, channels.allSatisfy({ $0 == first }), first == 0 || first == 1
    else { throw invalid("outline colours other than black or white are not supported") }
    return first
  }

  /// A Motion time `value timescale 1 0` as a whole frame count at `rate`.
  static func frames(_ text: String?, rate: Int, what: String) throws -> Int {
    let fields = (text ?? "").split(separator: " ").compactMap { Int($0) }
    guard fields.count == 4, fields[1] > 0, fields[2] == 1, fields[3] == 0,
      (fields[0] * rate) % fields[1] == 0
    else { throw invalid("\(what) \(text ?? "") is not a whole frame at \(rate) fps") }
    return fields[0] * rate / fields[1]
  }

  static func titleAndStyle(_ group: [XMLElement], rate: Int, index: Int) throws -> (
    MotionCaptionTitle, MotionCaptionStyle
  ) {
    let keyline = group[0]
    let face = group[1]
    let rim = group[2]
    let shared = [
      "/timing", "/parameter[1]/parameter[100]/parameter[101]", "/parameter[2]/parameter[369]",
      typeOn, "/style/parameter[3]", "/style/parameter[83]",
    ]
    for path in shared {
      let reference = try element(keyline, path).xmlString
      guard try group.allSatisfy({ try element($0, path).xmlString == reference }) else {
        throw invalid("title \(index): its three objects differ at \(path)")
      }
    }
    let timing = try element(keyline, "/timing")
    let start = try frames(timing.attribute(forName: "in")?.stringValue, rate: rate, what: "in")
    let last = try frames(timing.attribute(forName: "out")?.stringValue, rate: rate, what: "out")
    let offset = try frames(
      timing.attribute(forName: "offset")?.stringValue, rate: rate, what: "offset")
    guard offset == start, last >= start else { throw invalid("title \(index): invalid timing") }
    let text = ownText(try element(keyline, "/parameter[2]/parameter[369]/text"))
    let styleRun = try number(keyline, "/styleRun", "length")
    guard !text.isEmpty, styleRun == Double(text.utf16.count) else {
      throw invalid("title \(index): one style run must cover the whole text")
    }
    let onsets = try typeOn(keyline, graphemes: text.count, start: start, end: last + 1, rate: rate)
    let title = MotionCaptionTitle(
      text: text, startFrame: start, endFrame: last + 1,
      x: try number(keyline, "/parameter[1]/parameter[100]/parameter[101]/parameter[1]"),
      y: try number(keyline, "/parameter[1]/parameter[100]/parameter[101]/parameter[2]"),
      graphemeFrames: onsets)
    return (title, try style(keyline: keyline, face: face, rim: rim))
  }

  static func style(keyline: XMLElement, face: XMLElement, rim: XMLElement) throws
    -> MotionCaptionStyle
  {
    guard try outlineColour(keyline) == 0, try outlineColour(face) == 1,
      try outlineColour(rim) == 0
    else { throw invalid("expected a black keyline, a white band and a black rim") }
    let band = try hex(face, stop: "RGB1")
    guard band == "#FFFFFF", try hex(face, stop: "RGB2") == band else {
      throw invalid("the white band's face must be solid white")
    }
    let blurX = try number(rim, shadow + "/parameter[75]/parameter[1]")
    guard blurX == (try number(rim, shadow + "/parameter[75]/parameter[2]")) else {
      throw invalid("an anisotropic shadow blur is not supported")
    }
    guard abs(try number(rim, shadow + "/parameter[29]") - downward) < 1e-9 else {
      throw invalid("only a downward shadow is supported")
    }
    let font = ownText(try element(keyline, "/style/parameter[83]/font"))
    let style = MotionCaptionStyle(
      fontName: font, fontSize: try number(keyline, "/style/parameter[3]"), faceGrow: faceGrow,
      keyline: try number(keyline, outline + "/parameter[36]") * keylineScale,
      whiteBorder: try number(face, outline + "/parameter[36]"),
      rim: try number(rim, outline + "/parameter[36]"),
      shadowOpacity: try number(rim, shadow + "/parameter[26]") * shadowOpacityScale,
      shadowOffsetY: try number(rim, shadow + "/parameter[27]"),
      shadowSigma: blurX / 2.0.squareRoot(), gradientTop: try hex(keyline, stop: "RGB1"),
      gradientBottom: try hex(keyline, stop: "RGB2"),
      topLocation: try number(keyline, gradient + "/parameter[RGB1]/parameter[1]/curve"),
      bottomLocation: try number(keyline, gradient + "/parameter[RGB2]/parameter[1]/curve"),
      outputGamma: outputGamma)
    try style.validate()
    return style
  }

  /// The frame each grapheme appears on, from the Type On End keyframes: `(2k - 1) / 2N` shows
  /// k of the text's N graphemes; keys are linear (hold keys one frame before each step).
  static func typeOn(_ node: XMLElement, graphemes: Int, start: Int, end: Int, rate: Int) throws
    -> [Int]
  {
    let curve = try element(node, typeOn)
    var keys: [(frame: Int, value: Double)] = []
    for case let key as XMLElement in curve.children ?? [] where key.name == "keypoint" {
      let attributes = Set((key.attributes ?? []).compactMap(\.name))
      let parts = (key.children ?? []).compactMap { ($0 as? XMLElement)?.name }
      guard attributes == ["interpolation", "flags"],
        key.attribute(forName: "interpolation")?.stringValue == "1",
        key.attribute(forName: "flags")?.stringValue == "0",
        Set(parts).isSubset(of: keyParts), Set(parts).count == parts.count,
        let time = child(key, "time").map(ownText), let value = child(key, "value").map(ownText),
        let number = Double(value), (0...1).contains(number)
      else { throw invalid("Type On End keys must be linear values in 0–1") }
      keys.append((try frames(time, rate: rate, what: "key time") + start, number))
    }
    let count = child(curve, "numberOfKeypoints").map(ownText).flatMap { Int($0) }
    guard count == keys.count, !keys.isEmpty,
      zip(keys, keys.dropFirst()).allSatisfy({ $0.frame < $1.frame })
    else { throw invalid("Type On End keys must be counted and in time order") }
    var onsets: [Int] = []
    for frame in start..<end {
      let value = interpolated(keys, at: frame)
      let shown = min(
        graphemes, max(0, Int((value * Double(graphemes) + 0.5 + 1e-9).rounded(.down))))
      guard shown >= onsets.count else { throw invalid("the Type On reveal goes backwards") }
      onsets += Array(repeating: frame, count: shown - onsets.count)
    }
    guard onsets.count == graphemes else {
      throw invalid("the Type On reveal never shows the whole text")
    }
    return onsets
  }

  static func interpolated(_ keys: [(frame: Int, value: Double)], at frame: Int) -> Double {
    guard let after = keys.firstIndex(where: { $0.frame >= frame }) else {
      return keys[keys.count - 1].value
    }
    guard after > 0, keys[after].frame != frame else { return keys[after].value }
    let (low, high) = (keys[after - 1], keys[after])
    return low.value + (high.value - low.value) * Double(frame - low.frame)
      / Double(high.frame - low.frame)
  }
}
