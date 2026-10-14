import Foundation

/// Result of the FCPXML written alongside a native render.
public struct FCPXMLExport: Codable, Equatable, Sendable {
  public let path: String?
  public let written: Bool
  /// DTD validity against the installed Final Cut Pro; nil when it is not installed.
  public let validDTD: Bool?
  /// Recipe features present in the render but not represented as live FCP edits.
  public let unrepresented: [String]
  /// Why no FCPXML was written (an edit FCPXML would misrepresent).
  public let reason: String?

  public init(
    path: String?, written: Bool, validDTD: Bool?, unrepresented: [String], reason: String?
  ) {
    self.path = path
    self.written = written
    self.validDTD = validDTD
    self.unrepresented = unrepresented
    self.reason = reason
  }
}

/// Probed facts about one source file referenced by the timeline.
public struct FCPXMLSource: Equatable, Sendable {
  public let durationSeconds: Double
  public let hasAudio: Bool
  public init(durationSeconds: Double, hasAudio: Bool) {
    self.durationSeconds = durationSeconds
    self.hasAudio = hasAudio
  }
}

/// Builds an editable FCPXML 1.11 timeline from a native edit recipe: source clips
/// on the spine, captions and titles as Basic Title connected clips, and masks as
/// markers. Pure string building with XML escaping; no entities or external DTDs.
public enum FCPXMLTimeline {
  public static let version = "1.11"
  static let basicTitleUID =
    ".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti"
  static let sourceTimescale = 60_000
  static let titleStart = "3600s"
  static let defaultFont = "Helvetica Neue"

  struct TitleItem {
    let text: String
    let startSeconds: Double
    let endSeconds: Double
    /// Horizontal/vertical offset of the text block center from the frame center,
    /// in output pixels (positive y is up).
    let centerOffsetX: Double
    let centerOffsetY: Double
    let style: String
  }

  /// A reason the recipe cannot be represented faithfully, or nil.
  public static func unsupported(_ recipe: EditRecipe) -> String? {
    if recipe.video == nil { return "Audio-only renders have no FCPXML timeline" }
    if recipe.video?.foregroundVideoPath != nil {
      return "Foreground video is not exported to FCPXML"
    }
    if recipe.clips.contains(where: { $0.selection.rate != 1 }) {
      return "Retimed clips are not exported to FCPXML"
    }
    if recipe.clips.contains(where: { ($0.transitionInSeconds ?? 0) > 0 }) {
      return "Transitions are not exported to FCPXML"
    }
    if recipe.clips.contains(where: { $0.geometry != nil }) {
      return "Clip crop/rotation is not exported to FCPXML"
    }
    if !(recipe.additionalVideo ?? []).isEmpty || !(recipe.additionalAudio ?? []).isEmpty {
      return "Additional video/audio layers are not exported to FCPXML"
    }
    return nil
  }

  public static func document(
    _ recipe: EditRecipe, plan: EditPlan, sources: [String: FCPXMLSource],
    font: ResolvedCaptionFont?, name: String
  ) throws -> (xml: String, unrepresented: [String]) {
    if let reason = unsupported(recipe) { throw ProAppsError.invalid(reason) }
    guard let video = recipe.video else { throw ProAppsError.invalid("Missing canvas") }
    let rate = video.frameRate
    var unrepresented: [String] = []
    var assetIDs: [String: String] = [:]
    var resources = [
      "    <format id=\"r1\" frameDuration=\"\(rational(1, rate))\" width=\"\(video.width)\" height=\"\(video.height)\" colorSpace=\"1-1-1 (Rec. 709)\"/>"
    ]
    for clip in recipe.clips where assetIDs[clip.sourcePath] == nil {
      guard let source = sources[clip.sourcePath] else {
        throw ProAppsError.unavailable("Source was not probed for FCPXML")
      }
      let id = "a\(assetIDs.count + 1)"
      assetIDs[clip.sourcePath] = id
      let url = URL(fileURLWithPath: clip.sourcePath)
      resources.append(
        "    <asset id=\"\(id)\" name=\"\(escape(url.deletingPathExtension().lastPathComponent))\" start=\"0s\" duration=\"\(sourceTime(source.durationSeconds, roundingUp: true))\" hasVideo=\"1\" hasAudio=\"\(source.hasAudio ? 1 : 0)\">\n      <media-rep kind=\"original-media\" src=\"\(escape(url.absoluteString))\"/>\n    </asset>"
      )
    }
    let titles = try titleItems(
      video, font: font, duration: plan.durationSeconds, unrepresented: &unrepresented)
    if !titles.isEmpty {
      resources.append(
        "    <effect id=\"t1\" name=\"Basic Title\" uid=\"\(escape(basicTitleUID))\"/>")
    }
    if video.color != nil { unrepresented.append("color adjustments") }
    let masks = video.masks ?? []
    if !masks.isEmpty {
      unrepresented.append("masks (recorded as markers; blur/concealment is only in the render)")
    }
    var anchored: [Int: [String]] = [:]
    var laneEnds: [Double] = []
    for (index, title) in titles.enumerated() {
      let lane =
        laneEnds.firstIndex { $0 <= title.startSeconds + EditPlan.minimumTimeSeconds }
        ?? laneEnds.count
      if lane == laneEnds.count {
        laneEnds.append(title.endSeconds)
      } else {
        laneEnds[lane] = title.endSeconds
      }
      let (spanIndex, local) = anchor(title.startSeconds, plan: plan, recipe: recipe)
      anchored[spanIndex, default: []].append(
        titleXML(title, index: index + 1, lane: lane + 1, offset: local, rate: rate, video: video))
    }
    var markers: [Int: [String]] = [:]
    for mask in masks {
      let start = mask.startSeconds ?? 0
      let end = mask.endSeconds ?? plan.durationSeconds
      let (spanIndex, local) = anchor(start, plan: plan, recipe: recipe)
      let region = mask.region
      let kind = mask.blurRadius.map { "Blur r=\(format($0))" } ?? "Black"
      markers[spanIndex, default: []].append(
        "          <marker start=\"\(sourceTime(local))\" duration=\"\(frames(end - start, rate: rate))\" value=\"\(escape("\(kind) mask x=\(format(region.x)) y=\(format(region.y)) w=\(format(region.width)) h=\(format(region.height)) opacity=\(format(mask.opacity))"))\"/>"
      )
    }
    var spine: [String] = []
    for (index, span) in plan.spans.enumerated() {
      let clip = recipe.clips[span.clipIndex]
      guard let id = assetIDs[clip.sourcePath] else {
        throw ProAppsError.unavailable("Missing FCPXML asset")
      }
      let children =
        ["          <adjust-conform type=\"\(video.resizeMode.rawValue)\"/>"]
        + (anchored[index] ?? []) + (markers[index] ?? [])
      spine.append(
        "        <asset-clip ref=\"\(id)\" name=\"\(escape(URL(fileURLWithPath: clip.sourcePath).lastPathComponent))\" offset=\"\(frames(span.startSeconds, rate: rate))\" start=\"\(sourceTime(clip.selection.startSeconds))\" duration=\"\(frames(span.durationSeconds, rate: rate))\" tcFormat=\"NDF\">\n"
          + children.joined(separator: "\n") + "\n        </asset-clip>")
    }
    let project = escape(name)
    let xml = """
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE fcpxml>
      <fcpxml version="\(version)">
        <resources>
      \(resources.joined(separator: "\n"))
        </resources>
        <library>
          <event name="\(project)">
            <project name="\(project)">
              <sequence format="r1" duration="\(frames(plan.durationSeconds, rate: rate))" tcStart="0s" tcFormat="NDF">
                <spine>
      \(spine.joined(separator: "\n"))
                </spine>
              </sequence>
            </project>
          </event>
        </library>
      </fcpxml>

      """
    return (xml, unrepresented)
  }

  static func titleItems(
    _ video: EditVideoSettings, font: ResolvedCaptionFont?, duration: Double,
    unrepresented: inout [String]
  ) throws -> [TitleItem] {
    let width = Double(video.width)
    let height = Double(video.height)
    var items: [TitleItem] = []
    if let appearance = video.captionAppearance, let font {
      if appearance.rim > 0 { unrepresented.append("caption outer rim (FCP text has one stroke)") }
      let style = textStyle(
        font: font.family, face: font.face, size: font.pointSize, fill: try rgb(appearance.fill),
        stroke: appearance.border > 0 ? (try rgb(appearance.borderColor), appearance.border) : nil)
      for caption in video.styledCaptions ?? [] {
        let blockHeight = Double(caption.lines.count) * appearance.assFontSize
        items.append(
          TitleItem(
            text: caption.text, startSeconds: caption.startSeconds, endSeconds: caption.endSeconds,
            centerOffsetX: caption.x - width / 2,
            centerOffsetY: height / 2 - (caption.bottom - blockHeight / 2), style: style))
      }
    }
    let captionSize = video.captionStyle?.fontSize ?? min(48, max(16, height / 24))
    let outline = video.captionStyle?.outlineWidth ?? 0
    let captionStyle = textStyle(
      font: defaultFont, face: "Bold", size: captionSize, fill: white,
      stroke: outline > 0 ? (black, outline) : nil)
    for caption in video.captions ?? [] {
      items.append(
        TitleItem(
          text: caption.text, startSeconds: caption.startSeconds, endSeconds: caption.endSeconds,
          centerOffsetX: 0,
          centerOffsetY: -(height / 2 - TitleRenderer.captionMargin(video) - captionSize),
          style: captionStyle))
    }
    for title in video.titles ?? [] {
      items.append(
        TitleItem(
          text: title.text, startSeconds: 0, endSeconds: duration,
          centerOffsetX: title.x - width / 2,
          centerOffsetY: height / 2 - title.y - title.fontSize / 2,
          style: textStyle(
            font: defaultFont, face: "Bold", size: title.fontSize, fill: white, stroke: nil)))
    }
    return items.sorted { $0.startSeconds < $1.startSeconds }
  }

  static let white = EditRGB(red: 1, green: 1, blue: 1)
  static let black = EditRGB(red: 0, green: 0, blue: 0)

  static func rgb(_ hex: String) throws -> EditRGB {
    guard let value = EditRGB(hex: hex) else {
      throw ProAppsError.invalid("Caption colors must be #RRGGBB")
    }
    return value
  }

  static func textStyle(
    font: String, face: String?, size: Double, fill: EditRGB, stroke: (EditRGB, Double)?
  ) -> String {
    var attributes = [
      "font=\"\(escape(font))\"", "fontSize=\"\(format(size))\"",
      "fontColor=\"\(color(fill))\"", "alignment=\"center\"",
    ]
    if let face { attributes.insert("fontFace=\"\(escape(face))\"", at: 2) }
    if let (strokeColor, strokeWidth) = stroke {
      attributes.append("strokeColor=\"\(color(strokeColor))\"")
      attributes.append("strokeWidth=\"\(format(strokeWidth))\"")
    }
    return attributes.joined(separator: " ")
  }

  static func titleXML(
    _ title: TitleItem, index: Int, lane: Int, offset: Double, rate: Int, video: EditVideoSettings
  ) -> String {
    let height = Double(video.height)
    let duration = title.endSeconds - title.startSeconds
    // FCPXML transform positions are percentages of the frame height.
    let x = title.centerOffsetX / height * 100
    let y = title.centerOffsetY / height * 100
    return """
                <title ref="t1" lane="\(lane)" offset="\(sourceTime(offset))" name="\(escape("Caption \(index)"))" start="\(titleStart)" duration="\(frames(duration, rate: rate))">
                  <text>
                    <text-style ref="ts\(index)">\(escape(title.text))</text-style>
                  </text>
                  <text-style-def id="ts\(index)">
                    <text-style \(title.style)/>
                  </text-style-def>
                  <adjust-transform position="\(format(x)) \(format(y))"/>
                </title>
      """
  }

  /// The spine clip containing an output time, and that time in the clip's local
  /// (source) time base. Rate is always 1 for exported recipes.
  static func anchor(_ seconds: Double, plan: EditPlan, recipe: EditRecipe) -> (Int, Double) {
    let index =
      plan.spans.lastIndex { $0.startSeconds <= seconds + EditPlan.minimumTimeSeconds } ?? 0
    let span = plan.spans[index]
    let source = recipe.clips[span.clipIndex].selection.startSeconds
    return (index, source + max(0, seconds - span.startSeconds))
  }

  static func frames(_ seconds: Double, rate: Int) -> String {
    rational(Int((seconds * Double(rate)).rounded()), rate)
  }

  static func sourceTime(_ seconds: Double, roundingUp: Bool = false) -> String {
    let scaled = seconds * Double(sourceTimescale)
    return rational(Int(roundingUp ? scaled.rounded(.up) : scaled.rounded()), sourceTimescale)
  }

  static func rational(_ numerator: Int, _ denominator: Int) -> String {
    guard numerator != 0 else { return "0s" }
    var a = abs(numerator)
    var b = denominator
    while b != 0 { (a, b) = (b, a % b) }
    let top = numerator / a
    let bottom = denominator / a
    return bottom == 1 ? "\(top)s" : "\(top)/\(bottom)s"
  }

  static func color(_ rgb: EditRGB) -> String {
    "\(format(rgb.red)) \(format(rgb.green)) \(format(rgb.blue)) 1"
  }

  static func format(_ value: Double) -> String {
    let rounded = (value * 10_000).rounded() / 10_000
    return rounded == rounded.rounded() ? String(Int(rounded)) : String(rounded)
  }

  static func escape(_ text: String) -> String {
    var result = ""
    for character in text {
      switch character {
      case "&": result += "&amp;"
      case "<": result += "&lt;"
      case ">": result += "&gt;"
      case "\"": result += "&quot;"
      case "'": result += "&apos;"
      default: result.append(character)
      }
    }
    return result
  }
}
