import Foundation
import Testing

@testable import ProAppsCore

struct EditSinglePassTests {
  static let encoding = EditEncoding(
    codec: .h264, averageBitRate: 6_000_000, allowFrameReordering: false)
  static let appearance = EditCaptionAppearance(
    fontName: "HiraginoSans-W6", assFontSize: 66, fill: "#6CD4FF", border: 6,
    borderColor: "#FFFFFF", rim: 3, rimColor: "#000000")
  static let caption = EditStyledCaption(
    text: "一行目\n二行目", startSeconds: 0.5, endSeconds: 1.5, x: 540, bottom: 554)

  static func recipe(
    styled: [EditStyledCaption]? = [caption], appearance: EditCaptionAppearance? = appearance,
    encoding: EditEncoding? = encoding, audio: EditAudioAdjustment? = nil,
    additionalAudio: [EditAudioLayer]? = nil
  ) -> EditRecipe {
    EditRecipe(
      clips: [
        EditClip(
          sourcePath: "/synthetic/source.mp4",
          selection: EditSelection(startSeconds: 0, durationSeconds: 2, rate: 1), audio: audio)
      ],
      video: EditVideoSettings(
        width: 1080, height: 1920, frameRate: 30, resizeMode: .fit, styledCaptions: styled,
        captionAppearance: appearance, encoding: encoding),
      additionalAudio: additionalAudio)
  }

  @Test(arguments: [
    ("#6CD4FF", EditRGB(red: 0x6C / 255, green: 0xD4 / 255, blue: 1)),
    ("#000000", EditRGB(red: 0, green: 0, blue: 0)),
  ])
  func parsesHexColors(_ hex: String, _ expected: EditRGB) {
    #expect(EditRGB(hex: hex) == expected)
  }

  @Test(arguments: ["6CD4FF", "#6CD4F", "#GG0000", "#6CD4FFAA", ""])
  func rejectsMalformedHexColors(_ hex: String) {
    #expect(EditRGB(hex: hex) == nil)
  }

  @Test func acceptsSinglePassRecipesWithAndWithoutCaptions() throws {
    #expect(try EditPlan.build(Self.recipe()).durationSeconds == 2)
    #expect(try EditPlan.build(Self.recipe(styled: nil, appearance: nil)).durationSeconds == 2)
    #expect(try EditPlan.build(Self.recipe(styled: [])).durationSeconds == 2)
  }

  @Test func requiresEncodingForStyledCaptionsAndAppearance() {
    #expect(
      throws: ProAppsError.invalid("styledCaptions require video.encoding (single-pass render)")
    ) {
      try EditPlan.build(Self.recipe(encoding: nil))
    }
    #expect(
      throws: ProAppsError.invalid("styledCaptions require video.encoding (single-pass render)")
    ) {
      try EditPlan.build(Self.recipe(styled: nil, encoding: nil))
    }
    #expect(throws: ProAppsError.invalid("styledCaptions require captionAppearance")) {
      try EditPlan.build(Self.recipe(appearance: nil))
    }
  }

  @Test func refusesAudioAndOutOfRangeBitRates() {
    #expect(throws: ProAppsError.invalid("Encoding averageBitRate must be 500000–80000000 bits/s"))
    {
      try EditPlan.build(
        Self.recipe(
          encoding: EditEncoding(codec: .hevc, averageBitRate: 100, allowFrameReordering: true)))
    }
    let reason = "Single-pass encoding is video-only; remove audio adjustments and additional audio"
    #expect(throws: ProAppsError.invalid(reason)) {
      try EditPlan.build(Self.recipe(audio: .unity))
    }
    #expect(throws: ProAppsError.invalid(reason)) {
      try EditPlan.build(
        Self.recipe(
          additionalAudio: [
            EditAudioLayer(
              sourcePath: "/synthetic/a.wav",
              selection: EditSelection(startSeconds: 0, durationSeconds: 1, rate: 1),
              offsetSeconds: 0)
          ]))
    }
  }

  @Test func boundsStyledCaptionCount() {
    let many = Array(repeating: Self.caption, count: EditPlan.maximumStyledCaptions + 1)
    #expect(throws: ProAppsError.invalid("At most 400 styled captions are supported")) {
      try EditPlan.build(Self.recipe(styled: many))
    }
  }

  @Test(arguments: [
    EditStyledCaption(
      text: "1\n2\n3\n4\n5\n6\n7", startSeconds: 0, endSeconds: 1, x: 540, bottom: 900),
    EditStyledCaption(text: "a\n \nb", startSeconds: 0, endSeconds: 1, x: 540, bottom: 900),
    EditStyledCaption(text: "a\rb", startSeconds: 0, endSeconds: 1, x: 540, bottom: 900),
    EditStyledCaption(text: "a", startSeconds: 0, endSeconds: 3, x: 540, bottom: 900),
    EditStyledCaption(text: "a", startSeconds: -1, endSeconds: 1, x: 540, bottom: 900),
    EditStyledCaption(text: "a", startSeconds: 1, endSeconds: 1, x: 540, bottom: 900),
    EditStyledCaption(text: "a", startSeconds: 0, endSeconds: 1, x: 0, bottom: 900),
    EditStyledCaption(text: "a", startSeconds: 0, endSeconds: 1, x: 1080, bottom: 900),
    EditStyledCaption(text: "a", startSeconds: 0, endSeconds: 1, x: 540, bottom: 0),
    EditStyledCaption(text: "a", startSeconds: 0, endSeconds: 1, x: 540, bottom: 1921),
    EditStyledCaption(
      text: String(repeating: "あ", count: 201), startSeconds: 0, endSeconds: 1, x: 540,
      bottom: 900),
  ])
  func rejectsInvalidStyledCaptions(_ caption: EditStyledCaption) {
    #expect(throws: ProAppsError.self) { try EditPlan.build(Self.recipe(styled: [caption])) }
  }

  @Test(arguments: [
    EditCaptionAppearance(
      fontName: "Bad Name", assFontSize: 66, fill: "#FFFFFF", border: 1, borderColor: "#FFFFFF",
      rim: 1, rimColor: "#000000"),
    EditCaptionAppearance(
      fontName: "", assFontSize: 66, fill: "#FFFFFF", border: 1, borderColor: "#FFFFFF", rim: 1,
      rimColor: "#000000"),
    EditCaptionAppearance(
      fontName: "Font", assFontSize: 7, fill: "#FFFFFF", border: 1, borderColor: "#FFFFFF", rim: 1,
      rimColor: "#000000"),
    EditCaptionAppearance(
      fontName: "Font", assFontSize: 66, fill: "white", border: 1, borderColor: "#FFFFFF", rim: 1,
      rimColor: "#000000"),
    EditCaptionAppearance(
      fontName: "Font", assFontSize: 66, fill: "#FFFFFF", border: 17, borderColor: "#FFFFFF",
      rim: 1, rimColor: "#000000"),
    EditCaptionAppearance(
      fontName: "Font", assFontSize: 66, fill: "#FFFFFF", border: 1, borderColor: "#FFFFFF",
      rim: -1, rimColor: "#000000"),
    EditCaptionAppearance(
      fontName: "Font", assFontSize: 66, fill: "#FFFFFF", border: 1, borderColor: "#FFF",
      rim: 1, rimColor: "#000000"),
    EditCaptionAppearance(
      fontName: "Font", assFontSize: 66, fill: "#FFFFFF", border: 1, borderColor: "#FFFFFF",
      rim: 1, rimColor: "black"),
  ])
  func rejectsInvalidAppearances(_ appearance: EditCaptionAppearance) {
    #expect(throws: ProAppsError.self) { try EditPlan.build(Self.recipe(appearance: appearance)) }
  }

  @Test func exposesCaptionLinesAndHalfOpenActivity() {
    #expect(Self.caption.lines == ["一行目", "二行目"])
    #expect(!Self.caption.isActive(at: 0.49))
    #expect(Self.caption.isActive(at: 0.5))
    #expect(!Self.caption.isActive(at: 1.5))
  }

  @Test func singlePassFieldsRoundTripThroughJSON() throws {
    let recipe = Self.recipe()
    let decoded = try JSONDecoder().decode(EditRecipe.self, from: JSONEncoder().encode(recipe))
    #expect(decoded.video?.encoding == Self.encoding)
    #expect(decoded.video?.captionAppearance == Self.appearance)
    #expect(decoded.video?.styledCaptions == [Self.caption])
  }
}

struct FCPXMLTimelineTests {
  static let font = ResolvedCaptionFont(
    postScriptName: "HiraginoSans-W6", family: "Hiragino Sans", face: "W6", pointSize: 56,
    lineAscent: 50, lineAdvance: 66, renderedAscent: 48)
  static let sources = [
    "/synthetic/a & b.mp4": FCPXMLSource(durationSeconds: 100.00001, hasAudio: true),
    "/synthetic/c.mp4": FCPXMLSource(durationSeconds: 20, hasAudio: false),
  ]

  static func recipe(_ change: (inout EditClip) -> Void = { _ in }) -> EditRecipe {
    var first = EditClip(
      sourcePath: "/synthetic/a & b.mp4",
      selection: EditSelection(startSeconds: 10, durationSeconds: 2, rate: 1))
    change(&first)
    return EditRecipe(
      clips: [
        first,
        EditClip(
          sourcePath: "/synthetic/c.mp4",
          selection: EditSelection(startSeconds: 0.5, durationSeconds: 1, rate: 1)),
      ],
      video: EditVideoSettings(
        width: 1080, height: 1920, frameRate: 30, resizeMode: .fill,
        color: EditColor(brightness: 0.1, contrast: 1, saturation: 1),
        titles: [EditTitle(text: "Title <1>", x: 10, y: 20, fontSize: 32)],
        captions: [EditCaption(text: "plain", startSeconds: 2.2, endSeconds: 2.8)],
        masks: [
          EditMask(
            region: EditCrop(x: 0, y: 1300, width: 1080, height: 120), opacity: 1, blurRadius: 64,
            startSeconds: 0.5, endSeconds: 1),
          EditMask(region: EditCrop(x: 10, y: 10, width: 20, height: 20), opacity: 0.5),
        ],
        captionStyle: EditCaptionStyle(outlineWidth: 2, backgroundOpacity: 0),
        styledCaptions: [
          EditStyledCaption(text: "a\n\"b\"", startSeconds: 0, endSeconds: 1, x: 540, bottom: 554),
          EditStyledCaption(text: "c", startSeconds: 0.5, endSeconds: 2.5, x: 300, bottom: 800),
        ],
        captionAppearance: EditSinglePassTests.appearance,
        encoding: EditSinglePassTests.encoding))
  }

  @Test func writesAnEditableTimelineWithTitlesLanesAndMarkers() throws {
    let recipe = Self.recipe()
    let document = try FCPXMLTimeline.document(
      recipe, plan: EditPlan.build(recipe), sources: Self.sources, font: Self.font, name: "Part & 1"
    )
    let xml = document.xml
    #expect(try Interchange.inspect(Data(xml.utf8), kind: .fcpxml).root == "fcpxml")
    #expect(
      xml.contains("<format id=\"r1\" frameDuration=\"1/30s\" width=\"1080\" height=\"1920\""))
    #expect(xml.contains("src=\"file:///synthetic/a%20&amp;%20b.mp4\""))
    #expect(xml.contains("duration=\"6000001/60000s\" hasVideo=\"1\" hasAudio=\"1\""))
    #expect(xml.contains("hasAudio=\"0\""))
    #expect(xml.contains("<event name=\"Part &amp; 1\">"))
    #expect(xml.contains("<sequence format=\"r1\" duration=\"3s\""))
    #expect(xml.contains("offset=\"0s\" start=\"10s\" duration=\"2s\""))
    #expect(xml.contains("offset=\"2s\" start=\"1/2s\" duration=\"1s\""))
    #expect(xml.components(separatedBy: "<adjust-conform type=\"fill\"/>").count == 3)
    #expect(xml.components(separatedBy: "<title ref=\"t1\"").count == 5)
    #expect(xml.contains("lane=\"2\""))
    #expect(xml.contains("&quot;b&quot;"))
    #expect(xml.contains("Title &lt;1&gt;"))
    #expect(xml.contains("font=\"Hiragino Sans\" fontSize=\"56\" fontFace=\"W6\""))
    #expect(xml.contains("fontColor=\"0.4235 0.8314 1 1\""))
    #expect(xml.contains("strokeColor=\"1 1 1 1\" strokeWidth=\"6\""))
    #expect(xml.contains("value=\"Blur r=64 mask x=0 y=1300 w=1080 h=120 opacity=1\""))
    #expect(xml.contains("value=\"Black mask x=10 y=10 w=20 h=20 opacity=0.5\""))
    #expect(
      document.unrepresented == [
        "caption outer rim (FCP text has one stroke)", "color adjustments",
        "masks (recorded as markers; blur/concealment is only in the render)",
      ])
  }

  @Test func refusesMissingProbesAndUnsupportedRecipes() throws {
    let recipe = Self.recipe()
    #expect(throws: ProAppsError.unavailable("Source was not probed for FCPXML")) {
      try FCPXMLTimeline.document(
        recipe, plan: EditPlan.build(recipe), sources: [:], font: nil, name: "x")
    }
    let retimed = EditRecipe(
      clips: [
        EditClip(
          sourcePath: "/synthetic/c.mp4",
          selection: EditSelection(startSeconds: 0, durationSeconds: 2, rate: 2))
      ],
      video: EditVideoSettings(width: 64, height: 64, frameRate: 30, resizeMode: .fit))
    #expect(throws: ProAppsError.invalid("Retimed clips are not exported to FCPXML")) {
      try FCPXMLTimeline.document(
        retimed, plan: EditPlan.build(retimed), sources: Self.sources, font: nil, name: "x")
    }
  }

  @Test(arguments: [
    ("audio", "Audio-only renders have no FCPXML timeline"),
    ("rate", "Retimed clips are not exported to FCPXML"),
    ("transition", "Transitions are not exported to FCPXML"),
    ("geometry", "Clip crop/rotation is not exported to FCPXML"),
    ("layers", "Additional video/audio layers are not exported to FCPXML"),
    ("plain", ""),
  ])
  func explainsUnsupportedRecipes(_ kind: String, _ reason: String) {
    let selection = EditSelection(startSeconds: 0, durationSeconds: 2, rate: kind == "rate" ? 2 : 1)
    let clip = EditClip(
      sourcePath: "/s.mp4", selection: selection,
      geometry: kind == "geometry" ? EditGeometry(rotation: .clockwise90) : nil,
      transitionInSeconds: kind == "transition" ? 0.5 : nil)
    let recipe = EditRecipe(
      clips: [clip],
      video: kind == "audio"
        ? nil : EditVideoSettings(width: 64, height: 64, frameRate: 30, resizeMode: .fit),
      additionalVideo: kind == "layers"
        ? [EditVideoLayer(sourcePath: "/s.mp4", selection: selection, offsetSeconds: 0)] : nil)
    #expect(FCPXMLTimeline.unsupported(recipe) == (reason.isEmpty ? nil : reason))
  }

  @Test(arguments: [
    (0.0, 30, "0s"), (1.0, 30, "1s"), (0.5, 30, "1/2s"), (10.5, 30, "21/2s"),
    (1.0 / 30, 30, "1/30s"),
  ])
  func formatsFrameRationals(_ seconds: Double, _ rate: Int, _ expected: String) {
    #expect(FCPXMLTimeline.frames(seconds, rate: rate) == expected)
  }

  @Test func formatsSourceTimesAndValues() {
    #expect(FCPXMLTimeline.sourceTime(21.033333333) == "631/30s")
    #expect(FCPXMLTimeline.sourceTime(1.000001, roundingUp: true) == "60001/60000s")
    #expect(FCPXMLTimeline.format(2) == "2")
    #expect(FCPXMLTimeline.format(0.123456) == "0.1235")
    #expect(FCPXMLTimeline.escape("<a & 'b'>") == "&lt;a &amp; &apos;b&apos;&gt;")
    #expect(throws: ProAppsError.invalid("Caption colors must be #RRGGBB")) {
      try FCPXMLTimeline.rgb("nope")
    }
  }
}
