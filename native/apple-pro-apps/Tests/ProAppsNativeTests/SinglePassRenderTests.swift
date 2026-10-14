import AVFoundation
import CoreImage
import CoreText
import Foundation
import Testing

@testable import ProAppsCore

// Hardware codec sessions are a shared macOS resource; serialize like NativeEditorTests.
@Suite(.serialized)
struct SinglePassRenderTests {
  static let appearance = EditCaptionAppearance(
    fontName: "HiraginoSans-W6", assFontSize: 24, fill: "#6CD4FF", border: 2,
    borderColor: "#FFFFFF", rim: 1, rimColor: "#000000")

  private func directory() throws -> URL {
    let url = try Files.temporaryDirectory().appendingPathComponent(
      "single-pass-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
    return url
  }

  private func fixture(_ name: String) throws -> String {
    try #require(
      Bundle.module.url(forResource: name, withExtension: "mp4", subdirectory: "Fixtures")
    )
    .path
  }

  @Test func fillsAConstantCadenceFromASingleSourceFrame() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let recipe = EditRecipe(
      clips: [
        EditClip(
          sourcePath: try fixture("black"),
          selection: EditSelection(startSeconds: 0, durationSeconds: 1, rate: 1))
      ],
      video: EditVideoSettings(
        width: 320, height: 240, frameRate: 30, resizeMode: .fit,
        encoding: EditEncoding(codec: .h264, averageBitRate: 2_000_000, allowFrameReordering: false)
      ))
    let result = try await NativeEditor().render(recipe, directory: root.path, name: "black.mp4")
    #expect(result.frameCount == 30)
    #expect(result.audioTrackCount == 0)
    let verified = try await MediaProbe().verifyShortVideo(path: result.outputPath)
    #expect(verified.decodedFrames == 30)
    #expect(verified.media.width == 320)
    let fcpxml = try #require(result.fcpxml)
    #expect(fcpxml.written)
    #expect(fcpxml.validDTD == true)
    #expect(fcpxml.unrepresented == [])
    let path = try #require(fcpxml.path)
    #expect(FileManager.default.fileExists(atPath: path))
  }

  @Test(arguments: [EditEncoding.Codec.h264, .hevc])
  func composesMasksColorTitlesAndStyledCaptionsInOnePass(_ codec: EditEncoding.Codec)
    async throws
  {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let recipe = EditRecipe(
      clips: [
        EditClip(
          sourcePath: try fixture("quadrants-30"),
          selection: EditSelection(startSeconds: 0, durationSeconds: 1, rate: 1))
      ],
      video: EditVideoSettings(
        width: 320, height: 240, frameRate: 30, resizeMode: .fill,
        color: EditColor(brightness: 0, contrast: 1, saturation: 1),
        titles: [EditTitle(text: "T", x: 4, y: 4, fontSize: 12)],
        masks: [
          EditMask(
            region: EditCrop(x: 0, y: 0, width: 160, height: 120), opacity: 1, blurRadius: 8,
            startSeconds: 0, endSeconds: 0.5),
          EditMask(region: EditCrop(x: 160, y: 120, width: 160, height: 120), opacity: 1),
        ],
        styledCaptions: [
          EditStyledCaption(text: "あい\nう", startSeconds: 0.5, endSeconds: 1, x: 160, bottom: 200),
          EditStyledCaption(text: "え", startSeconds: 0.6, endSeconds: 0.8, x: 100, bottom: 120),
        ],
        captionAppearance: Self.appearance,
        encoding: EditEncoding(codec: codec, averageBitRate: 2_000_000, allowFrameReordering: true))
    )
    let result = try await NativeEditor().render(recipe, directory: root.path, name: "full.mp4")
    #expect(result.frameCount == 30)
    let region = EditCrop(x: 0, y: 160, width: 320, height: 40)
    let samples = try await FrameProbe().measure(
      path: result.outputPath,
      samples: [.init(timeSeconds: 0.2, region: region), .init(timeSeconds: 0.7, region: region)])
    let before = try #require(samples.first)
    let during = try #require(samples.last)
    #expect(during.meanBlue > before.meanBlue)
    let concealed = try await FrameProbe().measure(
      path: result.outputPath,
      samples: [.init(timeSeconds: 0.9, region: EditCrop(x: 200, y: 140, width: 60, height: 40))])
    #expect(try #require(concealed.first).meanRed < 0.05)
    let fcpxml = try #require(result.fcpxml)
    #expect(fcpxml.validDTD == true)
    #expect(fcpxml.unrepresented.contains("caption outer rim (FCP text has one stroke)"))
  }

  @Test func failuresDuringTheFrameLoopPublishNothing() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let recipe = EditRecipe(
      clips: [
        EditClip(
          sourcePath: try fixture("quadrants-30"),
          selection: EditSelection(startSeconds: 0, durationSeconds: 1, rate: 1))
      ],
      video: EditVideoSettings(
        width: 320, height: 240, frameRate: 30, resizeMode: .fit,
        styledCaptions: [
          EditStyledCaption(
            text: String(repeating: "あ", count: 40), startSeconds: 0.5, endSeconds: 1, x: 160,
            bottom: 200)
        ],
        captionAppearance: Self.appearance,
        encoding: EditEncoding(codec: .h264, averageBitRate: 2_000_000, allowFrameReordering: false)
      ))
    await #expect(
      throws: ProAppsError.invalid("Styled caption is wider than the canvas; wrap it into lines")
    ) {
      try await NativeEditor().render(recipe, directory: root.path, name: "wide.mp4")
    }
    let published = try FileManager.default.subpathsOfDirectory(atPath: root.path)
      .filter { $0.hasSuffix("wide.mp4") }
    #expect(published.isEmpty)
  }

  @Test func frameworkRefusalsBecomeExplicitErrors() throws {
    try NativeEditor.require(true, "unused")
    #expect(throws: ProAppsError.unavailable("refused")) {
      try NativeEditor.require(false, "refused")
    }
    #expect(try NativeEditor.required(7, "unused") == 7)
    #expect(throws: ProAppsError.unavailable("missing")) {
      let absent: Int? = nil
      _ = try NativeEditor.required(absent, "missing")
    }
  }

  @Test func reportsUnsupportedTimelinesInsteadOfWritingThem() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let selection = EditSelection(startSeconds: 0, durationSeconds: 1, rate: 2)
    let retimed = EditRecipe(
      clips: [EditClip(sourcePath: "/synthetic/never-read.mp4", selection: selection)],
      video: EditVideoSettings(width: 64, height: 64, frameRate: 30, resizeMode: .fit))
    let export = try await NativeEditor().exportTimeline(
      retimed, plan: EditPlan.build(retimed), directory: root, name: "x")
    #expect(export?.written == false)
    #expect(export?.reason == "Retimed clips are not exported to FCPXML")
    let audioOnly = EditRecipe(
      clips: [EditClip(sourcePath: "/synthetic/a.wav", selection: selection)])
    #expect(
      try await NativeEditor().exportTimeline(
        audioOnly, plan: EditPlan.build(audioOnly), directory: root, name: "y") == nil)
  }

  @Test func dtdValidationReportsInvalidDocuments() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let file = root.appendingPathComponent("bad.fcpxml")
    try Data(
      "<?xml version=\"1.0\"?>\n<!DOCTYPE fcpxml>\n<fcpxml version=\"1.11\"><wrong/></fcpxml>".utf8
    )
    .write(to: file)
    #expect(try await NativeEditor.validate(file) == false)
  }

  @Test func resolvesInstalledFontsWithLibassMetrics() throws {
    let font = try StyledCaptionRenderer.resolve(Self.appearance)
    #expect(font.family == "Hiragino Sans")
    #expect(font.face == "W6")
    #expect(font.pointSize < Self.appearance.assFontSize)
    #expect(font.lineAdvance == 24)
    #expect(font.lineAscent > 0 && font.lineAscent < 24)
    #expect(font.renderedAscent > 0)
    let missing = EditCaptionAppearance(
      fontName: "NoSuchFont-Regular", assFontSize: 24, fill: "#FFFFFF", border: 0,
      borderColor: "#FFFFFF", rim: 0, rimColor: "#000000")
    #expect(throws: ProAppsError.invalid("Caption font is not installed: NoSuchFont-Regular")) {
      try StyledCaptionRenderer.resolve(missing)
    }
  }

  @Test func readsOS2WinMetrics() throws {
    var table = [UInt8](repeating: 0, count: StyledCaptionRenderer.os2MinimumLength)
    table[74] = 0x03
    table[75] = 0xE8
    table[76] = 0x01
    table[77] = 0x2C
    let parsed = try #require(StyledCaptionRenderer.os2WinMetrics(Data(table)))
    #expect(parsed.0 == 1000)
    #expect(parsed.1 == 300)
    #expect(StyledCaptionRenderer.os2WinMetrics(Data(repeating: 0, count: 10)) == nil)
    let font = CTFontCreateWithName("HiraginoSans-W6" as CFString, 1, nil)
    let metrics = try StyledCaptionRenderer.verticalMetrics(font, unitsPerEm: 1000)
    #expect(metrics.0 > 0 && metrics.1 > 0)
    #expect(throws: ProAppsError.unavailable("Caption font reports no units per em")) {
      try StyledCaptionRenderer.verticalMetrics(font, unitsPerEm: 0)
    }
  }

  @MainActor
  @Test func cachesActiveCaptionsAndRejectsOverwideOnes() throws {
    let canvas = EditVideoSettings(width: 320, height: 240, frameRate: 30, resizeMode: .fit)
    let captions = [
      EditStyledCaption(text: "う", startSeconds: 0.5, endSeconds: 1, x: 160, bottom: 200),
      EditStyledCaption(text: "あ", startSeconds: 0, endSeconds: 1, x: 160, bottom: 100),
    ]
    let cache = try StyledCaptionCache(
      captions, appearance: Self.appearance, canvas: canvas,
      colorSpaceName: CGColorSpace.itur_709 as String)
    #expect(cache.resolvedFont.postScriptName == "HiraginoSans-W6")
    #expect(try cache.active(at: 0.25).count == 1)
    let both = try cache.active(at: 0.75)
    #expect(both.count == 2)
    #expect(both[0].extent.minY < both[1].extent.minY)
    #expect(cache.cachedCount == 2)
    #expect(try cache.active(at: 1.5).isEmpty)
    #expect(cache.cachedCount == 0)
    let wide = EditStyledCaption(
      text: String(repeating: "あ", count: 40), startSeconds: 0, endSeconds: 1, x: 160, bottom: 200)
    let overwide = try StyledCaptionCache([wide], appearance: Self.appearance, canvas: canvas)
    #expect(
      throws: ProAppsError.invalid("Styled caption is wider than the canvas; wrap it into lines")
    ) {
      try overwide.active(at: 0.5)
    }
  }

  @Test func tintsAndDilatesGlyphMasks() throws {
    let mask = CIImage(color: CIColor(red: 1, green: 1, blue: 1)).cropped(
      to: CGRect(x: 0, y: 0, width: 8, height: 8))
    #expect(try StyledCaptionRenderer.dilated(mask, radius: 0, extent: mask.extent) == mask)
    #expect(
      try StyledCaptionRenderer.dilated(mask, radius: 2, extent: mask.extent).extent == mask.extent)
    #expect(
      try StyledCaptionRenderer.tinted(mask, hex: "#FF0000", colorSpace: nil).extent == mask.extent)
    #expect(
      try StyledCaptionRenderer.tinted(
        mask, hex: "#FF0000", colorSpace: CGColorSpace(name: CGColorSpace.itur_709)
      ).extent == mask.extent)
    #expect(throws: ProAppsError.invalid("Caption colors must be #RRGGBB")) {
      try StyledCaptionRenderer.tinted(mask, hex: "red", colorSpace: nil)
    }
  }

  @Test func buildsFixedWriterSettings() throws {
    let h264 = VideoWriterSettings.output(
      EditEncoding(codec: .h264, averageBitRate: 6_000_000, allowFrameReordering: false),
      width: 1080, height: 1920, frameRate: 30)
    #expect(h264[AVVideoCodecKey] as? AVVideoCodecType == .h264)
    #expect(h264[AVVideoWidthKey] as? Int == 1080)
    let compression = try #require(h264[AVVideoCompressionPropertiesKey] as? [String: Any])
    #expect(compression[AVVideoAverageBitRateKey] as? Int == 6_000_000)
    #expect(compression[AVVideoAllowFrameReorderingKey] as? Bool == false)
    #expect(compression[AVVideoProfileLevelKey] as? String == AVVideoProfileLevelH264HighAutoLevel)
    let color = try #require(h264[AVVideoColorPropertiesKey] as? [String: String])
    #expect(color[AVVideoColorPrimariesKey] == AVVideoColorPrimaries_ITU_R_709_2)
    let hevc = VideoWriterSettings.output(
      EditEncoding(codec: .hevc, averageBitRate: 6_000_000, allowFrameReordering: true),
      width: 64, height: 64, frameRate: 60)
    #expect(hevc[AVVideoCodecKey] as? AVVideoCodecType == .hevc)
    #expect(
      VideoWriterSettings.pixelBuffers(width: 2, height: 4)[kCVPixelBufferHeightKey as String]
        as? Int == 4)
    #expect(
      VideoWriterSettings.readerPixels()[kCVPixelBufferPixelFormatTypeKey as String] as? OSType
        == kCVPixelFormatType_32BGRA)
  }
}
