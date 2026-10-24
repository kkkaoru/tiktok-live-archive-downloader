import Foundation
import Testing

@testable import ProAppsCore

struct MotionCaptionRendererTests {
  static let style = MotionCaptionStyle(
    fontName: "HiraginoSans-W6", fontSize: 60, faceGrow: 0.3, keyline: 1.4, whiteBorder: 6,
    rim: 8, shadowOpacity: 0.7, shadowOffsetY: 6, shadowSigma: 2, gradientTop: "#97D8F7",
    gradientBottom: "#2FB0EE", topLocation: 0.27, bottomLocation: 0.39, outputGamma: 1.961)
  static let title = MotionCaptionTitle(
    text: "口I\n日", startFrame: 2, endFrame: 10, x: 0, y: 0, graphemeFrames: [2, 4, 4, 6])

  static func renderer(_ title: MotionCaptionTitle = title) throws -> MotionCaptionRenderer {
    try MotionCaptionRenderer(title, style: style, canvasWidth: 320, canvasHeight: 320)
  }

  static func opaque(_ bitmap: MotionCaptionBitmap) -> Int {
    stride(from: 3, to: bitmap.rgba.count, by: 4).count { bitmap.rgba[$0] == 255 }
  }

  @Test func typeOnShowsEachGraphemeFromItsOnsetFrame() {
    #expect((0..<12).map(Self.title.visible(at:)) == [0, 0, 1, 1, 3, 3, 4, 4, 4, 4, 0, 0])
  }

  @Test func refusesTitlesWhoseOnsetsDoNotMatchTheirGraphemes() {
    let short = MotionCaptionTitle(
      text: "口I", startFrame: 0, endFrame: 4, x: 0, y: 0, graphemeFrames: [0])
    let late = MotionCaptionTitle(
      text: "口", startFrame: 0, endFrame: 4, x: 0, y: 0, graphemeFrames: [4])
    #expect(throws: ProAppsError.self) { try Self.renderer(short) }
    #expect(throws: ProAppsError.self) { try Self.renderer(late) }
  }

  @Test func refusesOutlinesThatDoNotNest() {
    let style = MotionCaptionStyle(
      fontName: "HiraginoSans-W6", fontSize: 60, faceGrow: 0, keyline: 7, whiteBorder: 6,
      rim: 8, shadowOpacity: 0.7, shadowOffsetY: 6, shadowSigma: 2, gradientTop: "#97D8F7",
      gradientBottom: "#2FB0EE", topLocation: 0.27, bottomLocation: 0.39, outputGamma: 1.961)
    #expect(throws: ProAppsError.self) {
      try MotionCaptionRenderer(Self.title, style: style, canvasWidth: 320, canvasHeight: 320)
    }
  }

  @Test func refusesAFontThatIsNotInstalled() {
    let style = MotionCaptionStyle(
      fontName: "NoSuchFont-Synthetic", fontSize: 60, faceGrow: 0, keyline: 1, whiteBorder: 6,
      rim: 8, shadowOpacity: 0.7, shadowOffsetY: 6, shadowSigma: 2, gradientTop: "#97D8F7",
      gradientBottom: "#2FB0EE", topLocation: 0.27, bottomLocation: 0.39, outputGamma: 1.961)
    #expect(throws: ProAppsError.self) {
      try MotionCaptionRenderer(Self.title, style: style, canvasWidth: 320, canvasHeight: 320)
    }
  }

  @Test func revealsGlyphsInTheirFinalLayout() throws {
    let renderer = try Self.renderer()
    let none = try renderer.bitmap(visible: 0)
    let first = try renderer.bitmap(visible: 1)
    let all = try renderer.bitmap(visible: 4)
    #expect(none.rgba.allSatisfy { $0 == 0 })
    #expect(Self.opaque(first) > 0)
    #expect(Self.opaque(all) > Self.opaque(first))
    // Every pixel the first glyph covers is covered once the whole title shows.
    let kept = stride(from: 3, to: first.rgba.count, by: 4).allSatisfy {
      first.rgba[$0] < 255 || all.rgba[$0] == 255
    }
    #expect(kept)
    #expect((all.originX, all.originY) == (first.originX, first.originY))
  }

  @Test func drawsTheWhiteBandBlackOutlinesAndTheEncodedFill() throws {
    let all = try Self.renderer().bitmap(visible: 4)
    let pixels: [[UInt8]] = stride(from: 0, to: all.rgba.count, by: 4).map { (index: Int) in
      Array(all.rgba[index..<(index + 4)])
    }
    #expect(pixels.contains([255, 255, 255, 255]))
    #expect(pixels.contains([0, 0, 0, 255]))
    // #2FB0EE in linear light encoded with gamma 1.961, as Motion writes it.
    #expect(pixels.contains([41, 166, 235, 255]))
    #expect(pixels.contains([140, 210, 246, 255]))
  }

  @Test func growsCoverageByASquare() throws {
    let renderer = try Self.renderer()
    var dot = [Float](repeating: 0, count: renderer.width * renderer.height)
    let centre = (renderer.height / 2) * renderer.width + renderer.width / 2
    dot[centre] = 1
    let grown = renderer.grown(dot, by: 2.5)
    #expect(grown.count { $0 == 1 } == 25)
    #expect(grown.count { $0 == 0.5 } == 20)
    #expect(grown.count { $0 == 0.25 } == 4)
    #expect(renderer.grown(dot, by: 0) == dot)
  }

  @Test func easesTheGradientBetweenItsStopsInEachLineBox() throws {
    let renderer = try Self.renderer()
    let top = renderer.gradient(at: renderer.boxTop + 0.1 * renderer.boxHeight)
    let bottom = renderer.gradient(at: renderer.boxTop + 0.9 * renderer.boxHeight)
    let nextLine = renderer.gradient(at: renderer.boxTop + 1.9 * renderer.boxHeight)
    let above = renderer.gradient(at: renderer.boxTop - 0.9 * renderer.boxHeight)
    #expect(bottom == nextLine)
    #expect(top == above)
    #expect(Int((renderer.encode(bottom.0) * 255).rounded()) == 41)
    #expect(Int((renderer.encode(top.0) * 255).rounded()) == 140)
  }

  @Test func writesAProResAlphaMovieOnceAndReportsItsStates() async throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "motion-caption-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let request = MotionCaptionRequest(
      output: directory.appendingPathComponent("captions.mov").path, width: 320, height: 320,
      rate: 30, frames: 12, style: Self.style, titles: [Self.title])
    let report = try await MotionCaptionMovie.render(request)
    #expect((report.frames, report.states) == (12, 3))
    #expect(FileManager.default.fileExists(atPath: request.output))
    await #expect(throws: ProAppsError.self) { try await MotionCaptionMovie.render(request) }
  }

  @Test func refusesTitlesBeyondTheMovie() async {
    let request = MotionCaptionRequest(
      output: "/synthetic/never.mov", width: 320, height: 320, rate: 30, frames: 5,
      style: Self.style, titles: [Self.title])
    await #expect(throws: ProAppsError.self) { try await MotionCaptionMovie.render(request) }
  }

  @Test func composesOverlappingTitlesWithStraightAlpha() {
    var pixels: [UInt8] = [10, 20, 30, 128]
    pixels.withUnsafeMutableBufferPointer { buffer in
      guard let base = buffer.baseAddress else { return }
      MotionCaptionMovie.over([200, 100, 0, 128], at: 0, into: base, at: 0)
    }
    #expect(pixels == [3, 73, 143, 192])
  }

  @Test func readsTheRequestNamedOnTheCommandLine() throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "motion-caption-request-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let request = MotionCaptionRequest(
      output: directory.appendingPathComponent("captions.mov").path, width: 320, height: 320,
      rate: 30, frames: 12, style: Self.style, titles: [Self.title])
    let file = directory.appendingPathComponent("request.json")
    try JSONEncoder().encode(request).write(to: file)
    #expect(
      try MotionCaptionMovie.command(arguments: ["caption-alpha-render", file.path])
        == .render(request))
    #expect(throws: ProAppsError.self) {
      try MotionCaptionMovie.command(arguments: ["caption-alpha-render"])
    }
    #expect(throws: ProAppsError.self) {
      try MotionCaptionMovie.command(arguments: ["caption-alpha-render", "--inspect"])
    }
    #expect(throws: ProAppsError.self) {
      try MotionCaptionMovie.command(
        arguments: ["caption-alpha-render", directory.appendingPathComponent("missing").path])
    }
  }
}
