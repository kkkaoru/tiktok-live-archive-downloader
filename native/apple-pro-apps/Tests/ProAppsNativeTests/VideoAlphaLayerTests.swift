import AVFoundation
import CoreImage
import Foundation
import Testing

@testable import ProAppsCore

struct VideoAlphaLayerTests {
  @Test(arguments: [false, true])
  func proResAlphaRetainsBackgroundAndBlendsTranslucentPixels(foreground: Bool) async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "video-alpha-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let before = try Data(contentsOf: source)
    let asset = AVURLAsset(url: source)
    let filters = try await AVVideoComposition.videoComposition(
      with: asset,
      applyingCIFiltersWithHandler: { request in
        let bounds = CGRect(x: 0, y: 0, width: 64, height: 64)
        let clear = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0))
          .cropped(to: bounds)
        let patch = CIImage(color: CIColor(red: 0, green: 0, blue: 1, alpha: 0.5))
          .cropped(
            to: CGRect(
              x: bounds.midX, y: bounds.midY, width: bounds.width / 2, height: bounds.height / 2))
        request.finish(with: patch.composited(over: clear), context: nil)
      })
    let export = try #require(
      AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetAppleProRes4444LPCM))
    let canvas = try #require(filters.mutableCopy() as? AVMutableVideoComposition)
    canvas.renderSize = CGSize(width: 64, height: 64)
    export.videoComposition = canvas
    let alpha = root.appendingPathComponent("synthetic-alpha.mov")
    try await export.export(to: alpha, as: .mov)
    let alphaBefore = try Data(contentsOf: alpha)
    let verifiedAlpha = try await MediaProbe().verifyShortVideo(path: alpha.path)
    try #require(verifiedAlpha.decodedFrames == 30)
    try #require(verifiedAlpha.media.width == 64 && verifiedAlpha.media.height == 64)
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: source.path,
          selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
      ],
      video: .init(
        width: 64, height: 64, frameRate: 30, resizeMode: .fit,
        masks: foreground
          ? [.init(region: .init(x: 0, y: 0, width: 64, height: 32), opacity: 1)] : nil,
        encoding: foreground
          ? .init(codec: .hevc, averageBitRate: 2_000_000, allowFrameReordering: false) : nil,
        foregroundVideoPath: foreground ? alpha.path : nil),
      additionalVideo: foreground
        ? nil
        : [
          .init(
            sourcePath: alpha.path,
            selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1), offsetSeconds: 0)
        ])
    let rendered = try await NativeEditor().render(recipe, directory: root.path, name: "alpha.mp4")
    let verified = try await MediaProbe().verifyShortVideo(path: rendered.outputPath)
    #expect(verified.decodedFrames == 30)
    #expect(verified.fullVideoDecoded)
    #expect(rendered.audioTrackCount == 0)
    let pixels = try await FrameProbe().measure(
      path: rendered.outputPath,
      samples: [
        .init(timeSeconds: 0.5, region: .init(x: 8, y: 8, width: 8, height: 8)),
        .init(timeSeconds: 0.5, region: .init(x: 48, y: 8, width: 8, height: 8)),
        .init(timeSeconds: 0.5, region: .init(x: 8, y: 48, width: 8, height: 8)),
      ])
    try #require(pixels.count == 3)
    if foreground {
      // The black source mask stays below the translucent blue foreground.
      #expect(pixels[0].meanRed < 0.05 && pixels[0].meanBlue < 0.05)
      #expect(pixels[1].meanGreen < 0.05 && pixels[1].meanBlue > 0.2 && pixels[1].meanRed < 0.05)
      #expect(rendered.frameCount == 30)
      #expect(rendered.fcpxml?.written == false)
      #expect(rendered.fcpxml?.reason == "Foreground video is not exported to FCPXML")
    } else {
      #expect(pixels[0].meanRed > 0.8 && pixels[0].meanBlue < 0.1)
      #expect(pixels[1].meanGreen > 0.2 && pixels[1].meanBlue > 0.2 && pixels[1].meanRed < 0.1)
    }
    #expect(pixels[2].meanBlue > 0.8 && pixels[2].meanRed < 0.1)
    #expect(try Data(contentsOf: source) == before)
    #expect(try Data(contentsOf: alpha) == alphaBefore)
  }
}
