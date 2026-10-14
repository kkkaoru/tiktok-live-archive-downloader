import CoreGraphics
import CoreImage
import Foundation
import Testing

@testable import ProAppsCore

struct TimedBlurTests {
  @Test func missingNativeFilterOutputIsAnExplicitFailure() {
    #expect(throws: ProAppsError.self) { try MaskRenderer.requireOutput(nil) }
  }

  @Test func nativeExportUsesHalfOpenMaskBeforeCaptions() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "timed-blur-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let before = try Data(contentsOf: source)
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: source.path,
          selection: .init(startSeconds: 0, durationSeconds: 1, rate: 0.5))
      ],
      video: .init(
        width: 320, height: 240, frameRate: 30, resizeMode: .fill,
        captions: [.init(text: "TEST", startSeconds: 0.5, endSeconds: 1)],
        masks: [
          .init(
            region: .init(x: 0, y: 0, width: 320, height: 240), opacity: 1,
            startSeconds: 0.5, endSeconds: 1),
          .init(
            region: .init(x: 0, y: 0, width: 320, height: 120), opacity: 0.9,
            blurRadius: 12, startSeconds: 1.5, endSeconds: 2),
        ]))
    let result = try await NativeEditor().render(recipe, directory: root.path, name: "timed.mp4")
    let decoded = try await MediaProbe().verifyShortVideo(path: result.outputPath)
    #expect(decoded.decodedFrames == 60)
    let area = EditCrop(x: 30, y: 30, width: 20, height: 20)
    let frames = try await FrameProbe().measure(
      path: result.outputPath,
      samples: [
        .init(timeSeconds: 0.4, region: area), .init(timeSeconds: 0.5, region: area),
        .init(timeSeconds: 1, region: area),
        .init(timeSeconds: 0.7, region: .init(x: 80, y: 200, width: 160, height: 30)),
      ])
    try #require(frames.count == 4)
    #expect(frames[0].meanRed > 0.8)
    #expect(frames[1].meanRed < 0.03)
    #expect(frames[2].meanRed > 0.8)
    #expect(frames[3].meanRed > 0.03)
    #expect(try Data(contentsOf: source) == before)
  }

  @Test func blurIsTimedCroppedAndBlendedWithoutChangingOutside() throws {
    let bounds = CGRect(x: 0, y: 0, width: 64, height: 64)
    let black = CIImage(color: .black).cropped(to: bounds)
    let white = CIImage(color: .white).cropped(to: CGRect(x: 32, y: 0, width: 32, height: 64))
    let source = white.composited(over: black)
    let mask = EditMask(
      region: .init(x: 16, y: 16, width: 32, height: 32), opacity: 1,
      blurRadius: 8, startSeconds: 1, endSeconds: 2)
    let active = try MaskRenderer.apply([mask], to: source, at: 1, canvasHeight: 64)
    let inactive = try MaskRenderer.apply([mask], to: source, at: 2, canvasHeight: 64)
    let before = try MaskRenderer.apply([mask], to: source, at: 0.999, canvasHeight: 64)
    let context = CIContext()
    let darkEdge = CGRect(x: 29, y: 30, width: 2, height: 4)
    let activeColor = try FrameProbe.average(active, region: darkEdge, context: context)
    let inactiveColor = try FrameProbe.average(inactive, region: darkEdge, context: context)
    let beforeColor = try FrameProbe.average(before, region: darkEdge, context: context)
    let outsideColor = try FrameProbe.average(
      active, region: CGRect(x: 29, y: 4, width: 2, height: 4), context: context)
    #expect(activeColor.red > 0.2 && activeColor.red < 0.8)
    #expect(inactiveColor.red == 0)
    #expect(beforeColor.red == 0)
    #expect(outsideColor.red == 0)
    #expect(active.extent == CGRect(x: 0, y: 0, width: 64, height: 64))

    let partial = EditMask(
      region: .init(x: 16, y: 16, width: 32, height: 32), opacity: 0.5, blurRadius: 8)
    let partialImage = try MaskRenderer.apply([partial], to: source, at: 1, canvasHeight: 64)
    let partialColor = try FrameProbe.average(partialImage, region: darkEdge, context: context)
    #expect(partialColor.red > 0 && partialColor.red < activeColor.red)
  }

  @Test func blackCompatibilityZeroOpacityAndTopLeftCoordinates() throws {
    let source = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))
    let mask = EditMask(region: .init(x: 0, y: 0, width: 32, height: 16), opacity: 1)
    let output = try MaskRenderer.apply([mask], to: source, at: 0, canvasHeight: 64)
    let context = CIContext()
    let upper = try FrameProbe.average(
      output, region: CGRect(x: 4, y: 52, width: 4, height: 4), context: context)
    let lower = try FrameProbe.average(
      output, region: CGRect(x: 4, y: 4, width: 4, height: 4), context: context)
    #expect(upper.red == 0)
    #expect(lower.red == 1)
    let transparent = EditMask(region: mask.region, opacity: 0, blurRadius: 8)
    let unchanged = try MaskRenderer.apply([transparent], to: source, at: 0, canvasHeight: 64)
    let color = try FrameProbe.average(
      unchanged, region: CGRect(x: 4, y: 52, width: 4, height: 4), context: context)
    #expect(color.red == 1)
  }
}
