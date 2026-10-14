import Foundation
import Testing

@testable import ProAppsCore

struct MaskRenderTests {
  @Test(arguments: [(seconds: 60, frames: 1800), (seconds: 90, frames: 2700)])
  func masksPersistAcrossBothRequiredDurations(_ sample: (seconds: Int, frames: Int)) async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "mask-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let before = try Data(contentsOf: source)
    let clip = EditClip(
      sourcePath: source.path, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 0.5))
    let recipe = EditRecipe(
      clips: Array(repeating: clip, count: sample.seconds / 2),
      video: .init(
        width: 32, height: 32, frameRate: 30, resizeMode: .fill,
        masks: [
          .init(region: .init(x: 0, y: 0, width: 16, height: 16), opacity: 1),
          .init(region: .init(x: 16, y: 0, width: 16, height: 16), opacity: 0.5),
        ]))
    let rendered = try await NativeEditor().render(recipe, directory: root.path, name: "masked.mp4")
    #expect(rendered.actualDurationSeconds == Double(sample.seconds))
    let decoded = try await MediaProbe().verifyShortVideo(
      path: rendered.outputPath,
      maximumFrames: 3000, maximumDurationSeconds: 100)
    #expect(decoded.decodedFrames == sample.frames)
    let opaque = EditCrop(x: 4, y: 4, width: 8, height: 8)
    let partial = EditCrop(x: 20, y: 4, width: 8, height: 8)
    let control = EditCrop(x: 4, y: 20, width: 8, height: 8)
    let measured = try await FrameProbe().measure(
      path: rendered.outputPath,
      samples: [
        .init(timeSeconds: 0.5, region: opaque),
        .init(timeSeconds: Double(sample.seconds) - 0.5, region: opaque),
        .init(timeSeconds: 0.5, region: partial),
        .init(timeSeconds: Double(sample.seconds) - 0.5, region: partial),
        .init(timeSeconds: 0.5, region: control),
        .init(timeSeconds: Double(sample.seconds) - 0.5, region: control),
      ])
    try #require(measured.count == 6)
    #expect(measured[0].meanRed < 0.03 && measured[1].meanRed < 0.03)
    #expect(measured[2].meanGreen > 0.2 && measured[2].meanGreen < 0.8)
    #expect(abs(measured[2].meanGreen - measured[3].meanGreen) < 0.02)
    #expect(measured[4].meanBlue > 0.8 && measured[5].meanBlue > 0.8)
    #expect(try Data(contentsOf: source) == before)
  }
}
