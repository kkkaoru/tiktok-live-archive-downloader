import Foundation
import Testing

@testable import ProAppsCore

// These cases use the shared macOS hardware codecs, as do the existing native suites.
@Suite(.serialized)
struct VideoLayerRenderTests {
  @Test func timedLayersHonorOpacityOrderingGeometryAndExplicitAudio() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "video-layers-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let before = try Data(contentsOf: source)
    let audio = try Files.writeNew(
      CueSound(durationSeconds: 1, onsetSeconds: [0.1], gain: 0.1).wave(),
      to: root.appendingPathComponent("cue.wav").path, extensions: ["wav"])
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: source.path, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
      ],
      video: .init(width: 64, height: 64, frameRate: 30, resizeMode: .fit),
      additionalAudio: [
        .init(
          sourcePath: audio.path, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
          offsetSeconds: 0)
      ],
      additionalVideo: [
        .init(
          sourcePath: source.path, selection: .init(startSeconds: 0, durationSeconds: 0.2, rate: 1),
          offsetSeconds: 0.1, geometry: .init(rotation: .clockwise90), opacity: 0),
        .init(
          sourcePath: source.path, selection: .init(startSeconds: 0, durationSeconds: 0.2, rate: 1),
          offsetSeconds: 0.3, geometry: .init(rotation: .clockwise90), opacity: 0.5),
        .init(
          sourcePath: source.path, selection: .init(startSeconds: 0, durationSeconds: 0.2, rate: 1),
          offsetSeconds: 0.5, geometry: .init(rotation: .clockwise90)),
        .init(
          sourcePath: source.path,
          selection: .init(startSeconds: 0, durationSeconds: 0.05, rate: 1), offsetSeconds: 0.4,
          geometry: .init(rotation: .clockwise180), opacity: 1),
      ])
    let result = try await NativeEditor().render(recipe, directory: root.path, name: "layers.mp4")
    let verified = try await MediaProbe().verifyShortVideo(path: result.outputPath)
    #expect(verified.fullVideoDecoded)
    #expect(verified.decodedFrames == 30)
    #expect(result.audioTrackCount == 1)
    let measuredAudio = try await AudioProbe().measure(
      path: result.outputPath,
      windows: [
        .init(startSeconds: 0.1, durationSeconds: 0.08),
        .init(startSeconds: 0.8, durationSeconds: 0.1),
      ])
    try #require(measuredAudio.windows.count == 2)
    #expect(measuredAudio.whole.frames == 16000)
    #expect(measuredAudio.windows[0].peak > 0.05)
    #expect(measuredAudio.windows[1].rms < 0.001)
    #expect(abs(result.actualDurationSeconds - 1) < 0.001)
    let pixels = try await FrameProbe().measure(
      path: result.outputPath,
      samples: [
        .init(timeSeconds: 0.05, region: .init(x: 8, y: 8, width: 8, height: 8)),
        .init(timeSeconds: 0.2, region: .init(x: 8, y: 8, width: 8, height: 8)),
        .init(timeSeconds: 0.35, region: .init(x: 8, y: 8, width: 8, height: 8)),
        .init(timeSeconds: 0.42, region: .init(x: 8, y: 8, width: 8, height: 8)),
        .init(timeSeconds: 0.6, region: .init(x: 8, y: 8, width: 8, height: 8)),
        .init(timeSeconds: 0.8, region: .init(x: 8, y: 8, width: 8, height: 8)),
      ])
    try #require(pixels.count == 6)
    #expect(pixels[0].meanRed > 0.8 && pixels[0].meanBlue < 0.1)
    #expect(pixels[1].meanRed > 0.8 && pixels[1].meanBlue < 0.1)
    #expect(pixels[2].meanRed > 0.2 && pixels[2].meanBlue > 0.2 && pixels[2].meanGreen < 0.1)
    #expect(pixels[3].meanRed > 0.8 && pixels[3].meanGreen > 0.8 && pixels[3].meanBlue < 0.1)
    #expect(pixels[4].meanBlue > 0.8 && pixels[4].meanRed < 0.1)
    #expect(pixels[5].meanRed > 0.8 && pixels[5].meanBlue < 0.1)
    #expect(try Data(contentsOf: source) == before)
    let saved = try JSONDecoder().decode(
      EditRequest.self, from: Data(contentsOf: URL(fileURLWithPath: result.projectPath)))
    #expect(saved.recipe.additionalVideo?.count == 4)
    #expect(
      !FileManager.default.fileExists(
        atPath: URL(fileURLWithPath: result.outputPath).deletingLastPathComponent()
          .appendingPathComponent(".rendering.mp4").path))
  }

  @Test func audioOnlyOverlaySourceIsRejectedWithoutPublication() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "video-layer-invalid-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let audio = try Files.writeNew(
      CueSound(durationSeconds: 1, onsetSeconds: [0.1], gain: 0.1).wave(),
      to: root.appendingPathComponent("cue.wav").path, extensions: ["wav"])
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: source.path, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
      ],
      video: .init(width: 64, height: 64, frameRate: 30, resizeMode: .fit),
      additionalVideo: [
        .init(
          sourcePath: audio.path, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
          offsetSeconds: 0)
      ])
    await #expect(throws: (any Error).self) {
      try await NativeEditor().render(recipe, directory: root.path, name: "invalid.mp4")
    }
    #expect(
      try FileManager.default.contentsOfDirectory(atPath: root.path).allSatisfy {
        !$0.hasPrefix("edit-")
      })
  }
}
