import AVFoundation
import Foundation
import Testing

@testable import ProAppsCore

extension NativeEditor {
  fileprivate func exerciseForeground(source: String, reads: Int, cancelImmediately: Bool)
    async throws -> Int
  {
    let video = EditVideoSettings(
      width: 16, height: 16, frameRate: 30, resizeMode: .fit, foregroundVideoPath: source)
    let foreground = try #require(try await prepareForeground(video, frames: 30))
    defer { foreground.cancel() }
    if cancelImmediately {
      foreground.cancel()
    } else {
      for slot in 0..<reads { _ = try foreground.image(at: slot) }
      try foreground.finish()
    }
    return foreground.reader.status.rawValue
  }
}

@Suite(.serialized)
struct ForegroundVideoTests {
  private func root() throws -> URL {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "foreground-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    return root
  }

  @Test(arguments: [false, true])
  func decoderCompletesOrCancelsDeterministically(cancel: Bool) async throws {
    let directory = try root()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: directory)
    let status = try await NativeEditor().exerciseForeground(
      source: source.path, reads: 30, cancelImmediately: cancel)
    #expect(
      status
        == (cancel
          ? AVAssetReader.Status.cancelled.rawValue : AVAssetReader.Status.completed.rawValue))
  }

  @Test(arguments: [
    (1, "Foreground video has extra frames or did not decode to end-of-stream"),
    (31, "Foreground video ended before the output clock"),
  ])
  func refusesExtraAndMissingDecodedFrames(_ reads: Int, _ message: String) async throws {
    let directory = try root()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: directory)
    await #expect(throws: ProAppsError.unavailable(message)) {
      try await NativeEditor().exerciseForeground(
        source: source.path, reads: reads, cancelImmediately: false)
    }
  }

  @Test(arguments: [(32, 30, 1.0), (16, 24, 1.0), (16, 30, 0.5)])
  func refusesMetadataMismatchBeforePublication(_ width: Int, _ rate: Int, _ duration: Double)
    async throws
  {
    let directory = try root()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: directory)
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: source.path,
          selection: .init(startSeconds: 0, durationSeconds: duration, rate: 1))
      ],
      video: .init(
        width: width, height: 16, frameRate: rate, resizeMode: .fit,
        encoding: .init(codec: .hevc, averageBitRate: 2_000_000, allowFrameReordering: false),
        foregroundVideoPath: source.path))
    await #expect(
      throws: ProAppsError.invalid(
        "Foreground video must match the output dimensions, rate and duration without transforms")
    ) {
      try await NativeEditor().render(recipe, directory: directory.path, name: "refused.mp4")
    }
    let files = try FileManager.default.subpathsOfDirectory(atPath: directory.path)
    #expect(
      !files.contains(where: { $0.hasSuffix("refused.mp4") || $0.hasSuffix(".rendering.mp4") }))
  }
}
