import AVFoundation
import Foundation
import Testing

@testable import ProAppsCore

@Suite(.serialized)
struct MoviePCMMuxTests {
  private func directory() throws -> URL {
    let root = try Files.temporaryDirectory().appendingPathComponent("pcm-mux-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    return root
  }

  private func audio(_ root: URL, seconds: Double) throws -> URL {
    let path = root.appendingPathComponent("audio.wav")
    try CueSound(durationSeconds: seconds, onsetSeconds: [0.1], gain: 0.1).wave().write(to: path)
    return path
  }

  @Test(arguments: [1.0, 0.99])
  func copiesPCMWithoutChangingOriginalsAndPreservesClocks(_ seconds: Double) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let video = try await ContinuousVideoFixture.make(in: root)
    let wave = try audio(root, seconds: seconds)
    let oldVideo = try Data(contentsOf: video)
    let oldAudio = try Data(contentsOf: wave)
    let output = root.appendingPathComponent("outputs")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    let request = MoviePCMMuxRequest(
      rootDirectory: root.path, videoPath: video.path, audioPath: wave.path,
      outputDirectory: output.path, outputName: "muxed.mov")
    let result = try await MoviePCMMux().write(request)
    #expect(try Data(contentsOf: video) == oldVideo)
    #expect(try Data(contentsOf: wave) == oldAudio)
    let checked = try await MediaProbe().verifyShortVideo(path: result.outputPath)
    #expect(checked.decodedFrames == 30)
    #expect(checked.media.audioTrackCount == 1)
    #expect(checked.media.durationSeconds == 1)
    let file = AVURLAsset(url: URL(fileURLWithPath: result.outputPath))
    let tracks = try await file.loadTracks(withMediaType: .audio)
    let track = try #require(tracks.first)
    #expect(try await track.load(.isSelfContained))
    #expect(try await track.load(.timeRange).duration.seconds == seconds)
    let firstBytes = try Data(contentsOf: URL(fileURLWithPath: result.outputPath))
    let second = try await MoviePCMMux().write(request)
    #expect(second.outputPath != result.outputPath)
    #expect(try Data(contentsOf: URL(fileURLWithPath: result.outputPath)) == firstBytes)
  }

  @Test(arguments: [0.5, 1.1])
  func rejectsMissingCoverageAndOverhang(_ seconds: Double) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let video = try await ContinuousVideoFixture.make(in: root)
    let wave = try audio(root, seconds: seconds)
    let request = MoviePCMMuxRequest(
      rootDirectory: root.path, videoPath: video.path, audioPath: wave.path,
      outputDirectory: root.path, outputName: "muxed.mov")
    await #expect(throws: (any Error).self) { try await MoviePCMMux().write(request) }
    #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).count == 2)
  }

  @Test func rejectsExistingAudioAndInvalidRoot() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let video = try await ContinuousVideoFixture.make(in: root)
    let wave = try audio(root, seconds: 1)
    let request = MoviePCMMuxRequest(
      rootDirectory: root.path, videoPath: video.path, audioPath: wave.path,
      outputDirectory: root.path, outputName: "first.mov")
    let first = try await MoviePCMMux().write(request)
    let duplicate = MoviePCMMuxRequest(
      rootDirectory: root.path, videoPath: first.outputPath, audioPath: wave.path,
      outputDirectory: root.path, outputName: "duplicate.mov")
    await #expect(throws: (any Error).self) { try await MoviePCMMux().write(duplicate) }
    let invalidRoot = MoviePCMMuxRequest(
      rootDirectory: "/", videoPath: video.path, audioPath: wave.path,
      outputDirectory: root.path, outputName: "invalid.mov")
    await #expect(throws: (any Error).self) { try await MoviePCMMux().write(invalidRoot) }
  }

  @Test func cancellationPreventsCloning() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let request = MoviePCMMuxRequest(
      rootDirectory: root.path, videoPath: "/unused.mov", audioPath: "/unused.wav",
      outputDirectory: root.path, outputName: "cancelled.mov")
    await #expect(throws: CancellationError.self) {
      try await withThrowingTaskGroup(of: MoviePCMMuxResult.self) { group in
        group.cancelAll()
        group.addTask { try await MoviePCMMux().write(request) }
        _ = try await group.next()
      }
    }
    #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
  }

  @Test func clockLCMRejectsOverflowAndInvalidScales() throws {
    #expect(try MoviePCMMux.commonTimescale(15360, 44100) == 11_289_600)
    #expect(try MoviePCMMux.commonTimescale(600, 16000) == 48000)
    #expect(throws: (any Error).self) { try MoviePCMMux.commonTimescale(0, 16000) }
    #expect(throws: (any Error).self) {
      try MoviePCMMux.commonTimescale(1_000_000_000, 999_999_999)
    }
  }
}
