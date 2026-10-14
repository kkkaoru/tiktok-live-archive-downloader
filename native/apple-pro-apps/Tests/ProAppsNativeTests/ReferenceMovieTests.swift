import AVFoundation
import Foundation
import Testing

@testable import ProAppsCore

@Suite(.serialized)
struct ReferenceMovieTests {
  private func directory() throws -> URL {
    let url = try Files.temporaryDirectory().appendingPathComponent(
      "reference-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
    return url
  }

  private func segment(_ path: String, at value: Int64 = 0, layer: Int = 0) -> ReferenceSegment {
    ReferenceSegment(
      sourcePath: path, sourceStart: .init(value: 0, timescale: 30),
      duration: .init(value: 15, timescale: 30),
      outputStart: .init(value: value, timescale: 30), layer: layer)
  }

  private func request(_ root: URL, _ segments: [ReferenceSegment]) -> ReferenceMovieRequest {
    ReferenceMovieRequest(
      rootDirectory: root.path, outputPath: root.appendingPathComponent("reference.mov").path,
      timescale: 90000, segments: segments)
  }

  @Test func concatenatesReferencedVideoWithoutCopyingOrChangingSource() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let before = try Data(contentsOf: source)
    let input = request(root, [segment(source.path), segment(source.path, at: 15)])
    let result = try await ReferenceMovieWriter().write(input)
    #expect(result.videoTrackCount == 1)
    #expect(result.sourcePaths == [source.path])
    #expect(result.duration == ForegroundTime(value: 90000, timescale: 90000))
    #expect(!result.selfContained && !result.rendered)
    #expect(result.headerBytes < before.count)
    #expect(try Data(contentsOf: source) == before)
    let decoded = try await MediaProbe().verifyShortVideo(path: result.outputPath)
    #expect(decoded.decodedFrames == 30)
    #expect(decoded.media.durationSeconds == 1)
    let header = try Data(contentsOf: URL(fileURLWithPath: result.outputPath))
    await #expect(throws: (any Error).self) { try await ReferenceMovieWriter().write(input) }
    #expect(try Data(contentsOf: URL(fileURLWithPath: result.outputPath)) == header)
  }

  @Test func retainsTwoTracksAndGapWithoutClaimingComposition() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let result = try await ReferenceMovieWriter().write(
      request(root, [segment(source.path), segment(source.path, at: 6, layer: 1)]))
    #expect(result.videoTrackCount == 2)
    #expect(result.duration == ForegroundTime(value: 63000, timescale: 90000))
    let movie = AVMovie(url: URL(fileURLWithPath: result.outputPath))
    let tracks = try await movie.loadTracks(withMediaType: .video)
    #expect(tracks.count == 2)
    #expect(!result.rendered)
  }

  @Test(arguments: [-1, 16]) func rejectsInvalidLayers(_ layer: Int) throws {
    let root = try Files.temporaryDirectory()
    #expect(throws: (any Error).self) {
      try request(root, [segment("/unused.mp4"), segment("/unused.mp4", layer: layer)]).validate()
    }
  }

  @Test func rejectsEmptyOverlapMissingBaseAndInexactClock() throws {
    let root = try Files.temporaryDirectory()
    #expect(throws: (any Error).self) { try request(root, []).validate() }
    #expect(throws: (any Error).self) {
      try request(root, [segment("/unused.mp4"), segment("/unused.mp4", at: 14)]).validate()
    }
    #expect(throws: (any Error).self) {
      try request(root, [segment("/unused.mp4", layer: 1)]).validate()
    }
    let invalid = ReferenceMovieRequest(
      rootDirectory: root.path, outputPath: root.appendingPathComponent("bad.mov").path,
      timescale: 1, segments: [segment("/unused.mp4")])
    #expect(throws: (any Error).self) { try invalid.validate() }
  }

  @Test func rejectsSourceOutsideRootAndSourceRangeOverflow() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let fixture = try #require(
      Bundle.module.url(forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures")
    )
    await #expect(throws: (any Error).self) {
      try await ReferenceMovieWriter().write(request(root, [segment(fixture.path)]))
    }
    let source = try await ContinuousVideoFixture.make(in: root)
    let excessive = ReferenceSegment(
      sourcePath: source.path, sourceStart: .init(value: 1, timescale: 1),
      duration: .init(value: 1, timescale: 1), outputStart: .init(value: 0, timescale: 1), layer: 0)
    await #expect(throws: (any Error).self) {
      try await ReferenceMovieWriter().write(request(root, [excessive]))
    }
    #expect(
      !FileManager.default.fileExists(atPath: root.appendingPathComponent("reference.mov").path))
  }

  @Test(arguments: [Int32(0), -1, 1_000_000_001])
  func rejectsInvalidTimescale(_ scale: Int32) throws {
    let root = try Files.temporaryDirectory()
    let invalid = ReferenceMovieRequest(
      rootDirectory: root.path, outputPath: root.appendingPathComponent("bad.mov").path,
      timescale: scale, segments: [segment("/unused.mp4")])
    #expect(throws: (any Error).self) { try invalid.validate() }
  }

  @Test(arguments: [Int64(0), -1])
  func rejectsNonpositiveDuration(_ value: Int64) throws {
    let root = try Files.temporaryDirectory()
    let invalid = ReferenceSegment(
      sourcePath: "/unused.mp4", sourceStart: .init(value: 0, timescale: 30),
      duration: .init(value: value, timescale: 30),
      outputStart: .init(value: 0, timescale: 30), layer: 0)
    #expect(throws: (any Error).self) { try request(root, [invalid]).validate() }
  }

  @Test func rejectsBounds() throws {
    let root = try Files.temporaryDirectory()
    #expect(throws: (any Error).self) {
      try request(root, [segment("/unused.mp4", at: 6 * 3600 * 30)]).validate()
    }
    #expect(throws: (any Error).self) {
      try request(root, Array(repeating: segment("/unused.mp4"), count: 1025)).validate()
    }
  }

  @Test func rejectsRootAndIncompatibleSources() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let invalidRoot = ReferenceMovieRequest(
      rootDirectory: "/", outputPath: root.appendingPathComponent("bad.mov").path,
      timescale: 90000, segments: [segment(source.path)])
    await #expect(throws: (any Error).self) { try await ReferenceMovieWriter().write(invalidRoot) }
    let fixture = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    let black = root.appendingPathComponent("black.mp4")
    try FileManager.default.copyItem(at: fixture, to: black)
    await #expect(throws: (any Error).self) {
      try await ReferenceMovieWriter().write(
        request(root, [segment(source.path), segment(black.path, at: 15)]))
    }
  }

  enum InvalidOutput: CaseIterable, Sendable { case outside, wrongExtension, sourceAlias }

  @Test(arguments: InvalidOutput.allCases)
  func rejectsInvalidOutput(_ kind: InvalidOutput) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let fixture = try await ContinuousVideoFixture.make(in: root)
    let source = root.appendingPathComponent("source.mov")
    try FileManager.default.copyItem(at: fixture, to: source)
    let output: URL
    switch kind {
    case .outside: output = root.deletingLastPathComponent().appendingPathComponent("outside.mov")
    case .wrongExtension: output = root.appendingPathComponent("invalid.mp4")
    case .sourceAlias: output = source
    }
    let invalid = ReferenceMovieRequest(
      rootDirectory: root.path, outputPath: output.path,
      timescale: 90000, segments: [segment(source.path)])
    await #expect(throws: (any Error).self) { try await ReferenceMovieWriter().write(invalid) }
  }

  @Test func cancellationPreventsAnyOutput() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let input = request(root, [segment("/unused.mp4")])
    await #expect(throws: CancellationError.self) {
      try await withThrowingTaskGroup(of: ReferenceMovieResult.self) { group in
        group.cancelAll()
        group.addTask { try await ReferenceMovieWriter().write(input) }
        _ = try await group.next()
      }
    }
    #expect(!FileManager.default.fileExists(atPath: input.outputPath))
  }

  @Test func refusesNestedReferences() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let linked = try await ReferenceMovieWriter().write(request(root, [segment(source.path)]))
    let nested = ReferenceMovieRequest(
      rootDirectory: root.path, outputPath: root.appendingPathComponent("nested.mov").path,
      timescale: 90000, segments: [segment(linked.outputPath)])
    await #expect(throws: (any Error).self) { try await ReferenceMovieWriter().write(nested) }
  }
}
