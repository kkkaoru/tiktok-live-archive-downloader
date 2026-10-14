import Foundation
import Testing

@testable import ProAppsCore

/// Copy an independently decoded, exact-clock synthetic fixture, not an export
/// whose frame cadence can itself change under codec/sanitizer load. The editor
/// under test must still preserve every frame of its actual native export.
enum ContinuousVideoFixture {
  static func make(in directory: URL) async throws -> URL {
    let seed = try #require(
      Bundle.module.url(
        forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures"))
    let destination = directory.appendingPathComponent("continuous-quadrants.mp4")
    try FileManager.default.copyItem(at: seed, to: destination)
    let decoded = try await MediaProbe().verifyShortVideo(path: destination.path)
    try #require(decoded.decodedFrames == 30)
    try #require(decoded.media.durationSeconds == 1)
    return destination
  }
}
