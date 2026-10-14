import CoreMedia
import Foundation
import Testing

@testable import ProAppsCore

/// Actual system Vision on public synthetic media, separate from injected unit tests.
/// Empty results are valid; these tests never require recognition of a person.
struct ForegroundVisionTests {
  @Test(
    .timeLimit(.minutes(1)),
    arguments: [
      ("black", 16, [ForegroundTime(value: 0, timescale: 90000)], [CMTime.zero]),
      (
        "tracking-object-30", 256,
        [ForegroundTime(value: 0, timescale: 90000), .init(value: 3000, timescale: 90000)],
        [CMTime.zero, CMTime(value: 1, timescale: 30)]
      ),
    ])
  func nativeRequestPreservesClocksAndSource(
    fixture: String, dimension: Int, samples: [ForegroundTime], expectedTimes: [CMTime]
  ) async throws {
    let source = try #require(
      Bundle.module.url(forResource: fixture, withExtension: "mp4", subdirectory: "Fixtures"))
    let original = try Data(contentsOf: source)
    let frames = try await FrameProbe().measureForeground(path: source.path, samples: samples)
    #expect(frames.map(\.requested) == samples)
    let actual = try frames.map { try $0.actual.time() }
    #expect(actual == expectedTimes)
    #expect(frames.allSatisfy { $0.width == dimension && $0.height == dimension })
    let consistent = frames.allSatisfy { frame in
      guard let observation = frame.observation else { return true }
      if observation.labels.isEmpty { return observation.mask == nil }
      guard let mask = observation.mask else { return false }
      guard mask.width == dimension, mask.height == dimension,
        mask.nonzeroPixels >= 0, mask.nonzeroPixels <= dimension * dimension
      else { return false }
      guard let bounds = mask.nonzeroBounds else { return mask.nonzeroPixels == 0 }
      return mask.nonzeroPixels > 0 && bounds.x >= 0 && bounds.y >= 0
        && bounds.width > 0 && bounds.height > 0
        && bounds.x + bounds.width <= dimension && bounds.y + bounds.height <= dimension
        && mask.nonzeroPixels <= bounds.width * bounds.height
    }
    #expect(consistent)
    #expect(try Data(contentsOf: source) == original)
  }
}
