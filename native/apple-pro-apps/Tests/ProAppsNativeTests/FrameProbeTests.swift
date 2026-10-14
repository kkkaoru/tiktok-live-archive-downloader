import CoreImage
import Foundation
import Testing

@testable import ProAppsCore

struct FrameProbeTests {
  @Test(arguments: [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
  func managedRGBAChannelOrderMatchesKnownColors(_ red: Double, _ green: Double, _ blue: Double)
    throws
  {
    let region = CGRect(x: 0, y: 0, width: 16, height: 16)
    let image = CIImage(color: CIColor(red: red, green: green, blue: blue)).cropped(to: region)
    let mean = try FrameProbe.average(image, region: region, context: CIContext())
    #expect(abs(mean.red - red) < 0.01)
    #expect(abs(mean.green - green) < 0.01)
    #expect(abs(mean.blue - blue) < 0.01)
  }

  @Test func displayTransformsCannotBypassTheDecodeBudget() throws {
    try FrameProbe.validateGeometry(
      size: CGSize(width: 360, height: 640),
      transform: CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 640, ty: 0))
    #expect(throws: (any Error).self) {
      try FrameProbe.validateGeometry(
        size: CGSize(width: 16, height: 16),
        transform: CGAffineTransform(scaleX: 1_000_000, y: 1_000_000))
    }
    #expect(throws: (any Error).self) {
      try FrameProbe.validateGeometry(
        size: CGSize(width: 16, height: 16),
        transform: CGAffineTransform(translationX: 1_000_000, y: 0))
    }
    #expect(throws: (any Error).self) {
      try FrameProbe.validateGeometry(
        size: CGSize(width: 16, height: 16), transform: CGAffineTransform(scaleX: 0, y: 0))
    }
    #expect(throws: (any Error).self) {
      try FrameProbe.validateGeometry(
        size: CGSize(width: CGFloat.infinity, height: 16), transform: .identity)
    }
  }

  @Test func topLeftRegionCoordinatesAreExplicit() throws {
    #expect(
      try FrameProbe.region(.init(x: 10, y: 20, width: 30, height: 40), width: 320, height: 240)
        == CGRect(x: 10, y: 180, width: 30, height: 40))
    #expect(
      try FrameProbe.region(nil, width: 320, height: 240)
        == CGRect(x: 0, y: 0, width: 320, height: 240))
    #expect(throws: (any Error).self) {
      try FrameProbe.region(.init(x: 300, y: 20, width: 30, height: 40), width: 320, height: 240)
    }
    #expect(throws: (any Error).self) {
      try FrameProbe.region(.init(x: 0, y: 0, width: -1, height: 40), width: 320, height: 240)
    }
  }

  @Test func decodesAndMeasuresOnlyTheRequestedSyntheticFrame() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    let samples = [
      FrameSample(timeSeconds: 0),
      FrameSample(timeSeconds: 0, region: .init(x: 0, y: 0, width: 8, height: 8)),
    ]
    let restored = try JSONDecoder().decode([FrameSample].self, from: JSONEncoder().encode(samples))
    let result = try await FrameProbe().measure(path: source.path, samples: restored)
    try #require(result.count == 2)
    #expect(result[0].width == 16)
    #expect(result[0].height == 16)
    #expect(result[0].actualTimeSeconds == 0)
    #expect(result[0].meanRed < 0.02)
    #expect(result[0].meanGreen < 0.02)
    #expect(result[0].meanBlue < 0.02)
    let decoded = try JSONDecoder().decode(
      [FrameMeasurement].self, from: JSONEncoder().encode(result))
    #expect(decoded.count == 2)
    let complete = try await MediaProbe().verifyShortVideo(path: source.path)
    #expect(complete.decodedFrames == 1)
    #expect(complete.fullVideoDecoded)
  }

  @Test(arguments: [Double.nan, -1.0, 2.0])
  func invalidFrameTimesCannotReachTheGenerator(_ time: Double) async throws {
    let source = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    await #expect(throws: (any Error).self) {
      try await FrameProbe().measure(path: source.path, samples: [.init(timeSeconds: time)])
    }
  }

  @Test(arguments: [0, 7201])
  func invalidVideoBudgetsAreRejectedBeforeOpeningTheSource(_ budget: Int) async {
    await #expect(throws: (any Error).self) {
      try await MediaProbe().verifyShortVideo(path: "/tmp/not-a-video", maximumFrames: budget)
    }
  }

  @Test(arguments: [0.0, -1.0, 120.001, Double.infinity, Double.nan])
  func invalidDurationBudgetsFailBeforeOpeningTheSource(_ seconds: Double) async {
    do {
      _ = try await MediaProbe().verifyShortVideo(
        path: "/tmp/not-a-video", maximumDurationSeconds: seconds)
      Issue.record("Invalid duration budget was accepted")
    } catch ProAppsError.invalid(let reason) {
      #expect(reason == "Video verification budgets must be 1–7200 frames and >0–120 seconds")
    } catch { Issue.record(error) }
  }

  @Test func emptyRequestsAndNonVideoSourcesAreRefused() async throws {
    await #expect(throws: (any Error).self) {
      try await FrameProbe().measure(path: "/tmp/not-a-video", samples: [])
    }
    let source = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    await #expect(throws: (any Error).self) {
      try await AudioProbe().measure(path: source.path, windows: [])
    }
  }
}
