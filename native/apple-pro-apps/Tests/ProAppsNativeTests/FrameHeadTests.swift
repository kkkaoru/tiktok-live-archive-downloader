import Foundation
import Testing
import Vision

@testable import ProAppsCore

struct FrameHeadTests {
  private struct Observation: BoundingBoxProviding {
    let boundingBox: NormalizedRect
  }

  @Test func mapsObservedRectanglesWithoutChoosingOrInventingASubject() throws {
    let measured = try FrameProbe.headRegions(
      [
        Observation(
          boundingBox: NormalizedRect(
            normalizedRect: CGRect(x: 0.125, y: 0.25, width: 0.25, height: 0.5))),
        Observation(
          boundingBox: NormalizedRect(
            normalizedRect: CGRect(x: 0.5, y: 0.125, width: 0.25, height: 0.125))),
      ],
      size: CGSize(width: 100, height: 100))
    let regions = measured.regions
    #expect(!measured.boundsClipped)
    try #require(regions.count == 2)
    #expect(regions[0].x == 12.5 && regions[0].y == 25)
    #expect(regions[0].width == 25 && regions[0].height == 50)
    #expect(regions[1].x == 50 && regions[1].y == 75)
    #expect(regions[1].width == 25 && regions[1].height == 12.5)
    #expect(
      try FrameProbe.headRegions([Observation](), size: CGSize(width: 100, height: 100)).regions
        .isEmpty)
    #expect(throws: ProAppsError.self) {
      try FrameProbe.headRegions(
        Array(
          repeating: Observation(
            boundingBox: NormalizedRect(normalizedRect: CGRect(x: 0, y: 0, width: 0.5, height: 0.5))
          ), count: 33),
        size: CGSize(width: 100, height: 100))
    }
    #expect(throws: ProAppsError.self) {
      try FrameProbe.headRegions(
        [
          Observation(
            boundingBox: NormalizedRect(normalizedRect: CGRect(x: 2, y: 0, width: 0.5, height: 0.5))
          )
        ],
        size: CGSize(width: 100, height: 100))
    }
  }

  @Test func containedFractionalRectanglesAreNotReportedAsBoundaryClipping() throws {
    let measured = try FrameProbe.headRegions(
      [
        Observation(
          boundingBox: NormalizedRect(
            normalizedRect: CGRect(x: 0.223, y: 0.123, width: 0.313, height: 0.321)))
      ],
      size: CGSize(width: 1080, height: 1920))
    #expect(!measured.boundsClipped)
    #expect(measured.regions.count == 1)
  }

  @Test func clippedGeometryIsExplicitEvenWhenTheNextObservationIsUnclipped() throws {
    let measured = try FrameProbe.headRegions(
      [
        Observation(
          boundingBox: NormalizedRect(
            normalizedRect: CGRect(x: -0.125, y: 0.25, width: 0.25, height: 0.5))),
        Observation(
          boundingBox: NormalizedRect(
            normalizedRect: CGRect(x: 0.5, y: 0.25, width: 0.25, height: 0.5))),
      ], size: CGSize(width: 100, height: 100))
    #expect(measured.boundsClipped)
    let first = try #require(measured.regions.first)
    #expect(first.x == 0 && first.width == 12.5)
    #expect(first.y == 25 && first.height == 50)
  }

  @Test func missingHeadsRetainRealFrameTimesAndRejectCropsAndInvalidSamples() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    let before = try Data(contentsOf: source)
    let probe = FrameProbe()
    let results = try await probe.detectHeads(path: source.path, samples: [.init(timeSeconds: 0)])
    let first = try #require(results.first)
    #expect(first.requestedTimeSeconds == 0 && first.actualTimeSeconds == 0)
    #expect(first.actualPresentationValue == 0 && first.actualPresentationTimescale > 0)
    #expect(!first.boundsClipped)
    #expect(first.width == 16 && first.height == 16)
    let encoded = try JSONEncoder().encode(results)
    #expect(try JSONDecoder().decode([FrameHeadMeasurement].self, from: encoded).count == 1)
    #expect(first.faces.isEmpty && first.bodies.isEmpty)
    #expect(first.faceDetectorComparison == nil)
    let compared = try await probe.detectHeads(
      path: source.path, samples: [.init(timeSeconds: 0)], compareFaceDetectors: true)
    let comparison = try #require(compared.first?.faceDetectorComparison)
    #expect(comparison.legacy.isEmpty && comparison.modern.isEmpty)
    #expect(comparison.requestedPresentationValue == 0)
    #expect(comparison.legacyRevision > 0)
    #expect(!comparison.operatingSystem.isEmpty)
    let generatorBitmap = try #require(comparison.generatorBitmap)
    let reader = try #require(comparison.reader)
    #expect(generatorBitmap.width == 16 && generatorBitmap.height == 16)
    #expect(reader.actualPresentationValue == 0)
    #expect(reader.actualPresentationTimescale > 0)
    #expect(reader.bitmap.width == 16 && reader.bitmap.height == 16)
    #expect(reader.legacyRevision > 0 && reader.legacy.isEmpty)
    let sourceColor = try #require(comparison.readerPreservingSourceColor)
    #expect(sourceColor.actualPresentationValue == 0)
    #expect(sourceColor.actualPresentationTimescale == reader.actualPresentationTimescale)
    #expect(sourceColor.bitmap.width == 16 && sourceColor.bitmap.height == 16)
    #expect(
      sourceColor.contextConfiguration == "CIContext() defaults; RGBA8; source CIImage color space")
    let roundTrip = try JSONDecoder().decode(
      [FrameHeadMeasurement].self, from: JSONEncoder().encode(compared))
    #expect(roundTrip.first?.faceDetectorComparison?.reader?.actualPresentationValue == 0)
    #expect(
      roundTrip.first?.faceDetectorComparison?.readerPreservingSourceColor?.actualPresentationValue
        == 0)
    #expect(try Data(contentsOf: source) == before)
    await #expect(throws: ProAppsError.self) {
      try await probe.detectHeads(path: source.path, samples: [])
    }
    await #expect(throws: ProAppsError.self) {
      try await probe.detectHeads(path: source.path, samples: [.init(timeSeconds: 100)])
    }
    await #expect(throws: ProAppsError.self) {
      try await probe.detectHeads(
        path: source.path,
        samples: [.init(timeSeconds: 0, region: .init(x: 0, y: 0, width: 1, height: 1))])
    }
    await #expect(throws: CancellationError.self) {
      try await withThrowingTaskGroup(of: Void.self) { group in
        group.cancelAll()
        group.addTask {
          _ = try await probe.detectHeads(path: source.path, samples: [.init(timeSeconds: 0)])
        }
        try await group.waitForAll()
      }
    }
  }
}
