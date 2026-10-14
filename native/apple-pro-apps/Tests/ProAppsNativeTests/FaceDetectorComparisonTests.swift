import AVFoundation
import Foundation
import Testing
import Vision

@testable import ProAppsCore

struct FaceDetectorComparisonTests {
  @Test(arguments: [(Float(0), false), (Float(0.3), false), (Float(0.31), true), (Float(1), true)])
  func keepsRawConfidenceAndUnclippedGeometry(_ confidence: Float, accepted: Bool) throws {
    let measured = try FaceDetectorCandidate(
      box: CGRect(x: -0.1, y: 0.25, width: 0.5, height: 0.5), confidence: confidence,
      size: CGSize(width: 100, height: 200))
    #expect(measured.confidence == confidence)
    #expect(measured.confidenceAccepted == accepted)
    #expect(!measured.geometryAccepted)
    #expect(measured.normalizedLowerLeft.x == -0.1)
    #expect(measured.displayTopLeft.x == -10 && measured.displayTopLeft.y == 50)
    #expect(measured.displayTopLeft.width == 50 && measured.displayTopLeft.height == 100)
  }

  @Test(arguments: [Float.nan, .infinity, -0.1, 1.1])
  func malformedConfidenceFails(_ confidence: Float) {
    #expect(throws: ProAppsError.self) {
      try FaceDetectorCandidate(
        box: CGRect(x: 0, y: 0, width: 1, height: 1), confidence: confidence,
        size: CGSize(width: 100, height: 200))
    }
  }

  @Test func rejectsMalformedGeometryAndPreservesContainedBounds() throws {
    #expect(throws: ProAppsError.self) {
      try FaceDetectorCandidate(
        box: CGRect(x: 0, y: 0, width: 0, height: 1), confidence: 0.5,
        size: CGSize(width: 100, height: 200))
    }
    let valid = try FaceDetectorCandidate(
      box: CGRect(x: 0, y: 0, width: 1, height: 1), confidence: 0.5,
      size: CGSize(width: 100, height: 200))
    #expect(valid.geometryAccepted && valid.confidenceAccepted)
    #expect(valid.displayTopLeft.y == 0)
    #expect(throws: ProAppsError.self) {
      try FaceDetectorCandidate(
        box: CGRect(x: CGFloat.greatestFiniteMagnitude, y: 0, width: 1, height: 1),
        confidence: 0.5, size: CGSize(width: 100, height: 200))
    }
  }

  @Test(arguments: [Double]([.nan, -.infinity, -1, 86401, 1.0 / 90001]))
  func rejectsUnrepresentableSampleClocks(_ seconds: Double) {
    #expect(throws: ProAppsError.self) {
      try FaceDetectorComparison.exactTime(seconds: seconds)
    }
  }

  @Test func oldReportsDecodeWithoutProducerEvidence() throws {
    let revision = String(
      decoding: try JSONEncoder().encode(DetectFaceRectanglesRequest.Revision.revision3),
      as: UTF8.self)
    let json = """
      {"requestedPresentationValue":0,"requestedPresentationTimescale":60000,
      "imageProducer":"synthetic legacy report","legacyRevision":3,
      "modernRevision":\(revision),"legacy":[],"modern":[],"operatingSystem":"synthetic"}
      """
    let report = try JSONDecoder().decode(FaceDetectorComparison.self, from: Data(json.utf8))
    #expect(report.generatorBitmap == nil && report.reader == nil)
    #expect(report.readerPreservingSourceColor == nil)
    #expect(report.modernRevision == .revision3 && report.requestedPresentationValue == 0)
  }

  @Test func exactClockDoesNotRoundToAnotherSourceFrame() throws {
    let time = try FaceDetectorComparison.exactTime(seconds: 0.125)
    #expect(time == CMTime(value: 1, timescale: 8))
    #expect(time.value == 7500 && time.timescale == 60000)
  }

  @Test func sameBitmapBoundaryRetainsBothDetectorsAndRequiresExactPTS() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    let generator = AVAssetImageGenerator(asset: AVURLAsset(url: source))
    let image = try await generator.image(at: .zero).image
    let box = CGRect(x: 0.125, y: 0.25, width: 0.5, height: 0.5)
    let modern = FaceObservation(boundingBox: NormalizedRect(normalizedRect: box))
    let legacy = VNFaceObservation(boundingBox: box)
    let result = try FaceDetectorComparison.collect(
      image: image, requested: .zero, actual: .zero, modern: [modern], revision: .revision3,
      legacyDetector: { input in
        #expect(input === image)
        return LegacyFaceDetections(observations: [legacy], revision: 3)
      })
    #expect(result.legacyRevision == 3 && result.modernRevision == .revision3)
    #expect(result.legacy.count == 1 && result.modern.count == 1)
    #expect(result.legacy.first?.confidence == 1 && result.modern.first?.confidence == 1)
    let data = try JSONEncoder().encode(result)
    #expect(try JSONDecoder().decode(FaceDetectorComparison.self, from: data).legacy.count == 1)
    #expect(throws: ProAppsError.self) {
      try FaceDetectorComparison.collect(
        image: image, requested: .zero, actual: CMTime(value: 1, timescale: 30),
        modern: [], revision: .revision3)
    }
    #expect(throws: ProAppsError.self) {
      try FaceDetectorComparison.collect(
        image: image, requested: .zero, actual: .zero,
        modern: Array(repeating: modern, count: 33), revision: .revision3,
        legacyDetector: { _ in LegacyFaceDetections(observations: [], revision: 3) })
    }
    #expect(throws: ProAppsError.self) {
      try FaceDetectorComparison.collect(
        image: image, requested: .zero, actual: .zero, modern: [], revision: .revision3,
        legacyDetector: { _ in throw ProAppsError.unavailable("Synthetic detector failure") })
    }
    #expect(throws: ProAppsError.self) {
      try FaceDetectorComparison.collect(
        image: image, requested: .zero, actual: .zero, modern: [], revision: .revision3,
        legacyDetector: { _ in
          LegacyFaceDetections(observations: Array(repeating: legacy, count: 33), revision: 3)
        })
    }
    #expect(throws: CancellationError.self) {
      try FaceDetectorComparison.collect(
        image: image, requested: .zero, actual: .zero, modern: [], revision: .revision3,
        legacyDetector: { _ in throw CancellationError() })
    }
  }
}
