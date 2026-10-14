import AVFoundation
import CoreImage
import CoreVideo
import Foundation
import Testing
import Vision

@testable import ProAppsCore

struct FaceProducerEvidenceTests {
  @Test func exactSamplePreservesClockAndMetadataAndConcurrentRequestsStayIndependent() async throws
  {
    let source = try #require(
      Bundle.module.url(forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures")
    )
    let probe = FrameProbe()
    async let first = probe.readerFaceEvidence(path: source.path, seconds: 0.5)
    async let second = probe.readerFaceEvidence(path: source.path, seconds: 0)
    let (evidence, zero) = try await (first, second)
    #expect(evidence.actualPresentationValue == 15000)
    #expect(evidence.actualPresentationTimescale == 30000)
    #expect(zero.actualPresentationValue == 0)
    #expect(evidence.pixelFormat == kCVPixelFormatType_32BGRA)
    #expect(evidence.bitmap.width == 16 && evidence.bitmap.height == 16)
    #expect(evidence.bitmap.bitsPerComponent == 8 && evidence.bitmap.bitsPerPixel == 32)
    #expect(evidence.bitmap.bytesPerRow >= 64)
    #expect(evidence.appliedTransform.a == 1 && evidence.appliedTransform.d == 1)
    #expect(evidence.appliedTransform.b == 0 && evidence.appliedTransform.c == 0)
    #expect(evidence.appliedTransform.tx == 0 && evidence.appliedTransform.ty == 0)
    #expect(evidence.contextConfiguration == "CIContext() defaults; createCGImage(_:from:)")
    #expect(evidence.legacyRevision > 0 && evidence.legacy.isEmpty)
    let decoded = try JSONDecoder().decode(
      FaceReaderEvidence.self, from: JSONEncoder().encode(evidence))
    #expect(decoded.actualPresentationValue == 15000)
    #expect(decoded.bitmap.bitmapInfo == evidence.bitmap.bitmapInfo)
    #expect(decoded.bitmap.alphaInfo == evidence.bitmap.alphaInfo)
    #expect(decoded.bitmap.colorSpaceName == evidence.bitmap.colorSpaceName)
    #expect(decoded.bitmap.colorSpaceModel == evidence.bitmap.colorSpaceModel)
    #expect(decoded.bitmap.colorSpaceComponentCount == evidence.bitmap.colorSpaceComponentCount)
  }

  @Test func explicitSourceColorVariantKeepsClockAndRecordsItsActualProfile() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures")
    )
    let evidence = try await FrameProbe().readerFaceEvidence(
      path: source.path, seconds: 0.5, preservingSourceColor: true)
    #expect(evidence.actualPresentationValue == 15000)
    #expect(evidence.actualPresentationTimescale == 30000)
    #expect(evidence.bitmap.width == 16 && evidence.bitmap.height == 16)
    #expect(evidence.bitmap.bitsPerComponent == 8 && evidence.bitmap.bitsPerPixel == 32)
    #expect(evidence.bitmap.colorSpaceModel == 1)
    #expect(evidence.bitmap.colorSpaceComponentCount == 3)
    // Valid ICC-based RGB spaces need not have a registered name. Check the
    // actual profile survives conversion, not an optional naming convention.
    let asset = AVURLAsset(url: source)
    let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
    let profileName = try TrackingFrameDecoder.withReader(
      asset: asset, track: track,
      timeRange: CMTimeRange(
        start: CMTime(value: 1, timescale: 2), duration: CMTime(value: 1, timescale: 60000))
    ) { _, output in
      let sample = try #require(output.copyNextSampleBuffer())
      #expect(CMSampleBufferGetPresentationTimeStamp(sample) == CMTime(value: 1, timescale: 2))
      let pixels = try #require(CMSampleBufferGetImageBuffer(sample))
      let sourceSpace = try #require(CIImage(cvPixelBuffer: pixels).colorSpace)
      let rendered = try TrackingFrameDecoder.bitmap(
        pixels: pixels, transform: .identity, context: CIContext(), preservingSourceColor: true)
      let renderedSpace = try #require(rendered.colorSpace)
      #expect(CFEqual(sourceSpace, renderedSpace))
      return renderedSpace.name.map { $0 as String }
    }
    #expect(evidence.bitmap.colorSpaceName == profileName)
    #expect(
      evidence.contextConfiguration == "CIContext() defaults; RGBA8; source CIImage color space")
    #expect(evidence.legacy.isEmpty)
  }

  @Test(arguments: [0.5, 1.0, 1.0 / 90001])
  func missingOutOfRangeAndUnrepresentableSamplesFail(_ seconds: Double) async throws {
    let source = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    await #expect(throws: ProAppsError.self) {
      try await FrameProbe().readerFaceEvidence(path: source.path, seconds: seconds)
    }
  }

  @Test(arguments: [AVAssetReader.Status.unknown, .failed, .cancelled])
  func rejectsReaderFailureStates(_ status: AVAssetReader.Status) {
    #expect(throws: ProAppsError.self) { try FaceReaderEvidence.validateStatus(status) }
  }

  @Test func acceptsPartialReadingWithoutClaimingCompletedDecode() throws {
    try FaceReaderEvidence.validateStatus(.reading)
    try FaceReaderEvidence.validateStatus(.completed)
    try FaceReaderEvidence.validateTime(
      requested: CMTime(value: 30000, timescale: 60000),
      actual: CMTime(value: 15000, timescale: 30000))
    #expect(throws: ProAppsError.self) {
      try FaceReaderEvidence.validateTime(requested: .zero, actual: .invalid)
    }
    #expect(throws: ProAppsError.self) {
      try FaceReaderEvidence.validateTime(
        requested: .zero,
        actual: CMTime(value: 0, timescale: 60000, flags: [.valid, .hasBeenRounded], epoch: 0))
    }
  }

  @Test func rawDetectionKeepsUnclippedGeometryAndRejectsMismatchedEvidence() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    let generator = AVAssetImageGenerator(asset: AVURLAsset(url: source))
    let image = try await generator.image(at: .zero).image
    let observation = VNFaceObservation(
      boundingBox: CGRect(x: -0.125, y: 0.25, width: 0.5, height: 0.5))
    let evidence = try FaceReaderEvidence.collect(
      image: image, requested: CMTime(value: 30000, timescale: 60000),
      actual: CMTime(value: 15000, timescale: 30000),
      pixelFormat: kCVPixelFormatType_32BGRA, transform: .identity,
      legacyDetector: { input in
        #expect(input === image)
        return LegacyFaceDetections(observations: [observation], revision: 3)
      })
    let candidate = try #require(evidence.legacy.first)
    #expect(evidence.legacyRevision == 3)
    #expect(candidate.confidence == observation.confidence)
    #expect(candidate.normalizedLowerLeft.x == -0.125)
    #expect(candidate.displayTopLeft.x == -2)
    #expect(!candidate.geometryAccepted)
    #expect(throws: ProAppsError.self) {
      try FaceDetectorComparison.collect(
        image: image, requested: .zero, actual: .zero, modern: [],
        revision: DetectFaceRectanglesRequest().revision, reader: evidence)
    }
    #expect(throws: ProAppsError.self) {
      try FaceDetectorComparison.collect(
        image: image, requested: .zero, actual: .zero, modern: [],
        revision: DetectFaceRectanglesRequest().revision, readerPreservingSourceColor: evidence)
    }
    #expect(throws: ProAppsError.self) {
      try FaceReaderEvidence.collect(
        image: image, requested: .zero, actual: CMTime(value: 1, timescale: 30),
        pixelFormat: kCVPixelFormatType_32BGRA, transform: .identity)
    }
    #expect(throws: ProAppsError.self) {
      try FaceReaderEvidence.collect(
        image: image, requested: .zero, actual: .zero, pixelFormat: 0, transform: .identity)
    }
    #expect(throws: ProAppsError.outputLimit) {
      try FaceReaderEvidence.collect(
        image: image, requested: .zero, actual: .zero,
        pixelFormat: kCVPixelFormatType_32BGRA, transform: .identity,
        legacyDetector: { _ in
          LegacyFaceDetections(observations: Array(repeating: observation, count: 33), revision: 3)
        })
    }
    #expect(throws: ProAppsError.unavailable("Synthetic detector failure")) {
      try FaceReaderEvidence.collect(
        image: image, requested: .zero, actual: .zero,
        pixelFormat: kCVPixelFormatType_32BGRA, transform: .identity,
        legacyDetector: { _ in throw ProAppsError.unavailable("Synthetic detector failure") })
    }
    #expect(throws: CancellationError.self) {
      try FaceReaderEvidence.collect(
        image: image, requested: .zero, actual: .zero,
        pixelFormat: kCVPixelFormatType_32BGRA, transform: .identity,
        legacyDetector: { _ in throw CancellationError() })
    }
  }

  @Test func cancelledRequestDoesNotReturnEvidence() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    await #expect(throws: CancellationError.self) {
      try await withThrowingTaskGroup(of: Void.self) { group in
        group.cancelAll()
        group.addTask {
          _ = try await FrameProbe().readerFaceEvidence(path: source.path, seconds: 0)
        }
        try await group.waitForAll()
      }
    }
  }

  @Test func readerIsCancelledOnThrowingBodyWithoutHidingTheFailure() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures")
    )
    let asset = AVURLAsset(url: source)
    let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
    var observedReader: AVAssetReader?
    #expect(throws: ProAppsError.unavailable("Synthetic body failure")) {
      let _: Int = try TrackingFrameDecoder.withReader(
        asset: asset, track: track,
        timeRange: CMTimeRange(start: .zero, duration: CMTime(value: 1, timescale: 1))
      ) { reader, _ in
        observedReader = reader
        throw ProAppsError.unavailable("Synthetic body failure")
      }
    }
    #expect(observedReader?.status == .cancelled)
  }
}
