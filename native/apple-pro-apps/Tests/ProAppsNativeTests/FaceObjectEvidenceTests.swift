import CoreGraphics
import Foundation
import Testing
import Vision

@testable import ProAppsCore

struct FaceObjectEvidenceTests {
  private let presentation = FaceObjectPresentation(value: 90_003, timescale: 90_000)

  private func evidence(
    box: CGRect, confidence: Float = 1, seed: FaceObjectSeed? = nil
  ) throws -> FaceObjectEvidence {
    try FaceObjectEvidence(
      raw: RawFaceObject(box: box, confidence: confidence, observationUUID: UUID()),
      presentation: presentation, stage: .trackingAdvance, seed: seed, width: 100, height: 200)
  }

  @Test func retainsUnclippedLeftGeometryAndSeparateVisibleSupport() throws {
    let result = try evidence(box: CGRect(x: -0.1, y: 0.2, width: 0.3, height: 0.3))
    #expect(result.normalizedLowerLeft.x == -0.1)
    #expect(result.displayTopLeft.x == -10)
    #expect(result.displayTopLeft.y == 100)
    #expect(result.displayTopLeft.width == 30)
    let visible = try #require(result.visibleIntersection)
    #expect(visible.x == 0 && visible.y == 100)
    #expect(abs(visible.width - 20) < 0.00001 && visible.height == 60)
    #expect(result.clipSides == [.left])
    #expect(result.disposition == .outOfFrameRejected)
    #expect(result.seed == nil)
  }

  @Test(arguments: [
    (CGRect(x: 0.8, y: 0.2, width: 0.3, height: 0.3), FaceObjectClipSide.right),
    (CGRect(x: 0.2, y: 0.8, width: 0.3, height: 0.3), FaceObjectClipSide.top),
    (CGRect(x: 0.2, y: -0.1, width: 0.3, height: 0.3), FaceObjectClipSide.bottom),
  ])
  func recordsEachDisplayClipSide(box: CGRect, side: FaceObjectClipSide) throws {
    let result = try evidence(box: box)
    #expect(result.clipSides == [side])
    #expect(result.visibleIntersection != nil)
    #expect(result.disposition == .outOfFrameRejected)
  }

  @Test func fullExitHasNoInventedVisibleIntersection() throws {
    let result = try evidence(box: CGRect(x: -1, y: 0.2, width: 0.2, height: 0.3))
    #expect(result.visibleIntersection == nil)
    #expect(result.displayTopLeft.x == -100)
    #expect(result.disposition == .outOfFrameRejected)
  }

  @Test func retainsSeedAssociationAndActualObservationUUID() throws {
    let observation = VNDetectedObjectObservation(
      boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.3))
    let seed = FaceObjectSeed(
      observationUUID: UUID(), presentation: .init(value: 1, timescale: 1))
    let result = try FaceObjectEvidence(
      observation: observation, presentation: presentation, stage: .trackingSeed,
      seed: seed, width: 100, height: 200)
    #expect(result.observationUUID == observation.uuid)
    #expect(result.seed?.observationUUID == seed.observationUUID)
    #expect(result.seed?.presentation.value == 1)
    #expect(result.presentation.value == 90_003 && result.presentation.timescale == 90_000)
    #expect(result.disposition == .accepted)
    #expect(result.clipSides.isEmpty)
  }

  @Test(arguments: [Float(0), 0.3, 0.31, 1])
  func confidenceIsRecordedWithoutChangingPolicy(_ confidence: Float) throws {
    let result = try evidence(
      box: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.3), confidence: confidence)
    #expect(result.confidence == confidence)
    #expect(result.disposition == (confidence > 0.3 ? .accepted : .confidenceRejected))
  }

  @Test(arguments: [Float.nan, .infinity, -1, 1.1])
  func malformedConfidenceIsNotSynthesized(_ confidence: Float) {
    #expect(throws: ProAppsError.self) {
      try evidence(box: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.3), confidence: confidence)
    }
  }

  @Test(arguments: [
    CGRect(x: 0, y: 0, width: 0, height: 1),
    CGRect(x: CGFloat.nan, y: 0, width: 1, height: 1),
    CGRect(x: 0, y: 0, width: CGFloat.infinity, height: 1),
    CGRect(x: 0, y: CGFloat.greatestFiniteMagnitude, width: 1, height: 1),
  ])
  func malformedOrOverflowingGeometryFails(_ box: CGRect) {
    #expect(throws: ProAppsError.self) { try evidence(box: box) }
  }

  @Test(arguments: [
    FaceObjectPresentation(value: -1, timescale: 1),
    FaceObjectPresentation(value: 1, timescale: 0),
    FaceObjectPresentation(value: 86_401, timescale: 1),
  ])
  func rejectsInvalidSourceClock(_ time: FaceObjectPresentation) {
    #expect(throws: ProAppsError.self) { try time.validate() }
  }

  @Test func seedCannotFollowCurrentFrameAndLargeTimesDoNotOverflow() throws {
    let future = FaceObjectSeed(
      observationUUID: UUID(), presentation: .init(value: 2, timescale: 1))
    #expect(throws: ProAppsError.self) {
      try evidence(box: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.3), seed: future)
    }
    let time = FaceObjectPresentation(value: 86_400_000_000_000, timescale: 1_000_000_000)
    let seed = FaceObjectSeed(observationUUID: UUID(), presentation: time)
    let result = try FaceObjectEvidence(
      raw: RawFaceObject(
        box: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.3), confidence: 1,
        observationUUID: UUID()),
      presentation: time, stage: .trackingAdvance, seed: seed, width: 100, height: 200)
    #expect(result.seed?.presentation.value == time.value)
  }

  @Test(arguments: [(-1, 0), (0, -1), (7201, 0), (7200, 1)])
  func refusesExcessiveEvidenceCounts(count: Int, adding: Int) {
    #expect(throws: ProAppsError.outputLimit) {
      try FaceObjectEvidence.count(after: count, adding: adding)
    }
  }

  @Test func recordBudgetIsFiniteAndOperationStagesRemainDistinct() throws {
    #expect(try FaceObjectEvidence.count(after: 7199, adding: 1) == 7200)
    #expect(FaceObjectStage(operation: .faceDetection) == .detection)
    #expect(FaceObjectStage(operation: .trackingSeed) == .trackingSeed)
    #expect(FaceObjectStage(operation: .trackingAdvance) == .trackingAdvance)
  }

  private func bitmap() throws -> CGImage {
    let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
    let context = try #require(
      CGContext(
        data: nil, width: 100, height: 200, bitsPerComponent: 8, bytesPerRow: 0,
        space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    return try #require(context.makeImage())
  }

  @Test func optInCapturePreservesRejectionTerminationAndClearsOnlyFrameEvidence() throws {
    let image = try bitmap()
    let observation = VNDetectedObjectObservation(
      boundingBox: CGRect(x: -0.1, y: 0.2, width: 0.3, height: 0.3))
    let request = VNTrackObjectRequest(detectedObjectObservation: observation)
    let sequence = FaceTrackingSequence()
    sequence.captureObjectEvidence = true
    sequence.beginFrame(presentation: presentation)
    #expect(try sequence.advance(request, observation: observation, in: image) == nil)
    #expect(request.isLastFrame)
    #expect(sequence.outOfFrameTrackingObservations == 1)
    let raw = try #require(sequence.objectEvidence.first)
    #expect(raw.observationUUID == observation.uuid)
    #expect(raw.disposition == .outOfFrameRejected)
    #expect(raw.presentation.value == presentation.value)
    sequence.beginFrame(presentation: .init(value: 90_006, timescale: 90_000))
    #expect(sequence.objectEvidence.isEmpty)
    #expect(sequence.outOfFrameTrackingObservations == 0)
  }

  @Test func optInRequiresActualPTSAndLegacyDoesNotCapture() throws {
    let image = try bitmap()
    let observation = VNDetectedObjectObservation(
      boundingBox: CGRect(x: -0.1, y: 0.2, width: 0.3, height: 0.3))
    let legacy = FaceTrackingSequence()
    let request = VNTrackObjectRequest(detectedObjectObservation: observation)
    #expect(try legacy.advance(request, observation: observation, in: image) == nil)
    #expect(legacy.objectEvidence.isEmpty)
    let captured = FaceTrackingSequence()
    captured.captureObjectEvidence = true
    let second = VNTrackObjectRequest(detectedObjectObservation: observation)
    #expect(throws: ProAppsError.self) {
      try captured.advance(second, observation: observation, in: image)
    }
    #expect(!second.isLastFrame)
  }

  @Test func wrapperKeepsOriginalMissingFaceAndUnapprovedFlags() throws {
    let frame = FaceTrackingFrame(
      presentationValue: 90_003, presentationTimescale: 90_000, width: 100, height: 200,
      method: .missing, detectionAttempted: true, outOfFrameTrackingObservations: 1,
      outOfFrameDetectionObservations: 0, faces: [])
    let raw = try evidence(box: CGRect(x: -0.1, y: 0.2, width: 0.3, height: 0.3))
    let wrapper = FaceObjectTrackingFrame(faceFrame: frame, objects: [raw])
    struct Flags: Decodable {
      let faceFrame: FaceTrackingFrame
      let faceObserved: Bool
      let objectEvidenceOnly: Bool
      let semanticHeadVerified: Bool
      let captionPlacementApproved: Bool
    }
    let flags = try JSONDecoder().decode(Flags.self, from: JSONEncoder().encode(wrapper))
    #expect(!flags.faceObserved && flags.objectEvidenceOnly)
    #expect(!flags.semanticHeadVerified && !flags.captionPlacementApproved)
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    #expect(try encoder.encode(flags.faceFrame) == encoder.encode(frame))
  }

  @Test func rejectedDetectionIsRecordedWithoutStartingAnObjectSequence() throws {
    let sequence = FaceTrackingSequence(detectFaces: { _ in
      [VNFaceObservation(boundingBox: CGRect(x: -0.1, y: 0.2, width: 0.3, height: 0.3))]
    })
    sequence.captureObjectEvidence = true
    sequence.beginFrame(presentation: presentation)
    #expect(try sequence.detect(in: bitmap()).isEmpty)
    #expect(sequence.outOfFrameDetectionObservations == 1)
    let object = try #require(sequence.objectEvidence.first)
    #expect(object.stage == .detection && object.seed == nil)
    #expect(object.disposition == .outOfFrameRejected)
    #expect(try sequence.track(in: bitmap()).isEmpty)
  }

  @Test func actualSeedChainIsSourceBoundAndLegacyFaceFramesStayEqual() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures")
    )
    let probe = FaceTrackProbe(makeSequence: {
      FaceTrackingSequence(detectFaces: { _ in
        [VNFaceObservation(boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.3))]
      })
    })
    let plan = FaceTrackingPlan(
      startSeconds: 0, durationSeconds: 0.1, maximumFrames: 3, sourceTimescale: 30000)
    let captured = try await probe.trackObjects(path: source.path, plan: plan)
    let first = try #require(captured.first)
    let detection = try #require(first.objects.first { $0.stage == .detection })
    let seeded = try #require(first.objects.first { $0.stage == .trackingSeed })
    let seed = try #require(seeded.seed)
    #expect(seed.observationUUID == detection.observationUUID)
    #expect(seed.presentation.value == 0)
    #expect(seeded.presentation.value == first.faceFrame.presentationValue)
    let legacy = try await probe.track(path: source.path, plan: plan)
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    #expect(try encoder.encode(captured.map(\.faceFrame)) == encoder.encode(legacy))
  }
}
