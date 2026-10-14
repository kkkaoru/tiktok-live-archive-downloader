import CoreGraphics
import CoreML
import CoreMedia
import Foundation
import Testing
import Vision

@testable import ProAppsCore

struct PartialObjectTrackingTests {
  private let policy = PartialObjectTrackingPlan(
    maximumSeeds: 4, minimumVisibleFraction: 0.25, boundaries: [])
  private let originalUUID = UUID(uuidString: "C1374BC0-76FD-473E-8432-9BEE25380FBC")

  private func bitmap(width: Int = 100, height: Int = 200) throws -> CGImage {
    let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
    let context = try #require(
      CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
        space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    return try #require(context.makeImage())
  }

  private func frame(
    value: Int64, objects: [FaceObjectEvidence] = [], width: Int = 100, height: Int = 200,
    faces: [TrackedFace] = []
  ) -> FaceObjectTrackingFrame {
    FaceObjectTrackingFrame(
      faceFrame: FaceTrackingFrame(
        presentationValue: value, presentationTimescale: 30, width: width, height: height,
        method: .missing, detectionAttempted: true, outOfFrameTrackingObservations: objects.count,
        outOfFrameDetectionObservations: 0, faces: faces), objects: objects)
  }

  private func rejection(
    value: Int64 = 0, box: CGRect = CGRect(x: -0.1, y: 0.2, width: 0.4, height: 0.3),
    seed: Bool = true, stage: FaceObjectStage = .trackingAdvance
  ) throws -> FaceObjectEvidence {
    let uuid = try #require(originalUUID)
    return try FaceObjectEvidence(
      raw: RawFaceObject(box: box, confidence: 1, observationUUID: uuid),
      presentation: .init(value: value, timescale: 30), stage: stage,
      seed: seed ? .init(observationUUID: uuid, presentation: .init(value: 0, timescale: 30)) : nil,
      width: 100, height: 200)
  }

  private func sequence(
    policy: PartialObjectTrackingPlan? = nil,
    perform: @escaping PartialObjectTrackingSequence.Perform = { _, request, _ in
      PartialObjectObservation(request.inputObservation)
    }
  ) -> PartialObjectTrackingSequence {
    PartialObjectTrackingSequence(
      policy: policy ?? self.policy, perform: perform,
      makeRequest: { VNTrackObjectRequest(detectedObjectObservation: $0) })
  }

  @Test func visibleCurrentSeedHasNewUUIDAndKeepsOriginalRejection() throws {
    let image = try bitmap()
    let rejected = try rejection()
    let tracker = sequence()
    let records = try tracker.process(image: image, legacy: frame(value: 0, objects: [rejected]))
    let result = try #require(records.first)
    #expect(result.status == .supportedSeed)
    let seed = try #require(result.derivedSeed)
    #expect(seed.observationUUID != originalUUID)
    #expect(seed.presentation.value == 0 && seed.ordinal == 1)
    #expect(seed.normalizedLowerLeft.x == 0)
    #expect(abs(seed.normalizedLowerLeft.width - 0.3) < 0.00001)
    #expect(seed.inputRawObservation.clipSides == [.left])
    #expect(result.originalRejection?.displayTopLeft.x == -10)
    #expect(result.raw?.seed?.observationUUID == originalUUID)
    #expect(result.raw?.clipSides.isEmpty == true)
    #expect(!result.semanticHeadVerified && !result.captionPlacementApproved)
    let advanced = try tracker.process(image: image, legacy: frame(value: 1))
    let next = try #require(advanced.first)
    #expect(next.status == .supportedAdvance && next.presentation.value == 1)
    #expect(next.raw?.presentation.value == 1)
    #expect(next.derivedSeed?.ordinal == 1)
    try tracker.finish()
  }

  @Test func initializesAppearanceOnExactCurrentImage() throws {
    let image = try bitmap()
    let tracker = sequence(perform: { _, request, current in
      #expect(current === image)
      #expect(request.inputObservation.boundingBox.minX == 0)
      return PartialObjectObservation(request.inputObservation)
    })
    let result = try tracker.process(image: image, legacy: frame(value: 0, objects: [rejection()]))
    #expect(result.first?.status == .supportedSeed)
  }

  @Test func missingFramesNeverBecomeHeldOrNewObservations() throws {
    let tracker = sequence()
    #expect(try tracker.process(image: bitmap(), legacy: frame(value: 0)).isEmpty)
    #expect(try tracker.process(image: bitmap(), legacy: frame(value: 1)).isEmpty)
  }

  @Test(arguments: [FaceObjectStage.detection, .trackingSeed])
  func onlyRejectedAdvancesWithActualDetectorSeedsCanStart(_ stage: FaceObjectStage) throws {
    let tracker = sequence()
    #expect(
      try tracker.process(
        image: bitmap(), legacy: frame(value: 0, objects: [rejection(stage: stage)])
      ).isEmpty)
  }

  @Test func rejectsSeedlessAndStaleRawInputs() throws {
    #expect(
      try sequence().process(
        image: bitmap(), legacy: frame(value: 0, objects: [rejection(seed: false)])
      ).isEmpty)
    #expect(throws: ProAppsError.self) {
      try sequence().process(image: bitmap(), legacy: frame(value: 1, objects: [rejection()]))
    }
  }

  @Test func ambiguousObjectsDoNotSelectAHUDOrLargestRectangle() throws {
    let rejected = try rejection()
    let result = try sequence().process(
      image: bitmap(), legacy: frame(value: 0, objects: [rejected, rejected]))
    #expect(result.first?.status == .ambiguousSeed)
    let faces = [
      TrackedFace(region: .init(x: 0, y: 0, width: 1, height: 1), confidence: 1),
      TrackedFace(region: .init(x: 2, y: 2, width: 1, height: 1), confidence: 1),
    ]
    let two = try sequence().process(
      image: bitmap(), legacy: frame(value: 0, objects: [rejected], faces: faces))
    #expect(two.first?.status == .ambiguousSeed)
  }

  @Test(arguments: [
    (CGRect(x: -2, y: 0.2, width: 0.4, height: 0.3), PartialObjectStatus.fullExit),
    (
      CGRect(x: -0.39, y: 0.2, width: 0.4, height: 0.3),
      PartialObjectStatus.insufficientVisibleSupport
    ),
  ])
  func unsupportedVisibilityDoesNotInitialize(box: CGRect, expected: PartialObjectStatus) throws {
    let tracker = sequence(perform: { _, _, _ in
      Issue.record("Unsupported visibility must not invoke Vision")
      return nil
    })
    let result = try tracker.process(
      image: bitmap(), legacy: frame(value: 0, objects: [rejection(box: box)]))
    #expect(result.first?.status == expected)
    #expect(result.first?.derivedSeed == nil)
  }

  @Test func noNativeResultEndsSupportInsteadOfHoldingInput() throws {
    let tracker = sequence(perform: { _, _, _ in nil })
    let result = try tracker.process(
      image: bitmap(), legacy: frame(value: 0, objects: [rejection()]))
    #expect(result.first?.status == .noNativeResult)
    #expect(result.first?.raw == nil)
    #expect(try tracker.process(image: bitmap(), legacy: frame(value: 1)).isEmpty)
  }

  @Test(arguments: [Float(0), 0.3])
  func nativeConfidenceLossIsRecordedAndTerminates(_ confidence: Float) throws {
    let tracker = sequence(perform: { _, _, _ in
      PartialObjectObservation(
        raw: RawFaceObject(
          box: CGRect(x: 0.1, y: 0.2, width: 0.4, height: 0.3), confidence: confidence,
          observationUUID: UUID()))
    })
    let result = try tracker.process(
      image: bitmap(), legacy: frame(value: 0, objects: [rejection()]))
    #expect(result.first?.status == .confidenceRejected)
    #expect(result.first?.raw?.confidence == confidence)
    #expect(try tracker.process(image: bitmap(), legacy: frame(value: 1)).isEmpty)
  }

  @Test(arguments: [
    (CGRect(x: -2, y: 0.2, width: 0.4, height: 0.3), PartialObjectStatus.fullExit),
    (
      CGRect(x: -0.39, y: 0.2, width: 0.4, height: 0.3),
      PartialObjectStatus.insufficientVisibleSupport
    ),
    (
      CGRect(x: -0.1, y: 0.2, width: 0.4, height: 0.3),
      PartialObjectStatus.insufficientVisibleSupport
    ),
  ])
  func initializationRequiresCurrentInsideImageResult(box: CGRect, expected: PartialObjectStatus)
    throws
  {
    let tracker = sequence(perform: { _, _, _ in
      PartialObjectObservation(VNDetectedObjectObservation(boundingBox: box))
    })
    let result = try tracker.process(
      image: bitmap(), legacy: frame(value: 0, objects: [rejection()]))
    #expect(result.first?.status == expected)
    #expect(try tracker.process(image: bitmap(), legacy: frame(value: 1)).isEmpty)
  }

  @Test func clipsAreReseededFromTheirActualCurrentImageAndBudgetIsFinite() throws {
    let limited = PartialObjectTrackingPlan(
      maximumSeeds: 1, minimumVisibleFraction: 0.25, boundaries: [])
    let tracker = sequence(
      policy: limited,
      perform: { _, request, _ in
        if request.inputObservation.boundingBox.minX == 0 {
          request.inputObservation = VNDetectedObjectObservation(
            boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.3))
          return PartialObjectObservation(request.inputObservation)
        }
        return PartialObjectObservation(
          VNDetectedObjectObservation(
            boundingBox: CGRect(x: -0.1, y: 0.2, width: 0.4, height: 0.3)))
      })
    #expect(
      try tracker.process(image: bitmap(), legacy: frame(value: 0, objects: [rejection()])).first?
        .status == .supportedSeed)
    let next = try tracker.process(image: bitmap(), legacy: frame(value: 1))
    #expect(next.first?.status == .seedLimit)
    #expect(next.first?.raw?.presentation.value == 1)
    #expect(try tracker.process(image: bitmap(), legacy: frame(value: 2)).isEmpty)
  }

  @Test func successfulReseedPreservesCurrentRawAndOriginalDetectorAssociation() throws {
    let tracker = sequence(perform: { _, request, _ in
      if request.inputObservation.boundingBox.minX == 0 {
        let observation = VNDetectedObjectObservation(
          boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.3))
        return PartialObjectObservation(observation)
      }
      return PartialObjectObservation(
        VNDetectedObjectObservation(
          boundingBox: CGRect(x: -0.1, y: 0.2, width: 0.4, height: 0.3)))
    })
    _ = try tracker.process(image: bitmap(), legacy: frame(value: 0, objects: [rejection()]))
    let next = try tracker.process(image: bitmap(), legacy: frame(value: 1))
    let seed = try #require(next.first?.derivedSeed)
    #expect(next.first?.status == .supportedSeed)
    #expect(seed.ordinal == 2 && seed.presentation.value == 1)
    #expect(seed.inputRawObservation.presentation.value == 1)
    #expect(seed.inputRawObservation.clipSides == [.left])
    #expect(next.first?.originalRejection?.presentation.value == 0)
  }

  @Test(arguments: [PartialObjectBoundaryReason.sourceCut, .scene, .zoom])
  func explicitBoundariesResetBeforeProcessingTheirFrame(_ reason: PartialObjectBoundaryReason)
    throws
  {
    let boundary = PartialObjectBoundary(
      presentation: .init(value: 1, timescale: 30), reason: reason)
    let policy = PartialObjectTrackingPlan(
      maximumSeeds: 4, minimumVisibleFraction: 0.25, boundaries: [boundary])
    try policy.validate(
      window: CMTimeRange(start: .zero, duration: CMTime(value: 3, timescale: 30)))
    let tracker = sequence(policy: policy)
    _ = try tracker.process(image: bitmap(), legacy: frame(value: 0, objects: [rejection()]))
    let reset = try tracker.process(image: bitmap(), legacy: frame(value: 1))
    #expect(reset.count == 1 && reset.first?.status == .boundaryReset)
    #expect(reset.first?.boundary?.reason == reason)
    #expect(try tracker.process(image: bitmap(), legacy: frame(value: 2)).isEmpty)
    try tracker.finish()
  }

  @Test func geometryChangeInvalidatesWithoutNormalizingAnOldBox() throws {
    let tracker = sequence()
    _ = try tracker.process(image: bitmap(), legacy: frame(value: 0, objects: [rejection()]))
    let resized = try tracker.process(
      image: bitmap(width: 200), legacy: frame(value: 1, width: 200))
    #expect(resized.first?.status == .geometryReset)
    #expect(resized.first?.raw == nil)
  }

  @Test func oldDetectorAssociationCannotCrossASceneBoundary() throws {
    let policy = PartialObjectTrackingPlan(
      maximumSeeds: 4, minimumVisibleFraction: 0.25,
      boundaries: [
        .init(presentation: .init(value: 1, timescale: 30), reason: .scene)
      ])
    let tracker = sequence(policy: policy)
    _ = try tracker.process(image: bitmap(), legacy: frame(value: 0))
    let results = try tracker.process(
      image: bitmap(), legacy: frame(value: 1, objects: [rejection(value: 1)]))
    #expect(results.count == 2)
    #expect(results.last?.status == .boundaryAssociationRejected)
    #expect(results.last?.raw?.seed?.presentation.value == 0)
    #expect(results.last?.derivedSeed == nil)
  }

  @Test func actualReacquisitionAndAmbiguityTerminatePartialAssociation() throws {
    let tracker = sequence()
    _ = try tracker.process(image: bitmap(), legacy: frame(value: 0, objects: [rejection()]))
    let detected = try rejection(
      value: 1, box: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.3), seed: false, stage: .detection
    )
    let next = try tracker.process(image: bitmap(), legacy: frame(value: 1, objects: [detected]))
    #expect(next.first?.status == .faceReacquired)
    #expect(try tracker.process(image: bitmap(), legacy: frame(value: 2)).isEmpty)
    let ambiguous = sequence()
    _ = try ambiguous.process(image: bitmap(), legacy: frame(value: 0, objects: [rejection()]))
    let reject = try rejection(value: 1)
    #expect(
      try ambiguous.process(image: bitmap(), legacy: frame(value: 1, objects: [reject, reject]))
        .first?.status == .ambiguousSeed)
    #expect(try ambiguous.process(image: bitmap(), legacy: frame(value: 2)).isEmpty)
  }

  @Test func unmatchedAndNonCurrentClockBoundariesFailClosed() throws {
    let policy = PartialObjectTrackingPlan(
      maximumSeeds: 1, minimumVisibleFraction: 0.5,
      boundaries: [
        .init(presentation: .init(value: 1, timescale: 30), reason: .scene)
      ])
    let tracker = sequence(policy: policy)
    _ = try tracker.process(image: bitmap(), legacy: frame(value: 0))
    #expect(throws: ProAppsError.self) { try tracker.finish() }
    #expect(throws: ProAppsError.self) {
      try tracker.process(image: bitmap(), legacy: frame(value: 2))
    }
    #expect(throws: ProAppsError.self) {
      try sequence().process(image: bitmap(), legacy: frame(value: -1))
    }
    let duplicate = sequence()
    _ = try duplicate.process(image: bitmap(), legacy: frame(value: 0))
    #expect(throws: ProAppsError.self) {
      try duplicate.process(image: bitmap(), legacy: frame(value: 0))
    }
    #expect(throws: ProAppsError.self) {
      try sequence().process(image: bitmap(), legacy: frame(value: 0, width: 200))
    }
  }

  @Test(arguments: [0, 257])
  func policySeedCountIsBounded(_ count: Int) {
    let policy = PartialObjectTrackingPlan(
      maximumSeeds: count, minimumVisibleFraction: 0.5, boundaries: [])
    #expect(throws: ProAppsError.self) {
      try policy.validate(
        window: CMTimeRange(start: .zero, duration: CMTime(value: 1, timescale: 1)))
    }
  }

  @Test(arguments: [Double.nan, 0, -0.1, 1.1])
  func policyVisibleSupportCannotBeSilentlyWeakened(_ fraction: Double) {
    let policy = PartialObjectTrackingPlan(
      maximumSeeds: 1, minimumVisibleFraction: fraction, boundaries: [])
    #expect(throws: ProAppsError.self) {
      try policy.validate(
        window: CMTimeRange(start: .zero, duration: CMTime(value: 1, timescale: 1)))
    }
  }

  @Test func boundaryValidationRejectsDuplicatesOutsideWindowAndExcessiveList() {
    let boundary = PartialObjectBoundary(
      presentation: .init(value: 1, timescale: 30), reason: .scene)
    let window = CMTimeRange(start: .zero, duration: CMTime(value: 3, timescale: 30))
    let duplicate = PartialObjectTrackingPlan(
      maximumSeeds: 1, minimumVisibleFraction: 1, boundaries: [boundary, boundary])
    #expect(throws: ProAppsError.self) { try duplicate.validate(window: window) }
    let outside = PartialObjectTrackingPlan(
      maximumSeeds: 1, minimumVisibleFraction: 1,
      boundaries: [
        .init(presentation: .init(value: 3, timescale: 30), reason: .zoom)
      ])
    #expect(throws: ProAppsError.self) { try outside.validate(window: window) }
    let excess = PartialObjectTrackingPlan(
      maximumSeeds: 1, minimumVisibleFraction: 1, boundaries: Array(repeating: boundary, count: 129)
    )
    #expect(throws: ProAppsError.self) { try excess.validate(window: window) }
  }

  @Test func typedNativeResultBoundaryRejectsInvalidOrAmbiguousArrays() throws {
    #expect(try PartialObjectTrackingSequence.decodeResults([]) == nil)
    #expect(throws: ProAppsError.self) { try PartialObjectTrackingSequence.decodeResults(nil) }
    let observation = VNDetectedObjectObservation(
      boundingBox: CGRect(x: 0, y: 0, width: 0.5, height: 0.5))
    #expect(throws: ProAppsError.self) {
      try PartialObjectTrackingSequence.decodeResults([observation, observation])
    }
    #expect(
      try PartialObjectTrackingSequence.decodeResults([observation])?.raw.observationUUID
        == observation.uuid)
  }

  @Test func frameworkFailureAndMissingNativeObservationAreNotSwallowed() throws {
    let failed = sequence(perform: { _, _, _ in
      throw ProAppsError.unavailable("synthetic tracking failure")
    })
    #expect(throws: ProAppsError.self) {
      try failed.process(image: bitmap(), legacy: frame(value: 0, objects: [rejection()]))
    }
    let invalid = sequence(perform: { _, _, _ in
      PartialObjectObservation(
        raw: RawFaceObject(
          box: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.3), confidence: 1,
          observationUUID: UUID()))
    })
    #expect(throws: ProAppsError.self) {
      try invalid.process(image: bitmap(), legacy: frame(value: 0, objects: [rejection()]))
    }
  }

  @Test func nativeRequestIsCPUConfiguredAndRunsOnASyntheticImage() throws {
    let observation = VNDetectedObjectObservation(
      boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.3))
    let request = try PartialObjectTrackingSequence.nativeRequest(observation)
    let devices = try request.supportedComputeStageDevices
    #expect(!devices.isEmpty)
    #expect(
      devices.keys.allSatisfy { stage in
        if case .cpu = request.computeDevice(for: stage) { return true }
        return false
      })
    let result = try PartialObjectTrackingSequence.nativePerform(
      handler: VNSequenceRequestHandler(), request: request, image: bitmap())
    #expect(result != nil)
  }

  @Test func defaultNativeControllerEncodesCurrentCandidateWithoutPromotingFaces() throws {
    let tracker = PartialObjectTrackingSequence(policy: policy)
    let legacy = frame(value: 0, objects: [try rejection()])
    let records = try tracker.process(image: bitmap(), legacy: legacy)
    #expect(records.count == 1)
    let current = try #require(records.first)
    #expect(current.presentation.value == 0)
    #expect(current.derivedSeed?.presentation.value == 0)
    struct RecordFlags: Decodable {
      let derivedPolicySupport: Bool
      let semanticHeadVerified: Bool
      let captionPlacementApproved: Bool
      let presentation: FaceObjectPresentation
    }
    struct Wrapper: Decodable {
      let faceObserved: Bool
      let partialObjects: [RecordFlags]
    }
    let result = try JSONDecoder().decode(
      Wrapper.self,
      from: JSONEncoder().encode(
        PartialObjectTrackingFrame(legacy: legacy, partialObjects: records)))
    #expect(!result.faceObserved)
    let flags = try #require(result.partialObjects.first)
    #expect(flags.derivedPolicySupport)
    #expect(!flags.semanticHeadVerified && !flags.captionPlacementApproved)
    #expect(flags.presentation.value == 0 && flags.presentation.timescale == 30)
  }

  @Test func optInProbePreservesLegacyAndCannotCreateSupportOnEmptyPixels() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    let probe = FaceTrackProbe()
    let plan = FaceTrackingPlan(
      startSeconds: 0, durationSeconds: 1, maximumFrames: 1, sourceTimescale: 30000)
    let partial = try await probe.trackPartialObjects(
      path: source.path, plan: plan, partialPlan: policy)
    let legacy = try await probe.trackObjects(path: source.path, plan: plan)
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    #expect(try encoder.encode(partial.map(\.legacy)) == encoder.encode(legacy))
    #expect(partial.first?.partialObjects.isEmpty == true)
    struct Flags: Decodable {
      let faceObserved: Bool
      let partialObjectEvidenceOnly: Bool
      let semanticHeadVerified: Bool
      let captionPlacementApproved: Bool
    }
    let flags = try JSONDecoder().decode([Flags].self, from: encoder.encode(partial))
    let first = try #require(flags.first)
    #expect(!first.faceObserved && first.partialObjectEvidenceOnly)
    #expect(!first.semanticHeadVerified && !first.captionPlacementApproved)
  }
}
