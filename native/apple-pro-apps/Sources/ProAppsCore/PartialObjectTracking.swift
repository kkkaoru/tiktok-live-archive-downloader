import CoreGraphics
import CoreMedia
import Foundation
import Vision

public enum PartialObjectBoundaryReason: String, Codable, Sendable {
  case sourceCut, scene, zoom
}

/// Explicit boundaries come from the caller's source-clock evidence, not scene inference.
public struct PartialObjectBoundary: Codable, Sendable {
  public let presentation: FaceObjectPresentation
  public let reason: PartialObjectBoundaryReason
}

/// Candidate object support policy. It never changes the ordinary face policy.
public struct PartialObjectTrackingPlan: Codable, Sendable {
  public static let reseedLimit = 256
  public static let boundaryLimit = 128
  public let maximumSeeds: Int
  public let minimumVisibleFraction: Double
  public let boundaries: [PartialObjectBoundary]

  public init(
    maximumSeeds: Int, minimumVisibleFraction: Double, boundaries: [PartialObjectBoundary]
  ) {
    self.maximumSeeds = maximumSeeds
    self.minimumVisibleFraction = minimumVisibleFraction
    self.boundaries = boundaries
  }

  func validate(window: CMTimeRange) throws {
    guard (1...Self.reseedLimit).contains(maximumSeeds), minimumVisibleFraction.isFinite,
      minimumVisibleFraction > 0, minimumVisibleFraction <= 1,
      boundaries.count <= Self.boundaryLimit
    else { throw ProAppsError.invalid("Partial object policy exceeds its finite budget") }
    var previous: CMTime?
    for boundary in boundaries {
      try boundary.presentation.validate()
      let time = CMTime(
        value: boundary.presentation.value, timescale: boundary.presentation.timescale)
      guard time >= window.start, time < window.end, previous.map({ time > $0 }) ?? true else {
        throw ProAppsError.invalid("Partial object boundaries must be unique ordered source PTS")
      }
      previous = time
    }
  }
}

public enum PartialObjectStatus: String, Encodable, Sendable {
  case supportedSeed, supportedAdvance, confidenceRejected, fullExit
  case insufficientVisibleSupport, noNativeResult, ambiguousSeed, seedLimit
  case boundaryReset, geometryReset
  case boundaryAssociationRejected, faceReacquired
}

/// A new request input derived from the current visible pixels. It is not a detector observation.
public struct PartialObjectDerivedSeed: Encodable, Sendable {
  public let observationUUID: UUID
  public let presentation: FaceObjectPresentation
  public let normalizedLowerLeft: EditCrop
  public let ordinal: Int
  public let inputRawObservation: FaceObjectEvidence
}

/// Current raw Vision evidence and derived request metadata are deliberately separate.
public struct PartialObjectRecord: Encodable, Sendable {
  public let presentation: FaceObjectPresentation
  public let status: PartialObjectStatus
  public let originalRejection: FaceObjectEvidence?
  public let derivedSeed: PartialObjectDerivedSeed?
  public let raw: FaceObjectEvidence?
  public let boundary: PartialObjectBoundary?
  public let derivedPolicySupport: Bool
  public let semanticHeadVerified: Bool
  public let captionPlacementApproved: Bool
}

public struct PartialObjectTrackingFrame: Encodable, Sendable {
  public let legacy: FaceObjectTrackingFrame
  public let partialObjects: [PartialObjectRecord]

  private enum CodingKeys: String, CodingKey {
    case faceFrame, objects, faceObserved, objectEvidenceOnly, partialObjects
    case semanticHeadVerified, captionPlacementApproved, partialObjectEvidenceOnly
  }

  public func encode(to encoder: any Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    try values.encode(legacy.faceFrame, forKey: .faceFrame)
    try values.encode(legacy.objects, forKey: .objects)
    try values.encode(!legacy.faceFrame.faces.isEmpty, forKey: .faceObserved)
    try values.encode(true, forKey: .objectEvidenceOnly)
    try values.encode(partialObjects, forKey: .partialObjects)
    try values.encode(true, forKey: .partialObjectEvidenceOnly)
    try values.encode(false, forKey: .semanticHeadVerified)
    try values.encode(false, forKey: .captionPlacementApproved)
  }
}

struct PartialObjectObservation {
  let raw: RawFaceObject
  let native: VNDetectedObjectObservation?

  init(_ observation: VNDetectedObjectObservation) {
    raw = RawFaceObject(
      box: observation.boundingBox, confidence: observation.confidence,
      observationUUID: observation.uuid)
    native = observation
  }

  init(raw: RawFaceObject, native: VNDetectedObjectObservation? = nil) {
    self.raw = raw
    self.native = native
  }
}

/// Confined to FaceTrackProbe's serial worker. No legacy request or handler is shared.
final class PartialObjectTrackingSequence {
  typealias Perform = (VNSequenceRequestHandler, VNTrackObjectRequest, CGImage) throws
    -> PartialObjectObservation?
  private let policy: PartialObjectTrackingPlan
  private let perform: Perform
  private let makeRequest: (VNDetectedObjectObservation) throws -> VNTrackObjectRequest
  private var handler = VNSequenceRequestHandler()
  private var request: VNTrackObjectRequest?
  private var original: FaceObjectEvidence?
  private var seed: PartialObjectDerivedSeed?
  private var previous: CMTime?
  private var dimensions: CGSize?
  private var boundaryIndex = 0
  private var seedCount = 0
  private var associationStart: CMTime?

  init(
    policy: PartialObjectTrackingPlan,
    perform: @escaping Perform = PartialObjectTrackingSequence.nativePerform,
    makeRequest: @escaping (VNDetectedObjectObservation) throws -> VNTrackObjectRequest =
      PartialObjectTrackingSequence.nativeRequest
  ) {
    self.policy = policy
    self.perform = perform
    self.makeRequest = makeRequest
  }

  static func nativeRequest(_ observation: VNDetectedObjectObservation) throws
    -> VNTrackObjectRequest
  {
    let request = VNTrackObjectRequest(detectedObjectObservation: observation)
    request.trackingLevel = .accurate
    try FaceTrackingSequence.configureCPU(request, supported: request.supportedComputeStageDevices)
    return request
  }

  static func nativePerform(
    handler: VNSequenceRequestHandler, request: VNTrackObjectRequest, image: CGImage
  ) throws -> PartialObjectObservation? {
    try handler.perform([request], on: image)
    return try decodeResults(request.results)
  }

  static func decodeResults(_ results: [VNObservation]?) throws -> PartialObjectObservation? {
    guard let results, results.count <= 1 else {
      throw ProAppsError.unavailable("Partial object tracker returned an ambiguous result array")
    }
    guard let first = results.first else { return nil }
    guard let observation = first as? VNDetectedObjectObservation else {
      throw ProAppsError.unavailable("Partial object tracker returned an invalid observation")
    }
    return PartialObjectObservation(observation)
  }

  func finish() throws {
    guard boundaryIndex == policy.boundaries.count else {
      throw ProAppsError.invalid("Partial object boundary did not match an actual decoded PTS")
    }
  }

  func process(image: CGImage, legacy: FaceObjectTrackingFrame) throws -> [PartialObjectRecord] {
    try Task.checkCancellation()
    let presentation = FaceObjectPresentation(
      value: legacy.faceFrame.presentationValue, timescale: legacy.faceFrame.presentationTimescale)
    try presentation.validate()
    let time = CMTime(value: presentation.value, timescale: presentation.timescale)
    guard previous.map({ time > $0 }) ?? true,
      image.width == legacy.faceFrame.width, image.height == legacy.faceFrame.height
    else { throw ProAppsError.invalid("Partial object frame is not current source geometry/PTS") }
    previous = time
    if associationStart == nil { associationStart = time }
    var records: [PartialObjectRecord] = []
    if boundaryIndex < policy.boundaries.count {
      let boundary = policy.boundaries[boundaryIndex]
      let boundaryTime = CMTime(
        value: boundary.presentation.value, timescale: boundary.presentation.timescale)
      guard time <= boundaryTime else {
        throw ProAppsError.invalid("Partial object boundary was not an actual decoded PTS")
      }
      if time == boundaryTime {
        records.append(record(presentation, status: .boundaryReset, boundary: boundary))
        clear()
        associationStart = time
        boundaryIndex += 1
      }
    }
    let size = CGSize(width: image.width, height: image.height)
    if let dimensions, dimensions != size {
      records.append(record(presentation, status: .geometryReset))
      clear()
      associationStart = time
    }
    dimensions = size
    for object in legacy.objects {
      guard object.presentation.value == presentation.value,
        object.presentation.timescale == presentation.timescale
      else { throw ProAppsError.invalid("Partial object input contains a stale raw observation") }
    }
    let candidates = legacy.objects.filter {
      $0.stage == .trackingAdvance && $0.disposition == .outOfFrameRejected && $0.seed != nil
    }
    guard candidates.count <= 1, legacy.faceFrame.faces.count <= 1 else {
      records.append(record(presentation, status: .ambiguousSeed))
      clear()
      return records
    }
    if request != nil,
      legacy.objects.contains(where: {
        $0.stage == .detection && $0.disposition == .accepted
      })
    {
      records.append(record(presentation, status: .faceReacquired))
      clear()
      return records
    }
    if let request {
      let observation = try NativeOperationFailure.perform(.trackingAdvance) {
        try perform(handler, request, image)
      }
      records.append(
        try receive(observation, image: image, presentation: presentation, initial: false))
      return records
    }
    guard let candidate = candidates.first else { return records }
    guard let originSeed = candidate.seed, let associationStart,
      CMTime(value: originSeed.presentation.value, timescale: originSeed.presentation.timescale)
        >= associationStart
    else {
      records.append(record(presentation, status: .boundaryAssociationRejected, raw: candidate))
      return records
    }
    original = candidate
    records.append(try initialize(candidate, image: image, presentation: presentation))
    return records
  }

  private func initialize(
    _ raw: FaceObjectEvidence, image: CGImage, presentation: FaceObjectPresentation
  ) throws -> PartialObjectRecord {
    guard let visible = raw.visibleIntersection else {
      let result = record(presentation, status: .fullExit, raw: raw)
      clear()
      return result
    }
    let fraction =
      visible.width * visible.height / (raw.displayTopLeft.width * raw.displayTopLeft.height)
    guard fraction >= policy.minimumVisibleFraction else {
      let result = record(presentation, status: .insufficientVisibleSupport, raw: raw)
      clear()
      return result
    }
    guard seedCount < policy.maximumSeeds else {
      let result = record(presentation, status: .seedLimit, raw: raw)
      clear()
      return result
    }
    let normalized = raw.normalizedLowerLeft
    let box = CGRect(
      x: normalized.x, y: normalized.y, width: normalized.width, height: normalized.height
    )
    .intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
    let observation = VNDetectedObjectObservation(boundingBox: box)
    seedCount += 1
    seed = PartialObjectDerivedSeed(
      observationUUID: observation.uuid, presentation: presentation,
      normalizedLowerLeft: EditCrop(x: box.minX, y: box.minY, width: box.width, height: box.height),
      ordinal: seedCount, inputRawObservation: raw)
    handler = VNSequenceRequestHandler()
    let newRequest = try makeRequest(observation)
    request = newRequest
    let measured = try NativeOperationFailure.perform(.trackingSeed) {
      try perform(handler, newRequest, image)
    }
    return try receive(measured, image: image, presentation: presentation, initial: true)
  }

  private func receive(
    _ observation: PartialObjectObservation?, image: CGImage,
    presentation: FaceObjectPresentation, initial: Bool
  ) throws -> PartialObjectRecord {
    guard let observation else {
      let result = record(presentation, status: .noNativeResult)
      clear()
      return result
    }
    let raw = try FaceObjectEvidence(
      raw: observation.raw, presentation: presentation,
      stage: initial ? .trackingSeed : .trackingAdvance,
      seed: original?.seed, width: image.width, height: image.height)
    guard FaceTrackingSequence.accepts(raw.confidence) else {
      let result = record(presentation, status: .confidenceRejected, raw: raw)
      clear()
      return result
    }
    guard let visible = raw.visibleIntersection else {
      let result = record(presentation, status: .fullExit, raw: raw)
      clear()
      return result
    }
    let fraction =
      visible.width * visible.height / (raw.displayTopLeft.width * raw.displayTopLeft.height)
    guard fraction >= policy.minimumVisibleFraction else {
      let result = record(presentation, status: .insufficientVisibleSupport, raw: raw)
      clear()
      return result
    }
    if !raw.clipSides.isEmpty {
      // A fresh visible-image initialization uses this frame, never a held rectangle.
      // The seed result itself must be inside the image to avoid recursive reseeding.
      if initial {
        let result = record(presentation, status: .insufficientVisibleSupport, raw: raw)
        clear()
        return result
      }
      return try initialize(raw, image: image, presentation: presentation)
    }
    guard let native = observation.native else {
      throw ProAppsError.invalid("Supported partial object has no current native observation")
    }
    request?.inputObservation = native
    return record(presentation, status: initial ? .supportedSeed : .supportedAdvance, raw: raw)
  }

  private func record(
    _ presentation: FaceObjectPresentation, status: PartialObjectStatus,
    raw: FaceObjectEvidence? = nil, boundary: PartialObjectBoundary? = nil
  ) -> PartialObjectRecord {
    PartialObjectRecord(
      presentation: presentation, status: status, originalRejection: original,
      derivedSeed: seed, raw: raw, boundary: boundary, derivedPolicySupport: true,
      semanticHeadVerified: false, captionPlacementApproved: false)
  }

  private func clear() {
    request?.isLastFrame = true
    request = nil
    original = nil
    seed = nil
    handler = VNSequenceRequestHandler()
  }
}
