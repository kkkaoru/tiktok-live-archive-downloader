import CoreGraphics
import CoreMedia
import Foundation
import Vision

/// An actual decoded source timestamp, not an inferred frame grid.
public struct FaceObjectPresentation: Codable, Sendable {
  public let value: Int64
  public let timescale: Int32

  func validate() throws {
    guard (1...FaceTrackingPlan.maximumTimescale).contains(timescale), value >= 0,
      value <= 86_400 * Int64(timescale)
    else { throw ProAppsError.invalid("Invalid raw object source presentation") }
  }
}

/// Request-local association with its actual detector seed; not a person identity.
public struct FaceObjectSeed: Codable, Sendable {
  public let observationUUID: UUID
  public let presentation: FaceObjectPresentation
}

public enum FaceObjectStage: String, Codable, Sendable {
  case detection, trackingSeed, trackingAdvance

  init(operation: NativeOperationFailure.Operation) {
    switch operation {
    case .faceDetection: self = .detection
    case .trackingSeed: self = .trackingSeed
    case .trackingAdvance: self = .trackingAdvance
    }
  }
}

public enum FaceObjectDisposition: String, Codable, Sendable {
  case accepted, confidenceRejected, outOfFrameRejected
}

public enum FaceObjectClipSide: String, Codable, Sendable {
  case left, top, right, bottom
}

struct RawFaceObject {
  let box: CGRect
  let confidence: Float
  let observationUUID: UUID
}

/// Raw object geometry remains unclipped. Visible intersection is a separate derivation.
/// Acceptance records the unchanged face policy, never a hair/head observation.
public struct FaceObjectEvidence: Encodable, Sendable {
  static let maximumRecords = FaceTrackingPlan.frameLimit
  public let presentation: FaceObjectPresentation
  public let stage: FaceObjectStage
  public let observationUUID: UUID
  public let seed: FaceObjectSeed?
  public let confidence: Float
  public let normalizedLowerLeft: EditCrop
  public let displayTopLeft: EditCrop
  public let visibleIntersection: EditCrop?
  public let clipSides: [FaceObjectClipSide]
  public let disposition: FaceObjectDisposition

  init(
    observation: VNDetectedObjectObservation, presentation: FaceObjectPresentation,
    stage: FaceObjectStage, seed: FaceObjectSeed?, width: Int, height: Int
  ) throws {
    try self.init(
      raw: RawFaceObject(
        box: observation.boundingBox, confidence: observation.confidence,
        observationUUID: observation.uuid),
      presentation: presentation, stage: stage, seed: seed, width: width, height: height)
  }

  init(
    raw: RawFaceObject, presentation: FaceObjectPresentation, stage: FaceObjectStage,
    seed: FaceObjectSeed?, width: Int, height: Int
  ) throws {
    try presentation.validate()
    if let seed {
      try seed.presentation.validate()
      guard
        CMTime(value: seed.presentation.value, timescale: seed.presentation.timescale)
          <= CMTime(value: presentation.value, timescale: presentation.timescale)
      else { throw ProAppsError.invalid("Raw object seed follows its observation") }
    }
    let box = raw.box
    guard width > 0, width <= 16384, height > 0, height <= 16384,
      width * height <= 16_777_216,
      box.minX.isFinite, box.minY.isFinite, box.width.isFinite, box.height.isFinite,
      box.maxX.isFinite, box.maxY.isFinite, box.width > 0, box.height > 0,
      raw.confidence.isFinite, (0...1).contains(raw.confidence)
    else { throw ProAppsError.invalid("Malformed raw object observation") }
    let x = box.minX * Double(width)
    let y = (1 - box.maxY) * Double(height)
    let w = box.width * Double(width)
    let h = box.height * Double(height)
    guard x.isFinite, y.isFinite, w.isFinite, h.isFinite,
      (x + w).isFinite, (y + h).isFinite
    else { throw ProAppsError.invalid("Raw object display geometry overflowed") }
    self.presentation = presentation
    self.stage = stage
    observationUUID = raw.observationUUID
    self.seed = seed
    confidence = raw.confidence
    normalizedLowerLeft = EditCrop(
      x: box.minX, y: box.minY, width: box.width, height: box.height)
    displayTopLeft = EditCrop(x: x, y: y, width: w, height: h)
    let left = max(0, x)
    let top = max(0, y)
    let right = min(Double(width), x + w)
    let bottom = min(Double(height), y + h)
    visibleIntersection =
      right > left && bottom > top
      ? EditCrop(x: left, y: top, width: right - left, height: bottom - top) : nil
    var sides: [FaceObjectClipSide] = []
    if box.minX < 0 { sides.append(.left) }
    if box.maxY > 1 { sides.append(.top) }
    if box.maxX > 1 { sides.append(.right) }
    if box.minY < 0 { sides.append(.bottom) }
    clipSides = sides
    if !FaceTrackingSequence.accepts(raw.confidence) {
      disposition = .confidenceRejected
    } else if !sides.isEmpty {
      disposition = .outOfFrameRejected
    } else {
      disposition = .accepted
    }
  }

  static func count(after count: Int, adding records: Int) throws -> Int {
    guard count >= 0, records >= 0, count <= maximumRecords,
      records <= maximumRecords - count
    else { throw ProAppsError.outputLimit }
    return count + records
  }
}

/// Opt-in response: its face frame is the ordinary result, with no object-to-face promotion.
public struct FaceObjectTrackingFrame: Encodable, Sendable {
  public let faceFrame: FaceTrackingFrame
  public let objects: [FaceObjectEvidence]

  private enum CodingKeys: String, CodingKey {
    case faceFrame, objects, faceObserved, objectEvidenceOnly
    case semanticHeadVerified, captionPlacementApproved
  }

  public func encode(to encoder: any Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    try values.encode(faceFrame, forKey: .faceFrame)
    try values.encode(objects, forKey: .objects)
    try values.encode(!faceFrame.faces.isEmpty, forKey: .faceObserved)
    try values.encode(true, forKey: .objectEvidenceOnly)
    try values.encode(false, forKey: .semanticHeadVerified)
    try values.encode(false, forKey: .captionPlacementApproved)
  }
}
