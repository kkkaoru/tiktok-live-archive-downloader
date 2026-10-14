import AVFoundation
import Foundation
import Vision

/// Raw detector evidence. Eligibility is the unchanged tracking policy, not approval.
public struct FaceDetectorCandidate: Codable, Sendable {
  public let confidence: Float
  public let normalizedLowerLeft: EditCrop
  public let displayTopLeft: EditCrop
  public let confidenceAccepted: Bool
  public let geometryAccepted: Bool

  init(box: CGRect, confidence: Float, size: CGSize) throws {
    guard box.origin.x.isFinite, box.origin.y.isFinite,
      box.width.isFinite, box.height.isFinite, box.width > 0, box.height > 0,
      box.maxX.isFinite, box.maxY.isFinite,
      confidence.isFinite, (0...1).contains(confidence),
      size.width.isFinite, size.height.isFinite, size.width > 0, size.height > 0
    else { throw ProAppsError.invalid("Malformed raw face-detector observation") }
    let display = CGRect(
      x: box.origin.x * size.width, y: (1 - box.maxY) * size.height,
      width: box.width * size.width, height: box.height * size.height)
    guard display.origin.x.isFinite, display.origin.y.isFinite,
      display.width.isFinite, display.height.isFinite
    else { throw ProAppsError.invalid("Raw detector display geometry overflowed") }
    self.confidence = confidence
    normalizedLowerLeft = EditCrop(
      x: box.origin.x, y: box.origin.y, width: box.width, height: box.height)
    displayTopLeft = EditCrop(
      x: display.origin.x, y: display.origin.y, width: display.width, height: display.height)
    confidenceAccepted = FaceTrackingSequence.accepts(confidence)
    geometryAccepted = box.minX >= 0 && box.minY >= 0 && box.maxX <= 1 && box.maxY <= 1
  }
}

/// Both detectors receive the exact same display-oriented CGImage, with no tracking.
public struct FaceDetectorComparison: Codable, Sendable {
  public let requestedPresentationValue: Int64
  public let requestedPresentationTimescale: Int32
  public let imageProducer: String
  public let legacyRevision: Int
  public let modernRevision: DetectFaceRectanglesRequest.Revision
  public let legacy: [FaceDetectorCandidate]
  public let modern: [FaceDetectorCandidate]
  public let operatingSystem: String
  public let generatorBitmap: FaceBitmapMetadata?
  public let reader: FaceReaderEvidence?
  public let readerPreservingSourceColor: FaceReaderEvidence?

  static func exactTime(seconds: Double) throws -> CMTime {
    guard seconds.isFinite, seconds >= 0, seconds <= 86400 else {
      throw ProAppsError.invalid("Invalid paired-detector sample time")
    }
    let scale = EditPlan.timeScale
    let value = Int64((seconds * Double(scale)).rounded())
    guard Double(value) / Double(scale) == seconds else {
      throw ProAppsError.invalid("Paired-detector request is not exact on the sample clock")
    }
    return CMTime(value: value, timescale: scale)
  }

  static func collect(
    image: CGImage, requested: CMTime, actual: CMTime,
    modern: [FaceObservation], revision: DetectFaceRectanglesRequest.Revision,
    reader: FaceReaderEvidence? = nil, readerPreservingSourceColor: FaceReaderEvidence? = nil,
    legacyDetector: (CGImage) throws -> LegacyFaceDetections = nativeLegacy
  ) throws -> FaceDetectorComparison {
    guard requested.isNumeric, actual.isNumeric, requested == actual else {
      throw ProAppsError.invalid("Paired-detector actual PTS differs from its exact request")
    }
    for case let evidence? in [reader, readerPreservingSourceColor] {
      try FaceReaderEvidence.validateTime(
        requested: requested,
        actual: CMTime(
          value: evidence.actualPresentationValue, timescale: evidence.actualPresentationTimescale))
    }
    try FrameProbe.validateGeometry(
      size: CGSize(width: image.width, height: image.height), transform: .identity)
    try Task.checkCancellation()
    let legacy = try legacyDetector(image)
    guard legacy.observations.count <= FaceTrackingSequence.maximumFaces,
      modern.count <= FaceTrackingSequence.maximumFaces
    else { throw ProAppsError.outputLimit }
    let size = CGSize(width: image.width, height: image.height)
    let old = try legacy.observations.map {
      try FaceDetectorCandidate(box: $0.boundingBox, confidence: $0.confidence, size: size)
    }
    let new = try modern.map {
      try FaceDetectorCandidate(box: $0.boundingBox.cgRect, confidence: $0.confidence, size: size)
    }
    try Task.checkCancellation()
    return FaceDetectorComparison(
      requestedPresentationValue: requested.value,
      requestedPresentationTimescale: requested.timescale,
      imageProducer: "AVAssetImageGenerator; preferred transform applied; orientation up",
      legacyRevision: legacy.revision, modernRevision: revision,
      legacy: old, modern: new,
      operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
      generatorBitmap: FaceBitmapMetadata(image), reader: reader,
      readerPreservingSourceColor: readerPreservingSourceColor)
  }

  static func nativeLegacy(_ image: CGImage) throws -> LegacyFaceDetections {
    let request = VNDetectFaceRectanglesRequest()
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    guard let observations = request.results else {
      throw ProAppsError.unavailable("Legacy face detector returned no result array")
    }
    return LegacyFaceDetections(observations: observations, revision: request.revision)
  }
}

struct LegacyFaceDetections {
  let observations: [VNFaceObservation]
  let revision: Int
}
