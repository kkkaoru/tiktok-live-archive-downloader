import CoreGraphics
import CoreML
import Foundation
import Vision

/// A measured face rectangle, not a person's identity or a hair boundary.
public struct TrackedFace: Codable, Sendable {
  public let region: EditCrop
  public let confidence: Float
}

/// Apple's face-detection/object-tracking sequence, confined to one serial worker.
/// https://developer.apple.com/documentation/vision/tracking-the-user-s-face-in-real-time
final class FaceTrackingSequence {
  static let minimumConfidence: Float = 0.3
  static let maximumFaces = 32
  private let detectFaces: (CGImage) throws -> [VNFaceObservation]
  private let performTracking:
    (VNSequenceRequestHandler, [VNTrackObjectRequest], CGImage) throws -> Void
  private var handler = VNSequenceRequestHandler()
  private var requests: [VNTrackObjectRequest] = []
  private(set) var outOfFrameTrackingObservations = 0
  private(set) var outOfFrameDetectionObservations = 0
  var captureObjectEvidence = false
  private(set) var objectEvidence: [FaceObjectEvidence] = []
  private var objectPresentation: FaceObjectPresentation?
  private var objectSeeds: [ObjectIdentifier: FaceObjectSeed] = [:]

  func beginFrame(presentation: FaceObjectPresentation? = nil) {
    outOfFrameTrackingObservations = 0
    outOfFrameDetectionObservations = 0
    objectEvidence = []
    objectPresentation = presentation
  }

  init(
    detectFaces: @escaping (CGImage) throws -> [VNFaceObservation] = FaceTrackingSequence
      .nativeDetections,
    performTracking:
      @escaping (VNSequenceRequestHandler, [VNTrackObjectRequest], CGImage) throws -> Void =
      FaceTrackingSequence.nativeTracking
  ) {
    self.detectFaces = detectFaces
    self.performTracking = performTracking
  }

  static func nativeTracking(
    handler: VNSequenceRequestHandler, requests: [VNTrackObjectRequest], image: CGImage
  ) throws {
    try handler.perform(requests, on: image)
  }

  static func nativeDetections(in image: CGImage) throws -> [VNFaceObservation] {
    let request = VNDetectFaceRectanglesRequest()
    try configureCPU(request, supported: request.supportedComputeStageDevices)
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    guard let observations = request.results else {
      throw ProAppsError.unavailable("Vision detection returned no result array")
    }
    return observations
  }

  /// Select only CPU devices explicitly advertised by every native compute stage.
  /// Unsupported requests fail before processing instead of falling back to Metal.
  static func configureCPU(
    _ request: VNRequest, supported: [VNComputeStage: [MLComputeDevice]]
  ) throws {
    guard !supported.isEmpty else {
      throw ProAppsError.unavailable("Vision request exposes no CPU compute stages")
    }
    for (stage, devices) in supported {
      guard let cpu = devices.first(where: { if case .cpu = $0 { true } else { false } }) else {
        throw ProAppsError.unavailable("Vision compute stage does not support CPU execution")
      }
      request.setComputeDevice(cpu, for: stage)
    }
  }

  func detect(in image: CGImage) throws -> [TrackedFace] {
    let observations = try NativeOperationFailure.perform(.faceDetection) {
      try detectFaces(image)
    }
    guard observations.count <= Self.maximumFaces else { throw ProAppsError.outputLimit }
    var accepted: [VNFaceObservation] = []
    var measurements: [TrackedFace] = []
    for observation in observations {
      try capture(observation, in: image, stage: .detection, seed: nil)
      guard Self.accepts(observation.confidence) else { continue }
      if Self.isOutOfFrame(observation.boundingBox) {
        outOfFrameDetectionObservations += 1
        continue
      }
      // Malformed geometry still fails before initializing native tracking.
      measurements.append(try Self.measure(observation, in: image))
      accepted.append(observation)
    }
    if !accepted.isEmpty { try seed(accepted, in: image) }
    return measurements
  }

  /// Called only with bounded, nonempty, validated detector observations.
  private func seed(_ observations: [VNFaceObservation], in image: CGImage) throws {
    handler = VNSequenceRequestHandler()
    objectSeeds = [:]
    requests = try observations.map {
      let request = VNTrackObjectRequest(detectedObjectObservation: $0)
      request.trackingLevel = .accurate
      try Self.configureCPU(request, supported: request.supportedComputeStageDevices)
      if captureObjectEvidence {
        guard let presentation = objectPresentation else {
          throw ProAppsError.invalid("Raw object evidence requires actual source PTS")
        }
        objectSeeds[ObjectIdentifier(request)] = FaceObjectSeed(
          observationUUID: $0.uuid, presentation: presentation)
      }
      return request
    }
    // Initialize appearance on the detection frame, not on an unrelated next frame.
    _ = try track(in: image, operation: .trackingSeed)
  }

  func track(
    in image: CGImage, operation: NativeOperationFailure.Operation = .trackingAdvance
  ) throws -> [TrackedFace] {
    guard !requests.isEmpty else { return [] }
    try NativeOperationFailure.perform(operation) {
      try performTracking(handler, requests, image)
    }
    var next: [VNTrackObjectRequest] = []
    var result: [TrackedFace] = []
    for request in requests {
      if request.isLastFrame {
        objectSeeds.removeValue(forKey: ObjectIdentifier(request))
        continue
      }
      guard let observation = request.results?.first as? VNDetectedObjectObservation else {
        throw ProAppsError.unavailable("Vision tracking returned no valid object observation")
      }
      if let measured = try advance(
        request, observation: observation, in: image, operation: operation)
      {
        result.append(measured)
      }
      next.append(request)
    }
    requests = next
    if requests.allSatisfy(\.isLastFrame) {
      // No surviving observation means no state remains to advance. Release the
      // exhausted sequence; only an actual detection may seed another one.
      requests = []
      objectSeeds = [:]
      handler = VNSequenceRequestHandler()
    }
    return result
  }

  /// Off-image tracking is loss, not a clipped face measurement. Preserve an
  /// explicit per-frame rejection count; malformed extents still fail closed.
  func advance(
    _ request: VNTrackObjectRequest, observation: VNDetectedObjectObservation, in image: CGImage,
    operation: NativeOperationFailure.Operation = .trackingAdvance
  ) throws -> TrackedFace? {
    try capture(
      observation, in: image, stage: FaceObjectStage(operation: operation),
      seed: objectSeeds[ObjectIdentifier(request)])
    guard Self.accepts(observation.confidence) else {
      request.isLastFrame = true
      return nil
    }
    if Self.isOutOfFrame(observation.boundingBox) {
      request.isLastFrame = true
      outOfFrameTrackingObservations += 1
      return nil
    }
    let measured = try Self.measure(observation, in: image, method: .tracking)
    request.inputObservation = observation
    return measured
  }

  private func capture(
    _ observation: VNDetectedObjectObservation, in image: CGImage, stage: FaceObjectStage,
    seed: FaceObjectSeed?
  ) throws {
    guard captureObjectEvidence else { return }
    guard let presentation = objectPresentation else {
      throw ProAppsError.invalid("Raw object evidence requires actual source PTS")
    }
    // 32 advancing tracks + 32 detections + 32 newly seeded tracks per source frame.
    guard objectEvidence.count < Self.maximumFaces * 3 else { throw ProAppsError.outputLimit }
    objectEvidence.append(
      try FaceObjectEvidence(
        observation: observation, presentation: presentation, stage: stage, seed: seed,
        width: image.width, height: image.height))
  }

  private static func isOutOfFrame(_ box: CGRect) -> Bool {
    box.minX.isFinite && box.minY.isFinite && box.width.isFinite && box.height.isFinite
      && box.maxX.isFinite && box.maxY.isFinite && box.width > 0 && box.height > 0
      && (box.minX < 0 || box.minY < 0 || box.maxX > 1 || box.maxY > 1)
  }

  static func accepts(_ confidence: Float) -> Bool {
    confidence.isFinite && confidence > minimumConfidence && confidence <= 1
  }

  static func measure(
    _ observation: VNDetectedObjectObservation, in image: CGImage,
    method: FaceObservationMethod = .detection
  ) throws -> TrackedFace {
    let box = observation.boundingBox
    guard box.minX.isFinite, box.minY.isFinite, box.width.isFinite, box.height.isFinite,
      box.width > 0, box.height > 0, box.minX >= 0, box.minY >= 0,
      box.maxX <= 1, box.maxY <= 1, observation.confidence.isFinite,
      (0...1).contains(observation.confidence)
    else {
      throw ProAppsError.invalid(
        "Vision face observation exceeds the decoded image; method=\(method.rawValue) "
          + "normalizedBox=(\(box.minX),\(box.minY),\(box.width),\(box.height)) "
          + "confidence=\(observation.confidence)")
    }
    return TrackedFace(
      region: EditCrop(
        x: box.minX * Double(image.width), y: (1 - box.maxY) * Double(image.height),
        width: box.width * Double(image.width), height: box.height * Double(image.height)),
      confidence: observation.confidence)
  }
}
