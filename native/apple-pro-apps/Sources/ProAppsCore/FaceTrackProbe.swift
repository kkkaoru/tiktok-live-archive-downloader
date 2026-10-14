import AVFoundation
import CoreVideo
import Dispatch
import Foundation

/// Explicit resource limits for one sequential, source-clock tracking window.
public struct FaceTrackingPlan: Codable, Sendable {
  public static let frameLimit = 7200
  public static let durationLimit = 120.0
  public static let maximumTimescale: Int32 = 1_000_000_000
  public let startSeconds: Double
  public let durationSeconds: Double
  public let maximumFrames: Int
  public let redetectIntervalSeconds: Double
  /// The verified source clock denominator; nil retains the legacy edit grid.
  public let sourceTimescale: Int32?

  public init(
    startSeconds: Double, durationSeconds: Double,
    maximumFrames: Int = FaceTrackingPlan.frameLimit,
    redetectIntervalSeconds: Double = 0.5,
    sourceTimescale: Int32? = nil
  ) {
    self.startSeconds = startSeconds
    self.durationSeconds = durationSeconds
    self.maximumFrames = maximumFrames
    self.redetectIntervalSeconds = redetectIntervalSeconds
    self.sourceTimescale = sourceTimescale
  }

  @discardableResult
  func validate(sourceDuration: Double) throws -> CMTimeRange {
    let maximumSourceDuration = 86_400.0
    let redetectionIntervalRange = 0.1...10.0
    let timescale = sourceTimescale ?? EditPlan.timeScale
    guard (1...Self.maximumTimescale).contains(timescale), sourceDuration.isFinite,
      sourceDuration > 0, sourceDuration <= maximumSourceDuration,
      startSeconds.isFinite, startSeconds >= 0, durationSeconds.isFinite,
      durationSeconds > 0, durationSeconds <= Self.durationLimit,
      startSeconds + durationSeconds <= sourceDuration,
      (1...Self.frameLimit).contains(maximumFrames), redetectIntervalSeconds.isFinite,
      redetectionIntervalRange.contains(redetectIntervalSeconds)
    else {
      throw ProAppsError.invalid("Face tracking window exceeds its source or resource budget")
    }
    // Recover integer grid coordinates, then require an exact floating-input
    // round trip. CMTime's seconds conversion can differ by an IEEE-754 ULP.
    let startValue = Int64((startSeconds * Double(timescale)).rounded())
    let durationValue = Int64((durationSeconds * Double(timescale)).rounded())
    guard Double(startValue) / Double(timescale) == startSeconds,
      Double(durationValue) / Double(timescale) == durationSeconds
    else {
      throw ProAppsError.invalid("Face tracking window is not representable on its source clock")
    }
    return CMTimeRange(
      start: CMTime(value: startValue, timescale: timescale),
      duration: CMTime(value: durationValue, timescale: timescale))
  }
}

public enum FaceObservationMethod: String, Codable, Sendable {
  case detection, tracking, missing
}

public struct FaceTrackingFrame: Codable, Sendable {
  public let presentationValue: Int64
  public let presentationTimescale: Int32
  public let width: Int
  public let height: Int
  public let method: FaceObservationMethod
  public let detectionAttempted: Bool
  public let outOfFrameTrackingObservations: Int
  public let outOfFrameDetectionObservations: Int
  public let faces: [TrackedFace]
}

/// AVAssetReader replaces the camera input in Apple's face-tracking example.
/// No camera, UI, model transfer, exported pixels or source modification is used.
public actor FaceTrackProbe {
  private let makeSequence: @Sendable () -> FaceTrackingSequence
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.face-tracking")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }

  public init() { makeSequence = { FaceTrackingSequence() } }

  // Injectable detection boundary for synthetic integration tests. Every request
  // still owns a fresh sequence; no mutable Vision state crosses worker isolation.
  init(makeSequence: @escaping @Sendable () -> FaceTrackingSequence) {
    self.makeSequence = makeSequence
  }

  public func track(path: String, plan: FaceTrackingPlan) async throws -> [FaceTrackingFrame] {
    try await trackFrames(path: path, plan: plan, objectEvidence: false).map(\.legacy.faceFrame)
  }

  /// Record raw object evidence separately; legacy face acceptance is unchanged.
  public func trackObjects(path: String, plan: FaceTrackingPlan) async throws
    -> [FaceObjectTrackingFrame]
  {
    try await trackFrames(path: path, plan: plan, objectEvidence: true).map(\.legacy)
  }

  /// Independent current-image object support. Partial objects never enter legacy faces.
  public func trackPartialObjects(
    path: String, plan: FaceTrackingPlan, partialPlan: PartialObjectTrackingPlan
  ) async throws -> [PartialObjectTrackingFrame] {
    try await trackFrames(
      path: path, plan: plan, objectEvidence: true, partialPlan: partialPlan)
  }

  private func trackFrames(
    path: String, plan: FaceTrackingPlan, objectEvidence: Bool,
    partialPlan: PartialObjectTrackingPlan? = nil
  ) async throws -> [PartialObjectTrackingFrame] {
    try Task.checkCancellation()
    let asset = AVURLAsset(url: try Files.existing(path))
    let timeRange = try plan.validate(sourceDuration: try await asset.load(.duration).seconds)
    try partialPlan?.validate(window: timeRange)
    let tracks = try await asset.loadTracks(withMediaType: .video)
    guard tracks.count == 1, let track = tracks.first else {
      throw ProAppsError.invalid("Face tracking requires exactly one video track")
    }
    let size = try await track.load(.naturalSize)
    let transform = try await track.load(.preferredTransform)
    try FrameProbe.validateGeometry(size: size, transform: transform)
    try Task.checkCancellation()
    return try TrackingFrameDecoder.withReader(
      asset: asset, track: track, timeRange: timeRange
    ) { reader, output in
      let result = try consume(
        output, plan: plan, transform: transform, timeRange: timeRange,
        objectEvidence: objectEvidence, partialPlan: partialPlan)
      guard reader.status == .completed, !result.isEmpty else {
        throw ProAppsError.unavailable("Face-tracking decoder did not complete its window")
      }
      return result
    }
  }

  private func consume(
    _ output: AVAssetReaderTrackOutput, plan: FaceTrackingPlan, transform: CGAffineTransform,
    timeRange: CMTimeRange, objectEvidence: Bool, partialPlan: PartialObjectTrackingPlan?
  ) throws -> [PartialObjectTrackingFrame] {
    let sequence = makeSequence()
    sequence.captureObjectEvidence = objectEvidence
    let partialSequence = partialPlan.map { PartialObjectTrackingSequence(policy: $0) }
    var frames: [PartialObjectTrackingFrame] = []
    var objectCount = 0
    var partialCount = 0
    var previousTime: CMTime?
    var lastDetectionTime: Double?
    while let sample = output.copyNextSampleBuffer() {
      try Task.checkCancellation()
      guard frames.count < plan.maximumFrames, CMSampleBufferIsValid(sample),
        CMSampleBufferDataIsReady(sample), let pixels = CMSampleBufferGetImageBuffer(sample)
      else { throw ProAppsError.invalid("Invalid frame or face-tracking frame budget exceeded") }
      let time = CMSampleBufferGetPresentationTimeStamp(sample)
      guard time.isNumeric, time.timescale > 0, time >= timeRange.start,
        time < timeRange.end,
        previousTime.map({ time > $0 }) ?? true
      else { throw ProAppsError.invalid("Face-tracking decoder returned invalid or unordered PTS") }
      previousTime = time
      let frame: PartialObjectTrackingFrame
      do {
        frame = try autoreleasepool {
          let bitmap = try FaceTrackingBitmap.render(pixels: pixels, transform: transform)
          sequence.beginFrame(
            presentation: FaceObjectPresentation(value: time.value, timescale: time.timescale))
          var faces = try sequence.track(in: bitmap)
          var method: FaceObservationMethod = faces.isEmpty ? .missing : .tracking
          let detectionDue =
            lastDetectionTime.map {
              time.seconds - $0 >= plan.redetectIntervalSeconds
            } ?? true
          // Retry a lost observation immediately, but do not run the detector on
          // every frame of a still-unobserved interval. Missing geometry stays missing.
          let lostObservation =
            faces.count < (frames.last?.legacy.faceFrame.faces.count ?? 0)
            || sequence.outOfFrameTrackingObservations > 0
          let detectionAttempted = detectionDue || lostObservation
          if detectionAttempted {
            let detected = try sequence.detect(in: bitmap)
            lastDetectionTime = time.seconds
            if !detected.isEmpty {
              faces = detected
              method = .detection
            }
          }
          let faceFrame = FaceTrackingFrame(
            presentationValue: time.value, presentationTimescale: time.timescale,
            width: bitmap.width, height: bitmap.height, method: method,
            detectionAttempted: detectionAttempted,
            outOfFrameTrackingObservations: sequence.outOfFrameTrackingObservations,
            outOfFrameDetectionObservations: sequence.outOfFrameDetectionObservations, faces: faces)
          let legacy = FaceObjectTrackingFrame(
            faceFrame: faceFrame, objects: sequence.objectEvidence)
          let partials = try partialSequence?.process(image: bitmap, legacy: legacy) ?? []
          return PartialObjectTrackingFrame(legacy: legacy, partialObjects: partials)
        }
      } catch let failure as NativeOperationFailure {
        throw failure.at(time)
      } catch ProAppsError.invalid(let reason) {
        throw ProAppsError.invalid(
          "\(reason); sourcePTS=\(time.value)/\(time.timescale)")
      }
      objectCount = try FaceObjectEvidence.count(
        after: objectCount, adding: frame.legacy.objects.count)
      partialCount = try FaceObjectEvidence.count(
        after: partialCount, adding: frame.partialObjects.count)
      frames.append(frame)
    }
    try Task.checkCancellation()
    try partialSequence?.finish()
    return frames
  }
}
