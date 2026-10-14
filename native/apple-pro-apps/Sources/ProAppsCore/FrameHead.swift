import AVFoundation
import Foundation
import Vision

/// Sampled display-oriented rectangles, not identity or exhaustive tracking.
public struct FrameHeadMeasurement: Codable, Sendable {
  public let requestedTimeSeconds: Double
  public let actualTimeSeconds: Double
  public let actualPresentationValue: Int64
  public let actualPresentationTimescale: Int32
  public let boundsClipped: Bool
  public let width: Int
  public let height: Int
  public let faces: [EditCrop]
  public let bodies: [EditCrop]
  public let faceDetectorComparison: FaceDetectorComparison?
}

extension FrameProbe {
  /// Detect faces and whole people locally. Empty arrays retain missing detections.
  /// Uses the same bounded, exact-time decoder as OCR; never chooses a subject.
  public func detectHeads(
    path: String, samples: [FrameSample], compareFaceDetectors: Bool = false
  ) async throws
    -> [FrameHeadMeasurement]
  {
    guard samples.allSatisfy({ $0.region == nil }) else {
      throw ProAppsError.invalid("Head detection requires full-frame samples")
    }
    let generator = try await source(path: path, samples: samples)
    var result: [FrameHeadMeasurement] = []
    for sample in samples {
      try Task.checkCancellation()
      let requested =
        compareFaceDetectors
        ? try FaceDetectorComparison.exactTime(seconds: sample.timeSeconds)
        : CMTime(seconds: sample.timeSeconds, preferredTimescale: EditPlan.timeScale)
      let frame = try await generator.image(at: requested)
      let size = CGSize(width: frame.image.width, height: frame.image.height)
      let faceRequest = DetectFaceRectanglesRequest()
      let faces = try await faceRequest.perform(on: frame.image)
      let comparison: FaceDetectorComparison?
      if compareFaceDetectors {
        try FaceReaderEvidence.validateTime(requested: requested, actual: frame.actualTime)
        let reader = try await readerFaceEvidence(path: path, seconds: sample.timeSeconds)
        let sourceColor = try await readerFaceEvidence(
          path: path, seconds: sample.timeSeconds, preservingSourceColor: true)
        comparison = try FaceDetectorComparison.collect(
          image: frame.image, requested: requested, actual: frame.actualTime,
          modern: faces, revision: faceRequest.revision, reader: reader,
          readerPreservingSourceColor: sourceColor)
      } else {
        comparison = nil
      }
      var request = DetectHumanRectanglesRequest()
      request.upperBodyOnly = false
      let bodies = try await request.perform(on: frame.image)
      try Task.checkCancellation()
      let faceRegions = try Self.headRegions(faces, size: size)
      let bodyRegions = try Self.headRegions(bodies, size: size)
      result.append(
        FrameHeadMeasurement(
          requestedTimeSeconds: sample.timeSeconds, actualTimeSeconds: frame.actualTime.seconds,
          actualPresentationValue: frame.actualTime.value,
          actualPresentationTimescale: frame.actualTime.timescale,
          boundsClipped: faceRegions.boundsClipped || bodyRegions.boundsClipped,
          width: frame.image.width, height: frame.image.height,
          faces: faceRegions.regions, bodies: bodyRegions.regions,
          faceDetectorComparison: comparison)
      )
    }
    return result
  }

  static func headRegions<Observation: BoundingBoxProviding>(
    _ observations: [Observation], size: CGSize
  ) throws -> (regions: [EditCrop], boundsClipped: Bool) {
    guard observations.count <= 32 else { throw ProAppsError.outputLimit }
    var regions: [EditCrop] = []
    var boundsClipped = false
    let imageBounds = CGRect(origin: .zero, size: size)
    for observation in observations {
      let raw = observation.boundingBox.toImageCoordinates(size, origin: .upperLeft)
      let box = try textRectangle(raw, imageSize: size)
      // Intersection can normalize a fully contained width by an IEEE-754 ULP.
      // Record boundary clipping, not incidental arithmetic normalization.
      boundsClipped = boundsClipped || !imageBounds.contains(raw)
      regions.append(EditCrop(x: box.minX, y: box.minY, width: box.width, height: box.height))
    }
    return (regions, boundsClipped)
  }
}
