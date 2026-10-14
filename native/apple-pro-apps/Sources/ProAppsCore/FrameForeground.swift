import AVFoundation
import CoreImage
import Foundation
import Vision

/// Observation is injectable; the non-Sendable generator remains operation-local.
struct ForegroundOperations: Sendable {
  var observe:
    @Sendable (isolated FrameProbe, CGImage, CIContext) async throws -> ForegroundObservation? = {
      probe, image, context in
      try await probe.foregroundObservation(image: image, context: context)
    }
  var head:
    @Sendable (isolated FrameProbe, CGImage, CIContext, HeadTopFaceRequest) async throws ->
      ForegroundHeadTopResult = {
        probe, image, context, face in
        try await probe.foregroundHeadTop(image: image, context: context, face: face)
      }
}

extension FrameProbe {
  /// Explicit rational samples only. No tracking, image export, or placement decision.
  public func measureForeground(
    path: String, samples: [ForegroundTime], headFace: HeadTopFaceRequest? = nil,
    headFaces: [HeadTopFaceRequest]? = nil
  ) async throws
    -> [ForegroundMeasurement]
  {
    try await measureForeground(
      path: path, samples: samples, operations: ForegroundOperations(), headFace: headFace,
      headFaces: headFaces)
  }

  func measureForeground(
    path: String, samples: [ForegroundTime], operations: ForegroundOperations,
    headFace: HeadTopFaceRequest? = nil, headFaces: [HeadTopFaceRequest]? = nil
  ) async throws -> [ForegroundMeasurement] {
    try Task.checkCancellation()
    let times = try ForegroundTime.validate(samples)
    let associatedFaces = try Self.associatedFaces(
      headFace: headFace, headFaces: headFaces, times: times)
    // Double seconds are only source() preflight inputs, never acquisition clocks.
    let generator = try await source(
      path: path, samples: times.map { FrameSample(timeSeconds: $0.seconds) })
    defer { generator.cancelAllCGImageGeneration() }
    try Task.checkCancellation()
    let context = CIContext()
    defer { context.clearCaches() }
    var results: [ForegroundMeasurement] = []
    var displaySize: CGSize?
    for (index, pair) in zip(samples, times).enumerated() {
      let (sample, requested) = pair
      let face = associatedFaces?[index]
      try Task.checkCancellation()
      let frame = try await generator.image(at: requested)
      try Task.checkCancellation()
      let actual = try ForegroundTime.actual(frame.actualTime, requested: requested)
      let size = CGSize(width: frame.image.width, height: frame.image.height)
      try Self.validateGeometry(size: size, transform: .identity)
      if let displaySize, displaySize != size {
        throw ProAppsError.invalid("Foreground display dimensions changed between samples")
      }
      displaySize = size
      if let face {
        _ = try HeadTopMeasurement.measure(
          masks: [], width: frame.image.width, height: frame.image.height,
          face: face.bounds, minimumFaceCoverage: face.minimumFaceCoverage)
      }
      try Task.checkCancellation()
      let observation: ForegroundObservation?
      let headTop: HeadTopCandidate?
      if let face {
        let measured = try await operations.head(self, frame.image, context, face)
        observation = measured.observation
        headTop = measured.candidate
      } else {
        observation = try await operations.observe(self, frame.image, context)
        headTop = nil
      }
      try Task.checkCancellation()
      let measurement = ForegroundMeasurement(
        requested: sample, actual: actual, width: frame.image.width, height: frame.image.height,
        observation: observation, headTop: headTop)
      results.append(measurement)
    }
    try Task.checkCancellation()
    return results
  }

  private static func associatedFaces(
    headFace: HeadTopFaceRequest?, headFaces: [HeadTopFaceRequest]?, times: [CMTime]
  ) throws -> [HeadTopFaceRequest]? {
    guard headFace == nil || headFaces == nil else {
      throw ProAppsError.invalid("headFace and headFaces are mutually exclusive")
    }
    let faces: [HeadTopFaceRequest]
    if let headFace {
      guard times.count == 1 else {
        throw ProAppsError.invalid("headFace requires one exact same-PTS face sample")
      }
      faces = [headFace]
    } else if let headFaces {
      guard headFaces.count == times.count else {
        throw ProAppsError.invalid("headFaces must match every requested sample")
      }
      faces = headFaces
    } else {
      return nil
    }
    for (face, requested) in zip(faces, times) {
      guard try face.presentation.time() == requested else {
        throw ProAppsError.invalid("Head-top face PTS differs from its requested sample")
      }
    }
    return faces
  }

  func foregroundObservation(image: CGImage, context: CIContext) async throws
    -> ForegroundObservation?
  {
    try Task.checkCancellation()
    let handler = ImageRequestHandler(image, orientation: .up)
    let request = GenerateForegroundInstanceMaskRequest(.revision1)
    let observation = try await handler.perform(request)
    try Task.checkCancellation()
    guard let observation else { return nil }
    return try ForegroundObservation.collect(labels: observation.allInstances) {
      try Task.checkCancellation()
      let pixels = try observation.generateScaledMask(
        for: observation.allInstances, scaledToImageFrom: handler)
      try Task.checkCancellation()
      return try ForegroundMaskStatistics.measure(
        CIImage(cvPixelBuffer: pixels), width: image.width, height: image.height, context: context)
    }
  }
}
