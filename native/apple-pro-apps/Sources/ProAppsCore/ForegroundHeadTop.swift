import CoreImage
import Foundation
import Vision

/// Explicit same-source face observation. Caller binds source and detector provenance.
public struct HeadTopFaceRequest: Decodable, Sendable {
  public let presentation: ForegroundTime
  public let x: Int
  public let y: Int
  public let width: Int
  public let height: Int
  public let minimumFaceCoverage: Double

  var bounds: ForegroundPixelBounds {
    .init(x: x, y: y, width: width, height: height)
  }
}

struct ForegroundHeadTopResult: Sendable {
  let observation: ForegroundObservation?
  let candidate: HeadTopCandidate
}

enum HeadTopRasterStorage {
  static func validateExtent(_ extent: CGRect, width: Int, height: Int) throws {
    guard extent == CGRect(x: 0, y: 0, width: width, height: height) else {
      throw ProAppsError.invalid("Invalid source-sized head-top mask extent")
    }
  }
  static func validateLabels(_ labels: IndexSet) throws {
    guard labels.count <= 256, labels.allSatisfy({ $0 > 0 }) else {
      throw ProAppsError.invalid("Foreground head-top instance labels exceed bounds")
    }
  }

  static func raster(
    label: Int, bitmap: CGImage?, width: Int, height: Int, retainedBytes: inout Int
  ) throws -> HeadTopMask {
    guard let bitmap else { throw ProAppsError.invalid("Missing source-sized head-top mask") }
    _ = try ForegroundMaskStatistics.bitmap(bitmap, width: width, height: height)
    guard let provider = bitmap.dataProvider?.data else {
      throw ProAppsError.unavailable("Missing head-top bitmap storage")
    }
    let size = try ForegroundMaskStatistics.storageSize(
      width: width, height: height, bytesPerRow: bitmap.bytesPerRow)
    let storage = provider as Data
    guard retainedBytes >= 0, retainedBytes <= ForegroundMaskStatistics.maximumBytes,
      storage.count <= ForegroundMaskStatistics.maximumBytes - retainedBytes
    else { throw ProAppsError.invalid("Head-top provider storage exceeds cumulative budget") }
    retainedBytes += storage.count
    return HeadTopMask(label: label, pixels: storage.prefix(size), bytesPerRow: bitmap.bytesPerRow)
  }
}

extension FrameProbe {
  /// Opt-in inference with union and individual masks from one observation.
  /// No mask is exported. Result is a geometric candidate, not semantic hair detection.
  func foregroundHeadTop(image: CGImage, context: CIContext, face: HeadTopFaceRequest) async throws
    -> ForegroundHeadTopResult
  {
    let handler = ImageRequestHandler(image, orientation: .up)
    let observation = try await handler.perform(GenerateForegroundInstanceMaskRequest(.revision1))
    try Task.checkCancellation()
    var masks: [HeadTopMask] = []
    var union: ForegroundObservation?
    var retainedBytes = 0
    if let observation {
      try HeadTopRasterStorage.validateLabels(observation.allInstances)
      union = try ForegroundObservation.collect(labels: observation.allInstances) {
        let pixels = try observation.generateScaledMask(
          for: observation.allInstances, scaledToImageFrom: handler)
        return try ForegroundMaskStatistics.measure(
          CIImage(cvPixelBuffer: pixels), width: image.width, height: image.height, context: context
        )
      }
      for label in observation.allInstances {
        try Task.checkCancellation()
        let pixels = try observation.generateScaledMask(
          for: IndexSet(integer: label), scaledToImageFrom: handler)
        let mask = CIImage(cvPixelBuffer: pixels)
        try HeadTopRasterStorage.validateExtent(
          mask.extent, width: image.width, height: image.height)
        let bitmap = context.createCGImage(
          mask, from: mask.extent, format: .L8, colorSpace: CGColorSpaceCreateDeviceGray())
        // Reuse the existing layout and provider-size validation, including padding.
        masks.append(
          try HeadTopRasterStorage.raster(
            label: label, bitmap: bitmap, width: image.width, height: image.height,
            retainedBytes: &retainedBytes))
      }
    }
    let candidate = try HeadTopMeasurement.measure(
      masks: masks, width: image.width, height: image.height, face: face.bounds,
      minimumFaceCoverage: face.minimumFaceCoverage)
    return ForegroundHeadTopResult(observation: union, candidate: candidate)
  }
}
