import CoreImage
import Dispatch
import Foundation
import Vision

/// Observed OCR context only: these rectangles are not repair masks or quality approval.
public struct ImageTextMeasurement: Codable, Sendable {
  public let width: Int
  public let height: Int
  public let lines: [FrameTextLine]
}

/// Read-only, sequential recognition of bounded local PNG regions, in request order.
public actor ImageTextProbe {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.image-text")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }
  private let context = CIContext()
  public init() {}

  public func recognize(paths: [String]) async throws -> [ImageTextMeasurement] {
    guard (1...8).contains(paths.count) else {
      throw ProAppsError.invalid("Supply 1–8 local PNG regions")
    }
    var result: [ImageTextMeasurement] = []
    var bytes = 0
    for path in paths {
      try Task.checkCancellation()
      let data = try Files.read(Files.existing(path, extensions: ["png"]))
      guard let image = CIImage(data: data), image.extent.origin == .zero,
        image.extent.width >= 8, image.extent.height >= 8,
        image.extent.width * image.extent.height <= 2_000_000,
        let bitmap = context.createCGImage(image, from: image.extent)
      else { throw ProAppsError.invalid("Invalid or excessive OCR region image") }
      var request = RecognizeTextRequest()
      request.recognitionLevel = .accurate
      request.recognitionLanguages = [
        Locale.Language(identifier: "ja-JP"), Locale.Language(identifier: "en-US"),
      ]
      request.usesLanguageCorrection = false
      let observations = try await request.perform(on: bitmap)
      guard observations.count <= 64 else { throw ProAppsError.outputLimit }
      var lines: [FrameTextLine] = []
      for observation in observations {
        try Task.checkCancellation()
        guard let candidate = observation.topCandidates(1).first else {
          throw ProAppsError.unavailable("OCR observation has no text candidate")
        }
        bytes += candidate.string.utf8.count
        guard bytes <= 32768 else { throw ProAppsError.outputLimit }
        let raw = observation.boundingBox.toImageCoordinates(
          CGSize(width: bitmap.width, height: bitmap.height), origin: .upperLeft)
        let rectangle = try FrameProbe.textRectangle(
          raw, imageSize: CGSize(width: bitmap.width, height: bitmap.height))
        lines.append(
          FrameTextLine(
            text: candidate.string, confidence: candidate.confidence,
            boundsClipped: rectangle != raw,
            region: EditCrop(
              x: rectangle.minX, y: rectangle.minY,
              width: rectangle.width, height: rectangle.height)))
      }
      result.append(
        ImageTextMeasurement(width: bitmap.width, height: bitmap.height, lines: lines))
    }
    return result
  }
}
