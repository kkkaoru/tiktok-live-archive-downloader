import CoreImage
import Dispatch
import Foundation
import Vision

/// Geometry-only diagnostics: no recognized text, paths, or image payloads.
public struct TextRectangleDiagnostics: Codable, Sendable {
  public let requestRevision: Int
  public let observationCount: Int
  public let charactersPerObservation: [Int]
  /// Zero-area native text envelopes; their measured characters remain available.
  public let degenerateTextObservationCount: Int?
  public let bitsPerComponent: Int
  public let bitsPerPixel: Int
  public let bytesPerRow: Int
  public let colorSpaceName: String?
}

public struct TextRectangleMeasurement: Codable, Sendable {
  public let width: Int
  public let height: Int
  public let characterBoxes: [EditCrop]
  /// Measured Vision text observations, never OCR-derived or inferred envelopes.
  public let textBoxes: [EditCrop]?
  public let diagnostics: TextRectangleDiagnostics?
}

/// Detect geometry only: neither OCR nor proof that every subtitle was found.
public actor TextRectangleProbe {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.text-rectangles")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }
  private let context = CIContext()
  public init() {}

  /// Sequential bounded batches reuse one actor/context without retaining decoded images.
  public func detect(paths: [String]) throws -> [TextRectangleMeasurement] {
    guard (1...8).contains(paths.count) else {
      throw ProAppsError.invalid("Supply 1–8 subtitle ROI paths")
    }
    return try paths.map { path in
      try autoreleasepool { try detect(path: path) }
    }
  }

  public func detect(path: String) throws -> TextRectangleMeasurement {
    try Task.checkCancellation()
    let data = try Files.read(Files.existing(path, extensions: ["png"]))
    guard let image = CIImage(data: data), image.extent.origin == .zero,
      image.extent.width >= 8, image.extent.height >= 8,
      image.extent.width * image.extent.height <= 2_000_000,
      let bitmap = context.createCGImage(image, from: image.extent)
    else { throw ProAppsError.invalid("Invalid or excessive subtitle ROI image") }
    let request = VNDetectTextRectanglesRequest()
    request.reportCharacterBoxes = true
    try VNImageRequestHandler(cgImage: bitmap, orientation: .up).perform([request])
    try Task.checkCancellation()
    guard let observations = request.results, observations.count <= 64 else {
      throw ProAppsError.outputLimit
    }
    var boxes: [EditCrop] = []
    var textBoxes: [EditCrop] = []
    var characterCounts: [Int] = []
    var degenerateTextObservations = 0
    for observation in observations {
      guard let characters = observation.characterBoxes, !characters.isEmpty,
        boxes.count + characters.count <= 4096
      else { throw ProAppsError.invalid("Missing or excessive character boxes") }
      characterCounts.append(characters.count)
      if let region = try Self.textBox(
        observation.boundingBox, width: bitmap.width, height: bitmap.height)
      {
        textBoxes.append(region)
      } else {
        degenerateTextObservations += 1
      }
      for character in characters {
        boxes.append(
          try Self.pixelBox(character.boundingBox, width: bitmap.width, height: bitmap.height))
      }
    }
    return TextRectangleMeasurement(
      width: bitmap.width, height: bitmap.height, characterBoxes: boxes, textBoxes: textBoxes,
      diagnostics: TextRectangleDiagnostics(
        requestRevision: request.revision, observationCount: observations.count,
        charactersPerObservation: characterCounts,
        degenerateTextObservationCount: degenerateTextObservations,
        bitsPerComponent: bitmap.bitsPerComponent,
        bitsPerPixel: bitmap.bitsPerPixel, bytesPerRow: bitmap.bytesPerRow,
        colorSpaceName: bitmap.colorSpace?.name.map { $0 as String }))
  }

  /// Vision can return a zero-area text envelope with valid measured characters.
  /// Preserve that distinction; never inflate the envelope or infer a replacement.
  static func textBox(_ box: CGRect, width: Int, height: Int) throws -> EditCrop? {
    if box.width > 0 && box.height > 0 {
      return try pixelBox(box, width: width, height: height)
    }
    guard box.minX.isFinite, box.minY.isFinite, box.width.isFinite, box.height.isFinite,
      box.minX >= 0, box.minY >= 0, box.width >= 0, box.height >= 0,
      box.maxX <= 1, box.maxY <= 1
    else { throw ProAppsError.invalid("Invalid Vision text rectangle") }
    return nil
  }

  /// Vision uses normalized bottom-left coordinates; masks use top-left pixels.
  static func pixelBox(_ box: CGRect, width: Int, height: Int) throws -> EditCrop {
    guard box.minX.isFinite, box.minY.isFinite, box.width.isFinite, box.height.isFinite,
      box.minX >= 0, box.minY >= 0, box.width > 0, box.height > 0,
      box.maxX <= 1, box.maxY <= 1
    else { throw ProAppsError.invalid("Invalid Vision character rectangle") }
    return EditCrop(
      x: box.minX * Double(width), y: (1 - box.maxY) * Double(height),
      width: box.width * Double(width), height: box.height * Double(height))
  }
}
