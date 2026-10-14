import AVFoundation
import CoreImage
import CoreVideo
import Foundation

/// Scalar bitmap provenance only; no pixels, paths, or recognized identities.
public struct FaceBitmapMetadata: Codable, Sendable {
  public let width: Int
  public let height: Int
  public let bitsPerComponent: Int
  public let bitsPerPixel: Int
  public let bytesPerRow: Int
  public let bitmapInfo: UInt32
  public let alphaInfo: UInt32
  public let colorSpaceName: String?
  public let colorSpaceModel: Int32?
  public let colorSpaceComponentCount: Int?

  init(_ image: CGImage) {
    width = image.width
    height = image.height
    bitsPerComponent = image.bitsPerComponent
    bitsPerPixel = image.bitsPerPixel
    bytesPerRow = image.bytesPerRow
    bitmapInfo = image.bitmapInfo.rawValue
    alphaInfo = image.alphaInfo.rawValue
    colorSpaceName = image.colorSpace?.name.map { $0 as String }
    colorSpaceModel = image.colorSpace?.model.rawValue
    colorSpaceComponentCount = image.colorSpace?.numberOfComponents
  }
}

public struct FaceAppliedTransform: Codable, Sendable {
  public let a: Double
  public let b: Double
  public let c: Double
  public let d: Double
  public let tx: Double
  public let ty: Double

  init(_ transform: CGAffineTransform) {
    a = Double(transform.a)
    b = Double(transform.b)
    c = Double(transform.c)
    d = Double(transform.d)
    tx = Double(transform.tx)
    ty = Double(transform.ty)
  }
}

/// One exact reader sample, not a complete decode or a tracking measurement.
public struct FaceReaderEvidence: Codable, Sendable {
  public let actualPresentationValue: Int64
  public let actualPresentationTimescale: Int32
  public let bitmap: FaceBitmapMetadata
  public let pixelFormat: UInt32
  public let appliedTransform: FaceAppliedTransform
  public let imageProducer: String
  public let contextConfiguration: String
  public let legacyRevision: Int
  public let legacy: [FaceDetectorCandidate]

  static func validateStatus(_ status: AVAssetReader.Status) throws {
    switch status {
    case .reading, .completed: return
    case .unknown, .failed, .cancelled:
      throw ProAppsError.unavailable("Comparison reader cannot supply a valid sample")
    @unknown default:
      throw ProAppsError.unavailable("Unknown comparison reader status")
    }
  }

  static func validateTime(requested: CMTime, actual: CMTime) throws {
    guard requested.isNumeric, actual.isNumeric,
      requested.timescale > 0, actual.timescale > 0,
      requested.epoch == 0, actual.epoch == 0,
      !requested.flags.contains(.hasBeenRounded), !actual.flags.contains(.hasBeenRounded),
      requested == actual
    else { throw ProAppsError.invalid("Comparison reader PTS differs from its exact request") }
  }

  static func collect(
    image: CGImage, requested: CMTime, actual: CMTime,
    pixelFormat: UInt32, transform: CGAffineTransform, preservingSourceColor: Bool = false,
    legacyDetector: (CGImage) throws -> LegacyFaceDetections = FaceDetectorComparison.nativeLegacy
  ) throws -> FaceReaderEvidence {
    try validateTime(requested: requested, actual: actual)
    guard pixelFormat == kCVPixelFormatType_32BGRA else {
      throw ProAppsError.invalid("Comparison reader did not produce 32BGRA")
    }
    let size = CGSize(width: image.width, height: image.height)
    try FrameProbe.validateGeometry(size: size, transform: .identity)
    try Task.checkCancellation()
    let detections = try legacyDetector(image)
    guard detections.observations.count <= FaceTrackingSequence.maximumFaces else {
      throw ProAppsError.outputLimit
    }
    let candidates = try detections.observations.map {
      try FaceDetectorCandidate(box: $0.boundingBox, confidence: $0.confidence, size: size)
    }
    try Task.checkCancellation()
    return FaceReaderEvidence(
      actualPresentationValue: actual.value, actualPresentationTimescale: actual.timescale,
      bitmap: FaceBitmapMetadata(image), pixelFormat: pixelFormat,
      appliedTransform: FaceAppliedTransform(transform),
      imageProducer: "AVAssetReader; 32BGRA; CIImage preferred transform; orientation up",
      contextConfiguration: preservingSourceColor
        ? "CIContext() defaults; RGBA8; source CIImage color space"
        : "CIContext() defaults; createCGImage(_:from:)",
      legacyRevision: detections.revision, legacy: candidates)
  }
}

/// Synchronous scoped use on the calling actor's existing serial executor.
/// Callers return scalar evidence, never decoder state or pixel buffers.
enum TrackingFrameDecoder {
  static func withReader<Result: Sendable>(
    asset: AVAsset, track: AVAssetTrack, timeRange: CMTimeRange,
    body: (AVAssetReader, AVAssetReaderTrackOutput) throws -> Result
  ) throws -> Result {
    try Task.checkCancellation()
    let reader = try AVAssetReader(asset: asset)
    defer { reader.cancelReading() }
    let output = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.timeRange = timeRange
    guard reader.canAdd(output) else {
      throw ProAppsError.unavailable("Cannot attach face-tracking decoder")
    }
    reader.add(output)
    guard reader.startReading() else {
      throw ProAppsError.unavailable("Cannot start face-tracking decoder")
    }
    return try body(reader, output)
  }

  static func bitmap(
    pixels: CVPixelBuffer, transform: CGAffineTransform, context: CIContext,
    preservingSourceColor: Bool = false
  ) throws -> CGImage {
    let source = CIImage(cvPixelBuffer: pixels)
    let image = source.transformed(by: transform)
    let rendered: CGImage?
    if preservingSourceColor {
      guard let colorSpace = source.colorSpace, colorSpace.model == .rgb else {
        throw ProAppsError.invalid("Comparison requires the decoded image's RGB color space")
      }
      rendered = context.createCGImage(
        image, from: image.extent, format: .RGBA8, colorSpace: colorSpace)
    } else {
      rendered = context.createCGImage(image, from: image.extent)
    }
    guard let bitmap = rendered else {
      throw ProAppsError.unavailable("Cannot prepare display-oriented tracking frame")
    }
    return bitmap
  }
}

extension FrameProbe {
  /// Copies only the first sample in one exact clock tick; never searches forward.
  func readerFaceEvidence(
    path: String, seconds: Double, preservingSourceColor: Bool = false
  ) async throws -> FaceReaderEvidence {
    let requested = try FaceDetectorComparison.exactTime(seconds: seconds)
    try Task.checkCancellation()
    let asset = AVURLAsset(url: try Files.existing(path))
    let duration = try await asset.load(.duration)
    let tracks = try await asset.loadTracks(withMediaType: .video)
    guard duration.isNumeric, duration.epoch == 0, duration > requested,
      tracks.count == 1, let track = tracks.first
    else {
      throw ProAppsError.invalid("Comparison reader requires a valid sample and one video track")
    }
    let size = try await track.load(.naturalSize)
    let transform = try await track.load(.preferredTransform)
    try Self.validateGeometry(size: size, transform: transform)
    let end = min(duration, requested + CMTime(value: 1, timescale: requested.timescale))
    return try TrackingFrameDecoder.withReader(
      asset: asset, track: track, timeRange: CMTimeRange(start: requested, end: end)
    ) { reader, output in
      try autoreleasepool {
        let sample = output.copyNextSampleBuffer()
        try Task.checkCancellation()
        try FaceReaderEvidence.validateStatus(reader.status)
        guard let sample, CMSampleBufferIsValid(sample), CMSampleBufferDataIsReady(sample),
          let pixels = CMSampleBufferGetImageBuffer(sample)
        else { throw ProAppsError.unavailable("Comparison reader returned no valid image sample") }
        let actual = CMSampleBufferGetPresentationTimeStamp(sample)
        try FaceReaderEvidence.validateTime(requested: requested, actual: actual)
        try Self.validateGeometry(
          size: CGSize(
            width: CVPixelBufferGetWidth(pixels), height: CVPixelBufferGetHeight(pixels)),
          transform: transform)
        let bitmap = try TrackingFrameDecoder.bitmap(
          pixels: pixels, transform: transform, context: CIContext(),
          preservingSourceColor: preservingSourceColor)
        let evidence = try FaceReaderEvidence.collect(
          image: bitmap, requested: requested, actual: actual,
          pixelFormat: CVPixelBufferGetPixelFormatType(pixels), transform: transform,
          preservingSourceColor: preservingSourceColor)
        try Task.checkCancellation()
        try FaceReaderEvidence.validateStatus(reader.status)
        return evidence
      }
    }
  }
}
