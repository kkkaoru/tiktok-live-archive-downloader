import AVFoundation
import CoreVideo
import Dispatch
import Foundation

public struct MediaSummary: Codable, Sendable {
  public let durationSeconds: Double
  public let width: Int
  public let height: Int
  public let frameRate: Double
  public let audioTrackCount: Int
  public let firstFrameDecoded: Bool
}

public struct VideoVerification: Codable, Sendable {
  public let media: MediaSummary
  public let decodedFrames: Int
  public let fullVideoDecoded: Bool
}

/// Runs in a disposable CLI child. Runner enforces the process deadline even if
/// a malformed asset stalls a synchronous system decoder. No image/audio leaves
/// this process: only scalar metadata and bounded video decode evidence return.
public actor MediaProbe {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.media")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }
  private static let maximumPixels = 16_777_216.0
  public static let defaultVerificationSeconds = 30.0
  public static let defaultVerificationFrames = 1800
  public static let maximumVerificationSeconds = 120.0
  public static let maximumVerificationFrames = 7200

  public init() {}

  public func inspect(path: String) async throws -> MediaSummary {
    try await decode(
      path: path, complete: false, maximumFrames: 1,
      maximumDurationSeconds: Self.defaultVerificationSeconds
    ).media
  }

  /// Defaults to 30 seconds/1800 frames. Longer clips require explicit opt-in;
  /// 120 seconds/7200 frames are hard ceilings, not an unlimited decode mode.
  /// Audio remains a separate check and the disposable child retains its deadline.
  public func verifyShortVideo(
    path: String, maximumFrames: Int = MediaProbe.defaultVerificationFrames,
    maximumDurationSeconds: Double = MediaProbe.defaultVerificationSeconds
  ) async throws -> VideoVerification {
    guard (1...Self.maximumVerificationFrames).contains(maximumFrames),
      maximumDurationSeconds.isFinite, maximumDurationSeconds > 0,
      maximumDurationSeconds <= Self.maximumVerificationSeconds
    else {
      throw ProAppsError.invalid(
        "Video verification budgets must be 1–7200 frames and >0–120 seconds")
    }
    return try await decode(
      path: path, complete: true, maximumFrames: maximumFrames,
      maximumDurationSeconds: maximumDurationSeconds)
  }

  private func decode(
    path: String, complete: Bool, maximumFrames: Int, maximumDurationSeconds: Double
  ) async throws
    -> VideoVerification
  {
    try Task.checkCancellation()
    let source = try Files.existing(path)
    let asset = AVURLAsset(url: source)
    let duration = try await asset.load(.duration).seconds
    let tracks = try await asset.loadTracks(withMediaType: .video)
    guard let track = tracks.first, duration.isFinite, duration > 0 else {
      throw ProAppsError.invalid("Expected a finite-duration video track")
    }
    guard !complete || duration <= maximumDurationSeconds else {
      throw ProAppsError.invalid("Video exceeds the requested verification duration budget")
    }
    let size = try await track.load(.naturalSize)
    let frameRate = try await track.load(.nominalFrameRate)
    guard size.width.isFinite, size.height.isFinite, size.width > 0, size.height > 0,
      size.width * size.height <= Self.maximumPixels,
      frameRate.isFinite, frameRate > 0
    else { throw ProAppsError.invalid("Unsupported video dimensions or frame rate") }
    let audio = try await asset.loadTracks(withMediaType: .audio)
    try Task.checkCancellation()
    let reader = try AVAssetReader(asset: asset)
    // The framework's NSDictionary contract receives only one fixed, typed pixel
    // format option. No caller-supplied heterogeneous values are admitted.
    let output = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
      ])
    guard reader.canAdd(output) else {
      throw ProAppsError.unavailable("Cannot attach video decoder")
    }
    reader.add(output)
    guard reader.startReading() else {
      throw ProAppsError.unavailable("Cannot start video decoder")
    }
    defer { reader.cancelReading() }
    guard let sample = output.copyNextSampleBuffer(), CMSampleBufferIsValid(sample),
      CMSampleBufferDataIsReady(sample), let image = CMSampleBufferGetImageBuffer(sample)
    else { throw ProAppsError.unavailable("First video frame did not decode") }
    var count = 1
    if complete {
      while let next = output.copyNextSampleBuffer() {
        try Task.checkCancellation()
        guard count < maximumFrames, CMSampleBufferIsValid(next),
          CMSampleBufferDataIsReady(next),
          let frame = CMSampleBufferGetImageBuffer(next), CVPixelBufferGetWidth(frame) > 0,
          CVPixelBufferGetHeight(frame) > 0,
          Double(CVPixelBufferGetWidth(frame)) * Double(CVPixelBufferGetHeight(frame))
            <= Self.maximumPixels
        else {
          throw ProAppsError.invalid(
            "Invalid decoded frame or requested verification frame budget exceeded")
        }
        count += 1
      }
      guard reader.status == .completed else {
        throw ProAppsError.unavailable("Video decoder did not reach end of stream")
      }
    }
    try Task.checkCancellation()
    let media = MediaSummary(
      durationSeconds: duration, width: CVPixelBufferGetWidth(image),
      height: CVPixelBufferGetHeight(image),
      frameRate: Double(frameRate), audioTrackCount: audio.count, firstFrameDecoded: true)
    return VideoVerification(media: media, decodedFrames: count, fullVideoDecoded: complete)
  }
}
