import AVFoundation
import CoreImage
import Foundation

extension NativeEditor {
  /// Owned only by one render on NativeEditor's serial executor. At most one
  /// decoded foreground frame is retained; no cache, retiming or fallback frames.
  struct ForegroundVideo {
    let reader: AVAssetReader
    let output: AVAssetReaderTrackOutput
    let width: Int
    let height: Int
    let rate: Int32

    func image(at slot: Int) throws -> CIImage {
      try Task.checkCancellation()
      let sample = try NativeEditor.required(
        output.copyNextSampleBuffer(), "Foreground video ended before the output clock")
      let buffer = try NativeEditor.required(
        CMSampleBufferGetImageBuffer(sample), "Foreground sample has no image")
      try Self.validateFrame(
        presentation: CMSampleBufferGetPresentationTimeStamp(sample),
        duration: CMSampleBufferGetDuration(sample), slot: slot, rate: rate,
        dimensionsMatch: CVPixelBufferGetWidth(buffer) == width
          && CVPixelBufferGetHeight(buffer) == height)
      return CIImage(cvPixelBuffer: buffer)
    }

    static func validateFrame(
      presentation: CMTime, duration: CMTime, slot: Int, rate: Int32, dimensionsMatch: Bool
    ) throws {
      // AVAssetReader's decoded frames can omit sample duration (observed invalid
      // even for the exact-clock golden fixture). Every PTS, the track end and
      // decoded EOF still must match; no missing frame is held or synthesized.
      let expectedDuration = CMTime(value: 1, timescale: rate)
      let durationMatches =
        duration == .invalid
        || (duration.isNumeric && duration.epoch == 0 && duration == expectedDuration)
      guard presentation.isNumeric, presentation.epoch == 0,
        presentation == CMTime(value: CMTimeValue(slot), timescale: rate),
        durationMatches, dimensionsMatch
      else {
        throw ProAppsError.invalid(
          "Foreground frame does not match the exact output clock/canvas: slot=\(slot), pts=\(presentation.value)/\(presentation.timescale) flags=\(presentation.flags.rawValue), duration=\(duration.value)/\(duration.timescale) flags=\(duration.flags.rawValue), canvas=\(dimensionsMatch)"
        )
      }
    }

    func finish() throws {
      try Task.checkCancellation()
      try NativeEditor.require(
        output.copyNextSampleBuffer() == nil && reader.status == .completed,
        "Foreground video has extra frames or did not decode to end-of-stream")
    }

    func cancel() {
      if reader.status == .reading { reader.cancelReading() }
    }
  }

  func prepareForeground(_ video: EditVideoSettings, frames: Int) async throws -> ForegroundVideo? {
    guard let path = video.foregroundVideoPath else { return nil }
    try Task.checkCancellation()
    let asset = AVURLAsset(url: try Files.existing(path))
    let tracks = try await asset.loadTracks(withMediaType: .video)
    guard tracks.count == 1, let track = tracks.first else {
      throw ProAppsError.invalid("Foreground video must have exactly one video track")
    }
    let (size, transform, range, rate) = try await track.load(
      .naturalSize, .preferredTransform, .timeRange, .nominalFrameRate)
    let duration = try await asset.load(.duration)
    let expected = CMTime(value: CMTimeValue(frames), timescale: CMTimeScale(video.frameRate))
    guard frames > 0, size == CGSize(width: video.width, height: video.height),
      transform == .identity, range.start == .zero, range.duration == expected,
      duration == expected, rate == Float(video.frameRate)
    else {
      throw ProAppsError.invalid(
        "Foreground video must match the output dimensions, rate and duration without transforms")
    }
    try Task.checkCancellation()
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(
      track: track, outputSettings: VideoWriterSettings.readerPixels())
    output.alwaysCopiesSampleData = false
    try Self.require(reader.canAdd(output), "Cannot read foreground video")
    reader.add(output)
    try Self.require(reader.startReading(), "Cannot start foreground video decoder")
    return ForegroundVideo(
      reader: reader, output: output, width: video.width, height: video.height,
      rate: CMTimeScale(video.frameRate))
  }
}
