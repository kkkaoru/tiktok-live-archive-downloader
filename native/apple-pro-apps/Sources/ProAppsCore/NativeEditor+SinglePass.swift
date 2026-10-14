import AVFoundation
import CoreImage
import Foundation

extension NativeEditor {
  static let writerPollInterval: Duration = .milliseconds(2)

  /// Reads the laid-out composition once, applies color, masks, titles and styled
  /// captions per frame, and encodes once with the requested hardware codec. The
  /// output is video-only (like `ffmpeg -an`). Returns the exact frame count written,
  /// which must equal the planned duration times the frame rate.
  func renderSinglePass(
    _ prepared: Prepared, video: EditVideoSettings, encoding: EditEncoding, duration: Double,
    to destination: URL
  ) async throws -> Int {
    try Task.checkCancellation()
    let expectedFrames = Int((duration * Double(video.frameRate)).rounded())
    let overlays = try await overlays(video)
    let foreground = try await prepareForeground(video, frames: expectedFrames)
    defer { foreground?.cancel() }
    // A mutable composition's tracks are in memory; no asynchronous load is needed.
    let tracks = prepared.composition.tracks(withMediaType: .video)
    let reader = try AVAssetReader(asset: prepared.composition)
    let output = AVAssetReaderVideoCompositionOutput(
      videoTracks: tracks, videoSettings: VideoWriterSettings.readerPixels())
    let composition = videoComposition(prepared, settings: video)
    // Emit frames on the recipe's cadence, not the (possibly variable) source timing.
    composition.sourceTrackIDForFrameTiming = kCMPersistentTrackID_Invalid
    output.videoComposition = composition
    output.alwaysCopiesSampleData = false
    reader.timeRange = CMTimeRange(start: .zero, duration: time(duration))
    try Self.require(reader.canAdd(output), "Cannot read the composed video")
    reader.add(output)
    let writer = try AVAssetWriter(outputURL: destination, fileType: .mp4)
    writer.shouldOptimizeForNetworkUse = true
    let input = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: VideoWriterSettings.output(
        encoding, width: video.width, height: video.height, frameRate: video.frameRate))
    input.expectsMediaDataInRealTime = false
    input.mediaTimeScale = CMTimeScale(video.frameRate) * VideoWriterSettings.timescalePerFrame
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: input,
      sourcePixelBufferAttributes: VideoWriterSettings.pixelBuffers(
        width: video.width, height: video.height))
    try Self.require(writer.canAdd(input), "The encoder rejected the requested video settings")
    writer.add(input)
    try Self.require(reader.startReading(), "Cannot start reading the composed video")
    try Self.require(writer.startWriting(), "Cannot start the native encoder")
    writer.startSession(atSourceTime: .zero)
    let context = CIContext()
    let bounds = CGRect(x: 0, y: 0, width: video.width, height: video.height)
    let rate = Double(video.frameRate)
    var frames = 0
    do {
      // Constant-rate output like FFmpeg's fps filter (round to nearest): a source
      // frame at pts p fills slot round(p * rate); empty slots repeat the latest
      // frame and the tail holds the last frame (tpad clone). Variable-rate sources
      // therefore still yield exactly `expectedFrames` frames on the output clock.
      var current: CMSampleBuffer?
      var pending = output.copyNextSampleBuffer()
      // Render back into the composed frames' own color space (BT.709 for these
      // sources) so pixel values round-trip unchanged, like FFmpeg, which does no
      // color management. Caption colors use the same space.
      let frameSpace = pending.flatMap(CMSampleBufferGetImageBuffer).flatMap {
        CIImage(cvPixelBuffer: $0).colorSpace
      }
      let colorSpace: CGColorSpace?
      if let frameSpace {
        colorSpace = frameSpace
      } else {
        colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
      }
      let captions: StyledCaptionCache?
      if let appearance = video.captionAppearance, let styled = video.styledCaptions,
        !styled.isEmpty
      {
        captions = try await StyledCaptionCache(
          styled, appearance: appearance, canvas: video,
          colorSpaceName: colorSpace?.name as String?)
      } else {
        captions = nil
      }
      for slot in 0..<expectedFrames {
        try Task.checkCancellation()
        while let next = pending,
          Int((CMSampleBufferGetPresentationTimeStamp(next).seconds * rate).rounded()) <= slot
        {
          current = next
          pending = output.copyNextSampleBuffer()
        }
        // Slots before the first source frame repeat that first frame.
        let sample: CMSampleBuffer?
        if let current { sample = current } else { sample = pending }
        let source = try Self.required(
          sample.flatMap(CMSampleBufferGetImageBuffer), "The composed video produced no frame")
        let presentation = CMTime(value: CMTimeValue(slot), timescale: CMTimeScale(video.frameRate))
        let seconds = Double(slot) / rate
        var image = try Self.composite(
          CIImage(cvPixelBuffer: source), settings: video, seconds: seconds, overlays: overlays)
        if let captions {
          for caption in try await captions.active(at: seconds) {
            image = caption.composited(over: image)
          }
        }
        if let foreground {
          image = try foreground.image(at: slot).composited(over: image)
        }
        let pool = try Self.required(
          adaptor.pixelBufferPool, "The encoder provided no pixel buffer pool")
        var allocated: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &allocated)
        let target = try Self.required(
          status == kCVReturnSuccess ? allocated : nil, "Cannot allocate an encoder frame")
        context.render(
          image.cropped(to: bounds), to: target, bounds: bounds, colorSpace: colorSpace)
        while !input.isReadyForMoreMediaData {
          try Task.checkCancellation()
          try await Task.sleep(for: Self.writerPollInterval)
        }
        try Self.require(
          adaptor.append(target, withPresentationTime: presentation),
          "The native encoder rejected a frame")
        frames += 1
      }
      try foreground?.finish()
      while pending != nil { pending = output.copyNextSampleBuffer() }
      try Self.require(
        reader.status == .completed, "Reading the composed video did not complete")
    } catch {
      reader.cancelReading()
      writer.cancelWriting()
      throw error
    }
    input.markAsFinished()
    await writer.finishWriting()
    try Self.require(writer.status == .completed, "The native encoder did not finish the file")
    return frames
  }

  /// Native framework refusals become explicit, non-retryable errors.
  static func require(_ condition: Bool, _ reason: String) throws {
    guard condition else { throw ProAppsError.unavailable(reason) }
  }

  static func required<T>(_ value: T?, _ reason: String) throws -> T {
    guard let value else { throw ProAppsError.unavailable(reason) }
    return value
  }
}
