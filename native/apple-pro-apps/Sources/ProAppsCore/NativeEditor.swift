import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins
import Dispatch
import Foundation

public struct EditRenderResult: Codable, Sendable {
  public let outputPath: String
  public let projectPath: String
  public let expectedDurationSeconds: Double
  public let actualDurationSeconds: Double
  public let videoTrackCount: Int
  public let audioTrackCount: Int
  /// Frames written by the single-pass render (absent for the export-session path).
  public let frameCount: Int?
  /// Editable FCPXML written next to every video render.
  public let fcpxml: FCPXMLExport?
}

/// One operation owns its mutable AVFoundation graph. Run through the existing
/// disposable child/deadline boundary; no composition/session is shared between
/// requests, played on hardware, or saved over an existing project or source.
public actor NativeEditor {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.edit")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }

  private struct VideoLayer {
    let range: CMTimeRange
    let instruction: AVMutableVideoCompositionLayerInstruction
  }

  struct TimedOverlay: Sendable {
    let image: CIImage
    let startSeconds: Double
    let endSeconds: Double
  }

  struct Prepared {
    let composition: AVMutableComposition
    let instructions: [AVVideoCompositionInstructionProtocol]
    let audioParameters: [AVAudioMixInputParameters]
  }

  public init() {}

  public func render(_ recipe: EditRecipe, directory: String, name: String) async throws
    -> EditRenderResult
  {
    try Task.checkCancellation()
    let plan = try EditPlan.build(recipe)
    let expectedExtension = plan.audioOnly ? "m4a" : "mp4"
    guard URL(fileURLWithPath: name).pathExtension.lowercased() == expectedExtension else {
      throw ProAppsError.invalid("Output extension must match audio-only M4A or video MP4")
    }
    let prepared = try await prepare(recipe, plan: plan)
    try Task.checkCancellation()
    let output = try Files.reserveOutput(directory: directory, name: name, kind: .edit)
    let staging = output.deletingLastPathComponent().appendingPathComponent(
      ".rendering.\(expectedExtension)")
    let graded = output.deletingLastPathComponent().appendingPathComponent(".graded.mp4")
    defer {
      for temporary in [staging, graded]
      where FileManager.default.fileExists(atPath: temporary.path) {
        Cleanup.perform { try FileManager.default.removeItem(at: temporary) }
      }
    }
    var finalStaging = staging
    var frameCount: Int?
    if let video = recipe.video, let encoding = video.encoding {
      frameCount = try await renderSinglePass(
        prepared, video: video, encoding: encoding, duration: plan.durationSeconds, to: staging)
    } else {
      try await exportSession(prepared, recipe: recipe, plan: plan, to: staging)
      try Task.checkCancellation()
      if let video = recipe.video,
        video.color != nil || !(video.titles ?? []).isEmpty || !(video.captions ?? []).isEmpty
          || !(video.masks ?? []).isEmpty
      {
        try await applyEffects(video, source: staging, destination: graded)
        finalStaging = graded
      }
    }
    try Task.checkCancellation()
    let completed = AVURLAsset(url: finalStaging)
    let duration = try await completed.load(.duration).seconds
    let videoTracks = try await completed.loadTracks(withMediaType: .video)
    let audioTracks = try await completed.loadTracks(withMediaType: .audio)
    guard duration.isFinite, duration > 0,
      plan.audioOnly ? !audioTracks.isEmpty : !videoTracks.isEmpty
    else { throw ProAppsError.unavailable("Export did not produce the requested media tracks") }
    let durationTolerance: Double
    if let video = recipe.video {
      durationTolerance = 1 / Double(video.frameRate)
    } else {
      durationTolerance = 0.1
    }
    guard abs(duration - plan.durationSeconds) <= durationTolerance else {
      throw ProAppsError.unavailable("Export duration does not match the edited timeline")
    }
    try Task.checkCancellation()
    let project = try Files.writeNew(
      JSONEncoder().encode(
        EditRequest(recipe: recipe, outputDirectory: directory, outputName: name)),
      to: output.deletingLastPathComponent().appendingPathComponent("edit-request.json").path,
      extensions: ["json"])
    // Hard-link publication fails if any destination appeared in the meantime.
    // Native media stays inside this operation's private 0700 directory.
    try FileManager.default.linkItem(at: finalStaging, to: output)
    let fcpxml = try await exportTimeline(
      recipe, plan: plan, directory: output.deletingLastPathComponent(),
      name: URL(fileURLWithPath: name).deletingPathExtension().lastPathComponent)
    return EditRenderResult(
      outputPath: output.path, projectPath: project.path,
      expectedDurationSeconds: plan.durationSeconds,
      actualDurationSeconds: duration, videoTrackCount: videoTracks.count,
      audioTrackCount: audioTracks.count, frameCount: frameCount, fcpxml: fcpxml)
  }

  private func exportSession(
    _ prepared: Prepared, recipe: EditRecipe, plan: EditPlan, to staging: URL
  ) async throws {
    let preset = plan.audioOnly ? AVAssetExportPresetAppleM4A : AVAssetExportPresetHighestQuality
    guard let session = AVAssetExportSession(asset: prepared.composition, presetName: preset) else {
      throw ProAppsError.unavailable("Cannot create native export session")
    }
    if let settings = recipe.video {
      session.videoComposition = videoComposition(prepared, settings: settings)
    }
    let mix = AVMutableAudioMix()
    mix.inputParameters = prepared.audioParameters
    session.audioMix = mix
    session.audioTimePitchAlgorithm = .spectral
    session.timeRange = CMTimeRange(start: .zero, duration: time(plan.durationSeconds))
    try await session.export(to: staging, as: plan.audioOnly ? .m4a : .mp4)
  }

  func videoComposition(_ prepared: Prepared, settings: EditVideoSettings)
    -> AVMutableVideoComposition
  {
    let composition = AVMutableVideoComposition()
    composition.renderSize = CGSize(width: settings.width, height: settings.height)
    composition.frameDuration = CMTime(value: 1, timescale: Int32(settings.frameRate))
    composition.instructions = prepared.instructions
    return composition
  }

  /// Static titles and timed captions, pre-rendered once within the pixel budget.
  func overlays(_ settings: EditVideoSettings) async throws -> [TimedOverlay] {
    try Task.checkCancellation()
    var preparedTitles: [TimedOverlay] = []
    var pixels = 0
    for title in settings.titles ?? [] {
      let bitmap = try await TitleRenderer.render(title, canvas: settings)
      try Task.checkCancellation()
      pixels = try TitleRenderer.addingPixels(
        width: bitmap.width, height: bitmap.height, to: pixels)
      let image = CIImage(cgImage: bitmap).transformed(
        by: CGAffineTransform(
          translationX: title.x, y: Double(settings.height) - title.y - Double(bitmap.height)))
      preparedTitles.append(
        TimedOverlay(image: image, startSeconds: 0, endSeconds: EditPlan.maximumDurationSeconds))
    }
    for caption in settings.captions ?? [] {
      let bitmap = try await TitleRenderer.renderCaption(caption, canvas: settings)
      try Task.checkCancellation()
      pixels = try TitleRenderer.addingPixels(
        width: bitmap.width, height: bitmap.height, to: pixels)
      let image = try TitleRenderer.captionImage(bitmap, canvas: settings).transformed(
        by: CGAffineTransform(
          translationX: Double(settings.width - bitmap.width) / 2,
          y: TitleRenderer.captionBottomMargin(settings, imageHeight: bitmap.height)))
      preparedTitles.append(
        TimedOverlay(
          image: image, startSeconds: caption.startSeconds, endSeconds: caption.endSeconds))
    }
    return preparedTitles
  }

  /// Color, timed masks, then pre-rendered overlays at one output time. Shared by the
  /// export-session effects pass and the single-pass writer, so both look the same.
  static func composite(
    _ source: CIImage, settings: EditVideoSettings, seconds: Double, overlays: [TimedOverlay]
  ) throws -> CIImage {
    // Color controls have an explicit SDR domain. Decoded YUV conversion can
    // overshoot 1 in extended working RGB, otherwise brightness -1 leaves colored
    // residuals instead of reaching the defined black bound.
    var image = source
    if let color = settings.color {
      let normalized = CIFilter.colorClamp()
      normalized.inputImage = image
      normalized.minComponents = CIVector(x: 0, y: 0, z: 0, w: 0)
      normalized.maxComponents = CIVector(x: 1, y: 1, z: 1, w: 1)
      let filter = CIFilter.colorControls()
      filter.inputImage = normalized.outputImage
      filter.brightness = Float(color.brightness)
      filter.contrast = Float(color.contrast)
      filter.saturation = Float(color.saturation)
      guard let adjusted = filter.outputImage else {
        throw ProAppsError.unavailable("Color filter produced no image")
      }
      image = adjusted
    }
    image = try MaskRenderer.apply(
      settings.masks ?? [], to: image, at: seconds, canvasHeight: settings.height)
    for overlay in overlays where seconds >= overlay.startSeconds && seconds < overlay.endSeconds {
      image = overlay.image.composited(over: image)
    }
    return image.cropped(to: source.extent)
  }

  private func applyEffects(_ settings: EditVideoSettings, source: URL, destination: URL)
    async throws
  {
    let overlays = try await overlays(settings)
    let asset = AVURLAsset(url: source)
    let composition = try await AVMutableVideoComposition.videoComposition(
      with: asset,
      applyingCIFiltersWithHandler: { request in
        // Each callback owns its mutable filter; only the Sendable settings cross
        // into AVFoundation's callback. The rendered extent remains bounded.
        do {
          let image = try Self.composite(
            request.sourceImage, settings: settings, seconds: request.compositionTime.seconds,
            overlays: overlays)
          request.finish(with: image, context: nil)
        } catch {
          request.finish(with: error)
        }
      })
    // Preserve the recipe's cadence rather than inheriting source sample timing
    // after rate changes. Apple's mutable CI composition explicitly supports this.
    composition.sourceTrackIDForFrameTiming = kCMPersistentTrackID_Invalid
    composition.frameDuration = CMTime(value: 1, timescale: CMTimeScale(settings.frameRate))
    guard
      let session = AVAssetExportSession(
        asset: asset, presetName: AVAssetExportPresetHighestQuality)
    else {
      throw ProAppsError.unavailable("Cannot create effects export session")
    }
    session.videoComposition = composition
    try Task.checkCancellation()
    try await session.export(to: destination, as: .mp4)
  }

  func prepare(_ recipe: EditRecipe, plan: EditPlan) async throws -> Prepared {
    let composition = AVMutableComposition()
    var videoLayers: [VideoLayer] = []
    var audioParameters: [AVAudioMixInputParameters] = []
    for span in plan.spans {
      try Task.checkCancellation()
      let clip = recipe.clips[span.clipIndex]
      let asset = try await load(clip.sourcePath, selection: clip.selection)
      if let canvas = recipe.video {
        videoLayers.append(
          try await videoLayer(
            asset: asset,
            placement: EditVideoLayer(
              sourcePath: clip.sourcePath, selection: clip.selection,
              offsetSeconds: span.startSeconds, geometry: clip.geometry),
            canvas: canvas, composition: composition, timelineEnd: plan.durationSeconds,
            transition: span.transitionInSeconds))
      }
      let audio = try await asset.loadTracks(withMediaType: .audio)
      if let source = audio.first {
        let destination = try await insert(
          source, selection: clip.selection, offset: span.startSeconds, into: composition,
          timelineEnd: plan.durationSeconds)
        audioParameters.append(
          audioMix(
            span.audio, track: destination, start: span.startSeconds,
            duration: span.durationSeconds))
      } else if recipe.muteOriginalAudio != true && (plan.audioOnly || clip.audio != nil) {
        throw ProAppsError.invalid("Requested clip audio is missing")
      }
    }
    if let canvas = recipe.video {
      for placement in recipe.additionalVideo ?? [] {
        try Task.checkCancellation()
        let asset = try await load(placement.sourcePath, selection: placement.selection)
        videoLayers.append(
          try await videoLayer(
            asset: asset, placement: placement, canvas: canvas,
            composition: composition, timelineEnd: plan.durationSeconds, transition: 0))
      }
    }
    for layer in recipe.additionalAudio ?? [] {
      try Task.checkCancellation()
      let asset = try await load(layer.sourcePath, selection: layer.selection)
      let tracks = try await asset.loadTracks(withMediaType: .audio)
      guard let source = tracks.first else {
        throw ProAppsError.invalid("Additional audio source has no audio track")
      }
      let destination = try await insert(
        source, selection: layer.selection, offset: layer.offsetSeconds, into: composition,
        timelineEnd: plan.durationSeconds)
      audioParameters.append(
        audioMix(
          layer.audio ?? .unity, track: destination, start: layer.offsetSeconds,
          duration: try EditPlan.outputDuration(layer.selection)))
    }
    return Prepared(
      composition: composition, instructions: instructions(videoLayers),
      audioParameters: audioParameters)
  }

  private func videoLayer(
    asset: AVURLAsset, placement: EditVideoLayer, canvas: EditVideoSettings,
    composition: AVMutableComposition, timelineEnd: Double, transition: Double
  ) async throws -> VideoLayer {
    let tracks = try await asset.loadTracks(withMediaType: .video)
    guard let source = tracks.first else {
      throw ProAppsError.invalid("Video clip has no video track")
    }
    let destination = try await insert(
      source, selection: placement.selection, offset: placement.offsetSeconds,
      into: composition, timelineEnd: timelineEnd)
    destination.preferredTransform = .identity
    let size = try await source.load(.naturalSize)
    let preferred = try await source.load(.preferredTransform)
    let geometry = try EditTransform.make(
      naturalSize: size, preferred: preferred, geometry: placement.geometry, canvas: canvas)
    let start = time(placement.offsetSeconds)
    let opacity = Float(placement.opacity ?? 1)
    let instruction = AVMutableVideoCompositionLayerInstruction(assetTrack: destination)
    instruction.setTransform(geometry.transform, at: start)
    instruction.setCropRectangle(geometry.sourceCrop, at: start)
    instruction.setOpacity(opacity, at: start)
    if transition > 0 {
      // Fade only the incoming alpha layer, avoiding a dark crossfade dip.
      instruction.setOpacityRamp(
        fromStartOpacity: 0, toEndOpacity: opacity,
        timeRange: CMTimeRange(start: start, duration: time(transition)))
    }
    return VideoLayer(
      range: CMTimeRange(
        start: start, duration: time(try EditPlan.outputDuration(placement.selection))),
      instruction: instruction)
  }

  private func instructions(_ layers: [VideoLayer]) -> [AVVideoCompositionInstructionProtocol] {
    // All times use EditPlan.timeScale. Partition at exact integer boundaries so
    // instructions are contiguous even when clip spans deliberately overlap.
    let boundaries = Set(layers.flatMap { [$0.range.start.value, $0.range.end.value] }).sorted()
    var result: [AVVideoCompositionInstructionProtocol] = []
    for (start, end) in zip(boundaries, boundaries.dropFirst()) {
      let instruction = AVMutableVideoCompositionInstruction()
      instruction.timeRange = CMTimeRange(
        start: CMTime(value: start, timescale: EditPlan.timeScale),
        duration: CMTime(value: end - start, timescale: EditPlan.timeScale))
      instruction.layerInstructions = layers.filter {
        $0.range.start.value <= start && $0.range.end.value >= end
      }.reversed().map(\.instruction)
      result.append(instruction)
    }
    return result
  }

  private func load(_ path: String, selection: EditSelection) async throws -> AVURLAsset {
    let asset = AVURLAsset(url: try Files.existing(path))
    let duration = try await asset.load(.duration).seconds
    guard duration.isFinite, duration > 0,
      selection.startSeconds + selection.durationSeconds <= duration + EditPlan.minimumTimeSeconds
    else {
      throw ProAppsError.invalid("Selection exceeds available source duration")
    }
    return asset
  }

  private func insert(
    _ source: AVAssetTrack, selection: EditSelection, offset: Double,
    into composition: AVMutableComposition, timelineEnd: Double
  ) async throws -> AVMutableCompositionTrack {
    let available = try await source.load(.timeRange)
    guard available.start.seconds.isFinite, available.duration.seconds.isFinite,
      available.duration.seconds > 0
    else {
      throw ProAppsError.invalid("Source track has no finite time range")
    }
    let requested = CMTimeRange(
      start: time(selection.startSeconds), duration: time(selection.durationSeconds))
    let selected = CMTimeRangeGetIntersection(requested, otherRange: available)
    guard selected.duration.seconds.isFinite, selected.duration.seconds > 0,
      let destination = composition.addMutableTrack(
        withMediaType: source.mediaType, preferredTrackID: kCMPersistentTrackID_Invalid)
    else { throw ProAppsError.unavailable("Cannot insert requested source track interval") }
    let leading = CMTimeMultiplyByFloat64(
      CMTimeSubtract(selected.start, requested.start), multiplier: 1 / selection.rate)
    let insertion = CMTimeAdd(time(offset), leading)
    if insertion > .zero {
      destination.insertEmptyTimeRange(CMTimeRange(start: .zero, duration: insertion))
    }
    try destination.insertTimeRange(selected, of: source, at: insertion)
    let scaled = CMTimeMultiplyByFloat64(selected.duration, multiplier: 1 / selection.rate)
    destination.scaleTimeRange(
      CMTimeRange(start: insertion, duration: selected.duration), toDuration: scaled)
    let end = CMTimeAdd(insertion, scaled)
    let desiredEnd = time(timelineEnd)
    if desiredEnd > end {
      destination.insertEmptyTimeRange(
        CMTimeRange(start: end, duration: CMTimeSubtract(desiredEnd, end)))
    }
    return destination
  }

  private func audioMix(
    _ adjustment: EditAudioAdjustment, track: AVCompositionTrack,
    start: Double, duration: Double
  ) -> AVAudioMixInputParameters {
    let parameters = AVMutableAudioMixInputParameters(track: track)
    let volume = Float(adjustment.volume)
    parameters.setVolume(volume, at: time(start))
    if adjustment.fadeInSeconds > 0 {
      parameters.setVolumeRamp(
        fromStartVolume: 0, toEndVolume: volume,
        timeRange: CMTimeRange(start: time(start), duration: time(adjustment.fadeInSeconds)))
    }
    if adjustment.fadeOutSeconds > 0 {
      parameters.setVolumeRamp(
        fromStartVolume: volume, toEndVolume: 0,
        timeRange: CMTimeRange(
          start: time(start + duration - adjustment.fadeOutSeconds),
          duration: time(adjustment.fadeOutSeconds)))
    }
    return parameters
  }

  func time(_ seconds: Double) -> CMTime {
    CMTime(seconds: seconds, preferredTimescale: EditPlan.timeScale)
  }
}
