import Foundation

/// Deterministic shape/timeline validation, not source readability or render proof.
public struct EditPlan: Codable, Sendable {
  public struct Span: Codable, Sendable {
    public let clipIndex: Int
    public let startSeconds: Double
    public let durationSeconds: Double
    public let transitionInSeconds: Double
    public let audio: EditAudioAdjustment
  }

  public static let maximumDurationSeconds: Double = 600
  public static let maximumTransitionSeconds = 5.0
  public static let maximumTitles = 8
  public static let maximumMasks = 8
  public static let maximumCaptions = 120
  public static let maximumClips = 60
  public static let maximumAudioLayers = 16
  public static let maximumVideoLayers = 16
  public static let maximumPixels = 8_294_400
  public static let maximumCanvasDimension = 3840
  public static let maximumFrameRate = 60
  public static let timeScale: Int32 = 60000
  public static let minimumTimeSeconds = 1.0 / Double(timeScale)
  private static let maximumSourceStartSeconds: Double = 86400
  private static let maximumCropCoordinate: Double = 16384
  private static let playbackRates = 0.25...4.0
  private static let timingTolerance = 0.000_001

  public let spans: [Span]
  public let durationSeconds: Double
  public let audioOnly: Bool

  public static func build(_ recipe: EditRecipe) throws -> EditPlan {
    guard (1...maximumClips).contains(recipe.clips.count),
      (recipe.additionalAudio?.count ?? 0) <= maximumAudioLayers
    else { throw ProAppsError.invalid("Editing requires 1–60 clips and at most 16 audio layers") }
    guard (recipe.additionalVideo?.count ?? 0) <= maximumVideoLayers,
      recipe.video != nil || (recipe.additionalVideo ?? []).isEmpty
    else { throw ProAppsError.invalid("Additional video requires a canvas and at most 16 layers") }
    if let video = recipe.video { try validate(video) }
    guard
      recipe.video != nil || recipe.muteOriginalAudio != true
        || !(recipe.additionalAudio ?? []).isEmpty
    else {
      throw ProAppsError.invalid("Audio-only output cannot mute every source without replacement")
    }
    let durations = try recipe.clips.map { try outputDuration($0.selection) }
    let transitions = try recipe.clips.enumerated().map { index, clip in
      let requested = clip.transitionInSeconds ?? 0
      guard requested.isFinite, (0...maximumTransitionSeconds).contains(requested),
        requested == 0 || requested >= minimumTimeSeconds,
        index > 0 || requested == 0
      else {
        throw ProAppsError.invalid(
          "Transition must be 0–5 seconds and cannot precede the first clip")
      }
      let overlap = (requested * Double(timeScale)).rounded() / Double(timeScale)
      if index > 0 {
        guard overlap <= min(durations[index - 1], durations[index]) / 2 else {
          throw ProAppsError.invalid("Transition cannot exceed half of either adjacent output clip")
        }
      }
      return overlap
    }
    var spans: [Span] = []
    var cursor: Double = 0
    for (index, clip) in recipe.clips.enumerated() {
      _ = try Files.absolute(clip.sourcePath)
      let duration = durations[index]
      let incoming = transitions[index]
      let outgoing = index + 1 < transitions.count ? transitions[index + 1] : 0
      let originalAudio = clip.audio ?? .unity
      try validate(originalAudio, duration: duration)
      // Crossfade and explicit fades share an envelope: take the longer fade,
      // not overlapping AVAudioMix ramps or an undocumented product of gains.
      let audio = EditAudioAdjustment(
        volume: recipe.muteOriginalAudio == true ? 0 : originalAudio.volume,
        fadeInSeconds: max(originalAudio.fadeInSeconds, incoming),
        fadeOutSeconds: max(originalAudio.fadeOutSeconds, outgoing))
      try validate(audio, duration: duration)
      cursor -= incoming
      if let geometry = clip.geometry {
        guard recipe.video != nil else {
          throw ProAppsError.invalid("Video transforms require a canvas")
        }
        if let crop = geometry.crop { try validate(crop) }
      }
      guard cursor + duration <= maximumDurationSeconds + timingTolerance else {
        throw ProAppsError.invalid("Edited timeline exceeds 600 seconds")
      }
      spans.append(
        Span(
          clipIndex: index, startSeconds: cursor, durationSeconds: duration,
          transitionInSeconds: incoming, audio: audio))
      cursor += duration
    }
    for layer in recipe.additionalAudio ?? [] {
      _ = try Files.absolute(layer.sourcePath)
      let duration = try outputDuration(layer.selection)
      guard layer.offsetSeconds.isFinite, layer.offsetSeconds >= 0,
        layer.offsetSeconds + duration <= cursor + timingTolerance
      else { throw ProAppsError.invalid("Additional audio must fit within the edited timeline") }
      try validate(layer.audio ?? .unity, duration: duration)
    }
    for layer in recipe.additionalVideo ?? [] {
      _ = try Files.absolute(layer.sourcePath)
      let duration = try outputDuration(layer.selection)
      let opacity = layer.opacity ?? 1
      guard layer.offsetSeconds.isFinite, layer.offsetSeconds >= 0,
        layer.offsetSeconds + duration <= cursor + timingTolerance,
        opacity.isFinite, (0...1).contains(opacity)
      else {
        throw ProAppsError.invalid("Additional video must fit the timeline with opacity 0–1")
      }
      if let crop = layer.geometry?.crop { try validate(crop) }
    }
    try validate(recipe.video?.captions ?? [], duration: cursor)
    try validateMaskTimes(recipe.video?.masks ?? [], duration: cursor)
    try validateSinglePass(recipe, duration: cursor)
    return EditPlan(spans: spans, durationSeconds: cursor, audioOnly: recipe.video == nil)
  }

  static func outputDuration(_ selection: EditSelection) throws -> Double {
    guard selection.startSeconds.isFinite,
      (0...maximumSourceStartSeconds).contains(selection.startSeconds),
      selection.durationSeconds.isFinite, selection.durationSeconds >= minimumTimeSeconds,
      selection.durationSeconds <= maximumDurationSeconds,
      selection.rate.isFinite, playbackRates.contains(selection.rate)
    else { throw ProAppsError.invalid("Invalid source selection or playback rate (0.25–4)") }
    let duration = selection.durationSeconds / selection.rate
    guard duration >= minimumTimeSeconds else {
      throw ProAppsError.invalid("Selection is shorter than one timeline tick")
    }
    // Quantize each segment once so adjacent instruction ranges share the same
    // timeline ticks rather than independently rounding overlapping durations.
    return (duration * Double(timeScale)).rounded() / Double(timeScale)
  }

  private static func validate(_ audio: EditAudioAdjustment, duration: Double) throws {
    guard audio.volume.isFinite, (0...1).contains(audio.volume),
      audio.fadeInSeconds.isFinite, audio.fadeInSeconds >= 0,
      audio.fadeOutSeconds.isFinite, audio.fadeOutSeconds >= 0,
      audio.fadeInSeconds == 0 || audio.fadeInSeconds >= minimumTimeSeconds,
      audio.fadeOutSeconds == 0 || audio.fadeOutSeconds >= minimumTimeSeconds,
      audio.fadeInSeconds + audio.fadeOutSeconds <= duration + timingTolerance
    else {
      throw ProAppsError.invalid("Invalid linear volume (0–1) or overlapping/out-of-range fades")
    }
  }

  static func validate(_ title: EditTitle, canvas: EditVideoSettings) throws {
    guard !title.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      title.text.count <= 120, title.text.utf8.count <= 1024,
      !title.text.unicodeScalars.contains(where: { CharacterSet.newlines.contains($0) }),
      title.x.isFinite, title.y.isFinite, title.x >= 0, title.y >= 0,
      title.x < Double(canvas.width), title.y < Double(canvas.height),
      title.fontSize.isFinite, (8...128).contains(title.fontSize)
    else {
      throw ProAppsError.invalid(
        "Title requires bounded single-line text, an in-canvas origin and font size 8–128")
    }
  }

  static func validateMaskTimes(_ masks: [EditMask], duration: Double) throws {
    for mask in masks {
      switch (mask.startSeconds, mask.endSeconds) {
      case (nil, nil): break
      case (.some(let start), .some(let end)):
        guard start.isFinite, end.isFinite, start >= 0, end <= duration,
          end - start >= minimumTimeSeconds
        else { throw ProAppsError.invalid("Mask interval must lie inside the output timeline") }
      default:
        throw ProAppsError.invalid("Mask startSeconds and endSeconds must be supplied together")
      }
    }
  }

  static func validate(_ captions: [EditCaption], duration: Double) throws {
    guard captions.count <= maximumCaptions else {
      throw ProAppsError.invalid("At most 120 timed captions are supported")
    }
    var previousEnd = 0.0
    for caption in captions {
      guard !caption.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
        caption.text.count <= 120, caption.text.utf8.count <= 1024, !caption.text.contains("\0"),
        caption.startSeconds.isFinite, caption.endSeconds.isFinite,
        caption.startSeconds >= previousEnd, caption.endSeconds <= duration,
        caption.endSeconds - caption.startSeconds >= minimumTimeSeconds
      else {
        throw ProAppsError.invalid(
          "Caption text/times must be bounded, ordered, nonoverlapping and inside the output timeline"
        )
      }
      previousEnd = caption.endSeconds
    }
  }

  static func validate(_ video: EditVideoSettings) throws {
    if let style = video.captionStyle {
      let maximumOutlineWidth = 6.0
      guard style.outlineWidth.isFinite, (0...maximumOutlineWidth).contains(style.outlineWidth),
        style.backgroundOpacity.isFinite, (0...1).contains(style.backgroundOpacity)
      else { throw ProAppsError.invalid("Caption outline requires 0–6 pixels and box opacity 0–1") }
      if let fontSize = style.fontSize {
        guard fontSize.isFinite, (16...64).contains(fontSize) else {
          throw ProAppsError.invalid("Caption font size must be 16–64 output pixels")
        }
      }
      if let center = style.centerY {
        guard center.isFinite, center > 0, center < Double(video.height), style.bottomMargin == nil
        else {
          throw ProAppsError.invalid(
            "Caption centerY must be inside the canvas and excludes bottomMargin")
        }
      }
      if let bottomMargin = style.bottomMargin {
        guard bottomMargin.isFinite, bottomMargin >= 0, bottomMargin < Double(video.height) else {
          throw ProAppsError.invalid("Caption bottom margin must be finite and inside the canvas")
        }
      }
    }
    let masks = video.masks ?? []
    try validateMaskTimes(masks, duration: maximumDurationSeconds)
    guard masks.count <= maximumMasks else {
      throw ProAppsError.invalid("At most eight source concealment masks are supported")
    }
    for mask in masks {
      try validate(mask.region)
      if let radius = mask.blurRadius {
        guard radius.isFinite, (1...64).contains(radius) else {
          throw ProAppsError.invalid("Mask blur radius must be 1–64 output pixels")
        }
      }
      guard mask.opacity.isFinite, (0...1).contains(mask.opacity),
        mask.region.x + mask.region.width <= Double(video.width),
        mask.region.y + mask.region.height <= Double(video.height)
      else { throw ProAppsError.invalid("Mask must fit the canvas with opacity 0–1") }
    }
    let titles = video.titles ?? []
    guard titles.count <= maximumTitles else {
      throw ProAppsError.invalid("At most eight static titles are supported")
    }
    for title in titles { try validate(title, canvas: video) }
    if let color = video.color {
      guard color.brightness.isFinite, (-1...1).contains(color.brightness),
        color.contrast.isFinite, (0...4).contains(color.contrast),
        color.saturation.isFinite, (0...2).contains(color.saturation)
      else {
        throw ProAppsError.invalid(
          "Color requires brightness -1–1, contrast 0–4 and saturation 0–2")
      }
    }
    guard (2...maximumCanvasDimension).contains(video.width),
      (2...maximumCanvasDimension).contains(video.height),
      video.width.isMultiple(of: 2), video.height.isMultiple(of: 2),
      video.width * video.height <= maximumPixels, (1...maximumFrameRate).contains(video.frameRate)
    else {
      throw ProAppsError.invalid(
        "Canvas requires even bounded dimensions, at most 4K pixels and 1–60 fps")
    }
    if !(video.captions ?? []).isEmpty, video.width < 160 || video.height < 90 {
      throw ProAppsError.invalid("Caption canvas must be at least 160 by 90 pixels")
    }
  }

  private static func validate(_ crop: EditCrop) throws {
    guard crop.x.isFinite, crop.y.isFinite, crop.width.isFinite, crop.height.isFinite,
      crop.x >= 0, crop.y >= 0, crop.width > 0, crop.height > 0,
      crop.x + crop.width <= maximumCropCoordinate, crop.y + crop.height <= maximumCropCoordinate
    else { throw ProAppsError.invalid("Crop requires a finite positive bounded rectangle") }
  }
}
