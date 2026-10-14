import Foundation

/// Local-file editing request. JSON selections explicitly supply start, duration,
/// and rate. Absent audio adjustment means unity gain without fades; absent video
/// settings requests audio-only output. These DTOs require EditPlan validation.
public struct EditRecipe: Codable, Sendable {
  public let clips: [EditClip]
  public let video: EditVideoSettings?
  public let additionalAudio: [EditAudioLayer]?
  public let muteOriginalAudio: Bool?
  public let additionalVideo: [EditVideoLayer]?

  public init(
    clips: [EditClip], video: EditVideoSettings? = nil,
    additionalAudio: [EditAudioLayer]? = nil, muteOriginalAudio: Bool? = nil,
    additionalVideo: [EditVideoLayer]? = nil
  ) {
    self.clips = clips
    self.video = video
    self.additionalAudio = additionalAudio
    self.muteOriginalAudio = muteOriginalAudio
    self.additionalVideo = additionalVideo
  }
}

/// Reusable local project request saved beside each successful render.
public struct EditRequest: Codable, Sendable {
  public let recipe: EditRecipe
  public let outputDirectory: String
  public let outputName: String

  public init(recipe: EditRecipe, outputDirectory: String, outputName: String) {
    self.recipe = recipe
    self.outputDirectory = outputDirectory
    self.outputName = outputName
  }
}

public struct EditSelection: Codable, Sendable {
  public let startSeconds: Double
  public let durationSeconds: Double
  public let rate: Double

  public init(startSeconds: Double, durationSeconds: Double, rate: Double) {
    self.startSeconds = startSeconds
    self.durationSeconds = durationSeconds
    self.rate = rate
  }
}

public struct EditAudioAdjustment: Codable, Sendable {
  public let volume: Double
  public let fadeInSeconds: Double
  public let fadeOutSeconds: Double

  public init(volume: Double, fadeInSeconds: Double, fadeOutSeconds: Double) {
    self.volume = volume
    self.fadeInSeconds = fadeInSeconds
    self.fadeOutSeconds = fadeOutSeconds
  }

  public static let unity = EditAudioAdjustment(volume: 1, fadeInSeconds: 0, fadeOutSeconds: 0)
}

/// Crop coordinates are pixels in the display-oriented, rotated source image.
public struct EditCrop: Codable, Sendable {
  public let x: Double
  public let y: Double
  public let width: Double
  public let height: Double

  public init(x: Double, y: Double, width: Double, height: Double) {
    self.x = x
    self.y = y
    self.width = width
    self.height = height
  }
}

public struct EditGeometry: Codable, Sendable {
  public enum Rotation: Int, Codable, Sendable {
    case none = 0
    case clockwise90 = 90
    case clockwise180 = 180
    case clockwise270 = 270
  }
  public let rotation: Rotation
  public let crop: EditCrop?

  public init(rotation: Rotation, crop: EditCrop? = nil) {
    self.rotation = rotation
    self.crop = crop
  }
}

public struct EditClip: Codable, Sendable {
  public let sourcePath: String
  public let selection: EditSelection
  public let audio: EditAudioAdjustment?
  public let geometry: EditGeometry?
  public let transitionInSeconds: Double?

  public init(
    sourcePath: String, selection: EditSelection,
    audio: EditAudioAdjustment? = nil, geometry: EditGeometry? = nil,
    transitionInSeconds: Double? = nil
  ) {
    self.sourcePath = sourcePath
    self.selection = selection
    self.audio = audio
    self.geometry = geometry
    self.transitionInSeconds = transitionInSeconds
  }
}

public struct EditAudioLayer: Codable, Sendable {
  public let sourcePath: String
  public let selection: EditSelection
  public let offsetSeconds: Double
  public let audio: EditAudioAdjustment?

  public init(
    sourcePath: String, selection: EditSelection, offsetSeconds: Double,
    audio: EditAudioAdjustment? = nil
  ) {
    self.sourcePath = sourcePath
    self.selection = selection
    self.offsetSeconds = offsetSeconds
    self.audio = audio
  }
}

/// Global SDR CoreImage color controls, applied after timeline/geometry rendering.
/// Input working RGB is clamped to 0–1; HDR headroom preservation is not promised.
/// Identity is brightness 0, contrast 1, saturation 1. An explicit setting uses
/// an additional encoding pass; omission preserves the original single pass.
public struct EditColor: Codable, Sendable {
  public let brightness: Double
  public let contrast: Double
  public let saturation: Double

  public init(brightness: Double, contrast: Double, saturation: Double) {
    self.brightness = brightness
    self.contrast = contrast
    self.saturation = saturation
  }
}

/// Static, single-line white bold system text in display-oriented output pixels.
/// Oversized text is rejected rather than silently clipped or truncated.
public struct EditTitle: Codable, Sendable {
  public let text: String
  public let x: Double
  public let y: Double
  public let fontSize: Double

  public init(text: String, x: Double, y: Double, fontSize: Double) {
    self.text = text
    self.x = x
    self.y = y
    self.fontSize = fontSize
  }
}

/// Half-open output-time interval [startSeconds, endSeconds). Captions are
/// wrapped white text on a translucent dark box, centered near the bottom.
/// Text is never silently clipped; these are supplied cues, not speech recognition.
public struct EditCaption: Codable, Sendable {
  public let text: String
  public let startSeconds: Double
  public let endSeconds: Double

  public init(text: String, startSeconds: Double, endSeconds: Double) {
    self.text = text
    self.startSeconds = startSeconds
    self.endSeconds = endSeconds
  }
}

/// Top-left output-pixel region applied before titles/captions.
/// Missing blurRadius preserves black concealment; a radius blends Gaussian blur
/// by opacity instead. Blur is not guaranteed text removal or reconstruction.
/// Optional start/end must be supplied together in output seconds.
public struct EditMask: Codable, Sendable {
  public let region: EditCrop
  public let opacity: Double
  public let blurRadius: Double?
  public let startSeconds: Double?
  public let endSeconds: Double?

  public init(
    region: EditCrop, opacity: Double, blurRadius: Double? = nil,
    startSeconds: Double? = nil, endSeconds: Double? = nil
  ) {
    self.region = region
    self.opacity = opacity
    self.blurRadius = blurRadius
    self.startSeconds = startSeconds
    self.endSeconds = endSeconds
  }

  /// Missing times mean the whole output; supplied times are half-open.
  public func isActive(at seconds: Double) -> Bool {
    seconds.isFinite && seconds >= (startSeconds ?? 0)
      && seconds < (endSeconds ?? EditPlan.maximumDurationSeconds)
  }
}

/// White captions with an expanded-alpha black outline (output pixels).
/// Background opacity applies to the caption box, independently of source masks.
public struct EditCaptionStyle: Codable, Sendable {
  public let outlineWidth: Double
  public let backgroundOpacity: Double
  public let fontSize: Double?
  public let bottomMargin: Double?
  public let centerY: Double?

  public init(
    outlineWidth: Double, backgroundOpacity: Double, fontSize: Double? = nil,
    bottomMargin: Double? = nil, centerY: Double? = nil
  ) {
    self.outlineWidth = outlineWidth
    self.backgroundOpacity = backgroundOpacity
    self.fontSize = fontSize
    self.bottomMargin = bottomMargin
    self.centerY = centerY
  }
}

public struct EditVideoSettings: Codable, Sendable {
  public enum ResizeMode: String, Codable, Sendable { case fit, fill }
  public let width: Int
  public let height: Int
  public let frameRate: Int
  public let resizeMode: ResizeMode
  public let color: EditColor?
  public let titles: [EditTitle]?
  public let captions: [EditCaption]?
  public let masks: [EditMask]?
  public let captionStyle: EditCaptionStyle?
  /// Per-cue positioned captions with a libass-compatible appearance (single pass only).
  public let styledCaptions: [EditStyledCaption]?
  public let captionAppearance: EditCaptionAppearance?
  /// Presence selects the single-pass, video-only reader/writer render.
  public let encoding: EditEncoding?
  /// Full-canvas, exact-clock video composited last, after masks and text. Single pass only.
  /// Source alpha is honored; opaque pixels cover the processed base. Audio is ignored.
  public let foregroundVideoPath: String?

  public init(
    width: Int, height: Int, frameRate: Int, resizeMode: ResizeMode, color: EditColor? = nil,
    titles: [EditTitle]? = nil, captions: [EditCaption]? = nil, masks: [EditMask]? = nil,
    captionStyle: EditCaptionStyle? = nil, styledCaptions: [EditStyledCaption]? = nil,
    captionAppearance: EditCaptionAppearance? = nil, encoding: EditEncoding? = nil,
    foregroundVideoPath: String? = nil
  ) {
    self.width = width
    self.height = height
    self.frameRate = frameRate
    self.resizeMode = resizeMode
    self.color = color
    self.titles = titles
    self.captions = captions
    self.masks = masks
    self.captionStyle = captionStyle
    self.styledCaptions = styledCaptions
    self.captionAppearance = captionAppearance
    self.encoding = encoding
    self.foregroundVideoPath = foregroundVideoPath
  }
}
