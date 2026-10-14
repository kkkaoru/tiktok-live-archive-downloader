import AVFoundation
import AudioToolbox
import Dispatch
import Foundation

public struct MoviePCMMuxRequest: Codable, Sendable {
  public let rootDirectory: String
  public let videoPath: String
  public let audioPath: String
  public let outputDirectory: String
  public let outputName: String
}

public struct MoviePCMMuxResult: Encodable, Sendable {
  public let outputPath: String
  public let videoDuration: ForegroundTime
  public let audioDuration: ForegroundTime
}

/// Append PCM to a new same-volume clone; never modify an existing source.
/// Only the new, private clone is mutable. Failed clones remain for diagnosis.
/// No video encoder, audio conversion, padding or resampling is invoked.
public actor MoviePCMMux {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.pcm-mux")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }

  public init() {}

  static func commonTimescale(_ left: Int32, _ right: Int32) throws -> Int32 {
    guard left > 0, right > 0 else { throw ProAppsError.invalid("Invalid movie clock") }
    var a = Int64(left)
    var b = Int64(right)
    while b != 0 { (a, b) = (b, a % b) }
    let scale = Int64(left) / a * Int64(right)
    guard scale <= Int64(ForegroundTime.maximumTimescale) else {
      throw ProAppsError.invalid("Exact movie clock exceeds its budget")
    }
    return Int32(scale)
  }

  public func write(_ request: MoviePCMMuxRequest) async throws -> MoviePCMMuxResult {
    try Task.checkCancellation()
    let root = try Files.absolute(request.rootDirectory)
    let directory = try Files.absolute(request.outputDirectory)
    let video = try Files.existing(request.videoPath, extensions: ["mov", "mp4"])
    let audio = try Files.existing(request.audioPath, extensions: ["wav"])
    guard root.path != "/", root == root.resolvingSymlinksInPath(),
      directory == directory.resolvingSymlinksInPath(),
      directory == root || directory.path.hasPrefix(root.path + "/"),
      video.path.hasPrefix(root.path + "/"), audio.path.hasPrefix(root.path + "/"),
      URL(fileURLWithPath: request.outputName).pathExtension == "mov"
    else { throw ProAppsError.invalid("PCM mux paths must stay inside the canonical root") }
    // cp -c falls back on cross-volume or non-cloning filesystems. Refuse both
    // documented fallback cases before dispatch; this is not a general file copy.
    let sourceVolume = try video.resourceValues(forKeys: [
      .volumeURLKey, .volumeSupportsFileCloningKey,
    ])
    let outputVolume = try directory.resourceValues(forKeys: [
      .volumeURLKey, .volumeSupportsFileCloningKey,
    ])
    guard sourceVolume.volumeSupportsFileCloning == true,
      outputVolume.volumeSupportsFileCloning == true,
      let volume = sourceVolume.volume, volume == outputVolume.volume
    else { throw ProAppsError.invalid("PCM mux requires one clone-capable filesystem") }
    let options = [
      AVURLAssetReferenceRestrictionsKey: AVAssetReferenceRestrictions.forbidAll.rawValue
    ]
    let videoAsset = AVURLAsset(url: video, options: options)
    let audioAsset = AVURLAsset(url: audio, options: options)
    let videos = try await videoAsset.loadTracks(withMediaType: .video)
    let existingAudio = try await videoAsset.loadTracks(withMediaType: .audio)
    let audios = try await audioAsset.loadTracks(withMediaType: .audio)
    guard videos.count == 1, let videoTrack = videos.first, existingAudio.isEmpty,
      audios.count == 1, let audioTrack = audios.first,
      try await videoTrack.load(.isSelfContained),
      try await audioTrack.load(.isSelfContained),
      try await !videoAsset.load(.containsFragments)
    else {
      throw ProAppsError.invalid("PCM mux requires self-contained video-only and audio inputs")
    }
    for sourceTrack in try await videoAsset.load(.tracks) {
      let type = sourceTrack.mediaType
      guard [.video, .timecode].contains(type), try await sourceTrack.load(.isSelfContained) else {
        throw ProAppsError.invalid(
          "Video input contains an unsupported or externally referenced track")
      }
    }
    let formats = try await audioTrack.load(.formatDescriptions)
    guard formats.count == 1, let format = formats.first,
      CMFormatDescriptionGetMediaSubType(format) == kAudioFormatLinearPCM
    else { throw ProAppsError.invalid("PCM mux accepts uncompressed PCM only") }
    let videoRange = try await videoTrack.load(.timeRange)
    let audioRange = try await audioTrack.load(.timeRange)
    let frame = try await videoTrack.load(.minFrameDuration)
    let gap = videoRange.duration - audioRange.duration
    guard videoRange.start == .zero, audioRange.start == .zero,
      videoRange.duration.isNumeric, audioRange.duration.isNumeric,
      audioRange.duration > .zero, videoRange.duration.seconds <= 6 * 3600,
      frame.isNumeric, frame > .zero, gap >= .zero, gap < frame
    else { throw ProAppsError.invalid("PCM must cover the zero-origin video to within one frame") }
    try Task.checkCancellation()
    let output = try Files.reserveOutput(
      directory: directory.path, name: request.outputName, kind: .edit)
    let copied = try await Runner.run(
      URL(fileURLWithPath: "/bin/cp"), ["-c", "-n", video.path, output.path], timeout: .seconds(30))
    guard copied.status == 0 else { throw ProAppsError.commandFailed(copied.status) }
    try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: output.path)
    try Task.checkCancellation()
    let movie = AVMutableMovie(url: output, options: nil)
    movie.timescale = try Self.commonTimescale(movie.timescale, audioRange.duration.timescale)
    movie.defaultMediaDataStorage = AVMediaDataStorage(url: output, options: nil)
    guard
      let track = movie.addMutableTrack(
        withMediaType: .audio, copySettingsFrom: audioTrack, options: nil)
    else { throw ProAppsError.unavailable("Cannot create PCM movie track") }
    try track.insertTimeRange(audioRange, of: audioTrack, at: .zero, copySampleData: true)
    try Task.checkCancellation()
    try movie.writeHeader(to: output, fileType: .mov, options: [])
    let final = AVURLAsset(url: output, options: options)
    let finalVideos = try await final.loadTracks(withMediaType: .video)
    let finalAudios = try await final.loadTracks(withMediaType: .audio)
    guard finalVideos.count == 1, finalAudios.count == 1,
      let finalVideo = finalVideos.first, let finalAudio = finalAudios.first,
      try await finalVideo.load(.isSelfContained), try await finalAudio.load(.isSelfContained),
      try await finalVideo.load(.timeRange) == videoRange,
      try await finalAudio.load(.timeRange) == audioRange
    else { throw ProAppsError.invalid("Muxed track containment or clocks changed") }
    return MoviePCMMuxResult(
      outputPath: output.path,
      videoDuration: .init(
        value: videoRange.duration.value, timescale: videoRange.duration.timescale),
      audioDuration: .init(
        value: audioRange.duration.value, timescale: audioRange.duration.timescale))
  }
}
