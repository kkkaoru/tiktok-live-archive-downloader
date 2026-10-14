import AVFoundation
import Dispatch
import Foundation

/// A video-only selection. Higher layer numbers are in front. All clocks are exact;
/// ForegroundTime is the existing validated rational source-clock representation.
public struct ReferenceSegment: Codable, Sendable {
  public let sourcePath: String
  public let sourceStart: ForegroundTime
  public let duration: ForegroundTime
  public let outputStart: ForegroundTime
  public let layer: Int
}

/// Local metadata-only interchange, not an encoded or self-contained delivery.
public struct ReferenceMovieRequest: Codable, Sendable {
  public static let maximumSegments = 1024
  public static let maximumLayers = 16
  public static let maximumSeconds = 6 * 3600
  public let rootDirectory: String
  public let outputPath: String
  public let timescale: Int32
  public let segments: [ReferenceSegment]

  func validate() throws {
    guard (1...ForegroundTime.maximumTimescale).contains(timescale),
      (1...Self.maximumSegments).contains(segments.count),
      segments.contains(where: { $0.layer == 0 })
    else { throw ProAppsError.invalid("Invalid reference movie limits or missing base layer") }
    var ends: [Int: CMTime] = [:]
    for segment in segments {
      let start = try segment.outputStart.time()
      let duration = try segment.duration.time()
      let source = try segment.sourceStart.time()
      let end = start + duration
      guard (0..<Self.maximumLayers).contains(segment.layer), duration > .zero,
        start >= (ends[segment.layer] ?? .zero), end.isNumeric,
        end.seconds <= Double(Self.maximumSeconds),
        [start, duration, source, source + duration, end].allSatisfy({
          $0.isNumeric && !$0.flags.contains(.hasBeenRounded)
            && !CMTimeConvertScale($0, timescale: timescale, method: .default).flags.contains(
              .hasBeenRounded)
        })
      else { throw ProAppsError.invalid("Invalid, overlapping or inexact reference selection") }
      ends[segment.layer] = end
    }
  }
}

public struct ReferenceMovieResult: Encodable, Sendable {
  public let outputPath: String
  public let sourcePaths: [String]
  public let videoTrackCount: Int
  public let headerBytes: Int
  public let duration: ForegroundTime
  public let selfContained: Bool
  public let rendered: Bool

  init(
    outputPath: String, sourcePaths: [String], videoTrackCount: Int, headerBytes: Int,
    duration: ForegroundTime
  ) {
    self.outputPath = outputPath
    self.sourcePaths = sourcePaths
    self.videoTrackCount = videoTrackCount
    self.headerBytes = headerBytes
    self.duration = duration
    selfContained = false
    rendered = false
  }
}

/// Owns one native graph on a bounded serial executor. Input sources are never
/// modified; callers retain and fingerprint every referenced file independently.
public actor ReferenceMovieWriter {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.reference-movie")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }

  public init() {}

  private struct Layer {
    let track: AVMutableMovieTrack
    let codec: FourCharCode
    let timescale: CMTimeScale
  }

  public func write(_ request: ReferenceMovieRequest) async throws -> ReferenceMovieResult {
    try Task.checkCancellation()
    try request.validate()
    let root = try Files.absolute(request.rootDirectory)
    guard root.path != "/", root == root.resolvingSymlinksInPath(),
      try root.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true
    else { throw ProAppsError.invalid("Reference root must be a canonical directory") }
    let output = try Files.absolute(request.outputPath)
    guard output.pathExtension == "mov",
      output.deletingLastPathComponent()
        == output.deletingLastPathComponent().resolvingSymlinksInPath(),
      output.path.hasPrefix(root.path + "/")
    else { throw ProAppsError.invalid("Reference output must be a local MOV within its root") }
    let movie = AVMutableMovie()
    movie.timescale = request.timescale
    var layers: [Int: Layer] = [:]
    var sources: [String] = []
    var canvas: CGSize?
    var expectedEnd = CMTime.zero
    for segment in request.segments {
      try Task.checkCancellation()
      let url = try Files.existing(segment.sourcePath, extensions: ["mov", "mp4"])
      guard url.path.hasPrefix(root.path + "/"), url != output else {
        throw ProAppsError.invalid("Reference source escapes its root or aliases the output")
      }
      let asset = AVURLAsset(
        url: url,
        options: [
          AVURLAssetReferenceRestrictionsKey: AVAssetReferenceRestrictions.forbidAll.rawValue
        ])
      let tracks = try await asset.loadTracks(withMediaType: .video)
      guard tracks.count == 1, let source = tracks.first else {
        throw ProAppsError.invalid("Reference input requires exactly one video track")
      }
      guard try await source.load(.isSelfContained),
        try await !asset.load(.containsFragments)
      else {
        throw ProAppsError.invalid("Reference input must be self-contained and unfragmented")
      }
      let size = try await source.load(.naturalSize)
      let transform = try await source.load(.preferredTransform)
      let formats = try await source.load(.formatDescriptions)
      let scale = try await source.load(.naturalTimeScale)
      guard transform.isIdentity, size.width >= 1, size.height >= 1,
        size.width <= 8192, size.height <= 8192, canvas == nil || canvas == size,
        formats.count == 1, let format = formats.first
      else { throw ProAppsError.invalid("Reference inputs must share an unrotated video canvas") }
      canvas = size
      let codec = CMFormatDescriptionGetMediaSubType(format)
      let range = CMTimeRange(
        start: try segment.sourceStart.time(), duration: try segment.duration.time())
      let available = try await source.load(.timeRange)
      guard available.containsTimeRange(range) else {
        throw ProAppsError.invalid("Reference selection exceeds its source track")
      }
      let layer: Layer
      if let existing = layers[segment.layer] {
        guard existing.codec == codec, existing.timescale == scale else {
          throw ProAppsError.invalid("Reference layer codec or media clock changed")
        }
        layer = existing
      } else {
        guard
          let track = movie.addMutableTrack(
            withMediaType: .video, copySettingsFrom: source, options: nil)
        else { throw ProAppsError.unavailable("Cannot create reference video track") }
        track.layer = -segment.layer
        layer = Layer(track: track, codec: codec, timescale: scale)
        layers[segment.layer] = layer
      }
      try Task.checkCancellation()
      try layer.track.insertTimeRange(
        range, of: source, at: segment.outputStart.time(), copySampleData: false)
      expectedEnd = max(expectedEnd, try segment.outputStart.time() + segment.duration.time())
      if !sources.contains(url.path) { sources.append(url.path) }
    }
    try Task.checkCancellation()
    let duration = try await movie.load(.duration)
    guard duration == expectedEnd else {
      throw ProAppsError.invalid("Reference movie duration differs from its exact plan")
    }
    let header = try movie.makeMovieHeader(fileType: .mov)
    let written = try Files.writeNew(header, to: output.path, extensions: ["mov"])
    return ReferenceMovieResult(
      outputPath: written.path, sourcePaths: sources, videoTrackCount: layers.count,
      headerBytes: header.count,
      duration: ForegroundTime(value: duration.value, timescale: duration.timescale))
  }
}
