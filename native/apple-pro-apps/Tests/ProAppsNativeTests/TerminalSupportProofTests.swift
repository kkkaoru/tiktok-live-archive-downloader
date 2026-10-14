import AVFoundation
import CoreImage
import Foundation
import Testing

@testable import ProAppsCore

// Synthetic-only proof. Keep artifacts; never use this helper to process recordings.
@Suite(.serialized, .timeLimit(.minutes(2)))
struct TerminalSupportProofTests {
  private enum Method { case adaptor, explicitTiming }

  private struct Support {
    let end: CMTime
    let scale: CMTimeScale
    let starts: [CMTime]

    init(end: CMTime, scale: CMTimeScale) throws {
      guard end.isNumeric, end > .zero, end <= CMTime(value: 1, timescale: 1),
        scale > 0, scale % 30 == 0
      else { throw ProAppsError.invalid("Unsupported synthetic support clock") }
      let converted = CMTimeConvertScale(end, timescale: scale, method: .default)
      guard converted == end, !converted.flags.contains(.hasBeenRounded) else {
        throw ProAppsError.invalid("Endpoint is not exactly representable")
      }
      self.end = converted
      self.scale = scale
      starts = (0..<30).map { CMTime(value: Int64($0), timescale: 30) }.filter { $0 < end }
    }
  }

  private struct Sample: Decodable {
    let pts: Int64
    let duration: Int64
  }

  private struct Stream: Decodable {
    let codecName: String
    let timeBase: String
    let durationTs: Int64
    let nbReadFrames: String?
  }

  private struct Probe: Decodable {
    let frames: [Sample]?
    let packets: [Sample]?
    let streams: [Stream]
  }

  @Test func rejectsInexactClocks() throws {
    #expect(throws: ProAppsError.self) {
      try Support(end: CMTime(value: 34, timescale: 125), scale: 15360)
    }
    #expect(throws: ProAppsError.self) {
      try Support(end: CMTime(value: 34, timescale: 125), scale: 0)
    }
    #expect(throws: ProAppsError.self) {
      try Support(end: .zero, scale: 60000)
    }
  }

  @Test(arguments: [16320, 18000, 16001])
  func explicitTimingPreservesTerminalSupport(endTicks: Int64) async throws {
    let support = try Support(end: CMTime(value: endTicks, timescale: 60000), scale: 60000)
    #expect(support.starts.count == 9)
    let root = try directory()
    let movie = root.appendingPathComponent("explicit-\(endTicks).mp4")
    try await write(support, to: movie, method: .explicitTiming)
    let frames = try await probe(movie, frames: true)
    let packets = try await probe(movie, frames: false)
    let rawPackets = try await probe(movie, frames: false, ignoreEditList: true)
    let expectedPTS: [Int64] = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
    let expectedDurations = Array(repeating: Int64(2000), count: 8) + [endTicks - 16000]
    #expect(try #require(frames.frames).map(\.pts) == expectedPTS)
    #expect(try #require(frames.frames).map(\.duration) == expectedDurations)
    #expect(try #require(packets.packets).map(\.pts) == expectedPTS)
    #expect(try #require(packets.packets).map(\.duration) == expectedDurations)
    #expect(try #require(rawPackets.packets).map(\.pts) == expectedPTS)
    #expect(try #require(rawPackets.packets).map(\.duration) == expectedDurations)
    let stream = try #require(frames.streams.first)
    #expect(stream.codecName == "hevc")
    #expect(stream.timeBase == "1/60000")
    #expect(stream.durationTs == endTicks)
    #expect(stream.nbReadFrames == "9")
    let hashes = try await decodedHashes(movie)
    #expect(hashes.count == 9)
    #expect(Set(hashes).count == 9)
  }

  @Test func sessionEndAloneDoesNotShortenPacketSupport() async throws {
    let support = try Support(end: CMTime(value: 34, timescale: 125), scale: 60000)
    let movie = try directory().appendingPathComponent("adaptor-exact-clocks.mp4")
    try await write(support, to: movie, method: .adaptor)
    let frames = try await probe(movie, frames: true)
    let packets = try await probe(movie, frames: false, ignoreEditList: true)
    #expect(try #require(frames.streams.first).durationTs == 16320)
    #expect(try #require(frames.frames).count == 9)
    #expect(try #require(packets.packets).count == 9)
    #expect(try #require(packets.packets?.last).duration == 2000)
    #expect(try #require(packets.packets?.last).duration != 320)
  }

  private func directory() throws -> URL {
    let configured = try #require(ProcessInfo.processInfo.environment["TMPDIR"])
    let root = try Files.temporaryDirectory(path: configured)
    let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let repository = package.deletingLastPathComponent().deletingLastPathComponent()
    let allowed = repository.appendingPathComponent("edits/work").resolvingSymlinksInPath()
    guard root.path.hasPrefix(allowed.path + "/"), root.path == configured else {
      throw ProAppsError.invalid("Proof scratch must be canonical repository-local storage")
    }
    let result = root.appendingPathComponent("terminal-support-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: result, withIntermediateDirectories: false)
    print("Terminal support evidence: \(result.path)")
    return result
  }

  private func write(_ support: Support, to movie: URL, method: Method) async throws {
    let writer = try AVAssetWriter(outputURL: movie, fileType: .mp4)
    writer.movieTimeScale = support.scale
    let input = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: VideoWriterSettings.output(
        EditEncoding(codec: .hevc, averageBitRate: 500_000, allowFrameReordering: false),
        width: 64, height: 64, frameRate: 30))
    input.mediaTimeScale = support.scale
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: input,
      sourcePixelBufferAttributes: VideoWriterSettings.pixelBuffers(width: 64, height: 64))
    try #require(writer.canAdd(input))
    writer.add(input)
    try #require(writer.startWriting())
    writer.startSession(atSourceTime: .zero)
    defer { if writer.status == .writing { writer.cancelWriting() } }
    let deadline = ContinuousClock.now.advanced(by: .seconds(20))
    let context = CIContext()
    let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
    let bounds = CGRect(x: 0, y: 0, width: 64, height: 64)
    for (index, pts) in support.starts.enumerated() {
      try Task.checkCancellation()
      let pool = try #require(adaptor.pixelBufferPool)
      var allocated: CVPixelBuffer?
      try #require(CVPixelBufferPoolCreatePixelBuffer(nil, pool, &allocated) == kCVReturnSuccess)
      let pixel = try #require(allocated)
      // Each genuine sample has a different gray level, including the short tail.
      let level = CGFloat(index + 1) / 10
      let image = CIImage(color: CIColor(red: level, green: level, blue: level))
      context.render(image.cropped(to: bounds), to: pixel, bounds: bounds, colorSpace: space)
      while !input.isReadyForMoreMediaData {
        guard ContinuousClock.now < deadline, writer.status == .writing else {
          throw ProAppsError.timedOut
        }
        try await Task.sleep(for: .milliseconds(2))
      }
      switch method {
      case .adaptor:
        try #require(adaptor.append(pixel, withPresentationTime: pts))
      case .explicitTiming:
        let end = min(CMTimeAdd(pts, CMTime(value: 1, timescale: 30)), support.end)
        try #require(input.append(try timedSample(pixel, start: pts, end: end)))
      }
    }
    writer.endSession(atSourceTime: support.end)
    input.markAsFinished()
    await writer.finishWriting()
    try #require(writer.status == .completed)
  }

  private func timedSample(
    _ pixel: CVPixelBuffer, start: CMTime, end: CMTime
  ) throws -> CMSampleBuffer {
    var description: CMVideoFormatDescription?
    try #require(
      CMVideoFormatDescriptionCreateForImageBuffer(
        allocator: kCFAllocatorDefault, imageBuffer: pixel, formatDescriptionOut: &description)
        == noErr)
    let format = try #require(description)
    var timing = CMSampleTimingInfo(
      duration: CMTimeSubtract(end, start), presentationTimeStamp: start, decodeTimeStamp: .invalid)
    var sample: CMSampleBuffer?
    try #require(
      CMSampleBufferCreateReadyWithImageBuffer(
        allocator: kCFAllocatorDefault, imageBuffer: pixel, formatDescription: format,
        sampleTiming: &timing, sampleBufferOut: &sample) == noErr)
    return try #require(sample)
  }

  private func decodedHashes(_ movie: URL) async throws -> [String] {
    let executable = try Files.existing("/opt/homebrew/bin/ffmpeg")
    let result = try await Runner.run(
      executable,
      [
        "-v", "error", "-nostdin", "-i", movie.path, "-map", "0:v:0", "-fps_mode",
        "passthrough", "-f", "framemd5", "-"
      ], timeout: .seconds(20))
    let evidence = movie.deletingPathExtension().appendingPathExtension("framemd5")
    _ = try Files.writeNew(Data(result.stdout.utf8), to: evidence.path, extensions: ["framemd5"])
    try #require(result.status == 0)
    return try result.stdout.split(separator: "\n").filter { !$0.hasPrefix("#") }.map { line in
      let fields = line.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
      try #require(fields.count == 6)
      return try #require(fields.last)
    }
  }

  private func probe(
    _ movie: URL, frames: Bool, ignoreEditList: Bool = false
  ) async throws -> Probe {
    let executable = try Files.existing("/opt/homebrew/bin/ffprobe")
    let kind = frames ? "frames" : "packets"
    let entries = frames ? "frame=pts,duration" : "packet=pts,duration"
    let arguments = [
      "-ignore_editlist", ignoreEditList ? "1" : "0",
      "-v", "error", "-select_streams", "v:0", "-show_\(kind)", "-count_frames",
      "-show_streams", "-show_entries",
      entries + ":stream=codec_name,time_base,duration_ts,nb_read_frames", "-of", "json",
      movie.path
    ]
    let result = try await Runner.run(executable, arguments, timeout: .seconds(20))
    let suffix = ignoreEditList ? "raw-\(kind)" : kind
    let evidence = movie.deletingPathExtension().appendingPathExtension("\(suffix).json")
    _ = try Files.writeNew(Data(result.stdout.utf8), to: evidence.path, extensions: ["json"])
    try #require(result.status == 0)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(Probe.self, from: Data(result.stdout.utf8))
  }
}
