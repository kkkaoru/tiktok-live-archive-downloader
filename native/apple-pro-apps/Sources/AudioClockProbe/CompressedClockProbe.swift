import AVFoundation
import CryptoKit
import Foundation

struct ProbeTime: Codable {
  let value: Int64
  let timescale: Int32

  init(_ time: CMTime) {
    value = time.value
    timescale = time.timescale
  }
}

struct ProbeEdit: Codable {
  let empty: Bool
  let sourceStart: ProbeTime
  let sourceDuration: ProbeTime
  let targetStart: ProbeTime
  let targetDuration: ProbeTime
}

struct ProbePacket: Codable {
  let pts: ProbeTime
  let duration: ProbeTime
  let offset: Int64
  let bytes: Int64
  let sha256: String
  let refreshSamples: Int
}

struct CompressedClockResult: Codable {
  let edits: [ProbeEdit]
  let packets: [ProbePacket]
}

/// Read-only sample-table identity. These clocks do not certify decoded presentation.
enum CompressedClockProbe {
  static func read(_ request: ClockProbeRequest) async throws -> CompressedClockResult {
    let source = URL(fileURLWithPath: request.source)
    let asset = AVURLAsset(url: source)
    let tracks = try await asset.loadTracks(withMediaType: .audio)
    guard tracks.count == 1, let track = tracks.first else {
      throw ClockProbeError.unsupportedAudio
    }
    let segments = try await track.load(.segments)
    let edits = segments.map {
      ProbeEdit(
        empty: $0.isEmpty, sourceStart: ProbeTime($0.timeMapping.source.start),
        sourceDuration: ProbeTime($0.timeMapping.source.duration),
        targetStart: ProbeTime($0.timeMapping.target.start),
        targetDuration: ProbeTime($0.timeMapping.target.duration))
    }
    let start = CMTime(value: request.startValue, timescale: request.timescale)
    let end = start + CMTime(value: request.durationValue, timescale: request.timescale)
    guard let cursor = track.makeSampleCursor(presentationTimeStamp: start) else {
      throw ClockProbeError.unsupportedAudio
    }
    let handle = try FileHandle(forReadingFrom: source)
    defer { handle.closeFile() }
    var packets: [ProbePacket] = []
    let deadline = ContinuousClock.now.advanced(by: .seconds(30))
    while cursor.presentationTimeStamp < end {
      try Task.checkCancellation()
      guard packets.count < 10000, ContinuousClock.now < deadline else {
        throw ClockProbeError.budgetExceeded
      }
      let range = cursor.currentSampleStorageRange
      let data = try packetData(
        source: source, location: cursor.currentChunkStorageURL, range: range, handle: handle)
      packets.append(
        ProbePacket(
          pts: ProbeTime(cursor.presentationTimeStamp),
          duration: ProbeTime(cursor.currentSampleDuration),
          offset: range.offset, bytes: range.length,
          sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
          refreshSamples: cursor.samplesRequiredForDecoderRefresh))
      if cursor.stepInDecodeOrder(byCount: 1) == 0 { break }
    }
    guard !packets.isEmpty else { throw ClockProbeError.readerFailed }
    return CompressedClockResult(edits: edits, packets: packets)
  }

  static func packetData(
    source: URL, location: URL?, range: AVSampleCursorStorageRange, handle: FileHandle
  ) throws -> Data {
    // Missing storage identity cannot certify which file owns the cursor's byte range.
    guard let location, location.standardizedFileURL == source.standardizedFileURL,
      range.offset >= 0, range.length > 0, range.length <= 1_048_576
    else { throw ClockProbeError.unsupportedAudio }
    try handle.seek(toOffset: UInt64(range.offset))
    guard let data = try handle.read(upToCount: Int(range.length)), data.count == range.length
    else {
      throw ClockProbeError.readerFailed
    }
    return data
  }
}
