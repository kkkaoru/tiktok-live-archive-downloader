import Foundation

/// Window endpoints are rounded to the nearest decoded sample. Returned start
/// and duration report that quantization, not an assumed decimal interval.
public struct AudioWindow: Codable, Sendable {
  public let startSeconds: Double
  public let durationSeconds: Double
  public init(startSeconds: Double, durationSeconds: Double) {
    self.startSeconds = startSeconds
    self.durationSeconds = durationSeconds
  }
}

public struct AudioLevel: Codable, Sendable {
  public let startSeconds: Double
  public let durationSeconds: Double
  public let frames: Int
  public let rms: Double
  public let peak: Double
  /// Signal zero-crossing rate, not a speech/music pitch or loudness estimate.
  public let zeroCrossingRateHz: Double
}

public struct PCMMeasurement: Codable, Sendable {
  public let sampleRate: Int
  public let whole: AudioLevel
  public let windows: [AudioLevel]

  public static let defaultDurationSeconds = 30.0
  public static let maximumDurationSeconds = 120.0
  public static let maximumWindows = 16

  /// Strict RIFF/WAVE mono signed little-endian PCM16 analysis. No raw pointers,
  /// alignment assumptions, arbitrary codec guesses or implicit channel mixing.
  public static func analyze(
    _ data: Data, windows: [AudioWindow],
    maximumDurationSeconds: Double = PCMMeasurement.defaultDurationSeconds
  ) throws -> PCMMeasurement {
    guard maximumDurationSeconds.isFinite, maximumDurationSeconds > 0,
      maximumDurationSeconds <= Self.maximumDurationSeconds
    else { throw ProAppsError.invalid("Audio duration budget must be >0–120 seconds") }
    guard data.count <= Files.maximumBytes, data.count >= 44, windows.count <= maximumWindows else {
      throw ProAppsError.invalid("PCM input/window limit exceeded")
    }
    // Normalize Data slice indices with a bounded safe collection.
    let bytes = [UInt8](data)
    guard tag(bytes, 0) == "RIFF", tag(bytes, 8) == "WAVE", integer32(bytes, 4) + 8 == bytes.count
    else {
      throw ProAppsError.invalid("Expected a complete bounded RIFF/WAVE file")
    }
    var sampleRate: Int?
    var pcm: Range<Int>?
    var offset = 12
    while offset <= bytes.count - 8 {
      let kind = tag(bytes, offset)
      let length = integer32(bytes, offset + 4)
      let body = offset + 8
      guard length <= bytes.count - body else { throw ProAppsError.invalid("Truncated WAV chunk") }
      if kind == "fmt " {
        guard sampleRate == nil, length >= 16 else {
          throw ProAppsError.invalid("Missing or duplicate WAV format")
        }
        let rate = integer32(bytes, body + 4)
        let formatCode = integer16(bytes, body)
        guard formatCode == 1 || formatCode == 65534, integer16(bytes, body + 2) == 1,
          (1...192000).contains(rate), integer32(bytes, body + 8) == rate * 2,
          integer16(bytes, body + 12) == 2, integer16(bytes, body + 14) == 16
        else {
          throw ProAppsError.invalid(
            "Expected normalized mono PCM16: format=\(integer16(bytes, body)), channels=\(integer16(bytes, body + 2)), rate=\(rate), byteRate=\(integer32(bytes, body + 8)), alignment=\(integer16(bytes, body + 12)), bits=\(integer16(bytes, body + 14))"
          )
        }
        if formatCode == 65534 {
          // WAVE_FORMAT_EXTENSIBLE: exact PCM subtype, valid bit depth and a
          // single/unspecified speaker. Never reinterpret a float/compressed GUID.
          guard length >= 40 else { throw ProAppsError.invalid("Truncated extensible WAV format") }
          let extensionBytes = integer16(bytes, body + 16)
          let channelMask = integer32(bytes, body + 20)
          let pcmGUID: [UInt8] = [1, 0, 0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113]
          guard extensionBytes >= 22, extensionBytes <= length - 18,
            integer16(bytes, body + 18) == 16,
            channelMask == 0 || (channelMask <= 131072 && (channelMask & (channelMask - 1)) == 0),
            Array(bytes[(body + 24)..<(body + 40)]) == pcmGUID
          else { throw ProAppsError.invalid("Invalid extensible mono PCM16 descriptor") }
        }
        sampleRate = rate
      } else if kind == "data" {
        guard pcm == nil, length > 0, length.isMultiple(of: 2) else {
          throw ProAppsError.invalid("Invalid or duplicate PCM data")
        }
        pcm = body..<(body + length)
      }
      offset = body + length + length % 2
    }
    guard offset == bytes.count, let rate = sampleRate, let pcm else {
      throw ProAppsError.invalid("Incomplete WAV structure")
    }
    let frames = pcm.count / 2
    guard Double(frames) / Double(rate) <= maximumDurationSeconds else {
      throw ProAppsError.invalid("Audio exceeds the requested measurement duration budget")
    }
    var measured: [AudioLevel] = []
    for window in windows {
      guard window.startSeconds.isFinite, window.durationSeconds.isFinite,
        window.startSeconds >= 0, window.durationSeconds > 0,
        window.startSeconds + window.durationSeconds <= Double(frames) / Double(rate)
      else { throw ProAppsError.invalid("Audio window lies outside decoded samples") }
      let lower = Int((window.startSeconds * Double(rate)).rounded())
      let upper = Int(
        ((window.startSeconds + window.durationSeconds) * Double(rate)).rounded())
      guard lower < upper, upper <= frames else {
        throw ProAppsError.invalid("Audio window rounds to no samples")
      }
      measured.append(level(bytes, pcmStart: pcm.lowerBound, frames: lower..<upper, rate: rate))
    }
    return PCMMeasurement(
      sampleRate: rate,
      whole: level(bytes, pcmStart: pcm.lowerBound, frames: 0..<frames, rate: rate),
      windows: measured)
  }

  /// Decode samples only after the shared RIFF validator has accepted the file.
  /// Bounds, duplicate chunks, PCM encoding and duration are validated by analyze.
  public static func decodedSamples(_ data: Data) throws -> (sampleRate: Int, samples: [Double]) {
    let measurement = try analyze(data, windows: [], maximumDurationSeconds: 60)
    let bytes = [UInt8](data)
    var offset = 12
    while tag(bytes, offset) != "data" {
      let length = integer32(bytes, offset + 4)
      offset += 8 + length + length % 2
    }
    let lower = offset + 8
    let count = integer32(bytes, offset + 4) / 2
    var samples: [Double] = []
    samples.reserveCapacity(count)
    for index in 0..<count {
      if index.isMultiple(of: 4096) { try Task.checkCancellation() }
      let position = lower + 2 * index
      let bits = UInt16(bytes[position]) | UInt16(bytes[position + 1]) << 8
      samples.append(Double(Int16(bitPattern: bits)) / 32768)
    }
    return (measurement.sampleRate, samples)
  }

  private static func level(_ bytes: [UInt8], pcmStart: Int, frames: Range<Int>, rate: Int)
    -> AudioLevel
  {
    let signed16Scale = 32768.0
    var squares = 0.0
    var peak = 0.0
    var previous: Double?
    var crossings = 0
    for frame in frames {
      let offset = pcmStart + frame * 2
      let bits = UInt16(bytes[offset]) | UInt16(bytes[offset + 1]) << 8
      let sample = Double(Int16(bitPattern: bits)) / signed16Scale
      squares += sample * sample
      peak = max(peak, abs(sample))
      if let previous, (previous < 0 && sample >= 0) || (previous >= 0 && sample < 0) {
        crossings += 1
      }
      previous = sample
    }
    let duration = Double(frames.count) / Double(rate)
    return AudioLevel(
      startSeconds: Double(frames.lowerBound) / Double(rate), durationSeconds: duration,
      frames: frames.count, rms: sqrt(squares / Double(frames.count)), peak: peak,
      zeroCrossingRateHz: Double(crossings) / duration)
  }

  private static func tag(_ bytes: [UInt8], _ offset: Int) -> String {
    String(decoding: bytes[offset..<(offset + 4)], as: UTF8.self)
  }

  private static func integer16(_ bytes: [UInt8], _ offset: Int) -> Int {
    Int(bytes[offset]) | Int(bytes[offset + 1]) << 8
  }

  private static func integer32(_ bytes: [UInt8], _ offset: Int) -> Int {
    Int(bytes[offset]) | Int(bytes[offset + 1]) << 8 | Int(bytes[offset + 2]) << 16 | Int(
      bytes[offset + 3]) << 24
  }
}
