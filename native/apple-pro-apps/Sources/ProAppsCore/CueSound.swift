import Foundation

/// A local synthesized cue track. Times are output seconds, rounded to PCM samples.
/// Each nonoverlapping cue is an 80ms, 880Hz sine with a smooth zero-endpoint
/// envelope. No playback, recording, downloaded assets or source audio is involved.
public struct CueSound: Codable, Sendable {
  public static let maximumCues = 120
  public let durationSeconds: Double
  public let onsetSeconds: [Double]
  public let gain: Double

  public init(durationSeconds: Double, onsetSeconds: [Double], gain: Double) {
    self.durationSeconds = durationSeconds
    self.onsetSeconds = onsetSeconds
    self.gain = gain
  }

  public func wave() throws -> Data {
    try Task.checkCancellation()
    let sampleRate = 16000
    let maximumSeconds = 120.0
    let maximumGain = 0.25
    let cueFrames = 1280
    let frequency = 880.0
    let bytesPerFrame = 2
    guard durationSeconds.isFinite, durationSeconds > 0, durationSeconds <= maximumSeconds,
      gain.isFinite, (0...maximumGain).contains(gain), onsetSeconds.count <= Self.maximumCues
    else {
      throw ProAppsError.invalid(
        "Cue track requires 0–120 seconds, at most 120 cues and gain 0–0.25")
    }
    let frames = Int((durationSeconds * Double(sampleRate)).rounded())
    guard frames > 0 else {
      throw ProAppsError.invalid("Cue track must contain at least one PCM frame")
    }
    var starts: [Int] = []
    var previousEnd = 0
    for onset in onsetSeconds {
      guard onset.isFinite, onset >= 0, onset <= durationSeconds else {
        throw ProAppsError.invalid("Cue onset must be finite and inside the track")
      }
      let start = Int((onset * Double(sampleRate)).rounded())
      guard start >= previousEnd, start + cueFrames <= frames else {
        throw ProAppsError.invalid("Ordered 80ms cues must not overlap or exceed the track")
      }
      starts.append(start)
      previousEnd = start + cueFrames
    }
    var pcm = Data(repeating: 0, count: frames * bytesPerFrame)
    for start in starts {
      try Task.checkCancellation()
      for index in 0..<cueFrames {
        let phase = Double(index) / Double(cueFrames - 1)
        let envelope = pow(sin(Double.pi * phase), 2)
        let tone = sin(2 * Double.pi * frequency * Double(index) / Double(sampleRate))
        let value = Int16((Double(Int16.max) * gain * envelope * tone).rounded())
        let bits = UInt16(bitPattern: value)
        let offset = (start + index) * bytesPerFrame
        pcm[offset] = UInt8(truncatingIfNeeded: bits)
        pcm[offset + 1] = UInt8(truncatingIfNeeded: bits >> 8)
      }
    }
    // Fixed little-endian mono PCM16 RIFF header, using checked bounded lengths
    // and safe value shifts, never raw pointers or native-memory serialization.
    var header = Data("RIFF".utf8)
    let riffHeaderBytes: UInt32 = 36
    let riffSize = UInt32(pcm.count) + riffHeaderBytes
    header.append(contentsOf: (0..<4).map { UInt8(truncatingIfNeeded: riffSize >> ($0 * 8)) })
    header.append(Data("WAVEfmt ".utf8))
    header.append(contentsOf: [16, 0, 0, 0, 1, 0, 1, 0, 128, 62, 0, 0, 0, 125, 0, 0, 2, 0, 16, 0])
    header.append(Data("data".utf8))
    let byteCount = UInt32(pcm.count)
    header.append(contentsOf: (0..<4).map { UInt8(truncatingIfNeeded: byteCount >> ($0 * 8)) })
    header.append(pcm)
    return header
  }
}
