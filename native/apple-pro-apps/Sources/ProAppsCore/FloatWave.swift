import Foundation

/// Bounded IEEE Float32 stereo WAV. No implicit clipping or normalization;
/// callers retain headroom measurements before lossy final encoding.
enum FloatWave {
  static func stereo(left: [Float], right: [Float], sampleRate: Int) throws -> Data {
    guard left.count == right.count, !left.isEmpty, left.count <= 441_000,
      (8000...96000).contains(sampleRate), left.allSatisfy(\.isFinite), right.allSatisfy(\.isFinite)
    else { throw ProAppsError.invalid("Invalid bounded stereo floating-point WAV samples or rate") }
    let byteCount = UInt32(left.count * 8)
    var data = Data("RIFF".utf8)
    append(byteCount + 36, to: &data)
    data.append(Data("WAVEfmt ".utf8))
    append(16, to: &data)
    data.append(contentsOf: [3, 0, 2, 0])
    append(UInt32(sampleRate), to: &data)
    append(UInt32(sampleRate * 8), to: &data)
    data.append(contentsOf: [8, 0, 32, 0])
    data.append(Data("data".utf8))
    append(byteCount, to: &data)
    data.reserveCapacity(Int(byteCount) + 44)
    for index in left.indices {
      if index.isMultiple(of: 4096) { try Task.checkCancellation() }
      append(left[index].bitPattern, to: &data)
      append(right[index].bitPattern, to: &data)
    }
    return data
  }

  private static func append(_ value: UInt32, to data: inout Data) {
    data.append(UInt8(truncatingIfNeeded: value))
    data.append(UInt8(truncatingIfNeeded: value >> 8))
    data.append(UInt8(truncatingIfNeeded: value >> 16))
    data.append(UInt8(truncatingIfNeeded: value >> 24))
  }
}
