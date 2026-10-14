import Foundation

/// Exact fixed model boundary: normalized 4096-point periodic-Hann STFT,
/// 1024-sample hop, 336 frames, no Nyquist bin, channel-major real/imag planes.
/// Frame padding matches HTDemucs _spec (left 1536, right 1620, reflection).
enum DemucsSpectrum {
  static let sampleCount = 343_980
  static let fftSize = 4096
  static let hop = 1024
  static let frames = 336
  static let bins = 2048
  static let planeCount = bins * frames
  private static let leftPadding = 1536
  private static let normalization: Float = 64
  private static let window = (0..<fftSize).map {
    Float(0.5 * (1 - cos(2 * .pi * Double($0) / Double(fftSize))))
  }

  static func forward(_ signal: [Float]) throws -> (real: [Float], imaginary: [Float]) {
    guard signal.count == sampleCount, signal.allSatisfy(\.isFinite) else {
      throw ProAppsError.invalid("Demucs input requires exactly 343980 finite samples per channel")
    }
    let fft = try AudioFFT(size: fftSize)
    let zeros = [Float](repeating: 0, count: fftSize)
    var real = [Float](repeating: 0, count: planeCount)
    var imaginary = real
    for frame in 0..<frames {
      try Task.checkCancellation()
      let samples = (0..<fftSize).map { index in
        let position = frame * hop + index - leftPadding
        let reflected: Int
        if position < 0 {
          reflected = -position
        } else if position >= sampleCount {
          reflected = 2 * sampleCount - 2 - position
        } else {
          reflected = position
        }
        return signal[reflected] * window[index]
      }
      let transformed = try fft.transform(real: samples, imaginary: zeros)
      for bin in 0..<bins {
        real[bin * frames + frame] = transformed.real[bin] / normalization
        imaginary[bin * frames + frame] = transformed.imaginary[bin] / normalization
      }
    }
    return (real, imaginary)
  }

  /// Restore omitted Nyquist as zero, then overlap-add and remove reflection
  /// padding. Equivalent to _ispec's zero time-frame padding and center trim.
  static func inverse(real: [Float], imaginary: [Float]) throws -> [Float] {
    guard real.count == planeCount, imaginary.count == planeCount,
      real.allSatisfy(\.isFinite), imaginary.allSatisfy(\.isFinite)
    else { throw ProAppsError.invalid("Demucs spectrum shape or finite-value check failed") }
    let fft = try AudioFFT(size: fftSize)
    var output = [Float](repeating: 0, count: sampleCount)
    var weights = output
    for frame in 0..<frames {
      try Task.checkCancellation()
      var r = [Float](repeating: 0, count: fftSize)
      var i = r
      r[0] = real[frame] * normalization
      for bin in 1..<bins {
        let index = bin * frames + frame
        r[bin] = real[index] * normalization
        i[bin] = imaginary[index] * normalization
        r[fftSize - bin] = r[bin]
        i[fftSize - bin] = -i[bin]
      }
      let samples = try fft.transform(real: r, imaginary: i, inverse: true).real
      let start = frame * hop - leftPadding
      let lower = max(0, -start)
      let upper = min(fftSize, sampleCount - start)
      for index in lower..<upper {
        output[start + index] += samples[index] * window[index]
        weights[start + index] += window[index] * window[index]
      }
    }
    return try normalize(output: output, weights: weights)
  }

  static func normalize(output: [Float], weights: [Float]) throws -> [Float] {
    guard output.count == weights.count, !output.isEmpty else {
      throw ProAppsError.invalid("Overlap-add arrays must be nonempty and equal in length")
    }
    return try zip(output, weights).map { sample, weight in
      guard sample.isFinite, weight.isFinite, weight > 1e-8 else {
        throw ProAppsError.invalid("Overlap-add contains uncovered or nonfinite samples")
      }
      let normalized = sample / weight
      guard normalized.isFinite else { throw ProAppsError.invalid("Overlap-add overflowed") }
      return normalized
    }
  }
}
