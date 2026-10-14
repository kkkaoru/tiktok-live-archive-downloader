import Foundation

/// Bounded radix-2 complex FFT using safe Swift arrays, without raw vDSP pointers.
/// Inverse transforms include 1/N normalization. Plans are immutable and reusable.
struct AudioFFT: Sendable {
  let size: Int
  private let permutation: [Int]
  private let cosine: [Float]
  private let sine: [Float]

  init(size: Int) throws {
    guard size >= 2, size <= 4096, size.nonzeroBitCount == 1 else {
      throw ProAppsError.invalid("Audio FFT requires a power of two from 2 to 4096")
    }
    self.size = size
    let bits = size.trailingZeroBitCount
    permutation = (0..<size).map { value in
      var remaining = value
      var reversed = 0
      for _ in 0..<bits {
        reversed = (reversed << 1) | (remaining & 1)
        remaining >>= 1
      }
      return reversed
    }
    cosine = (0..<(size / 2)).map { Float(cos(2 * .pi * Double($0) / Double(size))) }
    sine = (0..<(size / 2)).map { Float(sin(2 * .pi * Double($0) / Double(size))) }
  }

  func transform(real: [Float], imaginary: [Float], inverse: Bool = false)
    throws -> (real: [Float], imaginary: [Float])
  {
    try Task.checkCancellation()
    guard real.count == size, imaginary.count == size,
      real.allSatisfy(\.isFinite), imaginary.allSatisfy(\.isFinite)
    else { throw ProAppsError.invalid("FFT input must have the planned size and finite samples") }
    var real = permutation.map { real[$0] }
    var imaginary = permutation.map { imaginary[$0] }
    var width = 2
    while width <= size {
      let half = width / 2
      let step = size / width
      for base in stride(from: 0, to: size, by: width) {
        for index in 0..<half {
          let twiddle = index * step
          let sign: Float = inverse ? 1 : -1
          let c = cosine[twiddle]
          let s = sign * sine[twiddle]
          let lower = base + index
          let upper = lower + half
          let r = real[upper] * c - imaginary[upper] * s
          let i = real[upper] * s + imaginary[upper] * c
          real[upper] = real[lower] - r
          imaginary[upper] = imaginary[lower] - i
          real[lower] += r
          imaginary[lower] += i
        }
      }
      width *= 2
    }
    if inverse {
      let scale = Float(size)
      real = real.map { $0 / scale }
      imaginary = imaginary.map { $0 / scale }
    }
    guard real.allSatisfy(\.isFinite), imaginary.allSatisfy(\.isFinite) else {
      throw ProAppsError.invalid("FFT output overflowed")
    }
    return (real, imaginary)
  }
}
