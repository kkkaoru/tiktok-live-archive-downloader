import Testing

@testable import ProAppsCore

struct AudioFFTTests {
  @Test func impulseAndInverseHaveKnownScaling() throws {
    let fft = try AudioFFT(size: 4)
    let frequency = try fft.transform(real: [1, 0, 0, 0], imaginary: [0, 0, 0, 0])
    #expect(frequency.real == [1, 1, 1, 1])
    #expect(frequency.imaginary == [0, 0, 0, 0])
    let output = try fft.transform(real: [1, 1, 1, 1], imaginary: [0, 0, 0, 0], inverse: true)
    #expect(output.real == [1, 0, 0, 0])
    #expect(output.imaginary == [0, 0, 0, 0])
  }

  @Test func sineUsesNegativeForwardExponent() throws {
    let output = try AudioFFT(size: 4).transform(real: [0, 1, 0, -1], imaginary: [0, 0, 0, 0])
    #expect(abs(output.real[1]) < 0.000001)
    #expect(output.imaginary == [0, -2, 0, 2])
  }

  @Test(arguments: [0, 1, 3, 8192, Int.max])
  func rejectsInvalidSize(_ size: Int) {
    #expect(throws: ProAppsError.self) { try AudioFFT(size: size) }
  }

  @Test func rejectsMismatchedNonfiniteAndOverflowingData() throws {
    let fft = try AudioFFT(size: 2)
    #expect(throws: ProAppsError.self) { try fft.transform(real: [1], imaginary: [0, 0]) }
    #expect(throws: ProAppsError.self) { try fft.transform(real: [.nan, 0], imaginary: [0, 0]) }
    #expect(throws: ProAppsError.self) {
      try fft.transform(
        real: [.greatestFiniteMagnitude, .greatestFiniteMagnitude], imaginary: [0, 0])
    }
  }
}
