import Testing

@testable import ProAppsCore

struct DemucsSpectrumTests {
  @Test func constantSignalHasExpectedNormalizedSpectrumAndRoundTripsEdges() throws {
    let spectrum = try DemucsSpectrum.forward(Array(repeating: 0.125, count: 343_980))
    #expect(spectrum.real.count == 688_128)
    #expect(spectrum.imaginary.count == 688_128)
    #expect(abs(spectrum.real[0] - 4) < 0.0001)
    #expect(abs(spectrum.real[336] + 2) < 0.0001)
    #expect(abs(spectrum.imaginary[336]) < 0.0001)
    let restored = try DemucsSpectrum.inverse(real: spectrum.real, imaginary: spectrum.imaginary)
    #expect(restored.count == 343_980)
    #expect(abs(restored[0] - 0.125) < 0.0001)
    #expect(abs(restored[343_979] - 0.125) < 0.0001)
    #expect(restored.allSatisfy { abs($0 - 0.125) < 0.0001 })
  }

  @Test func invalidDimensionsAndNormalizationAreRefused() {
    #expect(throws: ProAppsError.self) { try DemucsSpectrum.forward([0]) }
    #expect(throws: ProAppsError.self) { try DemucsSpectrum.inverse(real: [], imaginary: []) }
    #expect(throws: ProAppsError.self) { try DemucsSpectrum.normalize(output: [], weights: []) }
    #expect(throws: ProAppsError.self) { try DemucsSpectrum.normalize(output: [1], weights: [0]) }
    #expect(throws: ProAppsError.self) {
      try DemucsSpectrum.normalize(output: [.nan], weights: [1])
    }
    #expect(throws: ProAppsError.self) {
      try DemucsSpectrum.normalize(output: [.greatestFiniteMagnitude], weights: [0.01])
    }
  }
}
