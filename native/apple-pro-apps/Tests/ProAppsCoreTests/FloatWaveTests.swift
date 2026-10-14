import Foundation
import Testing

@testable import ProAppsCore

struct FloatWaveTests {
  @Test func encodesExactIEEEFloatStereoWithoutClipping() throws {
    let wave = try FloatWave.stereo(left: [1, 0.5], right: [-1, -0.5], sampleRate: 44100)
    #expect(wave.count == 60)
    #expect(Array(wave[20..<24]) == [3, 0, 2, 0])
    #expect(Array(wave[24..<28]) == [68, 172, 0, 0])
    #expect(Array(wave[44..<60]) == [0, 0, 128, 63, 0, 0, 128, 191, 0, 0, 0, 63, 0, 0, 0, 191])
  }

  @Test func rejectsMalformedInputs() {
    #expect(throws: ProAppsError.self) {
      try FloatWave.stereo(left: [], right: [], sampleRate: 44100)
    }
    #expect(throws: ProAppsError.self) {
      try FloatWave.stereo(left: [0], right: [0, 1], sampleRate: 44100)
    }
    #expect(throws: ProAppsError.self) {
      try FloatWave.stereo(left: [.nan], right: [0], sampleRate: 44100)
    }
    #expect(throws: ProAppsError.self) {
      try FloatWave.stereo(left: [0], right: [0], sampleRate: 1)
    }
  }
}
