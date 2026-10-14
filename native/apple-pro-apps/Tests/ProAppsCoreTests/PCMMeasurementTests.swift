import Foundation
import Testing

@testable import ProAppsCore

struct PCMMeasurementTests {
  private func little(_ value: Int) -> [UInt8] {
    [
      UInt8(truncatingIfNeeded: value), UInt8(truncatingIfNeeded: value >> 8),
      UInt8(truncatingIfNeeded: value >> 16), UInt8(truncatingIfNeeded: value >> 24),
    ]
  }

  private func wave(
    samples: [Int16] = [0, 32767, -32768, 16384], rate: Int = 4, extra: [UInt8] = []
  ) -> Data {
    let pcm = samples.flatMap { sample in
      let bits = UInt16(bitPattern: sample)
      return [UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)]
    }
    let format =
      Array("fmt ".utf8) + little(16) + [1, 0, 1, 0] + little(rate) + little(rate * 2) + [
        2, 0, 16, 0,
      ]
    let body = Array("WAVE".utf8) + format + extra + Array("data".utf8) + little(pcm.count) + pcm
    return Data(Array("RIFF".utf8) + little(body.count) + body)
  }

  @Test func measuresSignedExtremaRMSAndWindowsWithoutOverflow() throws {
    let report = try PCMMeasurement.analyze(
      wave(), windows: [.init(startSeconds: 0.5, durationSeconds: 0.5)])
    #expect(report.sampleRate == 4)
    #expect(report.whole.frames == 4)
    #expect(report.whole.durationSeconds == 1)
    #expect(report.whole.peak == 1)
    #expect(abs(report.whole.rms - 0.749989828) < 0.00000001)
    #expect(report.whole.zeroCrossingRateHz == 2)
    let window = try #require(report.windows.first)
    #expect(window.startSeconds == 0.5)
    #expect(window.frames == 2)
    #expect(abs(window.rms - 0.790569415) < 0.00000001)
    let decoded = try JSONDecoder().decode(PCMMeasurement.self, from: JSONEncoder().encode(report))
    #expect(decoded.whole.peak == 1)
    #expect(
      try JSONDecoder().decode(
        AudioWindow.self,
        from: JSONEncoder().encode(AudioWindow(startSeconds: 0, durationSeconds: 1))
      ).durationSeconds == 1)
  }

  private func extensible() -> Data {
    var data = wave()
    data.replaceSubrange(16..<20, with: little(40))
    data.replaceSubrange(20..<22, with: [254, 255])
    let descriptor: [UInt8] = [
      22, 0, 16, 0, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113,
    ]
    data.insert(contentsOf: descriptor, at: 36)
    data.replaceSubrange(4..<8, with: little(data.count - 8))
    return data
  }

  @Test func extensiblePCMRequiresTheExactIntegerSubtype() throws {
    var data = extensible()
    #expect(try PCMMeasurement.analyze(data, windows: []).whole.peak == 1)
    data.replaceSubrange(40..<44, with: [0, 0, 0, 0])
    #expect(try PCMMeasurement.analyze(data, windows: []).whole.frames == 4)
  }

  @Test(arguments: [(36, UInt8(21)), (38, 24), (40, 3), (44, 3)])
  func malformedExtensibleDescriptorsAreNotGuessed(_ offset: Int, _ value: UInt8) {
    var data = extensible()
    data[offset] = value
    #expect(throws: (any Error).self) { try PCMMeasurement.analyze(data, windows: []) }
  }

  @Test func shortExtensibleHeadersFailBeforeReadingTheirGUID() {
    var data = wave()
    data.replaceSubrange(20..<22, with: [254, 255])
    #expect(throws: (any Error).self) { try PCMMeasurement.analyze(data, windows: []) }
  }

  @Test func silenceAndPaddedUnknownChunksHaveDefinedSemantics() throws {
    let junk: [UInt8] = Array("JUNK".utf8) + [1, 0, 0, 0, 99, 0]
    let data = wave(samples: [0, 0, 0, 0], extra: junk)
    let report = try PCMMeasurement.analyze(data, windows: [])
    #expect(report.whole.rms == 0)
    #expect(report.whole.peak == 0)
    #expect(report.whole.zeroCrossingRateHz == 0)
    let prefixed = Data([255]) + data
    #expect(try PCMMeasurement.analyze(prefixed.dropFirst(), windows: []).whole.frames == 4)
  }

  @Test(arguments: [
    (0, UInt8(88)), (4, 0), (16, 8), (20, 3), (22, 2), (24, 0), (28, 1), (32, 4), (34, 32),
    (40, 255),
  ])
  func malformedHeadersAndLengthsAreRejected(_ offset: Int, _ value: UInt8) {
    var data = wave()
    data[offset] = value
    #expect(throws: (any Error).self) { try PCMMeasurement.analyze(data, windows: []) }
  }

  @Test func duplicateChunksMissingDataOddSamplesAndTrailingBytesFailClosed() {
    let original = wave()
    let format = Array(original[12..<36])
    #expect(throws: (any Error).self) {
      try PCMMeasurement.analyze(wave(extra: format), windows: [])
    }
    let extraData = Array("data".utf8) + [2, 0, 0, 0, 0, 0]
    #expect(throws: (any Error).self) {
      try PCMMeasurement.analyze(wave(extra: extraData), windows: [])
    }
    #expect(throws: (any Error).self) { try PCMMeasurement.analyze(wave(samples: []), windows: []) }
    #expect(throws: (any Error).self) { try PCMMeasurement.analyze(Data(), windows: []) }
    var odd = original
    odd[40] = 7
    #expect(throws: (any Error).self) { try PCMMeasurement.analyze(odd, windows: []) }
    var missingFormat = original
    missingFormat.replaceSubrange(12..<16, with: Array("JUNK".utf8))
    #expect(throws: (any Error).self) { try PCMMeasurement.analyze(missingFormat, windows: []) }
    var trailing = original + Data([0])
    trailing.replaceSubrange(4..<8, with: little(trailing.count - 8))
    #expect(throws: (any Error).self) { try PCMMeasurement.analyze(trailing, windows: []) }
  }

  @Test(arguments: [
    AudioWindow(startSeconds: -1, durationSeconds: 1),
    AudioWindow(startSeconds: .nan, durationSeconds: 1),
    AudioWindow(startSeconds: 0, durationSeconds: .infinity),
    AudioWindow(startSeconds: 0, durationSeconds: 0),
    AudioWindow(startSeconds: 0.5, durationSeconds: 1),
    AudioWindow(startSeconds: 0, durationSeconds: 0.01),
  ])
  func invalidOrSubSampleWindowsNeverIndexOutsideThePCM(_ window: AudioWindow) {
    #expect(throws: (any Error).self) { try PCMMeasurement.analyze(wave(), windows: [window]) }
  }

  @Test func decimalWindowBoundariesDoNotLoseASampleToFloatingPointRounding() throws {
    let result = try PCMMeasurement.analyze(
      wave(samples: Array(repeating: 0, count: 32000), rate: 16000),
      windows: [.init(startSeconds: 1.65, durationSeconds: 0.2)])
    let window = try #require(result.windows.first)
    #expect(window.frames == 3200)
    #expect(window.durationSeconds == 0.2)
    #expect(window.startSeconds == 1.65)
  }

  @Test func extendedDurationRequiresOptInAndHasAHardCeiling() throws {
    let minute = wave(samples: Array(repeating: 0, count: 240), rate: 4)
    #expect(throws: ProAppsError.self) { try PCMMeasurement.analyze(minute, windows: []) }
    let report = try PCMMeasurement.analyze(
      minute, windows: [.init(startSeconds: 20, durationSeconds: 20)], maximumDurationSeconds: 60)
    #expect(report.whole.frames == 240)
    #expect(report.whole.durationSeconds == 60)
    #expect(report.windows.first?.frames == 80)
    #expect(
      try PCMMeasurement.analyze(
        wave(samples: Array(repeating: 0, count: 480), rate: 4), windows: [],
        maximumDurationSeconds: 120
      ).whole.durationSeconds == 120)
  }

  @Test(arguments: [0.0, -1.0, 120.001, Double.infinity, Double.nan])
  func invalidExtendedDurationBudgetsAreRefused(_ seconds: Double) {
    #expect(throws: ProAppsError.self) {
      try PCMMeasurement.analyze(wave(), windows: [], maximumDurationSeconds: seconds)
    }
  }

  @Test func durationAndWindowCountsAreBounded() {
    #expect(throws: (any Error).self) {
      try PCMMeasurement.analyze(wave(samples: Array(repeating: 0, count: 121)), windows: [])
    }
    #expect(throws: (any Error).self) {
      try PCMMeasurement.analyze(
        wave(), windows: Array(repeating: .init(startSeconds: 0, durationSeconds: 1), count: 17))
    }
  }
}
