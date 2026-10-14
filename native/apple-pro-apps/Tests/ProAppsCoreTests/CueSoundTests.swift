import Foundation
import Testing

@testable import ProAppsCore

struct CueSoundTests {
  @Test(arguments: [(seconds: 60.0, frames: 960000), (seconds: 90.0, frames: 1_440_000)])
  func bothRequiredDurationsHaveExactCueAndSilenceWindows(_ sample: (seconds: Double, frames: Int))
    throws
  {
    let track = CueSound(
      durationSeconds: sample.seconds, onsetSeconds: [1, 20, sample.seconds - 1], gain: 0.12)
    let wave = try track.wave()
    let measured = try PCMMeasurement.analyze(
      wave,
      windows: [
        .init(startSeconds: 0, durationSeconds: 1),
        .init(startSeconds: 1, durationSeconds: 0.08),
        .init(startSeconds: 1.08, durationSeconds: 1),
        .init(startSeconds: sample.seconds - 1, durationSeconds: 0.08),
        .init(startSeconds: sample.seconds - 0.92, durationSeconds: 0.5),
      ], maximumDurationSeconds: 120)
    #expect(measured.whole.frames == sample.frames)
    #expect(measured.whole.peak <= 0.121)
    try #require(measured.windows.count == 5)
    #expect(measured.windows[0].rms == 0)
    #expect(measured.windows[1].rms > 0.04 && measured.windows[1].rms < 0.07)
    #expect(
      measured.windows[1].zeroCrossingRateHz > 1400 && measured.windows[1].zeroCrossingRateHz < 2000
    )
    #expect(measured.windows[2].rms == 0)
    #expect(measured.windows[3].rms == measured.windows[1].rms)
    #expect(measured.windows[4].rms == 0)
    let restored = try JSONDecoder().decode(CueSound.self, from: JSONEncoder().encode(track))
    #expect(try restored.wave() == wave)
  }

  @Test(arguments: [0.0, -1.0, 120.01, Double.nan, Double.infinity, 0.000001])
  func invalidDurationIsRejected(_ duration: Double) {
    #expect(throws: ProAppsError.self) {
      try CueSound(durationSeconds: duration, onsetSeconds: [], gain: 0.1).wave()
    }
  }

  @Test(arguments: [-0.1, 0.251, Double.nan, Double.infinity])
  func invalidGainIsRejected(_ gain: Double) {
    #expect(throws: ProAppsError.self) {
      try CueSound(durationSeconds: 1, onsetSeconds: [], gain: gain).wave()
    }
  }

  @Test(arguments: [
    [-1.0], [Double.nan], [Double.infinity], [2.0], [0.95], [0.0, 0.04], [0.5, 0.0],
    Array(repeating: 0.0, count: 121),
  ])
  func invalidOnsetsAreRejected(_ onsets: [Double]) {
    #expect(throws: ProAppsError.self) {
      try CueSound(durationSeconds: 1, onsetSeconds: onsets, gain: 0.1).wave()
    }
  }

  @Test func emptyTrackAndBoundaryGainAreSupported() throws {
    let empty = try CueSound(durationSeconds: 1, onsetSeconds: [], gain: 0).wave()
    #expect(try PCMMeasurement.analyze(empty, windows: []).whole.rms == 0)
    let boundary = try CueSound(durationSeconds: 0.08, onsetSeconds: [0], gain: 0.25).wave()
    let measurement = try PCMMeasurement.analyze(boundary, windows: [])
    #expect(measurement.whole.frames == 1280)
    #expect(measurement.whole.peak <= 0.25)
  }

  @Test func alreadyCancelledWorkDoesNotAllocateOrPublish() async {
    await withThrowingTaskGroup(of: Data.self) { group in
      group.cancelAll()
      group.addTask { try CueSound(durationSeconds: 90, onsetSeconds: [1], gain: 0.1).wave() }
      await #expect(throws: CancellationError.self) { _ = try await group.next() }
    }
  }
}
