import Foundation
import Testing

@testable import ProAppsCore

struct SpeechTimingTests {
  @Test func nativeClockRoundingIsClippedAndOriginalTimingIsPreserved() throws {
    let raw = SpeechSegment(text: "unchanged", startSeconds: 18.42, durationSeconds: 11.602875)
    let normalized = try raw.normalizingNativeEnd(
      audioDuration: 30.02285714285714, timeScale: 16000)
    #expect(normalized.text == "unchanged")
    #expect(normalized.startSeconds == 18.42)
    #expect(abs(normalized.durationSeconds - 11.60285714285714) < 0.000000000001)
    #expect(normalized.originalDurationSeconds == 11.602875)
    let transcript = try SpeechTranscript(
      locale: "ja_JP", durationSeconds: 30.02285714285714, segments: [normalized])
    #expect(!transcript.humanReviewed)
    let restored = try JSONDecoder().decode(
      SpeechSegment.self, from: JSONEncoder().encode(normalized))
    #expect(restored == normalized)
    #expect(throws: ProAppsError.self) {
      try SpeechTranscript(locale: "ja_JP", durationSeconds: 30.02285714285714, segments: [raw])
    }
  }

  @Test func interiorAndExactEndpointRemainUnchanged() throws {
    let segment = SpeechSegment(text: "same", startSeconds: 1, durationSeconds: 1)
    #expect(try segment.normalizingNativeEnd(audioDuration: 3, timeScale: 16000) == segment)
    #expect(try segment.normalizingNativeEnd(audioDuration: 2, timeScale: 16000) == segment)
  }

  @Test func exactlyOneTickIsAllowedButLargerExcessIsRejected() throws {
    let oneTick = SpeechSegment(text: "same", startSeconds: 0, durationSeconds: 1.0625)
    let normalized = try oneTick.normalizingNativeEnd(audioDuration: 1, timeScale: 16)
    #expect(normalized.durationSeconds <= 1 && normalized.durationSeconds > 0.999999999)
    #expect(normalized.originalDurationSeconds == 1.0625)
    #expect(throws: ProAppsError.self) {
      try oneTick.normalizingNativeEnd(audioDuration: 1, timeScale: 16000)
    }
    #expect(throws: ProAppsError.self) {
      try SpeechSegment(text: "outside", startSeconds: 1, durationSeconds: 0.00001)
        .normalizingNativeEnd(audioDuration: 1, timeScale: 16000)
    }
  }

  @Test(arguments: [Double.nan, Double.infinity, -1.0, 0.0])
  func invalidNativeDurationIsRejected(_ duration: Double) {
    #expect(throws: ProAppsError.self) {
      try SpeechSegment(text: "x", startSeconds: 0, durationSeconds: duration).normalizingNativeEnd(
        audioDuration: 1, timeScale: 16000)
    }
  }

  @Test func invalidClockAndVanishingIntervalAreRejected() {
    #expect(throws: ProAppsError.self) {
      try SpeechSegment(text: "x", startSeconds: 0, durationSeconds: 1).normalizingNativeEnd(
        audioDuration: 1, timeScale: 0)
    }
    #expect(throws: ProAppsError.self) {
      try SpeechSegment(
        text: "x", startSeconds: 0, durationSeconds: Double.leastNonzeroMagnitude * 2
      ).normalizingNativeEnd(audioDuration: Double.leastNonzeroMagnitude, timeScale: 16000)
    }
  }
}
