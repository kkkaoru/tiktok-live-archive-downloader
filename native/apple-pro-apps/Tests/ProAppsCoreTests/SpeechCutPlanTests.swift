import Foundation
import Testing

@testable import ProAppsCore

struct SpeechCutPlanTests {
  @Test func mergesOverlapsAndMapsOutputWithoutClaimingSpeechVerification() throws {
    let plan = try SpeechCutPlan.make(
      .init(
        sourceDurationSeconds: 10,
        spans: [
          .init(startSeconds: 2, endSeconds: 3), .init(startSeconds: 2.5, endSeconds: 4),
          .init(startSeconds: 7, endSeconds: 8),
        ], paddingSeconds: 0.5,
        minimumRemovedGapSeconds: 0.25))
    #expect(plan.segments.count == 2)
    #expect(plan.segments[0].sourceStartSeconds == 1.5)
    #expect(plan.segments[0].sourceEndSeconds == 4.5)
    #expect(plan.segments[0].outputStartSeconds == 0)
    #expect(plan.segments[0].outputEndSeconds == 3)
    #expect(plan.segments[1].sourceStartSeconds == 6.5)
    #expect(plan.segments[1].sourceEndSeconds == 8.5)
    #expect(plan.segments[1].outputStartSeconds == 3)
    #expect(plan.outputDurationSeconds == 5)
    #expect(plan.removedSpans.count == 3)
    #expect(plan.removedSpans[0].startSeconds == 0)
    #expect(plan.removedSpans[0].endSeconds == 1.5)
    #expect(plan.removedSpans[1].startSeconds == 4.5)
    #expect(plan.removedSpans[1].endSeconds == 6.5)
    #expect(plan.removedSpans[2].startSeconds == 8.5)
    #expect(plan.removedSpans[2].endSeconds == 10)
    #expect(!plan.speechVerified)
  }

  @Test func clampsPaddingAndRetainsShortInteriorGaps() throws {
    let plan = try SpeechCutPlan.make(
      .init(
        sourceDurationSeconds: 3,
        spans: [.init(startSeconds: 0, endSeconds: 1), .init(startSeconds: 2, endSeconds: 3)],
        paddingSeconds: 0.25, minimumRemovedGapSeconds: 0.5))
    #expect(plan.segments.count == 1)
    #expect(plan.outputDurationSeconds == 3)
    #expect(plan.removedSpans.isEmpty)
    let decoded = try JSONDecoder().decode(SpeechCutPlan.self, from: JSONEncoder().encode(plan))
    #expect(decoded.sourceDurationSeconds == 3)
  }

  @Test(arguments: [
    [SpeechSpan(startSeconds: -1, endSeconds: 1)],
    [SpeechSpan(startSeconds: 1, endSeconds: 1)],
    [SpeechSpan(startSeconds: 0, endSeconds: 11)],
    [SpeechSpan(startSeconds: .nan, endSeconds: 1)],
    [SpeechSpan(startSeconds: 2, endSeconds: 3), SpeechSpan(startSeconds: 1, endSeconds: 2)],
  ])
  func rejectsInvalidSpans(_ spans: [SpeechSpan]) {
    #expect(throws: ProAppsError.self) {
      try SpeechCutPlan.make(
        .init(
          sourceDurationSeconds: 10, spans: spans,
          paddingSeconds: 0, minimumRemovedGapSeconds: 0))
    }
  }

  @Test(arguments: [-1.0, 0, Double.nan, Double.infinity, 21_601])
  func rejectsInvalidDuration(_ duration: Double) {
    #expect(throws: ProAppsError.self) {
      try SpeechCutPlan.make(
        .init(
          sourceDurationSeconds: duration,
          spans: [.init(startSeconds: 0, endSeconds: 1)], paddingSeconds: 0,
          minimumRemovedGapSeconds: 0))
    }
  }

  @Test(arguments: [
    (-1.0, 0.0), (2.1, 0.0), (Double.nan, 0.0), (0.0, -1.0),
    (0.0, 5.1), (0.0, Double.infinity),
  ])
  func rejectsInvalidPaddingAndGap(_ values: (Double, Double)) {
    #expect(throws: ProAppsError.self) {
      try SpeechCutPlan.make(
        .init(
          sourceDurationSeconds: 10,
          spans: [.init(startSeconds: 1, endSeconds: 2)], paddingSeconds: values.0,
          minimumRemovedGapSeconds: values.1))
    }
  }

  @Test func rejectsExcessiveSpanCount() {
    #expect(throws: ProAppsError.self) {
      try SpeechCutPlan.make(
        .init(
          sourceDurationSeconds: 10,
          spans: Array(repeating: .init(startSeconds: 1, endSeconds: 2), count: 30_001),
          paddingSeconds: 0, minimumRemovedGapSeconds: 0))
    }
  }

  @Test func refusesEmptySpansRatherThanDeletingEntireVideo() {
    #expect(throws: ProAppsError.self) {
      try SpeechCutPlan.make(
        .init(
          sourceDurationSeconds: 10, spans: [],
          paddingSeconds: 0, minimumRemovedGapSeconds: 0))
    }
  }
}
