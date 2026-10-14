import Foundation
import Testing

@testable import ProAppsCore

struct TimedMaskTests {
  @Test func halfOpenIntervalAndRoundTrip() throws {
    let mask = EditMask(
      region: .init(x: 0, y: 10, width: 20, height: 10), opacity: 0.9,
      blurRadius: 12, startSeconds: 1, endSeconds: 2)
    let decoded = try JSONDecoder().decode(EditMask.self, from: JSONEncoder().encode(mask))
    #expect(decoded.blurRadius == 12)
    #expect(decoded.startSeconds == 1)
    #expect(decoded.endSeconds == 2)
    #expect(!decoded.isActive(at: 0.999))
    #expect(decoded.isActive(at: 1))
    #expect(decoded.isActive(at: 1.999))
    #expect(!decoded.isActive(at: 2))
    #expect(!decoded.isActive(at: .nan))
    try EditPlan.validateMaskTimes([decoded], duration: 2)
  }

  @Test func legacyMaskHasNoEffectOptions() throws {
    let data = Data(#"{"region":{"x":0,"y":0,"width":20,"height":10},"opacity":1}"#.utf8)
    let mask = try JSONDecoder().decode(EditMask.self, from: data)
    #expect(mask.blurRadius == nil)
    #expect(mask.startSeconds == nil)
    #expect(mask.endSeconds == nil)
    #expect(mask.isActive(at: 0))
    #expect(mask.isActive(at: 599.999))
    #expect(!mask.isActive(at: -1))
    try EditPlan.validateMaskTimes([mask], duration: 1)
  }

  @Test(arguments: [0.0, -1.0, 64.1, Double.nan, Double.infinity])
  func rejectsInvalidBlurRadius(_ radius: Double) {
    let settings = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      masks: [
        .init(region: .init(x: 0, y: 0, width: 10, height: 10), opacity: 1, blurRadius: radius)
      ])
    #expect(throws: ProAppsError.self) { try EditPlan.validate(settings) }
  }

  @Test(arguments: [1.0, 64.0])
  func acceptsBlurRadiusBounds(_ radius: Double) throws {
    let settings = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      masks: [
        .init(region: .init(x: 0, y: 0, width: 10, height: 10), opacity: 1, blurRadius: radius)
      ])
    try EditPlan.validate(settings)
  }

  private static let invalidIntervals: [(Double?, Double?)] = [
    (0, nil), (nil, 1), (-1, 1), (1, 1), (2, 1), (0, 3), (.nan, 1), (0, .infinity),
  ]

  @Test(arguments: invalidIntervals)
  func rejectsInvalidIntervals(_ interval: (Double?, Double?)) {
    let mask = EditMask(
      region: .init(x: 0, y: 0, width: 10, height: 10), opacity: 1,
      startSeconds: interval.0, endSeconds: interval.1)
    #expect(throws: ProAppsError.self) {
      try EditPlan.validateMaskTimes([mask], duration: 2)
    }
  }
}
