import Foundation
import Testing

@testable import ProAppsCore

struct MaskTests {
  @Test(arguments: [0.0, 0.5, 1.0])
  func validOpacityRoundTrips(_ opacity: Double) throws {
    let settings = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      masks: [.init(region: .init(x: 0, y: 200, width: 320, height: 40), opacity: opacity)])
    let decoded = try JSONDecoder().decode(
      EditVideoSettings.self, from: JSONEncoder().encode(settings))
    try EditPlan.validate(decoded)
    #expect(decoded.masks?.count == 1)
    #expect(decoded.masks?.first?.opacity == opacity)
  }

  @Test(arguments: [-0.01, 1.01, Double.nan, Double.infinity])
  func invalidOpacityIsRejected(_ opacity: Double) {
    let settings = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      masks: [.init(region: .init(x: 0, y: 0, width: 10, height: 10), opacity: opacity)])
    #expect(throws: ProAppsError.self) { try EditPlan.validate(settings) }
  }

  @Test(arguments: [
    (-0.1, 0.0), (6.1, 0.0), (Double.nan, 0.0), (3.0, -0.1), (3.0, 1.1), (3.0, Double.infinity),
  ])
  func invalidCaptionStyleIsRejected(_ values: (Double, Double)) {
    let settings = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      captionStyle: .init(outlineWidth: values.0, backgroundOpacity: values.1))
    #expect(throws: ProAppsError.self) { try EditPlan.validate(settings) }
  }

  @Test(arguments: [
    (15.0, 0.0), (65.0, 0.0), (Double.nan, 0.0), (32.0, -1.0), (32.0, 240.0),
    (32.0, Double.infinity),
  ])
  func invalidCaptionSizeOrPositionIsRejected(_ values: (Double, Double)) {
    let video = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      captionStyle: .init(
        outlineWidth: 3, backgroundOpacity: 0, fontSize: values.0, bottomMargin: values.1))
    #expect(throws: ProAppsError.self) { try EditPlan.validate(video) }
  }

  @Test func invalidGeometryAndCountAreRejected() {
    let outside = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      masks: [.init(region: .init(x: 0, y: 230, width: 320, height: 11), opacity: 1)])
    let negative = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      masks: [.init(region: .init(x: -1, y: 0, width: 10, height: 10), opacity: 1)])
    let excessive = EditVideoSettings(
      width: 320, height: 240, frameRate: 30, resizeMode: .fit,
      masks: Array(
        repeating: .init(region: .init(x: 0, y: 0, width: 10, height: 10), opacity: 1), count: 9))
    #expect(throws: ProAppsError.self) { try EditPlan.validate(outside) }
    #expect(throws: ProAppsError.self) { try EditPlan.validate(negative) }
    #expect(throws: ProAppsError.self) { try EditPlan.validate(excessive) }
  }
}
