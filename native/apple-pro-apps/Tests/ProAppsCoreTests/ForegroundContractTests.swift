import AVFoundation
import Foundation
import Testing

@testable import ProAppsCore

struct ForegroundContractTests {
  private func recipe(path: String, encoded: Bool = true) -> EditRecipe {
    EditRecipe(
      clips: [
        .init(
          sourcePath: "/synthetic/base.mp4",
          selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
      ],
      video: .init(
        width: 64, height: 64, frameRate: 30, resizeMode: .fit,
        encoding: encoded
          ? .init(codec: .hevc, averageBitRate: 2_000_000, allowFrameReordering: false) : nil,
        foregroundVideoPath: path))
  }

  @Test func exactForegroundContractRoundTripsWithoutReadingSources() throws {
    let source = recipe(path: "/synthetic/motion.mov")
    #expect(try EditPlan.build(source).durationSeconds == 1)
    let decoded = try JSONDecoder().decode(EditRecipe.self, from: JSONEncoder().encode(source))
    #expect(decoded.video?.foregroundVideoPath == "/synthetic/motion.mov")
    #expect(FCPXMLTimeline.unsupported(decoded) == "Foreground video is not exported to FCPXML")
  }

  @Test func foregroundCannotBeIgnoredByTheExportSessionPath() {
    #expect(throws: ProAppsError.invalid("foregroundVideoPath requires video.encoding")) {
      try EditPlan.build(recipe(path: "/synthetic/motion.mov", encoded: false))
    }
  }

  @Test(arguments: [
    "", "relative.mov", "/tmp/bad\0.mov", "/" + String(repeating: "x", count: 4096),
  ])
  func rejectsInvalidPaths(_ path: String) {
    #expect(throws: ProAppsError.self) { try EditPlan.build(recipe(path: path)) }
  }

  @Test func equivalentRationalFrameTimesAreAccepted() throws {
    try NativeEditor.ForegroundVideo.validateFrame(
      presentation: CMTime(value: 1000, timescale: 30000),
      duration: CMTime(value: 1000, timescale: 30000), slot: 1, rate: 30, dimensionsMatch: true)
  }

  @Test(arguments: [
    CMTime.invalid, CMTime.indefinite, CMTime.positiveInfinity,
    CMTime(value: 1, timescale: 30),
    CMTime(value: 0, timescale: 30, flags: .valid, epoch: 1),
  ])
  func rejectsMissingShiftedAndNonzeroEpochTimes(_ presentation: CMTime) {
    #expect(throws: ProAppsError.self) {
      try NativeEditor.ForegroundVideo.validateFrame(
        presentation: presentation, duration: CMTime(value: 1, timescale: 30),
        slot: 0, rate: 30, dimensionsMatch: true)
    }
  }

  @Test func decodedFramesMayOmitDurationButNotPresentationTime() throws {
    try NativeEditor.ForegroundVideo.validateFrame(
      presentation: .zero, duration: .invalid, slot: 0, rate: 30, dimensionsMatch: true)
  }

  @Test(arguments: [
    CMTime.indefinite, CMTime.positiveInfinity, CMTime.zero, CMTime(value: 1, timescale: 15),
  ])
  func rejectsInvalidOrRetimedDurations(_ duration: CMTime) {
    #expect(throws: ProAppsError.self) {
      try NativeEditor.ForegroundVideo.validateFrame(
        presentation: .zero, duration: duration, slot: 0, rate: 30, dimensionsMatch: true)
    }
  }

  @Test func rejectsCanvasChanges() {
    #expect(throws: ProAppsError.self) {
      try NativeEditor.ForegroundVideo.validateFrame(
        presentation: .zero, duration: CMTime(value: 1, timescale: 30),
        slot: 0, rate: 30, dimensionsMatch: false)
    }
  }
}
