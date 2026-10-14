import Foundation
import Testing

@testable import ProAppsCore

struct EditPlanTests {
  private let selection = EditSelection(startSeconds: 0, durationSeconds: 2, rate: 1)

  @Test(arguments: [
    EditColor(brightness: .nan, contrast: 1, saturation: 1),
    EditColor(brightness: -1.01, contrast: 1, saturation: 1),
    EditColor(brightness: 1.01, contrast: 1, saturation: 1),
    EditColor(brightness: 0, contrast: -0.01, saturation: 1),
    EditColor(brightness: 0, contrast: 4.01, saturation: 1),
    EditColor(brightness: 0, contrast: 1, saturation: .infinity),
    EditColor(brightness: 0, contrast: 1, saturation: -0.01),
    EditColor(brightness: 0, contrast: 1, saturation: 2.01),
  ])
  func colorBoundsAreValidatedBeforeNativeRendering(_ color: EditColor) {
    let recipe = EditRecipe(
      clips: [.init(sourcePath: "/tmp/source.mp4", selection: selection)],
      video: .init(width: 32, height: 32, frameRate: 30, resizeMode: .fit, color: color))
    #expect(throws: (any Error).self) { try EditPlan.build(recipe) }
  }

  @Test(arguments: [
    EditTitle(text: " ", x: 0, y: 0, fontSize: 32),
    EditTitle(text: "two\nlines", x: 0, y: 0, fontSize: 32),
    EditTitle(text: String(repeating: "A", count: 121), x: 0, y: 0, fontSize: 32),
    EditTitle(text: "Title", x: -1, y: 0, fontSize: 32),
    EditTitle(text: "Title", x: 0, y: .nan, fontSize: 32),
    EditTitle(text: "Title", x: 320, y: 0, fontSize: 32),
    EditTitle(text: "Title", x: 0, y: 0, fontSize: 129),
  ])
  func invalidTitlesFailBeforeFontRendering(_ title: EditTitle) {
    #expect(throws: (any Error).self) {
      try EditPlan.validate(
        .init(width: 320, height: 240, frameRate: 30, resizeMode: .fit, titles: [title]))
    }
  }

  @Test func titleCountIsBounded() {
    #expect(throws: (any Error).self) {
      try EditPlan.validate(
        .init(
          width: 320, height: 240, frameRate: 30, resizeMode: .fit,
          titles: Array(repeating: .init(text: "Title", x: 0, y: 0, fontSize: 16), count: 9)))
    }
  }

  @Test func transitionsShortenTheTimelineAndResolveAudioEnvelopes() throws {
    let clip = EditClip(
      sourcePath: "/tmp/a.mp4", selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
    let incoming = EditClip(
      sourcePath: "/tmp/b.mp4", selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
      transitionInSeconds: 0.5)
    let plan = try EditPlan.build(.init(clips: [clip, incoming, incoming]))
    #expect(plan.durationSeconds == 2)
    try #require(plan.spans.count == 3)
    #expect(plan.spans[1].startSeconds == 0.5)
    #expect(plan.spans[2].startSeconds == 1)
    #expect(plan.spans[0].audio.fadeOutSeconds == 0.5)
    #expect(plan.spans[1].audio.fadeInSeconds == 0.5)
    #expect(plan.spans[1].audio.fadeOutSeconds == 0.5)
    let decoded = try JSONDecoder().decode(EditPlan.self, from: JSONEncoder().encode(plan))
    #expect(decoded.spans[1].transitionInSeconds == 0.5)
  }

  @Test(arguments: [-1.0, Double.nan, 6.0, 0.51, 0.00000001])
  func invalidTransitionDurationsAreRefused(_ overlap: Double) {
    let clip = EditClip(
      sourcePath: "/tmp/a.mp4", selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
    let incoming = EditClip(
      sourcePath: "/tmp/b.mp4", selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
      transitionInSeconds: overlap)
    #expect(throws: (any Error).self) { try EditPlan.build(.init(clips: [clip, incoming])) }
  }

  @Test func transitionsCannotPrecedeTheFirstClipOrConflictWithAudioFades() {
    let incoming = EditClip(
      sourcePath: "/tmp/b.mp4", selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
      transitionInSeconds: 0.5)
    #expect(throws: (any Error).self) { try EditPlan.build(.init(clips: [incoming])) }
    let clip = EditClip(
      sourcePath: "/tmp/a.mp4", selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
      audio: .init(volume: 1, fadeInSeconds: 0.75, fadeOutSeconds: 0))
    #expect(throws: (any Error).self) { try EditPlan.build(.init(clips: [clip, incoming])) }
  }

  @Test func orderedTimelineUsesOutputTimeForSpeedAndMixing() throws {
    let recipe = EditRecipe(
      clips: [
        EditClip(
          sourcePath: "/tmp/one.mov",
          selection: .init(startSeconds: 1, durationSeconds: 4, rate: 2),
          audio: .init(volume: 0.5, fadeInSeconds: 0.25, fadeOutSeconds: 0.5),
          geometry: .init(rotation: .clockwise90, crop: .init(x: 0, y: 0, width: 64, height: 32))),
        EditClip(
          sourcePath: "/tmp/two.mov",
          selection: .init(startSeconds: 0, durationSeconds: 3, rate: 0.5)),
      ], video: .init(width: 640, height: 360, frameRate: 30, resizeMode: .fill),
      additionalAudio: [
        EditAudioLayer(
          sourcePath: "/tmp/music.wav", selection: selection, offsetSeconds: 6,
          audio: .init(volume: 0.25, fadeInSeconds: 0.5, fadeOutSeconds: 0.5))
      ], muteOriginalAudio: true)
    let decoded = try JSONDecoder().decode(EditRecipe.self, from: JSONEncoder().encode(recipe))
    let plan = try EditPlan.build(decoded)
    #expect(plan.durationSeconds == 8)
    try #require(plan.spans.count == 2)
    try #require(decoded.clips.count == 2)
    #expect(plan.spans[0].startSeconds == 0)
    #expect(plan.spans[0].durationSeconds == 2)
    #expect(plan.spans[1].clipIndex == 1)
    #expect(plan.spans[1].startSeconds == 2)
    #expect(plan.spans[1].durationSeconds == 6)
    #expect(!plan.audioOnly)
    let restored = try JSONDecoder().decode(EditPlan.self, from: JSONEncoder().encode(plan))
    #expect(restored.durationSeconds == 8)
    #expect(decoded.clips[0].geometry?.rotation == .clockwise90)
    #expect(decoded.additionalAudio?[0].audio?.volume == 0.25)
  }

  @Test func maximumCanvasAndTimelineBoundariesAreAccepted() throws {
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: "/tmp/a.mov",
          selection: .init(startSeconds: 86400, durationSeconds: 600, rate: 1))
      ], video: .init(width: 3840, height: 2160, frameRate: 60, resizeMode: .fit))
    let decoded = try JSONDecoder().decode(EditRecipe.self, from: JSONEncoder().encode(recipe))
    #expect(try EditPlan.build(decoded).durationSeconds == 600)
    #expect(decoded.video?.resizeMode == .fit)
  }

  @Test func audioOnlyDefaultsAndExplicitReplacementAreDistinct() throws {
    let clip = EditClip(sourcePath: "/tmp/voice.m4a", selection: selection)
    #expect(try EditPlan.build(EditRecipe(clips: [clip])).audioOnly)
    let replacement = EditRecipe(
      clips: [clip],
      additionalAudio: [
        .init(sourcePath: "/tmp/music.wav", selection: selection, offsetSeconds: 0)
      ], muteOriginalAudio: true)
    #expect(try EditPlan.build(replacement).durationSeconds == 2)
    #expect(throws: (any Error).self) {
      try EditPlan.build(EditRecipe(clips: [clip], muteOriginalAudio: true))
    }
  }

  @Test(arguments: [
    EditSelection(startSeconds: -1, durationSeconds: 1, rate: 1),
    EditSelection(startSeconds: .infinity, durationSeconds: 1, rate: 1),
    EditSelection(startSeconds: 86401, durationSeconds: 1, rate: 1),
    EditSelection(startSeconds: 0, durationSeconds: 0, rate: 1),
    EditSelection(startSeconds: 0, durationSeconds: 0.000000001, rate: 1),
    EditSelection(startSeconds: 0, durationSeconds: 0.00002, rate: 4),
    EditSelection(startSeconds: 0, durationSeconds: .nan, rate: 1),
    EditSelection(startSeconds: 0, durationSeconds: 601, rate: 1),
    EditSelection(startSeconds: 0, durationSeconds: 1, rate: 0),
    EditSelection(startSeconds: 0, durationSeconds: 1, rate: 4.01),
    EditSelection(startSeconds: 0, durationSeconds: 1, rate: .infinity),
  ])
  func invalidSelectionsNeverProduceAPlan(_ invalid: EditSelection) {
    #expect(throws: (any Error).self) {
      try EditPlan.build(EditRecipe(clips: [.init(sourcePath: "/tmp/a.mov", selection: invalid)]))
    }
  }

  @Test(arguments: [
    EditAudioAdjustment(volume: -0.1, fadeInSeconds: 0, fadeOutSeconds: 0),
    EditAudioAdjustment(volume: 1.1, fadeInSeconds: 0, fadeOutSeconds: 0),
    EditAudioAdjustment(volume: .nan, fadeInSeconds: 0, fadeOutSeconds: 0),
    EditAudioAdjustment(volume: 1, fadeInSeconds: -1, fadeOutSeconds: 0),
    EditAudioAdjustment(volume: 1, fadeInSeconds: 0.000000001, fadeOutSeconds: 0),
    EditAudioAdjustment(volume: 1, fadeInSeconds: 0, fadeOutSeconds: 0.000000001),
    EditAudioAdjustment(volume: 1, fadeInSeconds: .infinity, fadeOutSeconds: 0),
    EditAudioAdjustment(volume: 1, fadeInSeconds: 0, fadeOutSeconds: -1),
    EditAudioAdjustment(volume: 1, fadeInSeconds: 0, fadeOutSeconds: .nan),
    EditAudioAdjustment(volume: 1, fadeInSeconds: 1.5, fadeOutSeconds: 1),
  ])
  func invalidAudioAdjustmentsAreRejected(_ adjustment: EditAudioAdjustment) {
    #expect(throws: (any Error).self) {
      try EditPlan.build(
        EditRecipe(clips: [.init(sourcePath: "/tmp/a.mov", selection: selection, audio: adjustment)]
        ))
    }
  }

  @Test(arguments: [
    (1, 360, 30), (641, 360, 30), (640, 361, 30), (3840, 3840, 30), (640, 360, 0), (640, 360, 61),
    (Int.max, 360, 30),
  ])
  func invalidCanvasesCannotOverflowOrReachAnEncoder(_ width: Int, _ height: Int, _ fps: Int) {
    #expect(throws: (any Error).self) {
      try EditPlan.build(
        EditRecipe(
          clips: [.init(sourcePath: "/tmp/a.mov", selection: selection)],
          video: .init(width: width, height: height, frameRate: fps, resizeMode: .fit)))
    }
  }

  @Test(arguments: [
    EditCrop(x: -1, y: 0, width: 10, height: 10), EditCrop(x: 0, y: -1, width: 10, height: 10),
    EditCrop(x: 0, y: 0, width: 0, height: 10), EditCrop(x: 0, y: 0, width: 10, height: -1),
    EditCrop(x: .nan, y: 0, width: 10, height: 10),
    EditCrop(x: 0, y: .infinity, width: 10, height: 10),
    EditCrop(x: 0, y: 0, width: .infinity, height: 10),
    EditCrop(x: 0, y: 0, width: 10, height: .nan),
    EditCrop(x: 16000, y: 0, width: 1000, height: 10),
    EditCrop(x: 0, y: 16000, width: 10, height: 1000),
  ])
  func malformedCropRectanglesFailBeforeSourceLoading(_ crop: EditCrop) {
    #expect(throws: (any Error).self) {
      try EditPlan.build(
        EditRecipe(
          clips: [
            .init(
              sourcePath: "/tmp/a.mov", selection: selection,
              geometry: .init(rotation: .none, crop: crop))
          ],
          video: .init(width: 640, height: 360, frameRate: 30, resizeMode: .fit)))
    }
  }

  @Test(arguments: [EditGeometry.Rotation.none, .clockwise90, .clockwise180, .clockwise270])
  func rotationsRoundTripWithoutInventingArbitraryAngles(_ rotation: EditGeometry.Rotation) throws {
    let geometry = EditGeometry(rotation: rotation)
    let decoded = try JSONDecoder().decode(EditGeometry.self, from: JSONEncoder().encode(geometry))
    #expect(decoded.rotation == rotation)
    #expect(decoded.crop == nil)
    #expect(throws: (any Error).self) {
      try JSONDecoder().decode(EditGeometry.self, from: Data("{\"rotation\":45}".utf8))
    }
  }

  @Test func boundedCollectionsPathsAndRequiredJSONAreEnforced() {
    let clip = EditClip(sourcePath: "/tmp/a.mov", selection: selection)
    #expect(throws: (any Error).self) { try EditPlan.build(EditRecipe(clips: [])) }
    #expect(throws: (any Error).self) {
      try EditPlan.build(EditRecipe(clips: Array(repeating: clip, count: 61)))
    }
    #expect(throws: (any Error).self) {
      try EditPlan.build(
        EditRecipe(
          clips: [clip],
          additionalAudio: Array(
            repeating: .init(sourcePath: "/tmp/a.wav", selection: selection, offsetSeconds: 0),
            count: 17)))
    }
    #expect(throws: (any Error).self) {
      try EditPlan.build(
        EditRecipe(clips: [.init(sourcePath: "https://example.invalid/a.mov", selection: selection)]
        ))
    }
    #expect(throws: (any Error).self) {
      try JSONDecoder().decode(EditRecipe.self, from: Data("{}".utf8))
    }
    #expect(throws: (any Error).self) {
      try JSONDecoder().decode(EditSelection.self, from: Data("{\"durationSeconds\":2}".utf8))
    }
    #expect(throws: (any Error).self) {
      try EditPlan.build(
        EditRecipe(clips: [
          .init(sourcePath: "/tmp/a.mov", selection: selection, geometry: .init(rotation: .none))
        ]))
    }
    #expect(throws: (any Error).self) {
      try EditPlan.build(
        EditRecipe(clips: [
          .init(
            sourcePath: "/tmp/a.mov",
            selection: .init(startSeconds: 0, durationSeconds: 600, rate: 0.25))
        ]))
    }
  }

  @Test(arguments: [-1.0, 1.0, Double.infinity])
  func additionalAudioMayNotOverflowTheTimeline(_ offset: Double) {
    #expect(throws: (any Error).self) {
      try EditPlan.build(
        EditRecipe(
          clips: [.init(sourcePath: "/tmp/a.mov", selection: selection)],
          additionalAudio: [
            .init(sourcePath: "/tmp/a.wav", selection: selection, offsetSeconds: offset)
          ]))
    }
  }
}
