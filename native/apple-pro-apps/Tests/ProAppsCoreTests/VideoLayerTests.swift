import Foundation
import Testing

@testable import ProAppsCore

struct VideoLayerTests {
  @Test func videoLayersRoundTripAndLeaveTheBaseTimelineUnchanged() throws {
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: "/tmp/base.mp4",
          selection: .init(startSeconds: 0, durationSeconds: 60, rate: 1))
      ],
      video: .init(width: 720, height: 1280, frameRate: 30, resizeMode: .fit),
      additionalVideo: [
        .init(
          sourcePath: "/tmp/animation.mov",
          selection: .init(startSeconds: 0, durationSeconds: 6, rate: 1), offsetSeconds: 1),
        .init(
          sourcePath: "/tmp/animation.mov",
          selection: .init(startSeconds: 0, durationSeconds: 6, rate: 1), offsetSeconds: 20,
          geometry: .init(rotation: .clockwise90, crop: .init(x: 0, y: 0, width: 20, height: 20)),
          opacity: 0.5),
      ])
    let decoded = try JSONDecoder().decode(EditRecipe.self, from: JSONEncoder().encode(recipe))
    let layers = try #require(decoded.additionalVideo)
    #expect(layers.count == 2)
    #expect(layers[0].opacity == nil)
    #expect(layers[1].opacity == 0.5)
    #expect(layers[1].offsetSeconds == 20)
    #expect(layers[1].geometry?.rotation == .clockwise90)
    #expect(try EditPlan.build(decoded).durationSeconds == 60)
  }

  @Test(arguments: [-1.0, 1.01, Double.infinity, Double.nan])
  func rejectsInvalidOpacity(opacity: Double) {
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: "/tmp/base.mp4",
          selection: .init(startSeconds: 0, durationSeconds: 60, rate: 1))
      ],
      video: .init(width: 720, height: 1280, frameRate: 30, resizeMode: .fit),
      additionalVideo: [
        .init(
          sourcePath: "/tmp/animation.mov",
          selection: .init(startSeconds: 0, durationSeconds: 6, rate: 1), offsetSeconds: 0,
          opacity: opacity)
      ])
    #expect(throws: (any Error).self) { try EditPlan.build(recipe) }
  }

  @Test(arguments: [-1.0, 55.0, Double.infinity, Double.nan])
  func rejectsInvalidOrOutOfTimelineOffsets(offset: Double) {
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: "/tmp/base.mp4",
          selection: .init(startSeconds: 0, durationSeconds: 60, rate: 1))
      ],
      video: .init(width: 720, height: 1280, frameRate: 30, resizeMode: .fit),
      additionalVideo: [
        .init(
          sourcePath: "/tmp/animation.mov",
          selection: .init(startSeconds: 0, durationSeconds: 6, rate: 1), offsetSeconds: offset)
      ])
    #expect(throws: (any Error).self) { try EditPlan.build(recipe) }
  }

  @Test func requiresCanvasAndBoundsLayerCount() throws {
    let layer = EditVideoLayer(
      sourcePath: "/tmp/animation.mov",
      selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1), offsetSeconds: 0, opacity: 0)
    let clips = [
      EditClip(
        sourcePath: "/tmp/base.mp4", selection: .init(startSeconds: 0, durationSeconds: 60, rate: 1)
      )
    ]
    #expect(throws: (any Error).self) {
      try EditPlan.build(EditRecipe(clips: clips, additionalVideo: [layer]))
    }
    #expect(try EditPlan.build(EditRecipe(clips: clips, additionalVideo: [])).audioOnly)
    let canvas = EditVideoSettings(width: 720, height: 1280, frameRate: 30, resizeMode: .fit)
    #expect(
      try EditPlan.build(
        EditRecipe(clips: clips, video: canvas, additionalVideo: Array(repeating: layer, count: 16))
      ).durationSeconds == 60)
    #expect(throws: (any Error).self) {
      try EditPlan.build(
        EditRecipe(clips: clips, video: canvas, additionalVideo: Array(repeating: layer, count: 17))
      )
    }
  }

  @Test(arguments: [
    EditVideoLayer(
      sourcePath: "relative.mov", selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
      offsetSeconds: 0),
    EditVideoLayer(
      sourcePath: "/tmp/animation.mov",
      selection: .init(startSeconds: 0, durationSeconds: 1, rate: 0), offsetSeconds: 0),
    EditVideoLayer(
      sourcePath: "/tmp/animation.mov",
      selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1), offsetSeconds: 0,
      geometry: .init(rotation: .none, crop: .init(x: -1, y: 0, width: 20, height: 20))),
  ])
  func rejectsBadPathsSelectionsAndCrops(layer: EditVideoLayer) {
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: "/tmp/base.mp4",
          selection: .init(startSeconds: 0, durationSeconds: 60, rate: 1))
      ],
      video: .init(width: 720, height: 1280, frameRate: 30, resizeMode: .fit),
      additionalVideo: [layer])
    #expect(throws: (any Error).self) { try EditPlan.build(recipe) }
  }
}
