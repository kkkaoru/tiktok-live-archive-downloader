import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

struct VideoLayerServiceTests {
  @Test(arguments: [0.5, 1.1])
  func layerSchemaAndTimelineValidation(opacity: Double) async throws {
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: "/tmp/base.mp4",
          selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
      ],
      video: .init(width: 64, height: 64, frameRate: 30, resizeMode: .fit),
      additionalVideo: [
        .init(
          sourcePath: "/tmp/animation.mov",
          selection: .init(startSeconds: 0, durationSeconds: 0.5, rate: 1), offsetSeconds: 0.25,
          geometry: .init(rotation: .none), opacity: opacity)
      ])
    let result = await NativeService().call(
      .init(name: "media_edit_plan", arguments: ["recipe": try Value(recipe)]))
    #expect(result.isError == (opacity > 1))
  }
}
