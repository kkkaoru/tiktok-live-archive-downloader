import Foundation
import MCP
import ProAppsCore
import Synchronization
import Testing

@testable import AppleProApps

struct EditBatchServiceTests {
  private actor Tracker {
    var inFlight = 0
    var peak = 0
    var names: [String] = []
    func begin(_ name: String) {
      inFlight += 1
      peak = max(peak, inFlight)
      names.append(name)
    }
    func end() { inFlight -= 1 }
  }

  static func request(_ name: String, rate: Double = 1) -> EditRequest {
    EditRequest(
      recipe: EditRecipe(
        clips: [
          EditClip(
            sourcePath: "/tmp/synthetic.mov",
            selection: EditSelection(startSeconds: 0, durationSeconds: 1, rate: rate))
        ],
        video: EditVideoSettings(
          width: 320, height: 240, frameRate: 30, resizeMode: .fit,
          styledCaptions: [
            EditStyledCaption(text: "字幕", startSeconds: 0, endSeconds: 0.5, x: 160, bottom: 200)
          ],
          captionAppearance: EditCaptionAppearance(
            fontName: "HiraginoSans-W6", assFontSize: 24, fill: "#6CD4FF", border: 2,
            borderColor: "#FFFFFF", rim: 1, rimColor: "#000000"),
          encoding: EditEncoding(
            codec: .h264, averageBitRate: 2_000_000, allowFrameReordering: false))),
      outputDirectory: "/tmp", outputName: name)
  }

  static let rendered = Data(
    "{\"outputPath\":\"/tmp/out.mp4\",\"projectPath\":\"/tmp/edit-request.json\",\"expectedDurationSeconds\":1,\"actualDurationSeconds\":1,\"videoTrackCount\":1,\"audioTrackCount\":0,\"frameCount\":30}"
      .utf8)

  static func arguments(_ items: [EditRequest], concurrency: Int? = nil) throws -> [String: Value] {
    var fields: [String: Value] = ["items": try Value(items)]
    if let concurrency { fields["concurrency"] = .int(concurrency) }
    return fields
  }

  @Test func rendersItemsInOrderWithBoundedConcurrencyAndIsolatedFailures() async throws {
    let tracker = Tracker()
    var native = NativeInterfaces()
    native.editMedia = { path in
      let request = try JSONDecoder().decode(
        EditRequest.self, from: Files.read(URL(fileURLWithPath: path)))
      await tracker.begin(request.outputName)
      try await Task.sleep(for: .milliseconds(20))
      await tracker.end()
      switch request.outputName {
      case "b.mp4": throw ProAppsError.unavailable("Synthetic failure")
      case "c.mp4": throw CancellationError()
      default: return try JSONDecoder().decode(EditRenderResult.self, from: Self.rendered)
      }
    }
    let items = ["a.mp4", "b.mp4", "c.mp4", "d.mp4", "e.mp4"].map { Self.request($0) }
    let result = await NativeService(interfaces: native).call(
      .init(name: "media_edit_batch", arguments: try Self.arguments(items, concurrency: 2)))
    #expect(result.isError == false)
    let fields = try #require(result.structuredContent?.objectValue)
    #expect(fields["completed"] == .int(3))
    #expect(fields["failed"] == .int(2))
    let outcomes = try #require(fields["items"]?.arrayValue)
    #expect(outcomes.map { $0.objectValue?["index"] } == (0..<5).map { .int($0) })
    #expect(outcomes[1].objectValue?["error"] == .string("Unavailable: Synthetic failure"))
    #expect(
      outcomes[2].objectValue?["error"]
        == .string("Render failed; inspect the output directory before retrying"))
    #expect(
      outcomes[0].objectValue?["render"]?.objectValue?["frameCount"] == .int(30))
    #expect(await tracker.peak <= 2)
    #expect(await tracker.names.count == 5)
  }

  @Test func rejectsTheWholeBatchWhenAnyRecipeIsInvalid() async throws {
    let called = Mutex(false)
    var native = NativeInterfaces()
    native.editMedia = { _ in
      called.withLock { $0 = true }
      return try JSONDecoder().decode(EditRenderResult.self, from: Self.rendered)
    }
    let items = [Self.request("a.mp4"), Self.request("b.mp4", rate: 0.1)]
    let result = await NativeService(interfaces: native).call(
      .init(name: "media_edit_batch", arguments: try Self.arguments(items)))
    #expect(result.isError == true)
    #expect(!called.withLock { $0 })
  }

  @Test func schemaRejectsUnknownStyledCaptionFieldsAndOversizedBatches() async throws {
    var item = try #require(try Value(Self.request("a.mp4")).objectValue)
    var recipe = try #require(item["recipe"]?.objectValue)
    var video = try #require(recipe["video"]?.objectValue)
    video["styledCaptions"] = .array([
      .object([
        "text": .string("x"), "startSeconds": .double(0), "endSeconds": .double(1),
        "x": .double(1), "bottom": .double(1), "color": .string("#FFFFFF"),
      ])
    ])
    recipe["video"] = .object(video)
    item["recipe"] = .object(recipe)
    let service = NativeService(interfaces: NativeInterfaces())
    let unknown = await service.call(
      .init(name: "media_edit_batch", arguments: ["items": .array([.object(item)])]))
    #expect(unknown.isError == true)
    let tooMany = try Self.arguments(Array(repeating: Self.request("a.mp4"), count: 33))
    #expect(await service.call(.init(name: "media_edit_batch", arguments: tooMany)).isError == true)
    let badConcurrency = try Self.arguments([Self.request("a.mp4")], concurrency: 5)
    #expect(
      await service.call(.init(name: "media_edit_batch", arguments: badConcurrency)).isError
        == true)
  }

  @Test func omittedConcurrencyRendersWithTheDefault() async throws {
    var native = NativeInterfaces()
    native.editMedia = { _ in try JSONDecoder().decode(EditRenderResult.self, from: Self.rendered) }
    let result = await NativeService(interfaces: native).call(
      .init(name: "media_edit_batch", arguments: try Self.arguments([Self.request("a.mp4")])))
    #expect(result.structuredContent?.objectValue?["completed"] == .int(1))
  }

  @Test func defaultConcurrencyIsThree() async {
    #expect(NativeService.defaultBatchConcurrency == 3)
    let values = await NativeService.renderBatch([], concurrency: 3) { _ in
      try JSONDecoder().decode(EditRenderResult.self, from: Self.rendered)
    }
    #expect(values.isEmpty)
  }
}
