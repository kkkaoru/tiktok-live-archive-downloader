import Foundation
import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

struct EditingServiceTests {
  private actor Capture {
    var path: String?
    func record(_ path: String) { self.path = path }
  }

  private func request(directory: String) -> EditRequest {
    .init(
      recipe: .init(
        clips: [
          .init(
            sourcePath: "/tmp/synthetic.mov",
            selection: .init(startSeconds: 0, durationSeconds: 2, rate: 2))
        ],
        video: .init(
          width: 320, height: 240, frameRate: 30, resizeMode: .fit,
          captions: [
            .init(text: String(repeating: "字幕の検証", count: 12), startSeconds: 0.1, endSeconds: 0.9)
          ], captionStyle: .init(outlineWidth: 3, backgroundOpacity: 0, centerY: 120)),
        additionalAudio: []), outputDirectory: directory, outputName: "out.mp4")
  }

  @Test(arguments: [0.9, 1.1])
  func timedBlurSchemaAndTimelineBounds(_ end: Double) async throws {
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: "/tmp/synthetic.mov",
          selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
      ],
      video: .init(
        width: 320, height: 240, frameRate: 30, resizeMode: .fit,
        masks: [
          .init(
            region: .init(x: 0, y: 100, width: 320, height: 40),
            opacity: 0.9, blurRadius: 16, startSeconds: 0.1, endSeconds: end)
        ]))
    let result = await NativeService().call(
      .init(name: "media_edit_plan", arguments: ["recipe": try Value(recipe)]))
    #expect(result.isError == (end > 1))
  }

  @Test func captionProjectionIsReadOnlyAndDoesNotClaimWordAlignment() async throws {
    let request = CaptionProjectionRequest(
      sourceCaptions: [
        .init(text: "字幕", startSeconds: 0, endSeconds: 4)
      ], retainedSpans: [.init(startSeconds: 1, endSeconds: 3)], maximumCharacters: 24)
    let arguments = try #require(try Value(request).objectValue)
    let result = await NativeService().call(.init(name: "caption_cut_plan", arguments: arguments))
    #expect(result.isError == false)
    let value = try #require(result.structuredContent?.objectValue?["plan"])
    let plan = try JSONDecoder().decode(
      CaptionProjectionResult.self, from: JSONEncoder().encode(value))
    #expect(plan.captions.count == 1)
    #expect(plan.captions.first?.startSeconds == 0)
    #expect(plan.captions.first?.endSeconds == 2)
    #expect(plan.timingIsEstimated)
  }

  @Test func speechPlanningPreservesProvidedSpanVerificationBoundary() async throws {
    let request = SpeechCutRequest(
      sourceDurationSeconds: 10,
      spans: [.init(startSeconds: 2, endSeconds: 4)], paddingSeconds: 0,
      minimumRemovedGapSeconds: 0.5)
    let arguments = try #require(try Value(request).objectValue)
    let result = await NativeService().call(.init(name: "speech_cut_plan", arguments: arguments))
    #expect(result.isError == false)
    let value = try #require(result.structuredContent?.objectValue?["plan"])
    let plan = try JSONDecoder().decode(SpeechCutPlan.self, from: JSONEncoder().encode(value))
    #expect(plan.outputDurationSeconds == 2)
    #expect(plan.segments.first?.sourceStartSeconds == 2)
    #expect(!plan.speechVerified)
    let invalid = await NativeService().call(.init(name: "speech_cut_plan", arguments: [:]))
    #expect(invalid.isError == true)
  }

  @Test func planningDoesNotPretendToValidateSources() async throws {
    let result = await NativeService().call(
      .init(
        name: "media_edit_plan", arguments: ["recipe": try Value(request(directory: "/tmp").recipe)]
      ))
    #expect(result.isError == false)
    #expect(result.structuredContent?.objectValue?["sourceFilesValidated"] == .bool(false))
    #expect(
      result.structuredContent?.objectValue?["plan"]?.objectValue?["durationSeconds"]?.doubleValue
        == 1
        || result.structuredContent?.objectValue?["plan"]?.objectValue?["durationSeconds"]?.intValue
          == 1
    )
  }

  @Test(arguments: ["success", "failure", "cancelled"])
  func renderingUsesAPrivateRequestAndCleansItOnEveryResult(_ mode: String) async throws {
    let capture = Capture()
    var native = NativeInterfaces()
    native.editMedia = { path in
      await capture.record(path)
      let decoded = try JSONDecoder().decode(
        EditRequest.self, from: Files.read(URL(fileURLWithPath: path)))
      #expect(decoded.outputName == "out.mp4")
      #expect(decoded.recipe.clips.count == 1)
      #expect(decoded.recipe.video?.captions?.first?.text == String(repeating: "字幕の検証", count: 12))
      if mode == "failure" { throw ProAppsError.unavailable("Synthetic renderer failure") }
      if mode == "cancelled" { throw CancellationError() }
      return try JSONDecoder().decode(
        EditRenderResult.self,
        from: Data(
          "{\"outputPath\":\"/tmp/synthetic-out.mp4\",\"projectPath\":\"/tmp/edit-request.json\",\"expectedDurationSeconds\":1,\"actualDurationSeconds\":1,\"videoTrackCount\":1,\"audioTrackCount\":1}"
            .utf8))
    }
    let value = try Value(request(directory: "/tmp"))
    let arguments = try #require(value.objectValue)
    let result = await NativeService(interfaces: native).call(
      .init(name: "media_edit", arguments: arguments))
    #expect(result.isError == (mode != "success"))
    let path = try #require(await capture.path)
    #expect(
      !FileManager.default.fileExists(
        atPath: URL(fileURLWithPath: path).deletingLastPathComponent().path))
    if mode == "success" {
      #expect(result.structuredContent?.objectValue?["fullMediaVerified"] == .bool(false))
      #expect(
        result.structuredContent?.objectValue?["render"]?.objectValue?["outputPath"]
          == .string("/tmp/synthetic-out.mp4"))
    } else if mode == "cancelled" {
      #expect(result.structuredContent?.objectValue?["cancelled"] == .bool(true))
    }
  }

  @Test func savedProjectsShareTheCLISchemaAndRejectUnknownKeys() async throws {
    let output = try Files.reserveOutput(
      directory: Files.temporaryDirectory().path, name: "request.json", kind: .edit)
    defer {
      do { try FileManager.default.removeItem(at: output.deletingLastPathComponent()) } catch {
        Issue.record(error)
      }
    }
    _ = try Files.writeNew(
      JSONEncoder().encode(request(directory: "/tmp")), to: output.path, extensions: ["json"])
    let service = NativeService()
    let result = await service.call(
      .init(name: "media_project_read", arguments: ["path": .string(output.path)]))
    #expect(result.isError == false)
    #expect(
      result.structuredContent?.objectValue?["request"]?.objectValue?["outputName"]
        == .string("out.mp4"))
    let value = try Value(request(directory: "/tmp"))
    var fields = try #require(value.objectValue)
    fields["shell"] = .string("not permitted")
    let bad = try Files.writeNew(
      JSONEncoder().encode(Value.object(fields)),
      to: output.deletingLastPathComponent().appendingPathComponent("bad.json").path,
      extensions: ["json"])
    await #expect(throws: (any Error).self) { try await service.readEditRequest(bad.path) }
    await #expect(throws: (any Error).self) {
      try await service.editing(
        "unknown", .object([:]), render: { _ in throw ProAppsError.invalid("Must not run") })
    }
  }
}
