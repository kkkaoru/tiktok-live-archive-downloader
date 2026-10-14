import Foundation
import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

struct ServiceTests {
  @Test func logicAudioCatalogDoesNotPromiseImportOrPlayback() throws {
    let spec = try #require(ToolSpec.all.first { $0.name == "app_open_document" })
    #expect(spec.description.contains("Logic Pro audio (WAV, AIFF, CAF, M4A, MP3)"))
    #expect(spec.description.contains("not completion or playback verification"))
    #expect(!spec.readOnly)
  }

  @Test func logicAudioDeliveryRemainsUnverified() async throws {
    var interfaces = NativeInterfaces()
    interfaces.openDocument = { app, bundleID, path in
      #expect(app == .logicPro)
      #expect(bundleID == "com.apple.mobilelogic")
      #expect(path == "/synthetic/instrumental.wav")
      return "delivery accepted"
    }
    let service = NativeService(interfaces: interfaces)
    let result = await service.call(
      .init(
        name: "app_open_document",
        arguments: [
          "app": .string("logicPro"), "bundleID": .string("com.apple.mobilelogic"),
          "path": .string("/synthetic/instrumental.wav"),
        ]))
    #expect(result.isError == false)
    #expect(result.structuredContent?.objectValue?["completionVerified"] == .bool(false))
  }

  @Test func catalogIsClosedAndUsesHonestAnnotations() throws {
    #expect(ToolSpec.all.count == 58)
    #expect(Set(ToolSpec.all.map(\.name)).count == 58)
    #expect(ToolSpec.all.contains { $0.name == "app_activate" })
    #expect(ToolSpec.all.contains { $0.name == "fcp_save_navigate" })
    #expect(!ToolSpec.all.contains { $0.name == "shell" || $0.name == "agent" })
  }

  @Test func schemasRejectUnknownNestedAndWrongTypedValues() throws {
    let control = try #require(ToolSpec.all.first { $0.name == "compressor_status" })
    #expect(throws: (any Error).self) {
      try validate(
        .object(["id": .string("safe"), "action": .string("cancel")]),
        schema: control.tool.inputSchema)
    }
    #expect(throws: (any Error).self) {
      try validate(.object([:]), schema: control.tool.inputSchema)
    }
    let midi = try #require(ToolSpec.all.first { $0.name == "midi_send" })
    #expect(throws: (any Error).self) {
      try validate(.object(["channel": .double(1.2)]), schema: midi.tool.inputSchema)
    }
    let patch = try #require(ToolSpec.all.first { $0.name == "interchange_patch" })
    let payload: Value = .object([
      "kind": .string("fcpxml"), "inputPath": .string("/tmp/a.fcpxml"),
      "outputPath": .string("/tmp/b.fcpxml"),
      "changes": .array([
        .object([
          "xpath": .string("/fcpxml/@version"), "value": .string("1.12"), "shell": .string("no"),
        ])
      ]),
    ])
    #expect(throws: (any Error).self) { try validate(payload, schema: patch.tool.inputSchema) }
    let schema = ToolSpec.object(
      ["n": ToolSpec.integer(0, 2), "b": ToolSpec.boolean, "v": ToolSpec.number], ["n"])
    try validate(.object(["n": .int(1), "b": .bool(false), "v": .double(0.5)]), schema: schema)
  }

  @Test(arguments: ToolSpec.all.map(\.name))
  func toolAnnotationsMatchTheMutationContract(_ name: String) throws {
    let spec = try #require(ToolSpec.all.first { $0.name == name })
    let readOnlyNames: Set<String> = [
      "app_capabilities", "interchange_inspect", "interchange_query", "midi_destinations",
      "motion_text_inspect",
      "compressor_inspect", "compressor_status", "media_inspect", "media_edit_plan",
      "media_project_read", "media_verify_video", "audio_measure", "video_frame_measure",
      "audio_transcribe", "audio_transcribe_whisper", "video_text_recognize", "video_head_detect",
      "video_face_track", "video_foreground_measure", "caption_text_measure",
      "image_text_rectangles", "image_text_rectangles_batch", "image_text_recognize_batch",
      "audio_reference_analyze", "audio_sound_activity",
      "speech_cut_plan", "caption_cut_plan",
      "fcpxml_validate", "fcp_save_plan",
      "ui_windows", "ui_inspect", "ui_menu_list", "ui_wait", "input_source_status",
    ]
    #expect(spec.tool.inputSchema.objectValue?["additionalProperties"] == .bool(false))
    #expect(spec.tool.annotations.readOnlyHint == readOnlyNames.contains(name))
    #expect(spec.tool.annotations.destructiveHint == !readOnlyNames.contains(name))
  }

  @Test(arguments: [
    Value.object(["n": .double(1.5)]), .object(["n": .int(3)]),
    .object(["n": .int(1), "b": .string("true")]),
    .object(["n": .int(1), "v": .double(.infinity)]),
  ])
  func rejectsWrongScalarTypes(_ value: Value) {
    let schema = ToolSpec.object(
      ["n": ToolSpec.integer(0, 2), "b": ToolSpec.boolean, "v": ToolSpec.number], ["n"])
    #expect(throws: (any Error).self) { try validate(value, schema: schema) }
  }

  @Test @MainActor func invalidMutationsFailBeforeNativeDispatch() async {
    let service = NativeService()
    let unknown = await service.call(.init(name: "arbitrary-shell", arguments: [:]))
    #expect(unknown.isError == true)
    let injection = await service.call(
      .init(
        name: "compressor_status", arguments: ["id": .string("x"), "action": .string("cancel")]))
    #expect(injection.isError == true)
    let midi = await service.call(
      .init(
        name: "midi_send",
        arguments: [
          "destinationID": .int(0), "destinationName": .string("never-dispatch"),
          "kind": .string("controlChange"), "channel": .int(1), "number": .int(1),
          "value": .int(128),
        ]))
    #expect(midi.isError == true)
    let osc = await service.call(
      .init(
        name: "osc_send", arguments: ["port": .int(80), "path": .string("/no"), "value": .int(1)]))
    #expect(osc.isError == true)
  }

  @Test @MainActor func fileToolsRoundTripWithoutAnApp() async throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "pro-apps-service-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let service = NativeService()
    let path = directory.appendingPathComponent("test.fcpxml").path
    let write = await service.call(
      .init(
        name: "interchange_write",
        arguments: [
          "kind": .string("fcpxml"), "outputPath": .string(path),
          "xml": .string("<fcpxml version=\"1.12\"><resources/></fcpxml>"),
        ]))
    #expect(write.isError == false)
    let again = await service.call(
      .init(
        name: "interchange_write",
        arguments: [
          "kind": .string("fcpxml"), "outputPath": .string(path), "xml": .string("<fcpxml/>"),
        ]))
    #expect(again.isError == true)
    let inspect = await service.call(
      .init(
        name: "interchange_inspect", arguments: ["kind": .string("fcpxml"), "path": .string(path)]))
    #expect(inspect.isError == false)
    let query = await service.call(
      .init(
        name: "interchange_query",
        arguments: [
          "kind": .string("fcpxml"), "path": .string(path), "xpath": .string("/fcpxml/@version"),
        ]))
    #expect(query.isError == false)
    let patch = await service.call(
      .init(
        name: "interchange_patch",
        arguments: [
          "kind": .string("fcpxml"), "inputPath": .string(path),
          "outputPath": .string(directory.appendingPathComponent("copy.fcpxml").path),
          "changes": .array([
            .object(["xpath": .string("/fcpxml/@version"), "value": .string("1.11")])
          ]),
        ]))
    #expect(patch.isError == false)
    let midi = await service.call(
      .init(
        name: "midi_file_create",
        arguments: [
          "outputPath": .string(directory.appendingPathComponent("test.mid").path),
          "bpm": .int(120),
          "notes": .array([
            .object([
              "note": .int(60), "velocity": .int(80), "channel": .int(1), "startTick": .int(0),
              "durationTick": .int(480),
            ])
          ]),
        ]))
    #expect(midi.isError == false)
  }
}
