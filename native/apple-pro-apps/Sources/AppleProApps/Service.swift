import Dispatch
import Foundation
import MCP
import ProAppsCore

/// Explicit framework boundaries keep tests away from real projects, sound and jobs.
struct NativeInterfaces: Sendable {
  static func measurementTimeout(_ name: String) -> Duration {
    name == "audio_transcribe_whisper" ? .seconds(180) : .seconds(60)
  }

  var writeCueArtifact: @Sendable (Data, String) throws -> URL = { data, path in
    try Files.writeNew(data, to: path, extensions: ["json", "wav"])
  }
  var validateFCPXML: @Sendable (String, String?) async throws -> FCPXMLValidation = {
    path, bundleID in
    let app = try await Applications.resolve(.finalCutPro, bundleID: bundleID)
    let directory = URL(fileURLWithPath: app.path).appendingPathComponent(
      "Contents/Frameworks/Interchange.framework/Versions/A/Resources")
    return try await FCPXMLValidator(dtdDirectory: directory).validate(path: path)
  }
  var measureMedia: @Sendable (String, String) async throws -> Value = { name, path in
    guard let binary = Bundle.main.executableURL else {
      throw ProAppsError.unavailable("Cannot locate the measurement executable")
    }
    let result = try await Runner.run(
      binary, ["measure-media", name, path], timeout: measurementTimeout(name))
    guard result.status == 0 else { throw ProAppsError.commandFailed(result.status) }
    return try JSONDecoder().decode(Value.self, from: Data(result.stdout.utf8))
  }
  var editMedia: @Sendable (String) async throws -> EditRenderResult = { path in
    guard let binary = Bundle.main.executableURL else {
      throw ProAppsError.unavailable("Cannot locate the editing executable")
    }
    let result = try await Runner.run(binary, ["edit-media", path], timeout: .seconds(300))
    guard result.status == 0 else { throw ProAppsError.commandFailed(result.status) }
    return try JSONDecoder().decode(EditRenderResult.self, from: Data(result.stdout.utf8))
  }
  var inspectMedia: @Sendable (String) async throws -> MediaSummary = { path in
    guard let binary = Bundle.main.executableURL else {
      throw ProAppsError.unavailable("Cannot locate the media probe executable")
    }
    let text = try await Runner.checked(binary, ["inspect-media", path])
    return try JSONDecoder().decode(MediaSummary.self, from: Data(text.utf8))
  }
  var inventory: @Sendable () async -> [InstalledApp] = { await Applications.inventory() }
  var openDocument: @Sendable (ProApp, String?, String) async throws -> String = {
    try await Applications.openDocument(app: $0, bundleID: $1, path: $2)
  }
  var compressor: @Sendable (String?) async throws -> URL = {
    try await Compressor.executable(bundleID: $0)
  }
  var command: @Sendable (URL, [String]) async throws -> CommandResult = {
    try await Runner.run($0, $1)
  }
  var destinations: @Sendable () -> [MIDIDestination] = { MIDIControl.destinations() }
  var midi: @Sendable (Int32, String, UInt32) throws -> Void = {
    try MIDIControl.send(id: $0, name: $1, word: $2)
  }
  var osc: @Sendable (Int, String, Double) throws -> Void = {
    try OSC.send(port: $0, path: $1, value: $2)
  }
  var ui: @Sendable (UIRequest, Duration) async throws -> UIResponse = { request, timeout in
    guard let binary = Bundle.main.executableURL else {
      throw ProAppsError.unavailable("Cannot locate the UI automation executable")
    }
    let encoded = String(decoding: try JSONEncoder().encode(request), as: UTF8.self)
    let result = try await Runner.run(binary, ["ui-native", encoded], timeout: timeout)
    guard result.status == 0 else { throw ProAppsError.commandFailed(result.status) }
    let outcome = try JSONDecoder().decode(UIChildOutcome.self, from: Data(result.stdout.utf8))
    if let error = outcome.rethrown() { throw error }
    guard let response = outcome.response else {
      throw ProAppsError.unavailable("UI child returned no result")
    }
    return response
  }
}

actor NativeService {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.native")
  nonisolated var unownedExecutor: UnownedSerialExecutor { executor.asUnownedSerialExecutor() }
  let interfaces: NativeInterfaces
  private var callsInFlight = 0
  private var mutationInFlight = false

  init(interfaces: NativeInterfaces = NativeInterfaces()) { self.interfaces = interfaces }

  func call(_ params: CallTool.Parameters) async -> CallTool.Result {
    do {
      try Task.checkCancellation()
      guard callsInFlight < 4 else {
        throw ProAppsError.unavailable("Native request admission limit reached")
      }
      callsInFlight += 1
      defer { callsInFlight -= 1 }
      guard let spec = ToolSpec.all.first(where: { $0.name == params.name }) else {
        throw ProAppsError.invalid("Unknown tool")
      }
      let arguments = Value.object(params.arguments ?? [:])
      try validate(arguments, schema: spec.tool.inputSchema)
      if !spec.readOnly {
        guard !mutationInFlight else {
          throw ProAppsError.unavailable(
            "Another mutation is in flight; inspect it before submitting another")
        }
        mutationInFlight = true
      }
      defer { if !spec.readOnly { mutationInFlight = false } }
      return try await dispatch(params.name, arguments)
    } catch is CancellationError {
      return response(["cancelled": .bool(true), "retrySafe": .bool(false)], failed: true)
    } catch let error as ProAppsError {
      return response(
        ["error": .string(error.description), "retrySafe": .bool(false)], failed: true)
    } catch {
      // Native exceptions can contain input/project data. Do not echo them.
      return response(
        [
          "error": .string(
            "Operation failed. Check the selected file, app and permissions; do not blindly retry."),
          "retrySafe": .bool(false),
        ], failed: true)
    }
  }

  func writeCueArtifact(_ data: Data, to path: String) throws -> URL {
    try interfaces.writeCueArtifact(data, path)
  }

  func decode<T: Decodable>(_ type: T.Type, _ value: Value) throws -> T {
    try JSONDecoder().decode(type, from: JSONEncoder().encode(value))
  }

  func response(_ fields: [String: Value], failed: Bool = false) -> CallTool.Result {
    let value = Value.object(fields)
    let text: String
    do { text = String(decoding: try JSONEncoder().encode(value), as: UTF8.self) } catch {
      return .init(
        content: [.text(text: "Result encoding failed", annotations: nil, _meta: nil)],
        isError: true)
    }
    return .init(
      content: [.text(text: text, annotations: nil, _meta: nil)],
      structuredContent: Optional.some(value),
      isError: failed)
  }

  private func written(_ url: URL) -> CallTool.Result {
    response(["outputPath": .string(url.path), "created": .bool(true), "imported": .bool(false)])
  }

  private func native(_ binary: URL, _ args: [String], output: URL? = nil) async -> CallTool.Result
  {
    do {
      let result = try await interfaces.command(binary, args)
      var fields: [String: Value] = [
        "exitStatus": .int(Int(result.status)),
        "stdout": .string(String(result.stdout.prefix(16000))),
        "truncated": .bool(result.stdout.count > 16000), "completionVerified": .bool(false),
        "retrySafe": .bool(false),
      ]
      if let output { fields["outputPath"] = .string(output.path) }
      return response(fields, failed: result.status != 0)
    } catch is CancellationError {
      var fields: [String: Value] = ["cancelled": .bool(true), "retrySafe": .bool(false)]
      if let output { fields["outputPath"] = .string(output.path) }
      return response(fields, failed: true)
    } catch {
      var fields: [String: Value] = [
        "error": .string(
          "Native command failed or timed out; its effect may be partial. Do not resubmit blindly."),
        "retrySafe": .bool(false),
      ]
      if let output { fields["outputPath"] = .string(output.path) }
      return response(fields, failed: true)
    }
  }

  private func dispatch(_ name: String, _ args: Value) async throws -> CallTool.Result {
    switch name {
    case "fcp_save_plan": return try await fcpSavePlan(args)
    case "fcp_save_navigate": return try await fcpSaveNavigate(args)
    case "app_capabilities":
      return response([
        "applications": try Value(await interfaces.inventory()),
        "machineInterfaces": .object([
          "offlineMedia": .string(
            "Native MP4/M4A recipes: trim/order/speed, crop/rotation/fit/fill, gain/fades/mix, cross-dissolves, SDR color and static titles. Reusable JSON; bounded video/audio/region measurement. Not live proprietary-editor state."
          ),
          "motion": .string(
            "Project open + bounded XML inspect/query/copy/patch. Undocumented ozml writes are opt-in; no documented headless Motion render API."
          ),
          "finalCutPro": .string(
            "FCPXML inspect/query/write/copy/patch, separate installed Apple DTD validation and Open Document delivery. Not a live timeline editing/export API; DTD validity does not prove import."
          ),
          "compressor": .string(
            "Official CLI source inspection, submission, bounded status, pause/resume/cancel by ID."
          ),
          "logicPro": .string(
            "Project/MIDI file open, SMF generation, CoreMIDI CC/program/pitch bend and loopback OSC. Requires explicit controller routing/assignments."
          ),
          "mainStage": .string(
            "Concert/patch open and CoreMIDI mapped controls. No documented native OSC or full concert editing API."
          ),
        ]),
        "guiAutomation": .bool(false), "backgroundAccessibility": .bool(true),
        "allOperationsGuaranteed": .bool(false),
      ])
    case "media_verify_video", "audio_measure", "video_frame_measure", "audio_transcribe",
      "audio_reference_analyze", "audio_sound_activity", "audio_separate_vocals",
      "audio_cue_track", "video_text_recognize", "video_head_detect", "video_face_track",
      "speech_locale_reserve", "audio_transcribe_whisper", "caption_text_measure",
      "video_foreground_measure":
      return try await measurement(name, args, execute: interfaces.measureMedia)
    case "media_edit_plan", "media_edit", "media_project_read", "speech_cut_plan",
      "caption_cut_plan", "media_edit_batch":
      return try await editing(name, args, render: interfaces.editMedia)
    case let name where ToolSpec.uiNames.contains(name):
      return try await uiTool(name, args)
    case "media_inspect":
      struct Input: Decodable { let path: String }
      let input = try decode(Input.self, args)
      let source = try Files.existing(input.path)
      return response([
        "media": try Value(await interfaces.inspectMedia(source.path)),
        "verificationScope": .string("metadata-and-first-decoded-video-frame"),
        "fullVideoVerified": .bool(false), "sourceModified": .bool(false),
      ])
    case "app_open_document":
      struct Input: Decodable {
        let app: ProApp
        let bundleID: String?
        let path: String
      }
      let input = try decode(Input.self, args)
      return response([
        "delivery": .string(
          try await interfaces.openDocument(input.app, input.bundleID, input.path)),
        "completionVerified": .bool(false),
      ])
    case "fcpxml_validate":
      struct Input: Decodable {
        let path: String
        let bundleID: String?
      }
      let input = try decode(Input.self, args)
      return response([
        "validation": try await Value(interfaces.validateFCPXML(input.path, input.bundleID)),
        "sourceModified": .bool(false),
      ])
    case "motion_text_inspect":
      struct Input: Decodable { let path: String }
      let input = try decode(Input.self, args)
      let data = try Files.read(
        Files.existing(input.path, extensions: InterchangeKind.motion.extensions))
      return response([
        "inventory": try Value(MotionText.inspect(data)), "sourceModified": .bool(false),
      ])
    case "motion_text_copy":
      struct Input: Decodable {
        let inputPath: String
        let outputPath: String
        let expectedSHA256: String
        let changes: [MotionTextChange]
        let allowUndocumentedFormat: Bool?
      }
      let input = try decode(Input.self, args)
      return written(
        try MotionText.copy(
          input: input.inputPath, output: input.outputPath, expectedSHA256: input.expectedSHA256,
          changes: input.changes, allowUndocumented: input.allowUndocumentedFormat ?? false))
    case "interchange_inspect":
      struct Input: Decodable {
        let kind: InterchangeKind
        let path: String
      }
      let input = try decode(Input.self, args)
      let data = try Files.read(Files.existing(input.path, extensions: input.kind.extensions))
      return response(["summary": try Value(Interchange.inspect(data, kind: input.kind))])
    case "interchange_query":
      struct Input: Decodable {
        let kind: InterchangeKind
        let path: String
        let xpath: String
        let limit: Int?
      }
      let input = try decode(Input.self, args)
      let data = try Files.read(Files.existing(input.path, extensions: input.kind.extensions))
      return response([
        "fragments": try Value(
          Interchange.query(data, kind: input.kind, xpath: input.xpath, limit: input.limit ?? 10)),
        "bounded": .bool(true),
      ])
    case "interchange_write":
      struct Input: Decodable {
        let kind: InterchangeKind
        let xml: String
        let outputPath: String
        let allowUndocumentedFormat: Bool?
      }
      let input = try decode(Input.self, args)
      return written(
        try Interchange.write(
          input.xml, kind: input.kind, output: input.outputPath,
          allowUndocumented: input.allowUndocumentedFormat ?? false))
    case "interchange_patch":
      struct Input: Decodable {
        let kind: InterchangeKind
        let inputPath: String
        let outputPath: String
        let changes: [XMLChange]
        let allowUndocumentedFormat: Bool?
      }
      let input = try decode(Input.self, args)
      return written(
        try Interchange.patch(
          input: input.inputPath, output: input.outputPath, kind: input.kind,
          changes: input.changes, allowUndocumented: input.allowUndocumentedFormat ?? false))
    case "midi_file_create":
      struct Input: Decodable {
        let outputPath: String
        let bpm: Double
        let ticksPerQuarter: Int?
        let notes: [MIDINote]
      }
      let input = try decode(Input.self, args)
      let bytes = try MIDIFile.create(
        notes: input.notes, bpm: input.bpm, ticksPerQuarter: input.ticksPerQuarter ?? 480)
      return written(try Files.writeNew(bytes, to: input.outputPath, extensions: ["mid", "midi"]))
    case "midi_destinations":
      return response([
        "destinations": try Value(interfaces.destinations()), "routingVerified": .bool(false),
      ])
    case "midi_send":
      struct Input: Decodable {
        let destinationID: Int32
        let destinationName: String
        let kind: MIDIControl.Kind
        let channel: Int
        let number: Int
        let value: Int
      }
      let input = try decode(Input.self, args)
      let word = try MIDIControl.message(
        kind: input.kind, channel: input.channel, number: input.number, value: input.value)
      try interfaces.midi(input.destinationID, input.destinationName, word)
      return response([
        "dispatched": .bool(true), "effectVerified": .bool(false), "retrySafe": .bool(false),
      ])
    case "osc_send":
      struct Input: Decodable {
        let port: Int
        let path: String
        let value: Double
      }
      let input = try decode(Input.self, args)
      try interfaces.osc(input.port, input.path, input.value)
      return response([
        "datagramSent": .bool(true), "receiverVerified": .bool(false), "retrySafe": .bool(false),
      ])
    case "compressor_inspect":
      struct Input: Decodable {
        let path: String
        let bundleID: String?
      }
      let input = try decode(Input.self, args)
      let path = try Files.existing(input.path)
      return await native(
        try await interfaces.compressor(input.bundleID),
        Compressor.inspectionArguments(source: path))
    case "compressor_submit":
      struct Input: Decodable {
        let sourcePath: String
        let presetPath: String
        let outputDirectory: String
        let outputName: String
        let batchName: String
        let range: Compressor.TimeRange?
        let bundleID: String?
      }
      let input = try decode(Input.self, args)
      let source = try Files.existing(input.sourcePath)
      let preset = try Files.existing(
        input.presetPath, extensions: ["cmprstng", "setting", "compressorsetting"])
      let binary = try await interfaces.compressor(input.bundleID)
      let output = try Compressor.reserveOutput(
        directory: input.outputDirectory, name: input.outputName)
      let arguments = try Compressor.submissionArguments(
        source: source, preset: preset, output: output, batchName: input.batchName,
        range: input.range)
      return await native(binary, arguments, output: output)
    case "compressor_status", "compressor_control":
      struct Input: Decodable {
        let id: String
        let job: Bool?
        let action: Compressor.Control?
        let bundleID: String?
      }
      let input = try decode(Input.self, args)
      let arguments = try Compressor.monitoringArguments(
        id: input.id, job: input.job ?? false, control: input.action)
      return await native(try await interfaces.compressor(input.bundleID), arguments)
    default: throw ProAppsError.invalid("Unknown tool")
    }
  }
}
