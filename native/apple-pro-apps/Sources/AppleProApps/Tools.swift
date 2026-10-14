import Foundation
import MCP
import ProAppsCore

struct ToolSpec: Sendable {
  let name: String
  let description: String
  let properties: [String: Value]
  let required: [String]
  let readOnly: Bool

  var tool: Tool {
    Tool(
      name: name, description: description,
      inputSchema: Self.object(properties, required),
      annotations: .init(
        readOnlyHint: readOnly, destructiveHint: !readOnly,
        idempotentHint: readOnly, openWorldHint: true))
  }

  static func object(_ properties: [String: Value], _ required: [String]) -> Value {
    .object([
      "type": .string("object"), "properties": .object(properties),
      "required": .array(required.map(Value.string)), "additionalProperties": .bool(false),
    ])
  }
  static func string(_ values: [String]? = nil, maximum: Int = 4096) -> Value {
    var fields: [String: Value] = [
      "type": .string("string"), "minLength": .int(1), "maxLength": .int(maximum),
    ]
    if let values { fields["enum"] = .array(values.map(Value.string)) }
    return .object(fields)
  }
  static func integer(_ minimum: Int, _ maximum: Int) -> Value {
    .object(["type": .string("integer"), "minimum": .int(minimum), "maximum": .int(maximum)])
  }
  static func array(_ items: Value, maximum: Int, minimum: Int = 1) -> Value {
    .object([
      "type": .string("array"), "items": items, "minItems": .int(minimum),
      "maxItems": .int(maximum),
    ])
  }
  static let boolean: Value = .object(["type": .string("boolean")])
  static let number: Value = .object(["type": .string("number")])
  static let kind = string(["fcpxml", "motion"])
  static let app = string(ProApp.allCases.map(\.rawValue))

  static let all: [ToolSpec] =
    measurements + editing + ui + fcpSave + [
      .init(
        name: "fcpxml_validate",
        description:
          "Validate bounded FCPXML against the matching self-contained DTD from the selected installed Final Cut Pro edition, using private snapshots and system xmllint with network/catalog resolution disabled. No app launch or source writes. DTD validity does NOT prove media reference validity, editor import or visual fidelity.",
        properties: ["path": string(), "bundleID": string()], required: ["path"], readOnly: true),
      .init(
        name: "media_inspect",
        description:
          "Read local video metadata and decode the first frame using AVFoundation in a deadline-limited child process. Returns scalar evidence, no image/audio or playback. Not full-file validation or Apple app import verification.",
        properties: ["path": string()], required: ["path"], readOnly: true),
      .init(
        name: "app_capabilities",
        description:
          "Read installed editions and machine-integration support/limits. No GUI or application launch.",
        properties: [:], required: [], readOnly: true),
      .init(
        name: "app_open_document",
        description:
          "Deliver an existing supported project/MIDI/FCPXML document, or Logic Pro audio (WAV, AIFF, CAF, M4A, MP3), through LaunchServices/Open Document. May import/create data or show licensing dialogs; delivery is not completion or playback verification.",
        properties: ["app": app, "bundleID": string(), "path": string()],
        required: ["app", "path"],
        readOnly: false),
      .init(
        name: "motion_text_inspect",
        description:
          "Read Motion text-layer IDs, content, source SHA-256 and supported-layout flags without UI. Bounded single-style adapter; no render validation.",
        properties: ["path": string()], required: ["path"], readOnly: true),
      .init(
        name: "motion_text_copy",
        description:
          "Create a NEW fingerprint-bound Motion copy, updating text, character objects and style-run lengths together. Requires explicit experimental-format opt-in. Supports observed ozml 4.0 single-style, neutral-kerning, single-line BMP text only; rejects unsupported formatting. No UI, overwrite or automatic render.",
        properties: [
          "inputPath": string(), "outputPath": string(), "expectedSHA256": string(maximum: 64),
          "allowUndocumentedFormat": boolean,
          "changes": array(
            object(
              [
                "layerID": integer(1, Int(Int32.max)), "expectedText": string(maximum: 8192),
                "replacement": string(maximum: 120),
              ], ["layerID", "expectedText", "replacement"]), maximum: 32),
        ], required: ["inputPath", "outputPath", "expectedSHA256", "changes"], readOnly: false),
      .init(
        name: "interchange_inspect",
        description:
          "Inspect bounded local FCPXML or Motion XML without launching apps. Well-formedness only, NOT DTD validation.",
        properties: ["kind": kind, "path": string()], required: ["kind", "path"], readOnly: true),
      .init(
        name: "interchange_query",
        description:
          "Read bounded XPath-selected XML fragments from a local FCPXML/Motion document. Results are limited to 10 fragments of 2048 characters each; may be truncated.",
        properties: [
          "kind": kind, "path": string(), "xpath": string(maximum: 1024), "limit": integer(1, 10),
        ], required: ["kind", "path", "xpath"], readOnly: true),
      .init(
        name: "interchange_write",
        description:
          "Write a NEW FCPXML or Motion document, refusing overwrite/entities. Motion is undocumented and requires explicit format opt-in. Does not import or render.",
        properties: [
          "kind": kind, "xml": string(maximum: Files.maximumBytes), "outputPath": string(),
          "allowUndocumentedFormat": boolean,
        ], required: ["kind", "xml", "outputPath"], readOnly: false),
      .init(
        name: "interchange_patch",
        description:
          "Clone FCPXML/Motion XML to a NEW file while patching uniquely selected existing leaf/attribute values. No original overwrite; Motion requires undocumented-format opt-in.",
        properties: [
          "kind": kind, "inputPath": string(), "outputPath": string(),
          "allowUndocumentedFormat": boolean,
          "changes": array(
            object(
              [
                "xpath": string(maximum: 1024),
                "value": .object(["type": .string("string"), "maxLength": .int(65536)]),
              ], ["xpath", "value"]), maximum: 64),
        ], required: ["kind", "inputPath", "outputPath", "changes"], readOnly: false),
      .init(
        name: "midi_file_create",
        description:
          "Generate a NEW standard MIDI type-0 file with tempo and paired note events. No audio playback or application mutation; import into Logic separately.",
        properties: [
          "outputPath": string(), "bpm": number, "ticksPerQuarter": integer(1, 32767),
          "notes": array(
            object(
              [
                "note": integer(0, 127), "velocity": integer(1, 127),
                "startTick": integer(0, 0x07ff_ffff), "durationTick": integer(1, 0x07ff_ffff),
                "channel": integer(1, 16),
              ], ["note", "velocity", "startTick", "durationTick", "channel"]), maximum: 4096),
        ], required: ["outputPath", "bpm", "notes"], readOnly: false),
      .init(
        name: "midi_destinations",
        description:
          "Read current CoreMIDI destination IDs/names. Does not create virtual ports, change routes or send events.",
        properties: [:], required: [], readOnly: true),
      .init(
        name: "midi_send",
        description:
          "Send one CC/program/pitch-bend event to an exact destination ID AND name. Requires user-verified Logic/MainStage routing; can affect sound/recording. Not application-exclusive and no semantic acknowledgment.",
        properties: [
          "destinationID": integer(Int(Int32.min), Int(Int32.max)), "destinationName": string(),
          "kind": string(["controlChange", "programChange", "pitchBend"]),
          "channel": integer(1, 16),
          "number": integer(0, 127), "value": integer(0, 16383),
        ], required: ["destinationID", "destinationName", "kind", "channel", "number", "value"],
        readOnly: false),
      .init(
        name: "osc_send",
        description:
          "Send one float OSC datagram to an explicitly configured LOOPBACK port/path (Logic controller assignment). No discovery, remote hosts, wildcards or acknowledgment. MainStage needs a separately configured bridge.",
        properties: ["port": integer(1024, 65535), "path": string(maximum: 512), "value": number],
        required: ["port", "path", "value"], readOnly: false),
      .init(
        name: "compressor_inspect",
        description:
          "Run the installed Compressor CLI -checkstream against one regular source file, without submitting a job.",
        properties: ["path": string(), "bundleID": string()], required: ["path"], readOnly: true),
      .init(
        name: "compressor_submit",
        description:
          "Submit one local source with a trusted .cmprstng, .compressorsetting or Apple .setting preset through the official CLI. Optional range selects a bounded source interval. Creates a private unique output subdirectory. Return is submission evidence, NOT completion; never blindly retry.",
        properties: [
          "sourcePath": string(), "presetPath": string(), "outputDirectory": string(),
          "outputName": string(maximum: 180), "batchName": string(maximum: 200),
          "bundleID": string(),
          "range": object(
            ["startSeconds": integer(0, 86399), "durationSeconds": integer(1, 600)],
            ["startSeconds", "durationSeconds"]),
        ], required: ["sourcePath", "presetPath", "outputDirectory", "outputName", "batchName"],
        readOnly: false),
      .init(
        name: "compressor_status",
        description:
          "Read exactly one Compressor job/batch using bounded -monitor -once. Supply the ID returned by submission; no global polling loop.",
        properties: ["id": string(maximum: 128), "job": boolean, "bundleID": string()],
        required: ["id"], readOnly: true),
      .init(
        name: "compressor_control",
        description:
          "Pause/resume/cancel exactly one Compressor job or batch. Cancel is destructive; explicit user authorization is required. Never resets the service or cancels all jobs.",
        properties: [
          "id": string(maximum: 128), "job": boolean,
          "action": string(["pause", "resume", "cancel"]), "bundleID": string(),
        ], required: ["id", "action"], readOnly: false),
    ]
}

// Enforce the published schema at runtime, including unknown/nested keys. Codable
// alone ignores unknown keys and is not sufficient for a mutation boundary.
func validate(_ value: Value, schema: Value) throws {
  guard let fields = schema.objectValue, let type = fields["type"]?.stringValue else {
    throw ProAppsError.invalid("Invalid internal schema")
  }
  if let allowed = fields["enum"]?.arrayValue, !allowed.contains(value) {
    throw ProAppsError.invalid("Unknown enum value")
  }
  switch type {
  case "object":
    guard let object = value.objectValue, let properties = fields["properties"]?.objectValue,
      Set(object.keys).isSubset(of: Set(properties.keys))
    else { throw ProAppsError.invalid("Unknown fields or non-object arguments") }
    for key in fields["required"]?.arrayValue ?? [] {
      guard let key = key.stringValue, object[key] != nil else {
        throw ProAppsError.invalid("Missing required field")
      }
    }
    for (key, child) in object {
      if let childSchema = properties[key] { try validate(child, schema: childSchema) }
    }
  case "array":
    guard let array = value.arrayValue, let item = fields["items"],
      array.count >= (fields["minItems"]?.intValue ?? 0),
      array.count <= (fields["maxItems"]?.intValue ?? 0)
    else { throw ProAppsError.invalid("Invalid array size") }
    for child in array { try validate(child, schema: item) }
  case "string":
    guard let text = value.stringValue, !text.contains("\0"),
      text.utf8.count >= (fields["minLength"]?.intValue ?? 0),
      text.utf8.count <= (fields["maxLength"]?.intValue ?? 4096)
    else { throw ProAppsError.invalid("Invalid string or length") }
  case "integer":
    guard let integer = value.intValue, integer >= (fields["minimum"]?.intValue ?? Int.min),
      integer <= (fields["maximum"]?.intValue ?? Int.max)
    else { throw ProAppsError.invalid("Expected an integer in range") }
  case "number":
    guard value.intValue != nil || value.doubleValue?.isFinite == true else {
      throw ProAppsError.invalid("Expected finite number")
    }
  case "boolean":
    guard value.boolValue != nil else { throw ProAppsError.invalid("Expected boolean") }
  default: throw ProAppsError.invalid("Unsupported internal schema")
  }
}
