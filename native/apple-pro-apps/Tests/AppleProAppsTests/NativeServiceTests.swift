import Foundation
import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

struct NativeServiceTests {
  func interfaces() -> NativeInterfaces {
    var native = NativeInterfaces()
    native.inventory = { [] }
    native.inspectMedia = { _ in
      try JSONDecoder().decode(
        MediaSummary.self,
        from: Data(
          #"{"durationSeconds":1,"width":16,"height":16,"frameRate":1,"audioTrackCount":0,"firstFrameDecoded":true}"#
            .utf8))
    }
    native.openDocument = { _, _, _ in "synthetic-open" }
    native.compressor = { _ in URL(fileURLWithPath: "/synthetic/compressor") }
    native.command = { _, _ in CommandResult(stdout: #"{"batchID":"synthetic-job"}"#, status: 0) }
    native.destinations = { [] }
    native.midi = { _, _, _ in }
    native.osc = { _, _, _ in }
    return native
  }

  @Test(arguments: [
    "app_capabilities", "app_open_document", "midi_destinations", "midi_send", "osc_send",
    "compressor_inspect", "compressor_submit", "compressor_status", "compressor_control",
    "media_inspect",
  ])
  func routesTypedNativeToolsWithoutRealProjects(_ name: String) async throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "pro-apps-native-mock-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let source = directory.appendingPathComponent("source.wav")
    let preset = directory.appendingPathComponent("trusted.cmprstng")
    try Data().write(to: source)
    try Data().write(to: preset)
    let arguments: [String: Value]
    switch name {
    case "app_open_document":
      arguments = ["app": .string("logicPro"), "path": .string("/synthetic/project.logicx")]
    case "midi_send":
      arguments = [
        "destinationID": .int(42), "destinationName": .string("synthetic-only"),
        "kind": .string("controlChange"), "channel": .int(1), "number": .int(7), "value": .int(100),
      ]
    case "osc_send":
      arguments = ["port": .int(9000), "path": .string("/synthetic"), "value": .double(0.5)]
    case "compressor_inspect", "media_inspect": arguments = ["path": .string(source.path)]
    case "compressor_submit":
      arguments = [
        "sourcePath": .string(source.path), "presetPath": .string(preset.path),
        "outputDirectory": .string(directory.path), "outputName": .string("out.mov"),
        "batchName": .string("synthetic-only"),
      ]
    case "compressor_status": arguments = ["id": .string("synthetic-job"), "job": .bool(true)]
    case "compressor_control":
      arguments = ["id": .string("synthetic-job"), "action": .string("pause")]
    default: arguments = [:]
    }
    var native = interfaces()
    if name == "compressor_inspect" {
      native.command = { _, arguments in
        #expect(arguments.first == "-checkstream")
        #expect(arguments.last?.hasPrefix("file:///") == true)
        return CommandResult(stdout: "synthetic inspection", status: 0)
      }
    }
    let result = await NativeService(interfaces: native).call(
      .init(name: name, arguments: arguments))
    #expect(result.isError == false)
    if name == "app_capabilities" {
      #expect(result.structuredContent?.objectValue?["allOperationsGuaranteed"] == .bool(false))
    }
    if name == "media_inspect" {
      #expect(result.structuredContent?.objectValue?["fullVideoVerified"] == .bool(false))
      #expect(
        result.structuredContent?.objectValue?["media"]?.objectValue?["firstFrameDecoded"]
          == .bool(true))
    }
    if name == "compressor_submit" {
      let path = try #require(result.structuredContent?.objectValue?["outputPath"]?.stringValue)
      #expect(path.hasPrefix(directory.path + "/compressor-"))
      #expect(!FileManager.default.fileExists(atPath: path))
    }
  }

  @Test(arguments: ["cmprstng", "setting", "compressorsetting"])
  func boundedSubmissionSupportsTrustedCustomAndApplePresets(_ suffix: String) async throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "compressor-range-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let source = directory.appendingPathComponent("source.mov")
    let preset = directory.appendingPathComponent("trusted.\(suffix)")
    try Data().write(to: source)
    try Data().write(to: preset)
    var native = interfaces()
    native.command = { _, arguments in
      #expect(
        Array(arguments.dropFirst(4).prefix(4)) == ["-in", "00:00:00;00", "-out", "00:00:05;00"])
      return CommandResult(stdout: "synthetic-job", status: 0)
    }
    let result = await NativeService(interfaces: native).call(
      .init(
        name: "compressor_submit",
        arguments: [
          "sourcePath": .string(source.path), "presetPath": .string(preset.path),
          "outputDirectory": .string(directory.path), "outputName": .string("out.mp4"),
          "batchName": .string("synthetic"),
          "range": .object(["startSeconds": .int(0), "durationSeconds": .int(5)]),
        ]))
    #expect(result.isError == false)
    #expect(result.structuredContent?.objectValue?["completionVerified"] == .bool(false))
  }

  @Test(arguments: ["failed", "thrown", "cancelled", "large"])
  func reportsNativeOutcomesWithoutClaimingCompletion(_ mode: String) async throws {
    var native = interfaces()
    native.command = { _, _ in
      switch mode {
      case "failed": return CommandResult(stdout: "failure", status: 1)
      case "thrown": throw ProAppsError.timedOut
      case "cancelled": throw CancellationError()
      default: return CommandResult(stdout: String(repeating: "x", count: 20_000), status: 0)
      }
    }
    let result = await NativeService(interfaces: native).call(
      .init(name: "compressor_status", arguments: ["id": .string("test-id")]))
    #expect(result.isError == (mode != "large"))
    #expect(result.structuredContent?.objectValue?["retrySafe"] == .bool(false))
    if mode == "cancelled" {
      #expect(result.structuredContent?.objectValue?["cancelled"] == .bool(true))
    }
    if mode == "large" {
      #expect(result.structuredContent?.objectValue?["truncated"] == .bool(true))
      #expect(result.structuredContent?.objectValue?["stdout"]?.stringValue?.count == 16000)
    }
  }

  @Test(.timeLimit(.minutes(1)))
  func mutationAdmissionAndCancellationAreExplicit() async throws {
    let entered = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
    defer { entered.continuation.finish() }
    var native = interfaces()
    native.command = { _, _ in
      entered.continuation.yield(())
      try await Task.sleep(for: .seconds(30))
      return CommandResult(stdout: "unexpected", status: 0)
    }
    let service = NativeService(interfaces: native)
    let task = Task {
      await service.call(
        .init(
          name: "compressor_control",
          arguments: ["id": .string("test-id"), "action": .string("pause")]))
    }
    defer { task.cancel() }
    var iterator = entered.stream.makeAsyncIterator()
    try #require(await iterator.next() != nil)
    let refused = await service.call(
      .init(
        name: "osc_send",
        arguments: ["port": .int(9000), "path": .string("/synthetic"), "value": .int(1)]))
    #expect(refused.isError == true)
    task.cancel()
    let cancelled = await task.value
    #expect(cancelled.structuredContent?.objectValue?["cancelled"] == .bool(true))
    let next = await service.call(
      .init(
        name: "osc_send",
        arguments: ["port": .int(9000), "path": .string("/synthetic"), "value": .int(1)]))
    #expect(next.isError == false)
  }

  @Test func serializationFailureIsBounded() async {
    let result = await NativeService(interfaces: interfaces()).response(["bad": .double(.infinity)])
    #expect(result.isError == true)
  }
}
