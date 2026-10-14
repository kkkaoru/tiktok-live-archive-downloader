import Foundation
import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

struct MeasurementServiceTests {
  private actor Capture {
    var path: String?
    func record(_ path: String) { self.path = path }
  }

  @Test func speechPreparationIsAMutationAndRecognitionRemainsReadOnly() throws {
    let preparation = try #require(ToolSpec.all.first { $0.name == "speech_locale_reserve" })
    let recognition = try #require(ToolSpec.all.first { $0.name == "audio_transcribe" })
    #expect(!preparation.readOnly)
    #expect(recognition.readOnly)
  }

  @Test(arguments: ["success", "failure", "cancelled"])
  func requestFilesArePrivateAndRemovedOnAllOutcomes(_ mode: String) async throws {
    let capture = Capture()
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      await capture.record(path)
      #expect(name == "media_verify_video")
      let value = try JSONDecoder().decode(Value.self, from: Files.read(URL(fileURLWithPath: path)))
      #expect(value.objectValue?["path"] == .string("/tmp/synthetic.mp4"))
      #expect(value.objectValue?["maximumDurationSeconds"] == .int(60))
      #expect(value.objectValue?["maximumFrames"] == .int(7200))
      if mode == "failure" { throw ProAppsError.unavailable("Synthetic failure") }
      if mode == "cancelled" { throw CancellationError() }
      return .object(["fullVideoDecoded": .bool(true)])
    }
    let result = await NativeService(interfaces: interfaces).call(
      .init(
        name: "media_verify_video",
        arguments: [
          "path": .string("/tmp/synthetic.mp4"), "maximumDurationSeconds": .int(60),
          "maximumFrames": .int(7200),
        ]))
    #expect(result.isError == (mode != "success"))
    let path = try #require(await capture.path)
    #expect(
      !FileManager.default.fileExists(
        atPath: URL(fileURLWithPath: path).deletingLastPathComponent().path))
    if mode == "success" {
      #expect(result.structuredContent?.objectValue?["sourceModified"] == .bool(false))
      #expect(
        result.structuredContent?.objectValue?["measurement"]?.objectValue?["fullVideoDecoded"]
          == .bool(true))
    }
  }

  @Test(arguments: [
    "media_verify_video", "audio_measure", "video_frame_measure", "audio_transcribe",
    "speech_locale_reserve",
  ])
  func nativeMeasurementCLIUsesTheSameStrictSchema(_ name: String) async throws {
    let request = try Files.reserveOutput(
      directory: Files.temporaryDirectory().path, name: "request.json", kind: .edit)
    let root = request.deletingLastPathComponent()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let video = package.appendingPathComponent("Tests/ProAppsNativeTests/Fixtures/black.mp4")
    var arguments: [String: Value] = ["path": .string(video.path)]
    if name == "audio_measure" || name == "audio_transcribe" {
      // One second of silent mono PCM16 at 16000 Hz; literal RIFF header.
      let header: [UInt8] = [
        82, 73, 70, 70, 36, 125, 0, 0, 87, 65, 86, 69, 102, 109, 116, 32, 16, 0, 0, 0, 1, 0, 1, 0,
        128, 62, 0, 0, 0, 125, 0, 0, 2, 0, 16, 0, 100, 97, 116, 97, 0, 125, 0, 0,
      ]
      let audio = root.appendingPathComponent("silence.wav")
      try (Data(header) + Data(repeating: 0, count: 32000)).write(
        to: audio, options: .withoutOverwriting)
      arguments = ["path": .string(audio.path)]
      if name == "audio_transcribe" {
        arguments["locale"] = .string("ja-JP")
      } else {
        arguments["windows"] = .array([])
      }
    } else if name == "speech_locale_reserve" {
      arguments = ["locale": .string("ja-JP")]
    } else if name == "video_frame_measure" {
      arguments["samples"] = .array([.object(["timeSeconds": .int(0)])])
    }
    _ = try Files.writeNew(
      JSONEncoder().encode(Value.object(arguments)), to: request.path, extensions: ["json"])
    let service = NativeService()
    if name == "audio_transcribe" || name == "speech_locale_reserve" {
      try #require(try await SpeechProbe().reserve(locale: "ja-JP").readyForTranscription)
    }
    let text = try await service.measureLocalFile(name: name, path: request.path)
    let result = try JSONDecoder().decode(Value.self, from: Data(text.utf8))
    if name == "media_verify_video" {
      #expect(result.objectValue?["decodedFrames"] == .int(1))
      #expect(result.objectValue?["fullVideoDecoded"] == .bool(true))
      let extended = try Files.writeNew(
        JSONEncoder().encode(
          Value.object([
            "path": .string(video.path), "maximumDurationSeconds": .int(60),
            "maximumFrames": .int(7200),
          ])), to: root.appendingPathComponent("extended.json").path, extensions: ["json"])
      let extendedText = try await service.measureLocalFile(name: name, path: extended.path)
      #expect(try JSONDecoder().decode(Value.self, from: Data(extendedText.utf8)) == result)
    } else if name == "audio_measure" {
      #expect(result.objectValue?["whole"]?.objectValue?["frames"] == .int(16000))
      var extendedArguments = arguments
      extendedArguments["maximumDurationSeconds"] = .int(60)
      let extended = try Files.writeNew(
        JSONEncoder().encode(Value.object(extendedArguments)),
        to: root.appendingPathComponent("extended-audio.json").path, extensions: ["json"])
      let extendedText = try await service.measureLocalFile(name: name, path: extended.path)
      #expect(try JSONDecoder().decode(Value.self, from: Data(extendedText.utf8)) == result)
    } else if name == "audio_transcribe" {
      #expect(result.objectValue?["onDevice"] == .bool(true))
      #expect(result.objectValue?["humanReviewed"] == .bool(false))
      #expect(result.objectValue?["segments"]?.arrayValue?.isEmpty == true)
      var timedArguments = arguments
      timedArguments["wordTiming"] = .bool(true)
      timedArguments["contextualStrings"] = .array([.string("編集")])
      let timedPath = try Files.writeNew(
        JSONEncoder().encode(Value.object(timedArguments)),
        to: root.appendingPathComponent("timed-speech.json").path, extensions: ["json"])
      let timedText = try await service.measureLocalFile(name: name, path: timedPath.path)
      #expect(try JSONDecoder().decode(Value.self, from: Data(timedText.utf8)) == result)
    } else if name == "speech_locale_reserve" {
      #expect(result.objectValue?["locale"] == .string("ja_JP"))
      #expect(result.objectValue?["readyForTranscription"] == .bool(true))
      #expect(result.objectValue?["newlyReserved"] == .bool(false))
    } else {
      #expect(result.arrayValue?.count == 1)
    }
    let products = Bundle(for: TestBundle.self).bundleURL.deletingLastPathComponent()
    let binary = products.appendingPathComponent("apple-pro-apps")
    var environment = ProcessInfo.processInfo.environment
    environment["LLVM_PROFILE_FILE"] =
      products.appendingPathComponent("codecov/measurement-%p-%m.profraw").path
    if name == "audio_transcribe" || name == "speech_locale_reserve" {
      let prepare = try Files.writeNew(
        JSONEncoder().encode(Value.object(["locale": .string("ja-JP")])),
        to: root.appendingPathComponent("prepare.json").path, extensions: ["json"])
      let prepared = try await Runner.run(
        binary, ["measure-media", "speech_locale_reserve", prepare.path], environment: environment)
      try #require(prepared.status == 0)
      let readiness = try JSONDecoder().decode(Value.self, from: Data(prepared.stdout.utf8))
      try #require(readiness.objectValue?["readyForTranscription"] == .bool(true))
    }
    let process = try await Runner.run(
      binary, ["measure-media", name, request.path], environment: environment, timeout: .seconds(15)
    )
    #expect(process.status == 0)
    #expect(try JSONDecoder().decode(Value.self, from: Data(process.stdout.utf8)) == result)
    arguments["shell"] = .string("refused")
    let bad = try Files.writeNew(
      JSONEncoder().encode(Value.object(arguments)),
      to: root.appendingPathComponent("bad.json").path, extensions: ["json"])
    await #expect(throws: (any Error).self) {
      try await service.measureLocalFile(name: name, path: bad.path)
    }
  }

  @Test func cueTrackUsesTheNativeCLIAndCreatesOnlyNewOutputs() async throws {
    let request = try Files.reserveOutput(
      directory: Files.temporaryDirectory().path, name: "cue-input.json", kind: .edit)
    let root = request.deletingLastPathComponent()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let arguments: [String: Value] = [
      "outputDirectory": .string(root.path), "outputName": .string("cue.wav"),
      "track": .object([
        "durationSeconds": .int(1), "onsetSeconds": .array([.double(0.25)]), "gain": .double(0.12),
      ]),
    ]
    // XCTest's Bundle.main is the test runner, not the shipped command. Inject
    // the compiled CLI boundary while retaining real serialization and dispatch.
    let products = Bundle(for: TestBundle.self).bundleURL.deletingLastPathComponent()
    let binary = products.appendingPathComponent("apple-pro-apps")
    var environment = ProcessInfo.processInfo.environment
    environment["LLVM_PROFILE_FILE"] =
      products.appendingPathComponent("codecov/cue-%p-%m.profraw").path
    let childEnvironment = environment
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      let process = try await Runner.run(
        binary, ["measure-media", name, path], environment: childEnvironment)
      guard process.status == 0 else { throw ProAppsError.commandFailed(process.status) }
      return try JSONDecoder().decode(Value.self, from: Data(process.stdout.utf8))
    }
    let result = await NativeService(interfaces: interfaces).call(
      .init(name: "audio_cue_track", arguments: arguments))
    try #require(result.isError != true)
    let output = try #require(
      result.structuredContent?.objectValue?["measurement"]?.objectValue?["outputPath"]?.stringValue
    )
    let measured = try PCMMeasurement.analyze(Files.read(URL(fileURLWithPath: output)), windows: [])
    #expect(measured.whole.frames == 16000)
    #expect(measured.whole.rms > 0)
    #expect(output.hasPrefix(root.path + "/edit-"))
    let spec = try #require(ToolSpec.all.first { $0.name == "audio_cue_track" })
    #expect(!spec.readOnly)
    var bad = arguments
    bad["outputName"] = .string("bad.mp4")
    _ = try Files.writeNew(
      JSONEncoder().encode(Value.object(bad)), to: request.path, extensions: ["json"])
    await #expect(throws: ProAppsError.self) {
      try await NativeService().measureLocalFile(name: "audio_cue_track", path: request.path)
    }
  }

  @Test func cuePublicationFailureRemovesItsDirectoryAndRetainsTheError() async throws {
    let request = try Files.reserveOutput(
      directory: Files.temporaryDirectory().path, name: "cue-input.json", kind: .edit)
    let root = request.deletingLastPathComponent()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let arguments: Value = .object([
      "outputDirectory": .string(root.path), "outputName": .string("cue.wav"),
      "track": .object([
        "durationSeconds": .int(1), "onsetSeconds": .array([.double(0.25)]), "gain": .double(0.12),
      ]),
    ])
    _ = try Files.writeNew(JSONEncoder().encode(arguments), to: request.path, extensions: ["json"])
    var interfaces = NativeInterfaces()
    interfaces.writeCueArtifact = { data, path in
      if URL(fileURLWithPath: path).pathExtension == "wav" { throw ProAppsError.commandFailed(42) }
      return try Files.writeNew(data, to: path, extensions: ["json"])
    }
    do {
      _ = try await NativeService(interfaces: interfaces).measureLocalFile(
        name: "audio_cue_track", path: request.path)
      Issue.record("Expected the simulated audio publication failure")
    } catch ProAppsError.commandFailed(let code) {
      #expect(code == 42)
    }
    #expect(try FileManager.default.contentsOfDirectory(atPath: root.path) == ["cue-input.json"])
  }

  @Test func malformedCLIAndUnknownOperationFailClosed() async {
    #expect(throws: (any Error).self) { try Command.parse(["measure-media"]) }
    await #expect(throws: (any Error).self) {
      try await NativeService().measureLocalFile(name: "unknown", path: "/tmp/none.json")
    }
  }
}
