import Foundation
import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

/// Exercise the real CLI decoding/dispatch boundary without starting inference or
/// reserving Speech assets. Cancellation must survive every native adapter.
struct MeasurementCancellationTests {
  @Test(arguments: [
    "caption_text_measure", "media_verify_video", "audio_transcribe_whisper",
    "speech_locale_reserve", "audio_transcribe", "audio_separate_vocals",
    "audio_sound_activity", "audio_reference_analyze", "audio_measure",
    "image_text_recognize_batch", "image_text_rectangles_batch", "image_text_rectangles",
    "video_text_recognize", "video_foreground_measure", "video_head_detect",
    "video_face_track", "video_frame_measure",
  ])
  func cancellationPropagatesThroughNativeDispatch(name: String) async throws {
    let request = try Files.reserveOutput(
      directory: Files.temporaryDirectory().path, name: "request.json", kind: .edit)
    let root = request.deletingLastPathComponent()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let missing = root.appendingPathComponent("unavailable.wav").path
    let arguments = arguments(name: name, missing: missing, root: root.path)
    _ = try Files.writeNew(JSONEncoder().encode(arguments), to: request.path, extensions: ["json"])
    await withTaskGroup(of: Void.self) { group in
      group.cancelAll()
      group.addTask {
        await #expect(throws: CancellationError.self) {
          try await NativeService().measureLocalFile(name: name, path: request.path)
        }
      }
    }
    #expect(try FileManager.default.contentsOfDirectory(atPath: root.path) == ["request.json"])
  }

  private func arguments(name: String, missing: String, root: String) -> Value {
    let sample: Value = .object(["timeSeconds": .int(0)])
    switch name {
    case "caption_text_measure":
      return .object([
        "fontName": .string("ArialMT"), "assFontSize": .int(66),
        "border": .int(0), "rim": .int(0), "texts": .array([.string("test")]),
      ])
    case "audio_transcribe_whisper":
      return .object([
        "path": .string(missing), "modelDirectory": .string(root),
        "tokenizerDirectory": .string(root),
      ])
    case "speech_locale_reserve":
      return .object(["locale": .string("ja-JP")])
    case "audio_transcribe":
      return .object(["path": .string(missing), "locale": .string("ja-JP")])
    case "audio_separate_vocals":
      return .object([
        "leftPath": .string(missing), "rightPath": .string(missing),
        "compiledModelPath": .string(root), "outputDirectory": .string(root),
        "outputName": .string("result.wav"),
      ])
    case "audio_reference_analyze":
      return .object([
        "sourcePath": .string(missing), "referencePath": .string(missing),
        "search": .object([
          "referenceStartSample": .int(0), "searchRadiusSamples": .int(0),
          "minimumCorrelation": .double(0.5),
        ]),
      ])
    case "audio_measure":
      return .object(["path": .string(missing), "windows": .array([])])
    case "image_text_recognize_batch", "image_text_rectangles_batch":
      return .object(["paths": .array([.string(missing)])])
    case "video_text_recognize", "video_head_detect", "video_frame_measure":
      return .object(["path": .string(missing), "samples": .array([sample])])
    case "video_foreground_measure":
      return .object([
        "path": .string(missing),
        "samples": .array([.object(["value": .int(0), "timescale": .int(30)])]),
      ])
    case "video_face_track":
      return .object([
        "path": .string(missing),
        "plan": .object([
          "startSeconds": .int(0), "durationSeconds": .int(1), "maximumFrames": .int(30),
          "redetectIntervalSeconds": .double(0.5),
        ]),
      ])
    default:
      return .object(["path": .string(missing)])
    }
  }

  @Test func localCuePublishesWaveAndRequestWithoutNativeSubprocess() async throws {
    let request = try Files.reserveOutput(
      directory: Files.temporaryDirectory().path, name: "request.json", kind: .edit)
    let root = request.deletingLastPathComponent()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let arguments: Value = .object([
      "outputDirectory": .string(root.path), "outputName": .string("cue.wav"),
      "track": .object([
        "durationSeconds": .int(1), "onsetSeconds": .array([.double(0.25)]), "gain": .double(0.1),
      ]),
    ])
    _ = try Files.writeNew(JSONEncoder().encode(arguments), to: request.path, extensions: ["json"])
    let output = try await NativeService().measureLocalFile(
      name: "audio_cue_track", path: request.path)
    let result = try JSONDecoder().decode(Value.self, from: Data(output.utf8))
    let wave = try #require(result.objectValue?["outputPath"]?.stringValue)
    let project = try #require(result.objectValue?["projectPath"]?.stringValue)
    let measured = try PCMMeasurement.analyze(Files.read(URL(fileURLWithPath: wave)), windows: [])
    #expect(measured.whole.frames == 16000)
    #expect(measured.whole.rms > 0)
    #expect(
      try JSONDecoder().decode(CueSound.self, from: Files.read(URL(fileURLWithPath: project))).gain
        == 0.1)
  }
}
