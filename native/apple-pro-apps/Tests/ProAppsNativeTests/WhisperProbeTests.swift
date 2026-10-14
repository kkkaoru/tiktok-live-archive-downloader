import Foundation
import Testing

@testable import ProAppsCore

// Core ML model specialization and the installed Japanese TTS service are shared.
@Suite(.serialized)
struct WhisperProbeTests {
  @Test func localWhisperRecognizesSyntheticJapaneseAndPreservesTheInput() async throws {
    let model = try #require(ProcessInfo.processInfo.environment["WHISPER_TEST_MODEL"])
    let tokenizer = try #require(ProcessInfo.processInfo.environment["WHISPER_TEST_TOKENIZER"])
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "whisper-native-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let audio = root.appendingPathComponent("synthetic.aiff")
    let generated = try await Runner.run(
      URL(fileURLWithPath: "/usr/bin/say"), ["-v", "Kyoko", "-o", audio.path, "今日は動画の編集を確認します。"])
    try #require(generated.status == 0)
    let original = try Data(contentsOf: audio)
    let result = try await WhisperProbe.transcribe(
      .init(path: audio.path, modelDirectory: model, tokenizerDirectory: tokenizer))
    #expect(result.onDevice)
    #expect(!result.humanReviewed)
    #expect(result.segments.map(\.text).joined().contains("動画"))
    #expect(result.segments.count > 1)
    #expect(try Data(contentsOf: audio) == original)
    await #expect(throws: (any Error).self) {
      try await WhisperProbe.transcribe(
        .init(path: audio.path, modelDirectory: root.path, tokenizerDirectory: tokenizer))
    }
    for name in ["AudioEncoder.mlmodelc", "MelSpectrogram.mlmodelc", "TextDecoder.mlmodelc"] {
      try FileManager.default.createDirectory(
        at: root.appendingPathComponent(name), withIntermediateDirectories: false)
    }
    // Directory validation succeeds, then model loading must fail and unload.
    await #expect(throws: (any Error).self) {
      try await WhisperProbe.transcribe(
        .init(path: audio.path, modelDirectory: root.path, tokenizerDirectory: tokenizer))
    }
    let header: [UInt8] = [
      82, 73, 70, 70, 36, 201, 29, 0, 87, 65, 86, 69, 102, 109, 116, 32, 16, 0, 0, 0, 1, 0, 1, 0,
      128, 62, 0, 0, 0, 125, 0, 0, 2, 0, 16, 0, 100, 97, 116, 97, 0, 201, 29, 0,
    ]
    let long = root.appendingPathComponent("long.wav")
    try (Data(header) + Data(repeating: 0, count: 1_952_000)).write(
      to: long, options: .withoutOverwriting)
    await #expect(throws: ProAppsError.self) {
      try await WhisperProbe.transcribe(
        .init(path: long.path, modelDirectory: model, tokenizerDirectory: tokenizer))
    }
  }

  @Test func cancellationAndInvalidSourcesFailBeforeModelAccess() async throws {
    let request = WhisperProbe.Request(
      path: "/missing.wav", modelDirectory: "/missing-model",
      tokenizerDirectory: "/missing-tokenizer")
    await #expect(throws: (any Error).self) { try await WhisperProbe.transcribe(request) }
    await withTaskGroup(of: Void.self) { group in
      group.cancelAll()
      group.addTask {
        await #expect(throws: CancellationError.self) { try await WhisperProbe.transcribe(request) }
      }
    }
  }
}
