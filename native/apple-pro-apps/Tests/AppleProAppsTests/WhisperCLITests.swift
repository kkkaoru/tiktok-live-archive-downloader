import Foundation
import ProAppsCore
import Testing

struct WhisperCLITests {
  @Test func actualMeasurementChildEmitsOnlyDecodableJSON() async throws {
    let model = try #require(ProcessInfo.processInfo.environment["WHISPER_TEST_MODEL"])
    let tokenizer = try #require(ProcessInfo.processInfo.environment["WHISPER_TEST_TOKENIZER"])
    let root = try Files.temporaryDirectory().appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let audio = root.appendingPathComponent("synthetic.aiff")
    let generated = try await Runner.run(
      URL(fileURLWithPath: "/usr/bin/say"), ["-v", "Kyoko", "-o", audio.path, "今日は動画の編集を確認します。"])
    try #require(generated.status == 0)
    let before = try Data(contentsOf: audio)
    let request = root.appendingPathComponent("request.json")
    try JSONEncoder().encode(
      WhisperProbe.Request(
        path: audio.path, modelDirectory: model, tokenizerDirectory: tokenizer)
    ).write(to: request)
    let products = Bundle(for: TestBundle.self).bundleURL.deletingLastPathComponent()
    let binary = products.appendingPathComponent("apple-pro-apps")
    let result = try await Runner.run(
      binary, ["measure-media", "audio_transcribe_whisper", request.path], timeout: .seconds(180))
    try #require(result.status == 0, "Synthetic child exit status: \(result.status)")
    let transcript = try JSONDecoder().decode(SpeechTranscript.self, from: Data(result.stdout.utf8))
    #expect(transcript.onDevice && !transcript.humanReviewed)
    #expect(transcript.segments.count > 1)
    #expect(try Data(contentsOf: audio) == before)
  }
}
