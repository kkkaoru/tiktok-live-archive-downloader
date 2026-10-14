import Foundation
import Speech
import Testing

@testable import ProAppsCore

// Native Speech/TTS use the installed Japanese models and shared system services.
@Suite(.serialized)
struct SpeechProbeTests {
  private func directory() throws -> URL {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "speech-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    return root
  }

  @Test func recognizesSyntheticJapaneseFileWithoutPlaybackOrCloud() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = root.appendingPathComponent("synthetic.aiff")
    // The installed voice inventory was checked first. -o writes a file, not audio output.
    let generated = try await Runner.run(
      URL(fileURLWithPath: "/usr/bin/say"), ["-v", "Kyoko", "-o", source.path, "今日は動画の編集を確認します。"])
    #expect(generated.status == 0)
    let before = try Data(contentsOf: source)
    // User-approved app-scoped setup. Keep this test application's Japanese
    // reservation; do not release it underneath parallel native/CLI tests.
    let prepared = try await SpeechProbe().reserve(locale: "ja-JP")
    try #require(prepared.readyForTranscription)
    #expect(prepared.locale == "ja_JP")
    #expect(try await SpeechProbe().reserve(locale: "ja-JP").newlyReserved == false)
    await #expect(throws: ProAppsError.self) { try await SpeechProbe().reserve(locale: "zz-ZZ") }
    let transcript = try await SpeechProbe().transcribe(path: source.path, locale: "ja-JP")
    #expect(transcript.locale == "ja_JP")
    #expect(transcript.onDevice)
    #expect(!transcript.humanReviewed)
    #expect(transcript.segments.map(\.text).joined().contains("動画"))
    #expect(
      transcript.segments.allSatisfy {
        $0.startSeconds >= 0 && $0.durationSeconds > 0
          && $0.startSeconds + $0.durationSeconds <= transcript.durationSeconds
      })
    let timed = try await SpeechProbe().transcribe(
      path: source.path, locale: "ja-JP",
      options: SpeechRecognitionOptions(wordTiming: true, contextualStrings: ["動画", "編集"]))
    #expect(timed.segments.count > 1)
    #expect(timed.segments.map(\.text).joined().contains("動画"))
    #expect(timed.segments.allSatisfy { $0.durationSeconds > 0 })
    #expect(try Data(contentsOf: source) == before)
    await #expect(throws: ProAppsError.self) {
      try await SpeechProbe().transcribe(path: source.path, locale: "zz-ZZ")
    }
  }

  @Test func longAudioAndVideoContainersAreRefusedBeforeRecognition() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    // Literal mono PCM16/16kHz header; exactly 61 seconds of silence.
    let header: [UInt8] = [
      82, 73, 70, 70, 36, 201, 29, 0, 87, 65, 86, 69, 102, 109, 116, 32, 16, 0, 0, 0, 1, 0, 1, 0,
      128, 62, 0, 0, 0, 125, 0, 0, 2, 0, 16, 0, 100, 97, 116, 97, 0, 201, 29, 0,
    ]
    let long = root.appendingPathComponent("long.wav")
    try (Data(header) + Data(repeating: 0, count: 1_952_000)).write(
      to: long, options: .withoutOverwriting)
    await #expect(throws: ProAppsError.self) {
      try await SpeechProbe().transcribe(path: long.path, locale: "ja-JP")
    }
    let video = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    await #expect(throws: ProAppsError.self) {
      try await SpeechProbe().transcribe(path: video.path, locale: "ja-JP")
    }
  }

  @Test func unreservedLocaleIsRejectedWithoutChangingReservations() async throws {
    guard #available(macOS 26.0, *) else {
      Issue.record("Native Speech integration requires macOS 26")
      return
    }
    let reserved = await AssetInventory.reservedLocales
    let supported = await SpeechTranscriber.supportedLocales
    // Japanese is the only approved/prepared test locale. Select a supported
    // unreserved alternative without downloading, reserving or releasing it.
    let locale = try #require(
      supported.first { !reserved.contains($0) && $0.language.languageCode?.identifier != "ja" })
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let audio = root.appendingPathComponent("silence.wav")
    let header: [UInt8] = [
      82, 73, 70, 70, 36, 125, 0, 0, 87, 65, 86, 69, 102, 109, 116, 32, 16, 0, 0, 0, 1, 0, 1, 0,
      128, 62, 0, 0, 0, 125, 0, 0, 2, 0, 16, 0, 100, 97, 116, 97, 0, 125, 0, 0,
    ]
    try (Data(header) + Data(repeating: 0, count: 32000)).write(
      to: audio, options: .withoutOverwriting)
    do {
      _ = try await SpeechProbe().transcribe(path: audio.path, locale: locale.identifier)
      Issue.record("Read-only transcription accepted an unprepared locale")
    } catch ProAppsError.unavailable(let reason) {
      #expect(reason.contains("explicitly reserve the locale first"))
    }
    #expect(await AssetInventory.reservedLocales.contains(locale) == false)
  }

  @Test func nativeSessionFailureClosesItsResultStream() async throws {
    guard #available(macOS 26.0, *) else {
      Issue.record("Native Speech integration requires macOS 26")
      return
    }
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let work = NativeSpeechWork(
      source: root.appendingPathComponent("missing.wav"),
      transcriber: SpeechTranscriber(locale: Locale(identifier: "ja_JP"), preset: .transcription),
      audioDuration: 1)
    await #expect(throws: (any Error).self) { try await SpeechRun.collect(work) }
  }
}
