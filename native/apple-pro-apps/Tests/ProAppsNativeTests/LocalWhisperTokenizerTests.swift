import Foundation
import Testing
import WhisperKit

@testable import ProAppsCore

// Uses only the explicitly installed, hash-verified public tokenizer JSON.
@Suite(.serialized)
struct LocalWhisperTokenizerTests {
  @Test func localJapaneseTokenizerRoundTripsWithoutHubFallback() async throws {
    let path = try #require(ProcessInfo.processInfo.environment["WHISPER_TEST_TOKENIZER"])
    let tokenizer = try await LocalWhisperTokenizer.load(directory: path)
    let encoded = tokenizer.encode(text: "今日は編集します。")
    #expect(tokenizer.decode(tokens: encoded) == "今日は編集します。")
    let japanese = try #require(tokenizer.convertTokenToId("<|ja|>"))
    #expect(tokenizer.allLanguageTokens.contains(japanese))
    #expect(tokenizer.convertIdToToken(japanese) == "<|ja|>")
    #expect(tokenizer.convertTokenToId("<|synthetic-not-a-token|>") == nil)
    #expect(tokenizer.convertIdToToken(-1) == nil)
    let split = tokenizer.splitToWordTokens(tokenIds: encoded)
    #expect(split.words.joined() == "今日は編集します。")
    #expect(split.wordTokens.flatMap { $0 } == encoded)
    let engine = try await OfflineWhisper(
      WhisperKitConfig(verbose: false, prewarm: false, load: false, download: false))
    await #expect(throws: ProAppsError.self) { try await engine.loadTokenizerIfNeeded() }
    engine.tokenizer = tokenizer
    try await engine.loadTokenizerIfNeeded()
  }

  @Test func missingLocalAssetsAndCancellationNeverDownload() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "whisper-local-\(UUID().uuidString)")
    await #expect(throws: (any Error).self) {
      try await LocalWhisperTokenizer.load(directory: root.path)
    }
    await withTaskGroup(of: Void.self) { group in
      group.cancelAll()
      group.addTask {
        await #expect(throws: CancellationError.self) {
          try await LocalWhisperTokenizer.load(directory: root.path)
        }
      }
    }
  }
}
