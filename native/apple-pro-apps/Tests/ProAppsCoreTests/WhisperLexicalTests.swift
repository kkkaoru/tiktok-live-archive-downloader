import Testing

@testable import ProAppsCore

struct WhisperLexicalTests {
  @Test func completeJapaneseWordsAreGroupedBeforeAlignment() {
    let result = WhisperUnicodeWords.lexical([0, 1, 2]) { identifiers in
      identifiers.map { ["東", "京", "。"][$0] }.joined()
    }
    #expect(result.words == ["東京", "。"])
    #expect(result.wordTokens == [[0, 1], [2]])
  }

  @Test func aCoarseModelTokenIsNeverSplitIntoFabricatedWordTokens() {
    let result = WhisperUnicodeWords.lexical([0]) { _ in "今日は" }
    #expect(result.words == ["今日は"])
    #expect(result.wordTokens == [[0]])
  }

  @Test func specialTokensAndWhitespaceRemainSeparateFromLexicalText() {
    let result = WhisperUnicodeWords.lexical([0, 1, 2, 3, 4]) { identifiers in
      identifiers.map { ["<|0.00|>", "東", "京", " ", "<|endoftext|>"][$0] }.joined()
    }
    #expect(result.words == ["<|0.00|>", "東京", " ", "<|endoftext|>"])
    #expect(result.wordTokens == [[0], [1, 2], [3], [4]])
  }

  @Test func emptyInputAndIncompleteUnicodeArePreserved() {
    #expect(WhisperUnicodeWords.lexical([], decode: { _ in "" }).words == [])
    let incomplete = WhisperUnicodeWords.lexical([9]) { _ in "\u{fffd}" }
    #expect(incomplete.words == ["\u{fffd}"])
    #expect(incomplete.wordTokens == [[9]])
  }
}
