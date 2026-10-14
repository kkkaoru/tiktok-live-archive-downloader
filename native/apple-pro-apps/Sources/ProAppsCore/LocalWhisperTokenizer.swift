import Dispatch
import Foundation
import NaturalLanguage
import WhisperKit

/// Explicitly local tokenizer. Unlike the upstream convenience loader, a local
/// error never falls back to downloading from the Hub.
struct LocalWhisperTokenizer: WhisperTokenizer, Sendable {
  private let tokenizer: TokenizerWrapper
  let specialTokens: SpecialTokens
  let allLanguageTokens: Set<Int>

  static func load(directory: String) async throws -> Self {
    try Task.checkCancellation()
    let root = try await LocalTokenizerFiles().validate(directory: directory)
    // Audited local-folder overload: reads local JSON; no remote/pretrained call.
    let wrapped = try await AutoTokenizerWrapper.from(modelFolder: root, strict: true)
    return try Self(tokenizer: wrapped)
  }

  private init(tokenizer: TokenizerWrapper) throws {
    self.tokenizer = tokenizer
    func required(_ token: String) throws -> Int {
      guard let identifier = Self.identifier(token, in: tokenizer) else {
        throw ProAppsError.invalid("Local Whisper vocabulary is missing a required special token")
      }
      return identifier
    }
    let whitespace = tokenizer.encode(text: " ", addSpecialTokens: false)
    guard whitespace.count == 1, let whitespaceToken = whitespace.first else {
      throw ProAppsError.invalid("Local Whisper vocabulary has unsupported whitespace encoding")
    }
    specialTokens = try SpecialTokens(
      endToken: required("<|endoftext|>"), englishToken: required("<|en|>"),
      noSpeechToken: required("<|nospeech|>"), noTimestampsToken: required("<|notimestamps|>"),
      specialTokenBegin: required("<|endoftext|>"),
      startOfPreviousToken: required("<|startofprev|>"),
      startOfTranscriptToken: required("<|startoftranscript|>"),
      timeTokenBegin: required("<|0.00|>"),
      transcribeToken: required("<|transcribe|>"), translateToken: required("<|translate|>"),
      whitespaceToken: whitespaceToken)
    // A vocabulary may omit languages; omission here does not invent token IDs.
    allLanguageTokens = Set(
      Constants.languages.values.compactMap { Self.identifier("<|\($0)|>", in: tokenizer) })
    _ = try required("<|ja|>")
  }

  func encode(text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }
  func decode(tokens: [Int]) -> String { tokenizer.decode(tokens: tokens) }
  func convertTokenToId(_ token: String) -> Int? { Self.identifier(token, in: tokenizer) }

  private static func identifier(_ token: String, in tokenizer: TokenizerWrapper) -> Int? {
    guard let identifier = tokenizer.convertTokenToId(token),
      tokenizer.convertIdToToken(identifier) == token
    else { return nil }
    return identifier
  }
  func convertIdToToken(_ id: Int) -> String? { tokenizer.convertIdToToken(id) }

  // Align complete lexical groups, not individual Japanese character pieces.
  // Post-alignment display grouping cannot repair zero-length character clocks.
  func splitToWordTokens(tokenIds: [Int]) -> (words: [String], wordTokens: [[Int]]) {
    WhisperUnicodeWords.lexical(tokenIds, decode: { tokenizer.decode(tokens: $0) })
  }
}

private actor LocalTokenizerFiles {
  nonisolated private let executor = DispatchSerialQueue(
    label: "apple-pro-apps.whisper-tokenizer-files")
  nonisolated var unownedExecutor: UnownedSerialExecutor { executor.asUnownedSerialExecutor() }

  func validate(directory: String) throws -> URL {
    let root = try Files.absolute(directory).resolvingSymlinksInPath()
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
      let file = try Files.existing(root.appendingPathComponent(name).path, extensions: ["json"])
      guard file.deletingLastPathComponent() == root else {
        throw ProAppsError.invalid("Tokenizer files must remain inside their explicit directory")
      }
      _ = try Files.read(file)
    }
    return root
  }
}

enum WhisperUnicodeWords {
  /// Merge Unicode-complete token groups before Whisper computes word alignment.
  /// Never split a model token or assign/interpolate any time values here.
  static func lexical(_ tokens: [Int], decode: ([Int]) -> String)
    -> (words: [String], wordTokens: [[Int]])
  {
    let raw = split(tokens, decode: decode)
    // Whisper markers are alignment delimiters, not natural-language context.
    // Mask only for boundary discovery; original words and token IDs survive.
    let text = raw.words.map { word in
      if word.hasPrefix("<|"), word.hasSuffix("|>") {
        return String(repeating: " ", count: word.count)
      }
      return word
    }.joined()
    let tokenizer = NLTokenizer(unit: .word)
    tokenizer.setLanguage(.japanese)
    tokenizer.string = text
    let interiors = Set(
      tokenizer.tokens(for: text.startIndex..<text.endIndex).flatMap { range in
        let start = text.distance(from: text.startIndex, to: range.lowerBound)
        let end = text.distance(from: text.startIndex, to: range.upperBound)
        return Array((start + 1)..<end)
      })
    var words: [String] = []
    var groups: [[Int]] = []
    var pendingText = ""
    var pendingTokens: [Int] = []
    var consumed = 0
    for (word, identifiers) in zip(raw.words, raw.wordTokens) {
      pendingText += word
      pendingTokens += identifiers
      consumed += word.count
      if !interiors.contains(consumed) {
        words.append(pendingText)
        groups.append(pendingTokens)
        pendingText = ""
        pendingTokens = []
      }
    }
    return (words, groups)
  }

  static func split(_ tokens: [Int], decode: ([Int]) -> String)
    -> (words: [String], wordTokens: [[Int]])
  {
    var words: [String] = []
    var groups: [[Int]] = []
    var pending: [Int] = []
    for token in tokens {
      pending.append(token)
      let decoded = decode(pending)
      if !decoded.contains("\u{fffd}") {
        words.append(decoded)
        groups.append(pending)
        pending = []
      }
    }
    if !pending.isEmpty {
      // Keep malformed/incomplete final bytes visible in raw diagnostic output.
      words.append(decode(pending))
      groups.append(pending)
    }
    return (words, groups)
  }
}
