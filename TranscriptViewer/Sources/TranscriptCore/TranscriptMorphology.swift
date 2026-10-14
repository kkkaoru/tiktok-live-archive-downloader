import CryptoKit
import Foundation

/// Explicit morphology from the pinned Japanese tokenizer. Ranges count Swift Characters.
public struct TranscriptMorphology: Codable, Equatable, Sendable {
    public struct Dictionary: Codable, Equatable, Sendable {
        public let path: String
        public let sha256: String
    }
    public struct Token: Codable, Equatable, Sendable {
        public let text: String
        public let lowerCharacter: Int
        public let upperCharacter: Int
        public let partOfSpeech: String
        public let features: [String]
    }
    public struct Section: Codable, Equatable, Sendable {
        public let startSeconds: Double
        public let endSeconds: Double
        public let textSHA256: String
        public let tokens: [Token]
    }
    public let version: Int
    public let method: String
    public let transcriptFingerprint: String
    public let dictionary: Dictionary
    public let sections: [Section]

    public static let dictionarySHA256 =
        "f3ddd9cacbd6eff696e49bb9266824eed28dceaa89c55e75e4a13384cc7f9d02"
    private static let parts: Set<String> = [
        "名詞", "動詞", "形容詞", "助詞", "助動詞", "記号", "感動詞", "副詞", "接続詞", "連体詞", "接頭詞", "フィラー", "その他",
    ]

    public func validate(source: TranscriptSummary, fingerprint: String) throws {
        guard version == 1, method == "vibrato-ipadic-v1",
            transcriptFingerprint == fingerprint, dictionary.sha256 == Self.dictionarySHA256,
            !dictionary.path.isEmpty, sections.count == source.sections.count,
            !sections.isEmpty, sections.count <= 512
        else { throw SourceExcerptError.invalidEvidence }
        for (section, original) in zip(sections, source.sections) {
            guard section.startSeconds == original.startSeconds,
                section.endSeconds == original.endSeconds,
                section.textSHA256 == Self.textHash(original.text),
                !section.tokens.isEmpty, section.tokens.count <= 100_000
            else { throw SourceExcerptError.invalidEvidence }
            let characters = Array(original.text)
            var lower = 0
            for token in section.tokens {
                guard token.lowerCharacter == lower,
                    token.upperCharacter > lower, token.upperCharacter <= characters.count,
                    token.text == String(characters[lower..<token.upperCharacter]),
                    token.features.count >= 2, token.features.count <= 16,
                    token.features.first == token.partOfSpeech,
                    Self.parts.contains(token.partOfSpeech),
                    token.features.allSatisfy({ $0.utf8.count <= 16384 })
                else { throw SourceExcerptError.invalidEvidence }
                lower = token.upperCharacter
            }
            guard lower == characters.count else { throw SourceExcerptError.invalidEvidence }
        }
    }

    public static func textHash(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    public func fingerprint() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return SHA256.hash(data: try encoder.encode(self)).map { String(format: "%02x", $0) }
            .joined()
    }
}
