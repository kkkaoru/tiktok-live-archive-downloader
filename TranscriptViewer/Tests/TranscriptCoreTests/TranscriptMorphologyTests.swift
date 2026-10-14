import Foundation
import Testing

@testable import TranscriptCore

struct TranscriptMorphologyTests {
    private let text = "か\u{3099}👩‍👩‍👧‍👦\n京都。"

    private func source() -> TranscriptSummary {
        .init(
            cues: [
                .init(
                    id: 0, startSeconds: 600, endSeconds: 610, startLabel: "", endLabel: "",
                    text: text)
            ], inputKind: .tokenSequence)
    }

    private func tokens() -> [TranscriptMorphology.Token] {
        Array(text).enumerated().map { offset, character in
            .init(
                text: String(character), lowerCharacter: offset, upperCharacter: offset + 1,
                partOfSpeech: "名詞", features: ["名詞", "一般"])
        }
    }

    private func evidence(
        version: Int = 1, method: String = "vibrato-ipadic-v1", fingerprint: String = "p",
        dictionary: TranscriptMorphology.Dictionary? = nil,
        sections: [TranscriptMorphology.Section]? = nil
    ) -> TranscriptMorphology {
        .init(
            version: version, method: method, transcriptFingerprint: fingerprint,
            dictionary: dictionary
                ?? .init(
                    path: "/synthetic/system.dic", sha256: TranscriptMorphology.dictionarySHA256),
            sections: sections ?? [
                .init(
                    startSeconds: 600, endSeconds: 1200,
                    textSHA256: TranscriptMorphology.textHash(text), tokens: tokens())
            ])
    }

    @Test func graphemesAndSourceClockHaveExactCoverageAndStableFingerprint() throws {
        let valid = evidence()
        try valid.validate(source: source(), fingerprint: "p")
        #expect(valid.sections[0].tokens[0].text.count == 1)
        #expect(valid.sections[0].tokens[1].text.count == 1)
        #expect(valid.sections[0].tokens[1].text.utf16.count > 1)
        #expect(try valid.fingerprint().count == 64)
        let restored = try JSONDecoder().decode(
            TranscriptMorphology.self, from: JSONEncoder().encode(valid))
        #expect(restored == valid)
        #expect(try restored.fingerprint() == valid.fingerprint())
    }

    @Test func metadataSourceHashClockAndDictionaryMismatchesAreRejected() {
        let invalid: [TranscriptMorphology] = [
            evidence(version: 2), evidence(method: "guessed-pos"), evidence(fingerprint: "wrong"),
            evidence(dictionary: .init(path: "", sha256: TranscriptMorphology.dictionarySHA256)),
            evidence(dictionary: .init(path: "/synthetic/system.dic", sha256: "wrong")),
            evidence(sections: []),
            evidence(sections: [
                .init(
                    startSeconds: 601, endSeconds: 1200,
                    textSHA256: TranscriptMorphology.textHash(text), tokens: tokens())
            ]),
            evidence(sections: [
                .init(
                    startSeconds: 600, endSeconds: 1199,
                    textSHA256: TranscriptMorphology.textHash(text), tokens: tokens())
            ]),
            evidence(sections: [
                .init(startSeconds: 600, endSeconds: 1200, textSHA256: "wrong", tokens: tokens())
            ]),
            evidence(sections: [
                .init(
                    startSeconds: 600, endSeconds: 1200,
                    textSHA256: TranscriptMorphology.textHash(text), tokens: [])
            ]),
        ]
        for item in invalid {
            #expect(throws: SourceExcerptError.invalidEvidence) {
                try item.validate(source: source(), fingerprint: "p")
            }
        }
    }

    @Test func forgedLabelsGapsUtf16BoundsAndSurfaceMutationAreRejected() throws {
        let first = try #require(tokens().first)
        let mutations: [TranscriptMorphology.Token] = [
            .init(
                text: first.text, lowerCharacter: -1, upperCharacter: 1, partOfSpeech: "名詞",
                features: first.features),
            .init(
                text: first.text, lowerCharacter: 0, upperCharacter: 0, partOfSpeech: "名詞",
                features: first.features),
            .init(
                text: first.text, lowerCharacter: 0, upperCharacter: 100, partOfSpeech: "名詞",
                features: first.features),
            .init(
                text: "創作", lowerCharacter: 0, upperCharacter: 1, partOfSpeech: "名詞",
                features: first.features),
            .init(
                text: first.text, lowerCharacter: 0, upperCharacter: 1, partOfSpeech: "名詞",
                features: []),
            .init(
                text: first.text, lowerCharacter: 0, upperCharacter: 1, partOfSpeech: "名詞",
                features: ["動詞", "一般"]),
            .init(
                text: first.text, lowerCharacter: 0, upperCharacter: 1, partOfSpeech: "guessed",
                features: ["guessed", "一般"]),
            .init(
                text: first.text, lowerCharacter: 0, upperCharacter: 1, partOfSpeech: "名詞",
                features: ["名詞", String(repeating: "x", count: 16385)]),
        ]
        for mutation in mutations {
            let item = evidence(sections: [
                .init(
                    startSeconds: 600, endSeconds: 1200,
                    textSHA256: TranscriptMorphology.textHash(text),
                    tokens: [mutation] + tokens().dropFirst())
            ])
            #expect(throws: SourceExcerptError.invalidEvidence) {
                try item.validate(source: source(), fingerprint: "p")
            }
        }
        let truncated = evidence(sections: [
            .init(
                startSeconds: 600, endSeconds: 1200,
                textSHA256: TranscriptMorphology.textHash(text), tokens: Array(tokens().dropLast()))
        ])
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try truncated.validate(source: source(), fingerprint: "p")
        }
    }
}
