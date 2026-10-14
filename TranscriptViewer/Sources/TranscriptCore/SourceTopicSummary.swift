import Foundation
import NaturalLanguage

/// Source-constrained keywords and speech excerpts; never generated factual paraphrases.
public struct SourceTopicSummary: Codable, Equatable, Sendable {
    public struct Keyword: Codable, Equatable, Sendable {
        public let sectionStartSeconds: Double
        public let lowerCharacter: Int
        public let upperCharacter: Int
        public let text: String
        public let startSeconds: Double
        public let endSeconds: Double
    }

    public struct Section: Codable, Equatable, Sendable, Identifiable {
        public var id: Double { startSeconds }
        public let startSeconds: Double
        public let endSeconds: Double
        public let keywords: [Keyword]
        public let excerpts: [RepresentativeSourceSummary.Clock]
    }

    public let version: Int
    public let method: String
    public let label: String
    public let representative: RepresentativeSourceSummary
    public let overview: [Keyword]
    public let sections: [Section]

    public static let methodName = "source-keywords-excerpts-v2"
    public static let displayLabel = "キーワードと発話抜粋による要約・未検証"
    private static let keywordsPerSection = 3
    private static let overviewKeywordLimit = 6

    /// Visit canonical recognition input, retaining exact words, ranges and token clock envelopes.
    public static func generate(
        video: VideoTranscript, fingerprint: String,
        clockDomain: RepresentativeSourceSummary.ClockDomain
    ) throws -> Self {
        let representative = try RepresentativeSourceSummary.generate(
            video: video, fingerprint: fingerprint, clockDomain: clockDomain)
        return try build(video: video, representative: representative)
    }

    /// Validate both lexical provenance and the complete deterministic selection contract.
    public func validate(
        video: VideoTranscript, fingerprint: String,
        clockDomain: RepresentativeSourceSummary.ClockDomain
    ) throws {
        try representative.validate(
            video: video, fingerprint: fingerprint, clockDomain: clockDomain)
        guard
            self
                == (try Self.generate(
                    video: video, fingerprint: fingerprint, clockDomain: clockDomain))
        else {
            throw SourceExcerptError.invalidEvidence
        }
    }

    private static func build(video: VideoTranscript, representative: RepresentativeSourceSummary)
        throws -> Self
    {
        let layouts = try RepresentativeSourceSummary.layouts(video)
        let weights = weightedWords(representative)
        var sections: [Section] = []
        for section in representative.source.sections {
            try Task.checkCancellation()
            guard let layout = layouts[section.startSeconds] else {
                throw SourceExcerptError.invalidEvidence
            }
            let clocks = representative.clocks.filter {
                $0.span.sectionStartSeconds == section.startSeconds
            }
            sections.append(
                .init(
                    startSeconds: section.startSeconds, endSeconds: section.endSeconds,
                    keywords: try keywords(
                        weights.filter { $0.word.span.sectionStartSeconds == section.startSeconds },
                        layout: layout), excerpts: clocks))
        }
        let overviewSpans = representative.source.overview
        var seen: Set<String> = []
        let overview = sections.flatMap(\.keywords).filter { keyword in
            overviewSpans.contains { span in
                span.sectionStartSeconds == keyword.sectionStartSeconds
                    && span.lowerCharacter <= keyword.lowerCharacter
                    && span.upperCharacter >= keyword.upperCharacter
            } && seen.insert(keyword.text.lowercased()).inserted
        }
        return Self(
            version: 1, method: methodName, label: displayLabel, representative: representative,
            overview: Array(overview.prefix(overviewKeywordLimit)), sections: sections)
    }

    private struct Word {
        let text: String
        let normalized: String
        let span: SourceExcerptSummary.Span
        let lower: Int
        let upper: Int
    }

    private static func words(_ span: SourceExcerptSummary.Span) -> [Word] {
        let text = span.text
        let eligible = RepresentativeSourceSummary.terms(text)
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        tokenizer.setLanguage(.japanese)
        var words: [Word] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = String(text[range])
            let normalized = word.lowercased()
            // Single-character function fragments and bare counts are poor keyword labels.
            // This is an explicit word heuristic, not a claim of Japanese POS availability.
            if word.count >= 2, eligible.contains(normalized),
                !discourseWords.contains(normalized),
                hasTopicScript(word),
                !word.unicodeScalars.allSatisfy(CharacterSet.decimalDigits.contains)
            {
                words.append(
                    .init(
                        text: word, normalized: normalized, span: span,
                        lower: span.lowerCharacter
                            + text.distance(from: text.startIndex, to: range.lowerBound),
                        upper: span.lowerCharacter
                            + text.distance(from: text.startIndex, to: range.upperBound)))
            }
            return true
        }
        return words
    }

    private static let discourseWords: Set<String> = [
        "さん", "ちゃん", "思う", "思っ", "言う", "言っ", "そう", "なんか", "ちょっと",
        "やばい", "はい", "うん", "それで", "だから", "本当に", "ほんと", "けど", "みたい",
        "だけ", "まで", "から", "つい", "について", "として", "という",
        "今日", "昨日", "明日", "普通", "全部", "本当", "あの", "その", "この",
    ]

    /// Orthographic filtering omits hiragana-only keywords, never their source quotations.
    private static func hasTopicScript(_ word: String) -> Bool {
        let topicRanges: [ClosedRange<UInt32>] = [
            0x3400...0x4DBF, 0x4E00...0x9FFF, 0x30A1...0x30FF, 0x41...0x5A, 0x61...0x7A,
        ]
        return word.unicodeScalars.contains { scalar in
            topicRanges.contains { $0.contains(scalar.value) }
        }
    }

    private struct WeightedWord {
        let word: Word
        let weight: Double
    }

    private static func weightedWords(_ representative: RepresentativeSourceSummary)
        -> [WeightedWord]
    {
        let words = representative.clocks.flatMap { Self.words($0.span) }
        let totalSections = representative.source.sections.count
        var weighted: [WeightedWord] = []
        for group in Dictionary(grouping: words, by: \.normalized).values {
            let sectionCount = Set(group.map { $0.span.sectionStartSeconds }).count
            let rarity = log(Double(totalSections + 1) / Double(sectionCount))
            for sectionWords in Dictionary(grouping: group, by: { $0.span.sectionStartSeconds })
                .values
            {
                let frequency = Set(sectionWords.map { $0.span.lowerCharacter }).count
                weighted += sectionWords.map {
                    .init(
                        word: $0,
                        weight: log1p(Double(frequency)) * rarity * sqrt(Double($0.text.count)))
                }
            }
        }
        return weighted
    }

    private static func keywords(_ weighted: [WeightedWord], layout: TranscriptTokenLayout)
        throws -> [Keyword]
    {
        let ranked = weighted.sorted { left, right in
            if left.weight != right.weight { return left.weight > right.weight }
            if left.word.text.count != right.word.text.count {
                return left.word.text.count > right.word.text.count
            }
            return left.word.lower < right.word.lower
        }
        var seen: Set<String> = []
        let selected = ranked.filter { seen.insert($0.word.normalized).inserted }
            .prefix(keywordsPerSection).map(\.word)
        return try selected.map { word in
            let clock = try layout.clock(lower: word.lower, upper: word.upper)
            return Keyword(
                sectionStartSeconds: word.span.sectionStartSeconds,
                lowerCharacter: word.lower, upperCharacter: word.upper, text: word.text,
                startSeconds: clock.start, endSeconds: clock.end)
        }
    }
}
