import Foundation
import NaturalLanguage

/// Deterministic speech excerpts. Selection is a heuristic, not semantic quality approval.
public struct RepresentativeSourceSummary: Codable, Equatable, Sendable {
    public enum ClockDomain: String, Codable, Sendable {
        case editedOutput = "edited-output"
        case sourcePresentation = "source-presentation"
    }
    public struct Clock: Codable, Equatable, Sendable {
        public let span: SourceExcerptSummary.Span
        public let startSeconds: Double
        public let endSeconds: Double
    }

    public let version: Int
    public let method: String
    public let label: String
    public let clockDomain: ClockDomain
    public let source: SourceExcerptSummary
    public let clocks: [Clock]

    public static let methodName = "word-time-diversity-v1"
    public static let displayLabel = "発話抜粋による要約・未検証"

    public func validate(
        video: VideoTranscript, fingerprint: String, clockDomain expected: ClockDomain
    ) throws {
        guard version == 1, method == Self.methodName, label == Self.displayLabel,
            clockDomain == expected, video.summaryInputKind == .tokenSequence
        else { throw SourceExcerptError.invalidEvidence }
        try source.validate(source: video.summary, fingerprint: fingerprint)
        let selected = source.sections.flatMap(\.excerpts)
        guard clocks.map(\.span) == selected else { throw SourceExcerptError.invalidEvidence }
        let layouts = try Self.layouts(video)
        for clock in clocks {
            guard let layout = layouts[clock.span.sectionStartSeconds] else {
                throw SourceExcerptError.invalidEvidence
            }
            let expected = try layout.clock(
                lower: clock.span.lowerCharacter, upper: clock.span.upperCharacter)
            guard clock.startSeconds == expected.start, clock.endSeconds == expected.end else {
                throw SourceExcerptError.invalidEvidence
            }
        }
    }

    /// No model calls, generated prose, forced reduction of actual repeated words, or silent fallback.
    public static func generate(
        video: VideoTranscript, fingerprint: String, clockDomain: ClockDomain
    ) throws -> Self {
        guard !video.summary.sections.isEmpty else { throw SummaryGenerationError.emptyTranscript }
        let layouts = try layouts(video)
        var sections: [SourceExcerptSummary.Section] = []
        var selected: [Candidate] = []
        for section in video.summary.sections {
            try Task.checkCancellation()
            guard let layout = layouts[section.startSeconds] else {
                throw SourceExcerptError.invalidEvidence
            }
            let candidates = try SourceExcerptGenerator.spans(section).map { span in
                let clock = try layout.clock(lower: span.lowerCharacter, upper: span.upperCharacter)
                return Candidate(
                    span: span, start: clock.start, end: clock.end, terms: terms(span.text))
            }
            let chosen = select(candidates, range: section.startSeconds..<section.endSeconds)
            sections.append(
                .init(
                    startSeconds: section.startSeconds, endSeconds: section.endSeconds,
                    excerpts: chosen.map(\.span)))
            selected += chosen
        }
        let result = Self(
            version: 1, method: methodName, label: displayLabel, clockDomain: clockDomain,
            source: .init(
                mode: "source-excerpts-v1", fingerprint: fingerprint, sections: sections,
                overview: select(selected).map(\.span)),
            clocks: selected.map {
                .init(span: $0.span, startSeconds: $0.start, endSeconds: $0.end)
            })
        try result.validate(video: video, fingerprint: fingerprint, clockDomain: clockDomain)
        return result
    }

    static func layouts(_ video: VideoTranscript) throws -> [Double: TranscriptTokenLayout] {
        guard video.summaryInputKind == .tokenSequence else {
            throw SourceExcerptError.invalidEvidence
        }
        let sorted = video.cues.sorted { $0.startSeconds < $1.startSeconds }
        let groups = Dictionary(grouping: sorted) {
            floor($0.startSeconds / TranscriptSummary.intervalSeconds)
                * TranscriptSummary.intervalSeconds
        }
        var result: [Double: TranscriptTokenLayout] = [:]
        for section in video.summary.sections {
            guard let cues = groups[section.startSeconds] else {
                throw SourceExcerptError.invalidEvidence
            }
            let layout = TranscriptTokenLayout(cues: cues)
            guard layout.text == section.text else { throw SourceExcerptError.invalidEvidence }
            result[section.startSeconds] = layout
        }
        return result
    }

    struct Candidate: Sendable {
        let span: SourceExcerptSummary.Span
        let start: Double
        let end: Double
        let terms: Set<String>
    }

    static func terms(_ text: String) -> Set<String> {
        // Japanese lexicalClass is not available on every system. This explicit
        // word-based policy does not claim part-of-speech tagging or fetch assets.
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        tokenizer.setLanguage(.japanese)
        var result: Set<String> = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = String(text[range]).lowercased()
            if !functionWords.contains(word),
                word.unicodeScalars.contains(where: CharacterSet.alphanumerics.contains)
            {
                result.insert(word)
            }
            return true
        }
        return result
    }

    /// Only scoring ignores these common Japanese discourse/function words; saved text is untouched.
    private static let functionWords: Set<String> = [
        "は", "が", "を", "に", "へ", "と", "で", "の", "も", "や", "か", "ね", "よ", "な", "って",
        "です", "ます", "だ", "する", "した", "して", "いる", "ある", "これ", "それ", "あれ", "ここ", "そこ",
        "ありがとう", "こんにちは", "こんばんは", "おはよう", "ございます", "よろしく",
        "し", "た", "て", "ない", "ん", "ござい",
    ]

    /// One eligible candidate per temporal third, with lexical novelty between selections.
    /// Repeated identical spans do not multiply vocabulary frequency or occupy multiple slots.
    static func select(_ input: [Candidate], range: Range<Double>? = nil) -> [Candidate] {
        var seen: Set<String> = []
        let unique = input.filter {
            seen.insert($0.span.text.trimmingCharacters(in: .whitespacesAndNewlines)).inserted
        }
        let informative = unique.filter { !$0.terms.isEmpty }
        let candidates = informative.isEmpty ? unique : informative
        guard let first = candidates.map(\.start).min(), let last = candidates.map(\.start).max()
        else { return [] }
        var frequency: [String: Int] = [:]
        for candidate in candidates {
            for term in candidate.terms { frequency[term, default: 0] += 1 }
        }
        let origin = range?.lowerBound ?? first
        let span = max((range?.upperBound ?? last) - origin, 1)
        var chosen: [Candidate] = []
        var used: Set<String> = []
        for third in 0..<3 {
            let bucket = candidates.filter { min(2, Int(($0.start - origin) / span * 3)) == third }
            let ranked = bucket.sorted { left, right in
                let lhs = score(left, frequency: frequency, used: used)
                let rhs = score(right, frequency: frequency, used: used)
                return lhs == rhs ? left.start < right.start : lhs > rhs
            }
            if let candidate = ranked.first {
                chosen.append(candidate)
                used.formUnion(candidate.terms)
            }
        }
        return chosen.sorted { $0.start < $1.start }
    }

    private static func score(_ candidate: Candidate, frequency: [String: Int], used: Set<String>)
        -> Double
    {
        let weight = candidate.terms.reduce(0.0) { $0 + log1p(Double(frequency[$1, default: 0])) }
        let novelty =
            Double(candidate.terms.subtracting(used).count + 1) / Double(candidate.terms.count + 1)
        return weight * novelty / sqrt(Double(max(1, candidate.terms.count)))
    }
}
