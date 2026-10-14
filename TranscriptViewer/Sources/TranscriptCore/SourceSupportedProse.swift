import Foundation

/// A paragraph grounded in selected speech, not freely rewritten factual claims.
/// Quotes retain source ranges and native timing; the model only chooses existing IDs.
public struct SourceSupportedProse: Codable, Equatable, Sendable {
    public struct Section: Codable, Equatable, Sendable, Identifiable {
        public var id: Double { startSeconds }
        public let startSeconds: Double
        public let endSeconds: Double
        public let text: String
        public let evidence: [RepresentativeSourceSummary.Clock]
    }

    public let version: Int
    public let method: String
    public let label: String
    public let fingerprint: String
    public let morphologyFingerprint: String
    public let clockDomain: RepresentativeSourceSummary.ClockDomain
    public let overview: String
    public let overviewEvidence: [RepresentativeSourceSummary.Clock]
    public let sections: [Section]

    public static let methodName = "source-supported-quotation-prose-v2"
    public static let displayLabel = "発話抜粋による要約・未検証"
    private static let legacyLabel = "発話根拠付きの文章要約・未検証"

    /// A shared source-range/clock view, not a different generation or approval result.
    public var representativeEvidence: RepresentativeSourceSummary {
        .init(
            version: 1, method: RepresentativeSourceSummary.methodName,
            label: RepresentativeSourceSummary.displayLabel, clockDomain: clockDomain,
            source: .init(
                mode: "source-excerpts-v1", fingerprint: fingerprint,
                sections: sections.map {
                    .init(
                        startSeconds: $0.startSeconds, endSeconds: $0.endSeconds,
                        excerpts: $0.evidence.map(\.span))
                }, overview: overviewEvidence.map(\.span)),
            clocks: sections.flatMap(\.evidence))
    }

    /// All prose is reproducible from the pinned source, including question/negation wording.
    /// This proves textual support, not ASR accuracy or complete semantic coverage.
    public func validate(
        video: VideoTranscript, fingerprint expected: String,
        clockDomain expectedClock: RepresentativeSourceSummary.ClockDomain,
        morphology: TranscriptMorphology
    ) throws {
        try morphology.validate(source: video.summary, fingerprint: expected)
        guard [1, 2].contains(version), method == Self.methodName,
            label == (version == 1 ? Self.legacyLabel : Self.displayLabel),
            fingerprint == expected, morphologyFingerprint == (try morphology.fingerprint()),
            clockDomain == expectedClock,
            !sections.isEmpty, sections.count == video.summary.sections.count
        else { throw SourceExcerptError.invalidEvidence }
        let layouts = try RepresentativeSourceSummary.layouts(video)
        for (index, pair) in zip(sections, video.summary.sections).enumerated() {
            let (section, source) = pair
            guard section.startSeconds == source.startSeconds,
                section.endSeconds == source.endSeconds,
                let layout = layouts[source.startSeconds]
            else { throw SourceExcerptError.invalidEvidence }
            let candidates = try SourceSupportedProseGenerator.candidates(
                source, morphology: morphology.sections[index], layout: layout,
                policy: version == 1 ? .legacy : .contextual
            ).map(\.span)
            try Self.check(section.evidence, candidates: candidates, layouts: layouts)
            guard
                section.evidence.allSatisfy({ $0.span.sectionStartSeconds == source.startSeconds }),
                section.text == (try Self.paragraph(section.evidence))
            else { throw SourceExcerptError.invalidEvidence }
            // A token layout is independently reconstructed; no character/time interpolation.
            guard layout.text == source.text else { throw SourceExcerptError.invalidEvidence }
        }
        let selected = sections.flatMap(\.evidence)
        try Self.check(overviewEvidence, candidates: selected.map(\.span), layouts: layouts)
        guard overviewEvidence.allSatisfy(selected.contains),
            overview == (try Self.paragraph(overviewEvidence))
        else { throw SourceExcerptError.invalidEvidence }
    }

    static func paragraph(_ evidence: [RepresentativeSourceSummary.Clock]) throws -> String {
        guard !evidence.isEmpty, evidence.count <= SourceSupportedProseGenerator.selectionLimit
        else {
            throw SourceExcerptError.invalidEvidence
        }
        let sentences = evidence.enumerated().map { index, clock in
            let source = SourceSupportedProseGenerator.presentation(clock.span.text)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let prefix = index == 0 ? "配信では" : "また、"
            // These words report speech; they do not turn a quoted plan/question into an event.
            return "\(prefix)「\(source)」と話している。"
        }
        return try NarrativeSummaryGenerator.validateResponse(sentences.joined())
    }

    private static func check(
        _ evidence: [RepresentativeSourceSummary.Clock], candidates: [SourceExcerptSummary.Span],
        layouts: [Double: TranscriptTokenLayout]
    ) throws {
        guard !evidence.isEmpty, evidence.count <= SourceSupportedProseGenerator.selectionLimit,
            Set(evidence.map { SourceSupportedProseGenerator.presentation($0.span.text) }).count
                == evidence.count
        else { throw SourceExcerptError.invalidEvidence }
        var previous: (section: Double, lower: Int)?
        for clock in evidence {
            let span = clock.span
            guard candidates.contains(span), let layout = layouts[span.sectionStartSeconds],
                previous.map({
                    $0.section < span.sectionStartSeconds
                        || ($0.section == span.sectionStartSeconds
                            && $0.lower < span.lowerCharacter)
                }) ?? true
            else { throw SourceExcerptError.invalidEvidence }
            let measured = try layout.clock(lower: span.lowerCharacter, upper: span.upperCharacter)
            guard clock.startSeconds == measured.start, clock.endSeconds == measured.end else {
                throw SourceExcerptError.invalidEvidence
            }
            previous = (span.sectionStartSeconds, span.lowerCharacter)
        }
    }
}

/// Select informative, diverse source spans, then construct conservative Japanese prose.
/// Every reduction selects original evidence again; generated prose is never a new source.
public struct SourceSupportedProseGenerator: Sendable {
    static let selectionLimit = 3
    private static let quoteByteLimit = 480
    private static let rawUnitByteLimit = 16384
    private let morphology: TranscriptMorphology
    private let respond: NarrativeSummaryGenerator.Respond

    public init(
        morphology: TranscriptMorphology, respond: @escaping NarrativeSummaryGenerator.Respond
    ) {
        self.morphology = morphology
        self.respond = respond
    }

    public func generate(
        video: VideoTranscript, fingerprint: String,
        clockDomain: RepresentativeSourceSummary.ClockDomain
    ) async throws -> SourceSupportedProse {
        guard !video.summary.sections.isEmpty else { throw SummaryGenerationError.emptyTranscript }
        try morphology.validate(source: video.summary, fingerprint: fingerprint)
        let layouts = try RepresentativeSourceSummary.layouts(video)
        var sections: [SourceSupportedProse.Section] = []
        for (index, source) in video.summary.sections.enumerated() {
            try Task.checkCancellation()
            guard let layout = layouts[source.startSeconds] else {
                throw SourceExcerptError.invalidEvidence
            }
            let spans = try await select(
                try Self.candidates(source, morphology: morphology.sections[index], layout: layout))
            let clocks = try spans.map { span in
                let timing = try layout.clock(
                    lower: span.lowerCharacter, upper: span.upperCharacter)
                return RepresentativeSourceSummary.Clock(
                    span: span, startSeconds: timing.start, endSeconds: timing.end)
            }
            sections.append(
                .init(
                    startSeconds: source.startSeconds, endSeconds: source.endSeconds,
                    text: try SourceSupportedProse.paragraph(clocks), evidence: clocks))
        }
        let all = sections.flatMap(\.evidence)
        let allCandidates = try video.summary.sections.enumerated().flatMap { index, source in
            guard let layout = layouts[source.startSeconds] else {
                throw SourceExcerptError.invalidEvidence
            }
            return try Self.candidates(
                source, morphology: morphology.sections[index], layout: layout)
        }
        let spans = try await select(
            allCandidates.filter { candidate in all.contains { $0.span == candidate.span } })
        let overview = try spans.map { span in
            guard let clock = all.first(where: { $0.span == span }) else {
                throw SourceExcerptError.invalidEvidence
            }
            return clock
        }
        let result = SourceSupportedProse(
            version: 2, method: SourceSupportedProse.methodName,
            label: SourceSupportedProse.displayLabel, fingerprint: fingerprint,
            morphologyFingerprint: try morphology.fingerprint(),
            clockDomain: clockDomain, overview: try SourceSupportedProse.paragraph(overview),
            overviewEvidence: overview, sections: sections)
        try result.validate(
            video: video, fingerprint: fingerprint, clockDomain: clockDomain, morphology: morphology
        )
        return result
    }

    /// Only prosodic spelling is shortened in model input/presentation. Raw evidence is unchanged.
    static func presentation(_ source: String) -> String {
        source.replacingOccurrences(
            of: #"([ー〜～っッ])\1{3,}"#, with: "$1$1", options: .regularExpression)
    }

    public struct Candidate: Codable, Sendable {
        let span: SourceExcerptSummary.Span
        let score: Int
        let startSeconds: Double
        let endSeconds: Double
    }

    public struct CandidatePreview: Codable, Sendable {
        public struct Section: Codable, Sendable {
            let startSeconds: Double
            let legacy: [Candidate]
            let contextual: [Candidate]
        }
        let fingerprint: String
        let morphologyFingerprint: String
        let sections: [Section]
    }

    /// Compare candidate ranges and scores without inference or publication.
    public static func preview(
        video: VideoTranscript, fingerprint: String, morphology: TranscriptMorphology
    ) throws -> CandidatePreview {
        try morphology.validate(source: video.summary, fingerprint: fingerprint)
        let layouts = try RepresentativeSourceSummary.layouts(video)
        let sections = try video.summary.sections.enumerated().map { index, source in
            guard let layout = layouts[source.startSeconds] else {
                throw SourceExcerptError.invalidEvidence
            }
            return CandidatePreview.Section(
                startSeconds: source.startSeconds,
                legacy: try candidates(
                    source, morphology: morphology.sections[index], layout: layout, policy: .legacy),
                contextual: try candidates(
                    source, morphology: morphology.sections[index], layout: layout))
        }
        return .init(
            fingerprint: fingerprint, morphologyFingerprint: try morphology.fingerprint(),
            sections: sections)
    }

    enum SelectionPolicy {
        case legacy
        case contextual
    }

    static func candidates(
        _ section: TranscriptSummary.Section, morphology: TranscriptMorphology.Section,
        layout: TranscriptTokenLayout, policy: SelectionPolicy = .contextual
    ) throws
        -> [Candidate]
    {
        let source = section.text
        var boundaries: [String.Index] = []
        for (index, token) in morphology.tokens.enumerated() {
            guard token.text.utf8.count <= rawUnitByteLimit else {
                throw SummaryGenerationError.oversizedCharacter
            }
            let next = index + 1 < morphology.tokens.count ? morphology.tokens[index + 1] : nil
            if completesClause(token, next: next) {
                boundaries.append(source.index(source.startIndex, offsetBy: token.upperCharacter))
            }
        }
        for index in source.indices where "。！？!?\n".contains(source[index]) {
            boundaries.append(source.index(after: index))
        }
        boundaries = Array(Set(boundaries)).sorted()
        if boundaries.last != source.endIndex { boundaries.append(source.endIndex) }
        var lower = source.startIndex
        var offset = 0
        var spans: [SourceExcerptSummary.Span] = []
        for upper in boundaries {
            let unit = String(source[lower..<upper])
            lower = upper
            let start = offset
            offset += unit.count
            // A long clause may be omitted from a limited summary, never shortened mid-clause.
            // No arbitrary new sentence boundary, token time or inferred ending is introduced.
            guard !unit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                policy == .contextual || presentation(unit).utf8.count <= quoteByteLimit
            else { continue }
            spans.append(
                .init(
                    sectionStartSeconds: section.startSeconds, lowerCharacter: start,
                    upperCharacter: offset, text: unit))
        }
        if policy == .contextual {
            spans = try retainingReplies(spans, morphology: morphology, layout: layout)
        }
        var seen: Set<String> = []
        return try spans.compactMap { span in
            let normalized = presentation(span.text).trimmingCharacters(in: .whitespacesAndNewlines)
            let byteLimit = policy == .legacy ? quoteByteLimit : contextByteLimit
            guard normalized.utf8.count <= byteLimit else { return nil }
            let tokens = morphology.tokens.filter {
                $0.lowerCharacter >= span.lowerCharacter && $0.upperCharacter <= span.upperCharacter
            }
            let weight = score(tokens: tokens, text: normalized, policy: policy)
            guard weight > 0, seen.insert(normalized).inserted else { return nil }
            let timing = try layout.clock(lower: span.lowerCharacter, upper: span.upperCharacter)
            return Candidate(
                span: span, score: weight, startSeconds: timing.start, endSeconds: timing.end)
        }
    }

    /// Preserve a nearby, shared-topic response as one contiguous original quotation.
    /// This is a conservative context heuristic, not speaker attribution or entailment.
    static func retainingReplies(
        _ spans: [SourceExcerptSummary.Span], morphology: TranscriptMorphology.Section,
        layout: TranscriptTokenLayout
    ) throws -> [SourceExcerptSummary.Span] {
        var result: [SourceExcerptSummary.Span] = []
        var index = 0
        while index < spans.count {
            let current = spans[index]
            guard index + 1 < spans.count else {
                result.append(current)
                break
            }
            let next = spans[index + 1]
            let head = String(next.text.prefix(40))
            let response = ["でも", "だけど", "しかし", "いや", "それは", "そう"]
                .contains(where: head.contains)
            let firstWords = contentNouns(
                morphology.tokens.filter {
                    $0.lowerCharacter >= current.lowerCharacter
                        && $0.upperCharacter <= current.upperCharacter
                })
            let nextWords = contentNouns(
                morphology.tokens.filter {
                    $0.lowerCharacter >= next.lowerCharacter
                        && $0.upperCharacter <= next.upperCharacter
                })
            let firstClock = try layout.clock(
                lower: current.lowerCharacter, upper: current.upperCharacter)
            let nextClock = try layout.clock(lower: next.lowerCharacter, upper: next.upperCharacter)
            guard response, !firstWords.isDisjoint(with: nextWords),
                current.upperCharacter == next.lowerCharacter,
                nextClock.start <= firstClock.end + CueTimeline.mergeGapSeconds
            else {
                result.append(current)
                index += 1
                continue
            }
            let combined = current.text + next.text
            // Reserve space for two model candidates and their ID delimiters.
            // If the pair cannot fit, retain the response rather than a truncated assertion.
            if presentation(combined).utf8.count <= contextByteLimit {
                result.append(
                    .init(
                        sectionStartSeconds: current.sectionStartSeconds,
                        lowerCharacter: current.lowerCharacter, upperCharacter: next.upperCharacter,
                        text: combined))
            } else {
                result.append(next)
            }
            index += 2
        }
        return result
    }

    private static var contextByteLimit: Int { SummaryRequest.inputByteLimit / 2 - 32 }

    private static func contentNouns(_ tokens: [TranscriptMorphology.Token]) -> Set<String> {
        Set(
            tokens.filter {
                $0.partOfSpeech == "名詞"
                    && !["非自立", "代名詞", "数", "接尾"].contains($0.features[1])
            }.map(\.text))
    }

    /// Preserve predicate + auxiliary/negation chains and noun modifiers as a single quote.
    /// A terminal particle closes colloquial speech; inflection comes from the pinned dictionary.
    static func completesClause(
        _ token: TranscriptMorphology.Token, next: TranscriptMorphology.Token?
    ) -> Bool {
        let terminalParticle = token.partOfSpeech == "助詞" && token.features[1] == "終助詞"
        let predicate =
            ["動詞", "形容詞", "助動詞"].contains(token.partOfSpeech)
            && token.features.count > 5 && token.features[5] == "基本形"
        guard terminalParticle || predicate else { return false }
        guard let next else { return true }
        if ["助詞", "助動詞"].contains(next.partOfSpeech)
            || (["動詞", "形容詞"].contains(next.partOfSpeech) && next.features[1] == "非自立")
        {
            return false
        }
        // "行く予定" and "買った物" are noun modifiers, not completed assertions.
        return terminalParticle || !["名詞", "接頭詞"].contains(next.partOfSpeech)
    }

    /// POS comes from the pinned tokenizer, not word length or generated model tags.
    /// Scoring is a conservative selection heuristic, not a semantic-coverage guarantee.
    static func score(
        tokens: [TranscriptMorphology.Token], text: String,
        policy: SelectionPolicy = .contextual
    ) -> Int {
        let routine = ["ありがとう", "いらっしゃい", "おかえり", "こんにちは", "お疲れ様"]
        var nouns: Set<String> = []
        var predicates = 0
        for token in tokens {
            if token.partOfSpeech == "名詞",
                !["非自立", "代名詞", "数", "接尾"].contains(token.features[1]),
                token.text.range(of: #"^[ー〜～っッ]+$"#, options: .regularExpression) == nil
            {
                nouns.insert(token.text)
            } else if ["動詞", "形容詞"].contains(token.partOfSpeech), token.features[1] != "非自立" {
                predicates += 1
            }
        }
        guard !nouns.isEmpty, predicates > 0 || nouns.count >= 2 else { return 0 }
        let greetingCount = routine.reduce(0) { $0 + text.components(separatedBy: $1).count - 1 }
        let content = nouns.count * 3 + min(predicates, 4)
        if policy == .legacy { return max(0, content - greetingCount * 4) }
        let socialCount = ["初見", "新参", "しんざん", "フォロー", "よろしく"].reduce(0) {
            $0 + text.components(separatedBy: $1).count - 1
        }
        // Rank content density, rather than rewarding the raw length of mixed-topic speech.
        // A minimum denominator prevents tiny noun/name fragments winning by density alone.
        let density = content * 100 / max(40, text.count)
        // Nominal utterances remain eligible; unsupported name lists receive less weight.
        let supportedDensity = predicates > 0 ? density : density / 4
        return max(0, supportedDensity - (greetingCount + socialCount) * 12)
    }

    private func select(_ spans: [Candidate]) async throws
        -> [SourceExcerptSummary.Span]
    {
        var seen: Set<String> = []
        let spans = spans.filter { seen.insert(Self.presentation($0.span.text)).inserted }
        guard !spans.isEmpty else { throw SummaryGenerationError.emptyTranscript }
        // Native source-clock thirds spread coverage; factual POS density ranks within each third.
        // Duplicate speech is retained in the source, but is not repeated in a summary paragraph.
        guard let start = spans.map(\.startSeconds).min(), let end = spans.map(\.endSeconds).max()
        else {
            throw SourceExcerptError.invalidSelection
        }
        let duration = max(end - start, Double.ulpOfOne)
        let buckets = Dictionary(grouping: spans) {
            min(2, Int(($0.startSeconds - start) / duration * 3))
        }
        var selected: [SourceExcerptSummary.Span] = []
        for group in buckets.keys.sorted() {
            try Task.checkCancellation()
            let ranked = (buckets[group] ?? []).sorted {
                $0.score == $1.score
                    ? $0.span.lowerCharacter < $1.span.lowerCharacter : $0.score > $1.score
            }
            let batch = Array(ranked.prefix(2))
            guard let first = batch.first else { throw SourceExcerptError.invalidSelection }
            if batch.count == 1 {
                selected.append(first.span)
                continue
            }
            let input = batch.enumerated().map {
                "ID \($0.offset)\n\(Self.presentation($0.element.span.text))"
            }.joined(separator: "\n")
            guard input.utf8.count <= SummaryRequest.inputByteLimit else {
                throw SummaryGenerationError.oversizedCharacter
            }
            let raw = try await respond(
                SummaryRequest(
                    kind: .section, source: input,
                    sectionStartSeconds: first.span.sectionStartSeconds,
                    selectsSource: true, supportedSpeechSelection: true))
            let ids: [Int]
            do {
                ids = try JSONDecoder().decode([Int].self, from: Data(raw.utf8))
            } catch is DecodingError {
                throw SourceExcerptError.invalidSelection
            }
            guard ids.count == 1, let id = ids.first, batch.indices.contains(id) else {
                throw SourceExcerptError.invalidSelection
            }
            selected.append(batch[id].span)
        }
        return selected.sorted {
            $0.sectionStartSeconds == $1.sectionStartSeconds
                ? $0.lowerCharacter < $1.lowerCharacter
                : $0.sectionStartSeconds < $1.sectionStartSeconds
        }
    }
}
