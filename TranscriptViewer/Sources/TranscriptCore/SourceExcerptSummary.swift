import Foundation
import NaturalLanguage

/// Explicit source excerpts, not rewritten prose or a guarantee of ASR correctness.
public struct SourceExcerptSummary: Codable, Equatable, Sendable {
    public struct Span: Codable, Equatable, Sendable {
        public let sectionStartSeconds: Double
        public let lowerCharacter: Int
        public let upperCharacter: Int
        public let text: String
    }

    public struct Section: Codable, Equatable, Sendable {
        public let startSeconds: Double
        public let endSeconds: Double
        public let excerpts: [Span]
    }

    public let mode: String
    public let fingerprint: String
    public let sections: [Section]
    public let overview: [Span]

    /// Reconstruct every excerpt from the exact clock-bound input, including overview membership.
    public func validate(source: TranscriptSummary, fingerprint expected: String) throws {
        guard mode == "source-excerpts-v1", fingerprint == expected,
            sections.count == source.sections.count, !sections.isEmpty,
            !overview.isEmpty, overview.count <= 3
        else { throw SourceExcerptError.invalidEvidence }
        for (saved, original) in zip(sections, source.sections) {
            guard saved.startSeconds == original.startSeconds,
                saved.endSeconds == original.endSeconds, !saved.excerpts.isEmpty,
                saved.excerpts.count <= 3
            else { throw SourceExcerptError.invalidEvidence }
            let characters = Array(original.text)
            var previousEnd = 0
            for span in saved.excerpts {
                guard span.sectionStartSeconds == original.startSeconds,
                    span.lowerCharacter >= previousEnd,
                    span.upperCharacter > span.lowerCharacter,
                    span.upperCharacter <= characters.count,
                    span.text == String(characters[span.lowerCharacter..<span.upperCharacter])
                else { throw SourceExcerptError.invalidEvidence }
                previousEnd = span.upperCharacter
            }
        }
        let selected = sections.flatMap(\.excerpts)
        var previousIndex = -1
        for span in overview {
            guard let index = selected.firstIndex(of: span), index > previousIndex else {
                throw SourceExcerptError.invalidEvidence
            }
            previousIndex = index
        }
    }
}

public enum SourceExcerptError: Error, Equatable {
    case invalidSelection
    case invalidEvidence
}

/// Visits all source spans. Model output can select indices, never supply saved prose.
public struct SourceExcerptGenerator: Sendable {
    private let respond: NarrativeSummaryGenerator.Respond

    public init(respond: @escaping NarrativeSummaryGenerator.Respond) {
        self.respond = respond
    }

    public func generate(
        _ source: TranscriptSummary, fingerprint: String,
        progress: @Sendable (Int, Int) async -> Void = { _, _ in }
    ) async throws -> SourceExcerptSummary {
        guard !source.sections.isEmpty else { throw SummaryGenerationError.emptyTranscript }
        var sections: [SourceExcerptSummary.Section] = []
        for section in source.sections {
            try Task.checkCancellation()
            let selected = try await reduce(Self.spans(section), kind: .section)
            sections.append(
                .init(
                    startSeconds: section.startSeconds,
                    endSeconds: section.endSeconds, excerpts: selected))
            await progress(sections.count, source.sections.count)
        }
        let overview = try await reduce(sections.flatMap(\.excerpts), kind: .overview)
        let result = SourceExcerptSummary(
            mode: "source-excerpts-v1", fingerprint: fingerprint,
            sections: sections, overview: overview)
        try result.validate(source: source, fingerprint: fingerprint)
        return result
    }

    static func spans(_ section: TranscriptSummary.Section) throws
        -> [SourceExcerptSummary.Span]
    {
        var result: [SourceExcerptSummary.Span] = []
        var text = ""
        var start = 0
        var offset = 0
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = section.text
        tokenizer.setLanguage(.japanese)
        var boundaries: [String.Index] = []
        tokenizer.enumerateTokens(in: section.text.startIndex..<section.text.endIndex) { range, _ in
            boundaries.append(range.upperBound)
            return true
        }
        for index in section.text.indices where "。！？\n".contains(section.text[index]) {
            boundaries.append(section.text.index(after: index))
        }
        boundaries = Array(Set(boundaries)).sorted()
        if boundaries.last != section.text.endIndex { boundaries.append(section.text.endIndex) }
        var lower = section.text.startIndex
        for upper in boundaries {
            let unit = String(section.text[lower..<upper])
            lower = upper
            guard unit.utf8.count <= 600 else {
                throw SummaryGenerationError.oversizedCharacter
            }
            // Newlines are measured utterance gaps. Keep them as source boundaries.
            if !text.isEmpty,
                text.utf8.count + unit.utf8.count > 600 || unit.hasPrefix("\n")
            {
                result.append(
                    .init(
                        sectionStartSeconds: section.startSeconds,
                        lowerCharacter: start, upperCharacter: offset, text: text))
                start = offset
                text = ""
            }
            text.append(unit)
            offset += unit.count
            if unit.contains(where: { "。！？\n".contains($0) }) {
                if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    result.append(
                        .init(
                            sectionStartSeconds: section.startSeconds,
                            lowerCharacter: start, upperCharacter: offset, text: text))
                }
                start = offset
                text = ""
            }
        }
        if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            result.append(
                .init(
                    sectionStartSeconds: section.startSeconds,
                    lowerCharacter: start, upperCharacter: offset, text: text))
        }
        return result
    }

    private func reduce(_ spans: [SourceExcerptSummary.Span], kind: SummaryRequest.Kind)
        async throws -> [SourceExcerptSummary.Span]
    {
        guard !spans.isEmpty else { throw SummaryGenerationError.emptyTranscript }
        var candidates = spans
        // Two bounded spans per request; selecting one guarantees strict progress.
        while candidates.count > 3 {
            var next: [SourceExcerptSummary.Span] = []
            for start in stride(from: 0, to: candidates.count, by: 2) {
                let batch = Array(candidates[start..<min(start + 2, candidates.count)])
                next += try await select(batch, kind: kind)
            }
            candidates = next
        }
        return candidates
    }

    private func select(_ spans: [SourceExcerptSummary.Span], kind: SummaryRequest.Kind)
        async throws -> [SourceExcerptSummary.Span]
    {
        try Task.checkCancellation()
        if spans.count == 1 { return spans }
        let lines = spans.enumerated().map { "ID \($0.offset)\n\($0.element.text)" }
        let input = lines.joined(separator: "\n")
        guard input.utf8.count <= SummaryRequest.inputByteLimit else {
            throw SummaryGenerationError.oversizedCharacter
        }
        let raw = try await respond(
            SummaryRequest(
                kind: kind, source: input,
                sectionStartSeconds: kind == .overview ? nil : spans.first?.sectionStartSeconds,
                selectsSource: true))
        try Task.checkCancellation()
        let indices: [Int]
        do {
            indices = try JSONDecoder().decode([Int].self, from: Data(raw.utf8))
        } catch is DecodingError {
            throw SourceExcerptError.invalidSelection
        }
        guard indices.count == 1, let index = indices.first, spans.indices.contains(index)
        else { throw SourceExcerptError.invalidSelection }
        return [spans[index]]
    }
}
