import Foundation

/// The exact joined token text and conservative source-clock envelopes for its characters.
public struct TranscriptTokenLayout: Sendable {
    struct Token: Sendable {
        let lower: Int
        let upper: Int
        let start: Double
        let end: Double
    }

    public let text: String
    let tokens: [Token]

    public init(cues: [TranscriptCue]) {
        var joined = ""
        var ranges: [Token] = []
        var offset = 0
        var previousEnd: Double?
        for cue in cues {
            if let previousEnd, cue.startSeconds > previousEnd + CueTimeline.mergeGapSeconds {
                joined += "\n"
                offset += 1
            }
            let count = cue.text.count
            ranges.append(
                .init(
                    lower: offset, upper: offset + count, start: cue.startSeconds,
                    end: cue.endSeconds))
            joined += cue.text
            offset += count
            previousEnd = max(previousEnd ?? cue.endSeconds, cue.endSeconds)
        }
        let trimmed = joined.trimmingCharacters(in: .whitespacesAndNewlines)
        text = trimmed
        guard !trimmed.isEmpty, let range = joined.range(of: trimmed) else {
            tokens = []
            return
        }
        let leading = joined.distance(from: joined.startIndex, to: range.lowerBound)
        let length = trimmed.count
        tokens = ranges.compactMap { token in
            let lower = max(0, token.lower - leading)
            let upper = min(length, token.upper - leading)
            guard lower < upper else { return nil }  // Only trimmed whitespace is omitted.
            return .init(lower: lower, upper: upper, start: token.start, end: token.end)
        }
    }

    /// Bounds cover original tokens; no proportional character-to-time interpolation is used.
    public func clock(lower: Int, upper: Int) throws -> (start: Double, end: Double) {
        guard lower >= 0, lower < upper, upper <= text.count else {
            throw SourceExcerptError.invalidEvidence
        }
        let matched = tokens.filter { $0.lower < upper && $0.upper > lower }
        guard let start = matched.map(\.start).min(), let end = matched.map(\.end).max(),
            start.isFinite, end.isFinite, start >= 0, end >= start
        else { throw SourceExcerptError.invalidEvidence }
        return (start, end)
    }
}
