import Foundation

/// Full transcript input, grouped by ten-minute start clocks. No ranked excerpts.
public struct TranscriptSummary: Equatable, Sendable, Encodable {
    public enum InputKind: String, Sendable {
        case cumulativeCaptions
        case tokenSequence
    }
    public struct Section: Equatable, Sendable, Identifiable, Encodable {
        public var id: Double { startSeconds }
        public let startSeconds: Double
        public let endSeconds: Double
        public let text: String

        public var startLabel: String {
            Duration.seconds(startSeconds).formatted(.time(pattern: .hourMinuteSecond))
        }

        public var endLabel: String {
            Duration.seconds(endSeconds).formatted(.time(pattern: .hourMinuteSecond))
        }
    }

    /// Intervals without transcript text are omitted, not interpreted as silence.
    public let sections: [Section]
    public static let intervalSeconds = 600.0

    /// Clocks are validated by the parsers. A boundary-spanning utterance belongs
    /// to its starting interval. Only cumulative caption steps are collapsed.
    public init(cues: [TranscriptCue], inputKind: InputKind = .cumulativeCaptions) {
        let sorted = cues.sorted { $0.startSeconds < $1.startSeconds }
        let utterances = inputKind == .tokenSequence ? sorted : CueTimeline.utterances(from: sorted)
        let groups = Dictionary(grouping: utterances) {
            floor($0.startSeconds / Self.intervalSeconds) * Self.intervalSeconds
        }
        sections = groups.sorted { $0.key < $1.key }.compactMap { start, cues in
            let text =
                (inputKind == .tokenSequence
                ? Self.tokenText(cues) : cues.map(\.text).joined(separator: "\n"))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            return Section(
                startSeconds: start, endSeconds: start + Self.intervalSeconds, text: text)
        }
    }

    /// Preserve producer token order and repeats. A gap beyond the existing
    /// utterance continuity limit adds a boundary, never a word-internal newline.
    private static func tokenText(_ cues: [TranscriptCue]) -> String {
        TranscriptTokenLayout(cues: cues).text
    }
}

/// Supported rates are shared by the picker and AVPlayer's default/current rate.
public enum PlaybackSpeed: Float, CaseIterable, Sendable {
    case normal = 1
    case oneAndHalf = 1.5
    case double = 2

    public var label: String {
        switch self {
        case .normal: "1.0x"
        case .oneAndHalf: "1.5x"
        case .double: "2.0x"
        }
    }
}
