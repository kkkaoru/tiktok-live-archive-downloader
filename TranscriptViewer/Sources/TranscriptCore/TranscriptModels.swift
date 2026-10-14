import Foundation

public struct TranscriptCue: Equatable, Sendable, Identifiable {
    public var id: Int
    public var startSeconds: Double
    public var endSeconds: Double
    public var startLabel: String
    public var endLabel: String
    public var text: String

    public init(
        id: Int,
        startSeconds: Double,
        endSeconds: Double,
        startLabel: String,
        endLabel: String,
        text: String
    ) {
        self.id = id
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.startLabel = startLabel
        self.endLabel = endLabel
        self.text = text
    }
}

public struct VideoTranscript: Equatable, Sendable, Identifiable {
    public var id: URL { movieURL }
    public var title: String
    public var movieURL: URL
    public var transcriptURL: URL
    /// Validated JST start from source evidence or a provenance-bound canonical delivery.
    public var broadcastStartJST: String?
    /// Set only after the explicit candidate selection's provenance and SRT checks pass.
    public let isUnverifiedCandidate: Bool
    public var recordingID: String? { RecordingIdentity(movie: movieURL)?.recordingID }
    public var cues: [TranscriptCue] {
        didSet { summary = TranscriptSummary(cues: cues, inputKind: summaryInputKind) }
    }
    public let summaryInputKind: TranscriptSummary.InputKind
    public private(set) var summary: TranscriptSummary

    public init(
        title: String, movieURL: URL, transcriptURL: URL, cues: [TranscriptCue],
        broadcastStartJST: String? = nil,
        summaryInputKind: TranscriptSummary.InputKind = .cumulativeCaptions,
        isUnverifiedCandidate: Bool = false
    ) {
        self.title = title
        self.movieURL = movieURL
        self.transcriptURL = transcriptURL
        self.broadcastStartJST = broadcastStartJST
        self.isUnverifiedCandidate = isUnverifiedCandidate
        self.cues = cues
        self.summaryInputKind = summaryInputKind
        self.summary = TranscriptSummary(cues: cues, inputKind: summaryInputKind)
    }
}

public enum TranscriptError: Error, Equatable, Sendable, LocalizedError {
    case missingHeader
    case malformedRow(Int)
    case malformedSRT(Int)
    case invalidClock(String)
    case unreadableDirectory(String)

    public var errorDescription: String? { message }

    /// Non-optional form for callers that always need a reason string.
    public var message: String {
        switch self {
        case .missingHeader:
            "CSVの見出しが不正です"
        case .malformedRow(let row):
            "CSVの\(row)行目が不正です"
        case .malformedSRT(let block):
            "字幕の\(block)ブロックが不正です"
        case .invalidClock(let value):
            "時刻が不正です: \(value)"
        case .unreadableDirectory(let path):
            "フォルダを読めません: \(path)"
        }
    }
}
