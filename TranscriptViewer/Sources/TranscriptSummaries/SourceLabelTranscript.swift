import Foundation
import TranscriptCore

/// Explicit source-clock input for summaries before an edited movie is published.
/// This does not register a Viewer candidate or transform recognition timestamps.
public struct SourceLabelTranscript: Sendable {
    private struct Document: Decodable {
        struct Label: Decodable {
            let start: Double
            let end: Double
            let text: String
        }
        let labels: [Label]
    }

    public let labels: URL
    public let media: URL
    public let inputKind: TranscriptSummary.InputKind
    static let maximumBytes = 64 * 1024 * 1024
    // Leave headroom for rounded section ends before Duration's Int64-second boundary.
    static let maximumClockSeconds = Double(Int64.max / 2)

    public init(
        labels: URL, media: URL, clockDomain: String,
        inputKind: TranscriptSummary.InputKind = .cumulativeCaptions
    ) throws {
        guard clockDomain == "source-presentation" else { throw SummaryCommandError.arguments }
        self.labels = labels
        self.media = media
        self.inputKind = inputKind
    }

    public func load() throws -> VideoTranscript {
        for url in [labels, media] {
            guard url.isFileURL, url.path == url.standardizedFileURL.path,
                url.path == url.resolvingSymlinksInPath().path,
                !url.pathComponents.contains(where: { ["private", "downloads"].contains($0) }),
                try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true
            else { throw SummaryStoreError.unsafePath }
        }
        let handle = try FileHandle(forReadingFrom: labels)
        let read = Result { try handle.read(upToCount: Self.maximumBytes + 1) ?? Data() }
        try handle.close()
        let data = try read.get()
        guard data.count <= Self.maximumBytes else { throw SummaryStoreError.oversizedRecord }
        let document = try JSONDecoder().decode(Document.self, from: data)
        guard !document.labels.isEmpty else { throw SummaryGenerationError.emptyTranscript }
        var previousStart = 0.0
        var cues: [TranscriptCue] = []
        for label in document.labels {
            guard label.start.isFinite, label.end.isFinite, label.start >= previousStart,
                label.end >= label.start, label.end <= Self.maximumClockSeconds,
                !label.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { throw SummaryStoreError.invalidRecord }
            cues.append(
                TranscriptCue(
                    id: cues.count, startSeconds: label.start, endSeconds: label.end,
                    startLabel: String(label.start), endLabel: String(label.end), text: label.text))
            previousStart = label.start
        }
        return VideoTranscript(
            title: media.deletingPathExtension().lastPathComponent,
            movieURL: media, transcriptURL: labels, cues: cues, summaryInputKind: inputKind)
    }
}
