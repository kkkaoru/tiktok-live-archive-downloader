import Foundation

/// Exact identity from a canonical dated delivery, independent of folder labels.
struct RecordingIdentity: Equatable {
    private static let timestampLength = 19
    private static let maximumStemLength = 160
    private static let zoneMarker = "_JST_"
    private static let deliverySuffix = "-review"

    let recordingID: String
    let broadcastStartJST: String

    static func isRecordingID(_ value: String) -> Bool {
        value.range(
            of: #"^(replay-[0-9]{1,32}|youtube-[A-Za-z0-9_-]{11})$"#,
            options: .regularExpression) == value.startIndex..<value.endIndex
    }

    init?(movie: URL) {
        guard ["mp4", "mov", "m4v"].contains(movie.pathExtension) else { return nil }
        let stem = movie.deletingPathExtension().lastPathComponent
        guard stem.count <= Self.maximumStemLength, stem.hasSuffix(Self.deliverySuffix),
            stem.dropFirst(Self.timestampLength).hasPrefix(Self.zoneMarker)
        else { return nil }
        let recordingID = String(
            stem.dropFirst(Self.timestampLength + Self.zoneMarker.count)
                .dropLast(Self.deliverySuffix.count))
        guard Self.isRecordingID(recordingID),
            let start = TranscriptCatalog.jstLabel(stamp: String(stem.prefix(Self.timestampLength)))
        else { return nil }
        self.recordingID = recordingID
        broadcastStartJST = start
    }
}
