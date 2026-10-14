import Foundation
import Synchronization

public enum SearchScope: Equatable, Sendable {
    case currentVideo
    case allVideos
}

public struct SearchHit: Equatable, Sendable, Identifiable {
    public var id: String { "\(videoID.path(percentEncoded: false))#\(cue.id)" }
    public var videoID: URL
    public var videoTitle: String
    public var cue: TranscriptCue

    public init(videoID: URL, videoTitle: String, cue: TranscriptCue) {
        self.videoID = videoID
        self.videoTitle = videoTitle
        self.cue = cue
    }
}

public enum TranscriptSearch {
    private static let searchQueue = DispatchQueue(label: "TranscriptSearch", qos: .userInitiated)

    /// Search on a bounded worker; cancelled queued searches skip their scan.
    public static func hitsAsync(
        query: String,
        videos: [VideoTranscript],
        scope: SearchScope,
        currentVideoID: URL?
    ) async -> [SearchHit] {
        let cancelled = Mutex(false)
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                searchQueue.async {
                    guard !cancelled.withLock({ $0 }) else {
                        continuation.resume(returning: [])
                        return
                    }
                    // ponytail: an in-flight scan finishes; add chunk cancellation if it grows costly.
                    continuation.resume(
                        returning: hits(
                            query: query, videos: videos, scope: scope,
                            currentVideoID: currentVideoID))
                }
            }
        } onCancel: {
            cancelled.withLock { $0 = true }
        }
    }

    public static func hits(
        query: String,
        videos: [VideoTranscript],
        scope: SearchScope,
        currentVideoID: URL?
    ) -> [SearchHit] {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty else {
            return []
        }
        let selected: [VideoTranscript]
        switch scope {
        case .allVideos:
            selected = videos
        case .currentVideo:
            selected = videos.filter { $0.id == currentVideoID }
        }
        var results: [SearchHit] = []
        for video in selected {
            for cue in CueTimeline.utterances(from: video.cues)
            where cue.text.localizedStandardContains(needle) {
                results.append(SearchHit(videoID: video.id, videoTitle: video.title, cue: cue))
            }
        }
        return results
    }
}
