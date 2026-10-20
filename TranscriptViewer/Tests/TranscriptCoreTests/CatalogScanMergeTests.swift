import Foundation
import Testing

@testable import TranscriptCore

struct CatalogScanMergeTests {
    private func video(_ name: String, start: String?) -> VideoTranscript {
        VideoTranscript(
            title: name, movieURL: URL(fileURLWithPath: "/library/\(name).mp4"),
            transcriptURL: URL(fileURLWithPath: "/library/\(name).srt"), cues: [],
            broadcastStartJST: start)
    }

    @Test func mergeKeepsOneCopyOfAMovieAndSortsNewestFirst() {
        var first = CatalogScan()
        first.videos = [video("old", start: "2026-09-01 18:00:00 JST")]
        var second = CatalogScan()
        second.videos = [
            video("old", start: "2026-09-01 18:00:00 JST"),
            video("new", start: "2026-10-06 18:30:48 JST"),
            video("undated", start: nil),
        ]
        first.merge(second)
        #expect(first.videos.map(\.title) == ["new", "old", "undated"])
    }

    @Test func mergeCombinesEveryKindOfIssue() {
        var first = CatalogScan()
        first.pendingMovieLabels = ["a"]
        first.emptyFolders = ["e1"]
        var second = CatalogScan()
        second.pendingMovieLabels = ["b"]
        second.moviesWithoutTranscript = [URL(fileURLWithPath: "/library/x.mp4")]
        second.unreadableTranscriptLabels = ["u"]
        second.unknownBroadcastDateLabels = ["d"]
        second.ambiguousDeliveryLabels = ["m"]
        second.invalidSupersededRecordLabels = ["s"]
        second.invalidCandidateSelectionLabels = ["c"]
        second.emptyFolders = ["e2"]
        first.merge(second)
        #expect(first.pendingMovieLabels == ["a", "b"])
        #expect(first.emptyFolders == ["e1", "e2"])
        #expect(first.issues.count == 10)
    }

    @Test func aMoviePairedByAnotherLibraryIsNotReportedWithoutTranscript() {
        var store = CatalogScan()
        store.moviesWithoutTranscript = [
            URL(fileURLWithPath: "/library/old.mp4"), URL(fileURLWithPath: "/library/lone.mp4"),
        ]
        var edits = CatalogScan()
        edits.videos = [video("old", start: "2026-09-01 18:00:00 JST")]
        var reversed = edits
        edits.merge(store)
        store.merge(reversed)
        reversed = store
        #expect(edits.moviesWithoutTranscript.map(\.lastPathComponent) == ["lone.mp4"])
        #expect(reversed.moviesWithoutTranscript.map(\.lastPathComponent) == ["lone.mp4"])
    }

    private func recordingVideo(_ revision: String, date: String, candidate: Bool)
        -> VideoTranscript
    {
        VideoTranscript(
            title: revision,
            movieURL: URL(
                fileURLWithPath: "/library/\(revision)/\(date)_JST_replay-123-review.mp4"),
            transcriptURL: URL(fileURLWithPath: "/library/\(revision)/captions.srt"), cues: [],
            broadcastStartJST: date == "2026-10-04_14-30-09"
                ? "2026-10-04 14:30:09 JST" : "2026-10-05 14:30:09 JST",
            isUnverifiedCandidate: candidate)
    }

    @Test func currentCandidateReplacesOlderRecordingWhileUnrelatedMoviesStay() {
        var first = CatalogScan()
        first.videos = [
            recordingVideo("old", date: "2026-10-04_14-30-09", candidate: true),
            video("unrelated", start: nil),
        ]
        var second = CatalogScan()
        second.videos = [recordingVideo("current", date: "2026-10-04_14-30-09", candidate: true)]
        first.merge(second)
        #expect(first.videos.map(\.title) == ["current", "unrelated"])
        #expect(first.videos.first?.recordingID == "replay-123")
        #expect(first.videos.first?.isUnverifiedCandidate == true)
        #expect(first.issues.isEmpty)
    }

    @Test func conflictingCandidateDateDoesNotReplaceExistingRecording() {
        var first = CatalogScan()
        first.videos = [recordingVideo("old", date: "2026-10-04_14-30-09", candidate: false)]
        var second = CatalogScan()
        second.videos = [recordingVideo("conflict", date: "2026-10-05_14-30-09", candidate: true)]
        first.merge(second)
        #expect(first.videos.map(\.title) == ["old"])
        #expect(first.ambiguousDeliveryLabels == ["replay-123"])
    }

    @Test func unboundCandidateDoesNotInventARecordingIdentity() {
        var first = CatalogScan()
        var second = CatalogScan()
        var unknown = video("unknown", start: nil)
        unknown = VideoTranscript(
            title: unknown.title, movieURL: unknown.movieURL, transcriptURL: unknown.transcriptURL,
            cues: [], isUnverifiedCandidate: true)
        second.videos = [unknown]
        first.merge(second)
        #expect(first.videos.count == 1)
        #expect(first.videos.first?.recordingID == nil)
    }
}
