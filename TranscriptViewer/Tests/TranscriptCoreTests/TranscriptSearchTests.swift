import Foundation
import Testing

@testable import TranscriptCore

struct TranscriptSearchTests {
    private let first = VideoTranscript(
        title: "2026-09-14_a",
        movieURL: URL(fileURLWithPath: "/tmp/a.mov"),
        transcriptURL: URL(fileURLWithPath: "/tmp/a.csv"),
        cues: [
            TranscriptCue(
                id: 0,
                startSeconds: 1,
                endSeconds: 2,
                startLabel: "00:00:01.000",
                endLabel: "00:00:02.000",
                text: "大丈夫そう"
            ),
            TranscriptCue(
                id: 1,
                startSeconds: 3,
                endSeconds: 4,
                startLabel: "00:00:03.000",
                endLabel: "00:00:04.000",
                text: "ありがとう"
            ),
        ]
    )
    private let second = VideoTranscript(
        title: "2026-09-15_b",
        movieURL: URL(fileURLWithPath: "/tmp/b.mov"),
        transcriptURL: URL(fileURLWithPath: "/tmp/b.csv"),
        cues: [
            TranscriptCue(
                id: 0,
                startSeconds: 8,
                endSeconds: 9,
                startLabel: "00:00:08.000",
                endLabel: "00:00:09.000",
                text: "大丈夫かな"
            )
        ]
    )

    @Test func emptyQueryIsEmpty() {
        #expect(
            TranscriptSearch.hits(
                query: "  ",
                videos: [first, second],
                scope: .allVideos,
                currentVideoID: first.id
            ) == []
        )
    }

    @Test func currentVideoOnly() {
        let hits = TranscriptSearch.hits(
            query: "大丈夫",
            videos: [first, second],
            scope: .currentVideo,
            currentVideoID: first.id
        )
        #expect(hits.count == 1)
        #expect(hits[0].cue.text == "大丈夫そう")
        #expect(hits[0].videoTitle == "2026-09-14_a")
        #expect(hits[0].id.hasSuffix("#0"))
    }

    @Test func allVideos() {
        let hits = TranscriptSearch.hits(
            query: "大丈夫",
            videos: [first, second],
            scope: .allVideos,
            currentVideoID: first.id
        )
        #expect(hits.count == 2)
        #expect(hits[0].cue.startSeconds == 1)
        #expect(hits[1].cue.startSeconds == 8)
    }

    @Test func noMatch() {
        #expect(
            TranscriptSearch.hits(
                query: "zzz",
                videos: [first, second],
                scope: .allVideos,
                currentVideoID: nil
            ) == []
        )
    }

    @Test func backgroundSearchPreservesUnicodeMatchesAndScope() async {
        let hits = await TranscriptSearch.hitsAsync(
            query: "大丈夫", videos: [first, second], scope: .allVideos, currentVideoID: first.id)
        #expect(hits.map(\.cue.text) == ["大丈夫そう", "大丈夫かな"])
        let current = await TranscriptSearch.hitsAsync(
            query: "大丈夫", videos: [first, second], scope: .currentVideo,
            currentVideoID: second.id)
        #expect(current.map(\.cue.text) == ["大丈夫かな"])
    }

    @Test func cancelledSearchSkipsTheQueuedScan() async {
        let (gate, continuation) = AsyncStream<Void>.makeStream()
        let task = Task {
            for await _ in gate { break }
            return await TranscriptSearch.hitsAsync(
                query: "大丈夫", videos: [first, second], scope: .allVideos,
                currentVideoID: nil)
        }
        task.cancel()
        continuation.yield(())
        continuation.finish()
        #expect(await task.value.isEmpty)
    }
}

struct CueDurationTests {
    @Test func emptyDurationIsZero() {
        #expect(CueTimeline.duration(of: []) == 0)
    }

    @Test func usesLastEnd() {
        let cues = [
            TranscriptCue(
                id: 0, startSeconds: 1, endSeconds: 2, startLabel: "1", endLabel: "2", text: "a"),
            TranscriptCue(
                id: 1, startSeconds: 5, endSeconds: 9, startLabel: "5", endLabel: "9", text: "b"),
        ]
        #expect(CueTimeline.duration(of: cues) == 9)
    }
}

struct TranscriptErrorTests {
    @Test func japaneseDescriptions() {
        #expect(TranscriptError.missingHeader.errorDescription == "CSVの見出しが不正です")
        #expect(TranscriptError.malformedRow(2).errorDescription == "CSVの2行目が不正です")
        #expect(TranscriptError.invalidClock("x").errorDescription == "時刻が不正です: x")
        #expect(TranscriptError.invalidClock("0.000").errorDescription == "時刻が不正です: 0.000")
        #expect(TranscriptError.unreadableDirectory("/tmp").errorDescription == "フォルダを読めません: /tmp")
    }
}
