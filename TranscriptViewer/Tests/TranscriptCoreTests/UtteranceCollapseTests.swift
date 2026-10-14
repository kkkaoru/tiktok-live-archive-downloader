import Foundation
import Testing

@testable import TranscriptCore

struct UtteranceCollapseTests {
    @Test func dropsPrefixStepsAndKeepsFirstStart() {
        let cues = [
            TranscriptCue(
                id: 0, startSeconds: 1, endSeconds: 1.4, startLabel: "00:00:01.000",
                endLabel: "00:00:01.400",
                text: "お水"),
            TranscriptCue(
                id: 1, startSeconds: 1.3, endSeconds: 1.8, startLabel: "00:00:01.300",
                endLabel: "00:00:01.800",
                text: "お水を"),
            TranscriptCue(
                id: 2, startSeconds: 1.6, endSeconds: 2.5, startLabel: "00:00:01.600",
                endLabel: "00:00:02.500",
                text: "お水を飲みました"),
        ]
        let utterances = CueTimeline.utterances(from: cues)
        #expect(utterances.count == 1)
        #expect(utterances[0].id == 0)
        #expect(utterances[0].startSeconds == 1)
        #expect(utterances[0].endSeconds == 2.5)
        #expect(utterances[0].startLabel == "00:00:01.000")
        #expect(utterances[0].text == "お水を飲みました")
    }

    @Test func splitsWhenTextIsNotAPrefix() {
        let cues = [
            TranscriptCue(
                id: 0, startSeconds: 1, endSeconds: 2, startLabel: "1", endLabel: "2",
                text: "お水を飲みました"),
            TranscriptCue(
                id: 1, startSeconds: 2.1, endSeconds: 3, startLabel: "2.1", endLabel: "3",
                text: "ありがとう"),
        ]
        let utterances = CueTimeline.utterances(from: cues)
        #expect(utterances.count == 2)
        #expect(utterances[0].text == "お水を飲みました")
        #expect(utterances[1].text == "ありがとう")
    }

    @Test func splitsAcrossLongGap() {
        let cues = [
            TranscriptCue(
                id: 0, startSeconds: 1, endSeconds: 2, startLabel: "1", endLabel: "2", text: "お水"),
            TranscriptCue(
                id: 1, startSeconds: 8, endSeconds: 9, startLabel: "8", endLabel: "9", text: "お水を"),
        ]
        let utterances = CueTimeline.utterances(from: cues)
        #expect(utterances.count == 2)
        #expect(utterances[0].text == "お水")
        #expect(utterances[1].text == "お水を")
    }

    @Test func emptyTextStartsNewUtterance() {
        let cues = [
            TranscriptCue(
                id: 0, startSeconds: 1, endSeconds: 2, startLabel: "1", endLabel: "2", text: ""),
            TranscriptCue(
                id: 1, startSeconds: 2, endSeconds: 3, startLabel: "2", endLabel: "3", text: "お水"),
        ]
        let utterances = CueTimeline.utterances(from: cues)
        #expect(utterances.count == 2)
        #expect(utterances[1].text == "お水")
    }
}

struct CollapsedSearchTests {
    @Test func searchSkipsPrefixSteps() {
        let video = VideoTranscript(
            title: "a",
            movieURL: URL(fileURLWithPath: "/tmp/a.mov"),
            transcriptURL: URL(fileURLWithPath: "/tmp/a.csv"),
            cues: [
                TranscriptCue(
                    id: 0, startSeconds: 1, endSeconds: 1.4, startLabel: "00:00:01.000",
                    endLabel: "00:00:01.400",
                    text: "お水"),
                TranscriptCue(
                    id: 1, startSeconds: 1.3, endSeconds: 1.8, startLabel: "00:00:01.300",
                    endLabel: "00:00:01.800",
                    text: "お水を"),
                TranscriptCue(
                    id: 2, startSeconds: 1.6, endSeconds: 2.5, startLabel: "00:00:01.600",
                    endLabel: "00:00:02.500",
                    text: "お水を飲みました"),
            ]
        )
        let hits = TranscriptSearch.hits(
            query: "お水",
            videos: [video],
            scope: .allVideos,
            currentVideoID: video.id
        )
        #expect(hits.count == 1)
        #expect(hits[0].cue.text == "お水を飲みました")
        #expect(hits[0].cue.startSeconds == 1)
        #expect(hits[0].cue.startLabel == "00:00:01.000")
    }
}
