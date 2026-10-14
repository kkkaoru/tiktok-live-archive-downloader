import Testing

@testable import TranscriptCore

struct CueTimelineTests {
    private let cues = [
        TranscriptCue(
            id: 0, startSeconds: 1, endSeconds: 2, startLabel: "1", endLabel: "2", text: "a"),
        TranscriptCue(
            id: 1, startSeconds: 2, endSeconds: 4, startLabel: "2", endLabel: "4", text: "b"),
        TranscriptCue(
            id: 2, startSeconds: 5, endSeconds: 6, startLabel: "5", endLabel: "6", text: "c"),
    ]

    @Test func beforeFirstIsNil() {
        #expect(CueTimeline.index(at: 0.5, in: cues) == nil)
    }

    @Test func insideFirst() {
        #expect(CueTimeline.index(at: 1.0, in: cues) == 0)
        #expect(CueTimeline.index(at: 1.9, in: cues) == 0)
    }

    @Test func insideSecond() {
        #expect(CueTimeline.index(at: 2.0, in: cues) == 1)
        #expect(CueTimeline.index(at: 3.5, in: cues) == 1)
    }

    @Test func gapIsNil() {
        #expect(CueTimeline.index(at: 4.5, in: cues) == nil)
    }

    @Test func lastInclusiveEnd() {
        #expect(CueTimeline.index(at: 6.0, in: cues) == 2)
    }

    @Test func afterLastIsNil() {
        #expect(CueTimeline.index(at: 6.1, in: cues) == nil)
    }

    @Test func emptyAndNonFinite() {
        #expect(CueTimeline.index(at: 1, in: []) == nil)
        #expect(CueTimeline.index(at: .nan, in: cues) == nil)
    }

    @Test func windowAroundMiddle() {
        let slice = CueTimeline.window(around: 1, in: cues, radius: 1)
        #expect(slice.count == 3)
        #expect(slice[0].text == "a")
        #expect(slice[1].text == "b")
        #expect(slice[2].text == "c")
    }

    @Test func windowEmptyAndClamped() {
        #expect(CueTimeline.window(around: 0, in: [], radius: 2) == [])
        #expect(CueTimeline.window(around: 0, in: cues, radius: -1) == [])
        let first = CueTimeline.window(around: nil, in: cues, radius: 0)
        #expect(first.count == 1)
        #expect(first[0].text == "a")
        let clamped = CueTimeline.window(around: 99, in: cues, radius: 1)
        #expect(clamped.count == 2)
        #expect(clamped[0].text == "b")
        #expect(clamped[1].text == "c")
    }
}
