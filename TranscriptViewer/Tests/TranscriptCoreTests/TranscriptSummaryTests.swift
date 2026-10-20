import Foundation
import Testing

@testable import TranscriptCore

struct TranscriptSummaryTests {
    private func cue(_ start: Double, _ text: String, end: Double? = nil) -> TranscriptCue {
        TranscriptCue(
            id: Int(start * 10), startSeconds: start, endSeconds: end ?? start + 1,
            startLabel: "", endLabel: "", text: text)
    }

    @Test func emptyTranscript() {
        #expect(TranscriptSummary(cues: []).sections.isEmpty)
        #expect(TranscriptSummary(cues: [cue(0, " \n ")]).sections.isEmpty)
    }

    @Test func tenMinuteBoundariesAndJumpClocks() {
        let summary = TranscriptSummary(cues: [
            cue(1200, "最後の話題"), cue(600, "次の話題"),
            cue(599, "境界をまたぐ発話", end: 602), cue(0, "最初の話題"),
        ])
        #expect(summary.sections.map(\.startSeconds) == [0, 600, 1200])
        #expect(summary.sections.map(\.endSeconds) == [600, 1200, 1800])
        #expect(summary.sections.map(\.id) == [0, 600, 1200])
        #expect(summary.sections[0].text == "最初の話題\n境界をまたぐ発話")
        #expect(summary.sections[1].text == "次の話題")
        #expect(summary.sections[1].startLabel == "0:10:00")
        #expect(summary.sections[1].endLabel == "0:20:00")
    }

    @Test func collapseGrowingCaptionsButKeepRepeatedUtterances() {
        let summary = TranscriptSummary(cues: [
            cue(0, "今日"), cue(1, "今日は旅行"), cue(2, "今日は旅行の計画です。"),
            cue(10, "今日は旅行の計画です。"),
        ])
        #expect(summary.sections[0].text == "今日は旅行の計画です。\n今日は旅行の計画です。")
    }

    @Test func sparseTranscriptDoesNotInventSilentTopics() {
        let summary = TranscriptSummary(cues: [cue(0, "開始"), cue(3600, "再開")])
        #expect(summary.sections.map(\.startSeconds) == [0, 3600])
        #expect(summary.sections[1].startLabel == "1:00:00")
    }

    @Test func noExtractiveRankingOrTruncation() {
        let cues = (0..<20).map { cue(Double($0 * 30), "話題\($0)について説明します。") }
        let summary = TranscriptSummary(cues: cues)
        #expect(summary.sections[0].text == cues.map(\.text).joined(separator: "\n"))
        #expect(summary == TranscriptSummary(cues: cues))
    }

    @Test func editedTranscriptRegeneratesInput() {
        var video = VideoTranscript(
            title: "Synthetic", movieURL: URL(fileURLWithPath: "/synthetic.mp4"),
            transcriptURL: URL(fileURLWithPath: "/synthetic.srt"), cues: [cue(0, "旧内容")])
        #expect(video.summary.sections[0].text == "旧内容")
        video.cues = [cue(600, "新内容")]
        #expect(video.summary.sections[0].text == "新内容")
        #expect(video.summary.sections[0].startSeconds == 600)
    }

    @Test func supportedSpeeds() {
        #expect(PlaybackSpeed.allCases.map(\.rawValue) == [0.5, 1, 1.5, 2])
        #expect(PlaybackSpeed.allCases.map(\.label) == ["0.5x", "1.0x", "1.5x", "2.0x"])
        #expect(PlaybackSpeed.allCases.map(\.next) == [.normal, .oneAndHalf, .double, .half])
    }
}
