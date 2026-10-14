import Foundation
import Testing

@testable import TranscriptCore
@testable import TranscriptSummaries

struct SourceTopicSummaryTests {
    private func cue(_ start: Double, _ end: Double, _ text: String) -> TranscriptCue {
        .init(
            id: Int(start * 100), startSeconds: start, endSeconds: end,
            startLabel: "", endLabel: "", text: text)
    }

    private func video(_ cues: [TranscriptCue]) -> VideoTranscript {
        .init(
            title: "Synthetic", movieURL: URL(fileURLWithPath: "/synthetic.mov"),
            transcriptURL: URL(fileURLWithPath: "/synthetic.json"), cues: cues,
            summaryInputKind: .tokenSequence)
    }

    @Test func exactKeywordTextAndOriginalRecognitionClockArePreserved() throws {
        let input = video([cue(1, 2, "旅行"), cue(2, 2.2, "の"), cue(2.2, 3, "旅行。")])
        let result = try SourceTopicSummary.generate(
            video: input, fingerprint: "pin",
            clockDomain: .editedOutput)
        #expect(result.label == "キーワードと発話抜粋による要約・未検証")
        #expect(result.method == "source-keywords-excerpts-v2")
        #expect(result.sections.count == 1)
        #expect(result.sections.map(\.id) == [0])
        #expect(result.sections.first?.excerpts.first?.span.text == "旅行の旅行。")
        #expect(
            result.sections.first?.keywords == [
                .init(
                    sectionStartSeconds: 0, lowerCharacter: 0, upperCharacter: 2,
                    text: "旅行", startSeconds: 1, endSeconds: 2)
            ])
        #expect(result.overview == result.sections.first?.keywords)
        try result.validate(video: input, fingerprint: "pin", clockDomain: .editedOutput)
    }

    @Test func coarseTokenEnvelopeIsNotInterpolatedToKeywordCharacters() throws {
        let input = video([cue(17.25, 22.75, "旅行の交通費と映画について。")])
        let result = try SourceTopicSummary.generate(
            video: input, fingerprint: "pin",
            clockDomain: .sourcePresentation)
        let keyword = try #require(result.sections.first?.keywords.first)
        #expect(keyword.text == "旅行")
        #expect(keyword.startSeconds == 17.25)
        #expect(keyword.endSeconds == 22.75)
        #expect(result.representative.clocks.first?.span.text == "旅行の交通費と映画について。")
    }

    @Test func frequentWordsCountDistinctExcerptsAndKeepTheirOriginalSpelling() throws {
        let input = video([
            cue(1, 2, "Travelについて。"), cue(201, 202, "Travelとbudgetについて。"),
            cue(401, 402, "Travelとbudgetとplanningについて。"),
            cue(601, 602, "料理の献立。"),
        ])
        let result = try SourceTopicSummary.generate(
            video: input, fingerprint: "pin",
            clockDomain: .editedOutput)
        #expect(result.sections.count == 2)
        #expect(result.sections.first?.keywords.first?.text == "Travel")
        #expect(result.sections.first?.keywords.count == 3)
        #expect(result.overview.count <= 6)
        #expect(result.sections.last?.startSeconds == 600)
        #expect(result.sections.last?.excerpts.first?.startSeconds == 601)
        #expect(
            try JSONDecoder().decode(SourceTopicSummary.self, from: JSONEncoder().encode(result))
                == result)
    }

    @Test func onlyFunctionFragmentsAndBareCountsHaveNoInventedTopic() throws {
        let input = video([cue(1, 2, "ありがとうはい12345。")])
        let result = try SourceTopicSummary.generate(
            video: input, fingerprint: "pin",
            clockDomain: .sourcePresentation)
        #expect(result.sections.first?.keywords.isEmpty == true)
        #expect(result.overview.isEmpty)
        #expect(result.sections.first?.excerpts.first?.span.text == "ありがとうはい12345。")
        try result.validate(video: input, fingerprint: "pin", clockDomain: .sourcePresentation)
    }

    @Test func equallyFrequentWordsUseLiteralLengthThenSourceOrder() throws {
        let input = video([cue(1, 2, "Travelのplanning。")])
        let result = try SourceTopicSummary.generate(
            video: input, fingerprint: "pin",
            clockDomain: .editedOutput)
        #expect(result.sections.first?.keywords.map(\.text) == ["planning", "Travel"])
        #expect(result.sections.first?.excerpts.first?.span.text == "Travelのplanning。")
    }

    @Test func broadDiscourseAndHiraganaFragmentsDoNotDisplaceDistinctSectionTerms() throws {
        let input = video([
            cue(1, 2, "TravelとAIなんてる。"), cue(601, 602, "Travel普通今日。"),
        ])
        let result = try SourceTopicSummary.generate(
            video: input, fingerprint: "pin",
            clockDomain: .editedOutput)
        #expect(result.sections.first?.keywords.map(\.text) == ["AI", "Travel"])
        #expect(result.sections.last?.keywords.map(\.text) == ["Travel"])
        #expect(result.sections.first?.excerpts.first?.span.text == "TravelとAIなんてる。")
        #expect(result.sections.last?.excerpts.first?.span.text == "Travel普通今日。")
    }

    @Test(arguments: ["label", "range", "clock", "overview", "method", "version"])
    func changedKeywordContractOrEvidenceIsRejected(_ field: String) throws {
        let input = video([cue(1, 2, "旅行の旅行。")])
        let original = try SourceTopicSummary.generate(
            video: input, fingerprint: "pin",
            clockDomain: .editedOutput)
        let keyword = try #require(original.overview.first)
        let altered = SourceTopicSummary.Keyword(
            sectionStartSeconds: 0,
            lowerCharacter: field == "range" ? 1 : keyword.lowerCharacter,
            upperCharacter: keyword.upperCharacter, text: keyword.text,
            startSeconds: field == "clock" ? 1.5 : keyword.startSeconds,
            endSeconds: keyword.endSeconds)
        let changed = SourceTopicSummary(
            version: field == "version" ? 2 : 1,
            method: field == "method" ? "unverified-alternative" : original.method,
            label: field == "label" ? "Approved" : original.label,
            representative: original.representative,
            overview: field == "overview" ? [] : [altered], sections: original.sections)
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try changed.validate(video: input, fingerprint: "pin", clockDomain: .editedOutput)
        }
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try original.validate(video: input, fingerprint: "changed", clockDomain: .editedOutput)
        }
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try original.validate(
                video: input, fingerprint: "pin", clockDomain: .sourcePresentation)
        }
    }

    @Test func topicsCLIUsesSharedInputAndVerifierWithoutNativeInference() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let labels = fixture.root.appending(path: "tokens.json")
        let output = fixture.root.appending(path: "topics.json")
        try Data(#"{"labels":[{"start":1,"end":2,"text":"旅行の計画。"}]}"#.utf8).write(to: labels)
        let args = [
            "--source-labels", labels.path, "--source-media", fixture.video.movieURL.path,
            "--clock-domain", "source-presentation", "--source-label-layout", "tokens",
            "--excerpts-output", output.path, "--excerpt-selection", "topics",
        ]
        let neverInfer: NarrativeSummaryGenerator.Respond = { _ in
            throw SourceExcerptError.invalidSelection
        }
        #expect(
            try await SummaryCommand(arguments: ["generate"] + args)
                .run(respond: neverInfer, output: { _ in }) == 0)
        #expect(
            try await SummaryCommand(arguments: ["inspect"] + args)
                .run(respond: neverInfer, output: { _ in }) == 0)
        let result = try JSONDecoder().decode(
            SourceTopicSummary.self, from: Data(contentsOf: output))
        #expect(result.sections.first?.keywords.first?.text == "旅行")
        #expect(!FileManager.default.fileExists(atPath: fixture.current.path))
    }
}
