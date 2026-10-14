import Foundation
import Testing

@testable import TranscriptCore
@testable import TranscriptSummaries

struct RepresentativeSourceSummaryTests {
    @Test func japaneseContentWordsAreActuallyAvailable() {
        let terms = RepresentativeSourceSummary.terms("旅行の予定を考えます。交通費を計算します。")
        #expect(terms.contains("旅行"))
        // Apple's word tokenizer separates this compound; selected source text is never split here.
        #expect(terms.contains("交通") && terms.contains("費"))
    }
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

    private func candidate(_ start: Double, _ text: String, _ terms: Set<String>)
        -> RepresentativeSourceSummary.Candidate
    {
        .init(
            span: .init(
                sectionStartSeconds: 0, lowerCharacter: 0, upperCharacter: text.count, text: text),
            start: start, end: start + 1, terms: terms)
    }

    @Test func exactWordsRepeatedTokensAndNonzeroClockEnvelopes() throws {
        let layout = TranscriptTokenLayout(cues: [
            cue(0.738, 1, " "), cue(1, 1.1, "フォ"), cue(1.1, 1.2, "ロワー"),
            cue(1.2, 1.3, "さん"), cue(1.3, 1.4, "やばい"), cue(1.4, 1.5, "やばい"), cue(1.5, 1.6, " "),
        ])
        #expect(layout.text == "フォロワーさんやばいやばい")
        let clock = try layout.clock(lower: 0, upper: 7)
        #expect(clock.start == 1 && clock.end == 1.3)
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try layout.clock(lower: -1, upper: 2)
        }
        #expect(throws: SourceExcerptError.invalidEvidence) { try layout.clock(lower: 1, upper: 1) }
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try layout.clock(lower: 0, upper: 100)
        }
        let empty = TranscriptTokenLayout(cues: [cue(0, 1, " \n")])
        #expect(empty.text.isEmpty)
        #expect(throws: SourceExcerptError.invalidEvidence) { try empty.clock(lower: 0, upper: 1) }
    }

    @Test func repeatedGreetingsDoNotDisplaceMiddleAndEndTopics() {
        let greeting = (0..<40).map { candidate(Double($0), "こんにちは！", []) }
        let selected = RepresentativeSourceSummary.select(
            greeting + [
                candidate(200, "旅行の交通費を計算します。", ["旅行", "交通費", "計算"]),
                candidate(400, "仕事の予定を考えます。", ["仕事", "予定", "考える"]),
            ])
        #expect(selected.map(\.start) == [200, 400])
        #expect(RepresentativeSourceSummary.select([]).isEmpty)
        #expect(RepresentativeSourceSummary.select(greeting).count == 1)
    }

    @Test func timeCoverageAndLexicalNoveltyHaveDeterministicTies() {
        let selected = RepresentativeSourceSummary.select([
            candidate(0, "旅行の計画。", ["旅行", "計画"]),
            candidate(100, "旅行の準備。", ["旅行", "準備"]),
            candidate(250, "旅行の続き。", ["旅行", "計画"]),
            candidate(260, "仕事の相談。", ["仕事", "相談"]),
            candidate(500, "食事の説明。", ["食事", "説明"]),
        ])
        #expect(selected.map(\.start) == [0, 260, 500])
    }

    @Test func sectionsQuotesAndCrossBoundaryTokenClocksStayExact() throws {
        let input = video([
            cue(0.738, 2, "旅行の予定を考えます。"),
            cue(200, 202, "交通費について相談します。"),
            cue(599, 602, "仕事の日程は未定です。"),
            cue(610, 612, "別の日の食事について話します。"),
        ])
        let result = try RepresentativeSourceSummary.generate(
            video: input, fingerprint: "p", clockDomain: .editedOutput)
        try result.validate(video: input, fingerprint: "p", clockDomain: .editedOutput)
        #expect(result.label == "発話抜粋による要約・未検証")
        #expect(result.source.sections.map(\.startSeconds) == [0, 600])
        #expect(result.clocks.contains { $0.startSeconds == 599 && $0.endSeconds == 602 })
        #expect(result.source.sections[0].excerpts.count == 3)
        #expect(result.source.sections[0].excerpts.contains { $0.text.contains("交通費") })
        #expect(result.source.sections[1].excerpts.count == 1)
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try result.validate(video: input, fingerprint: "changed", clockDomain: .editedOutput)
        }
        let first = try #require(result.clocks.first)
        let bad = RepresentativeSourceSummary(
            version: 1, method: result.method, label: result.label, clockDomain: .editedOutput,
            source: result.source,
            clocks: [.init(span: first.span, startSeconds: 1, endSeconds: 2)]
                + result.clocks.dropFirst())
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try bad.validate(video: input, fingerprint: "p", clockDomain: .editedOutput)
        }
        let missing = RepresentativeSourceSummary(
            version: 1, method: result.method, label: result.label, clockDomain: .editedOutput,
            source: result.source, clocks: [])
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try missing.validate(video: input, fingerprint: "p", clockDomain: .editedOutput)
        }
        let mislabeled = RepresentativeSourceSummary(
            version: 1, method: result.method, label: "Approved", clockDomain: .editedOutput,
            source: result.source, clocks: result.clocks)
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try mislabeled.validate(video: input, fingerprint: "p", clockDomain: .editedOutput)
        }
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try result.validate(video: input, fingerprint: "p", clockDomain: .sourcePresentation)
        }
    }

    @Test func missingAndDisplayStageInputsAreNotSilentlyReinterpreted() {
        #expect(throws: SummaryGenerationError.emptyTranscript) {
            try RepresentativeSourceSummary.generate(
                video: video([]), fingerprint: "p", clockDomain: .editedOutput)
        }
        let captions = VideoTranscript(
            title: "Display", movieURL: URL(fileURLWithPath: "/a.mov"),
            transcriptURL: URL(fileURLWithPath: "/a.srt"),
            cues: [cue(0, 1, "今日"), cue(1, 2, "今日は")])
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try RepresentativeSourceSummary.generate(
                video: captions, fingerprint: "p", clockDomain: .editedOutput)
        }
    }

    @Test func cliWritesAndVerifiesWithoutModelCalls() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let labels = fixture.root.appendingPathComponent("tokens.json")
        let output = fixture.root.appendingPathComponent("representative.json")
        try Data(
            #"{"labels":[{"start":0.738,"end":1,"text":"旅行の計画。"},{"start":210,"end":212,"text":"仕事の相談。"},{"start":500,"end":502,"text":"食事の説明。"}]}"#
                .utf8
        ).write(to: labels)
        let args = [
            "--source-labels", labels.path, "--source-media", fixture.video.movieURL.path,
            "--clock-domain", "source-presentation", "--source-label-layout", "tokens",
            "--excerpts-output", output.path, "--excerpt-selection", "representative",
        ]
        #expect(
            try await SummaryCommand(arguments: ["generate"] + args).run(
                respond: { _ in throw SourceExcerptError.invalidSelection }, output: { _ in }) == 0)
        #expect(
            try await SummaryCommand(arguments: ["inspect"] + args).run(
                respond: { _ in throw SourceExcerptError.invalidSelection }, output: { _ in }) == 0)
        let result = try JSONDecoder().decode(
            RepresentativeSourceSummary.self, from: Data(contentsOf: output))
        #expect(result.source.sections.count == 1 && result.clocks.count == 3)
        #expect(throws: SummaryCommandError.arguments) {
            try SummaryCommand(arguments: [
                "generate", "--library", fixture.root.path, "--excerpt-selection", "representative",
            ])
        }
        #expect(throws: SummaryCommandError.arguments) {
            try SummaryCommand(arguments: ["generate"] + Array(args.dropLast()) + ["unknown"])
        }
    }
}
