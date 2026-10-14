import Foundation
import Testing

@testable import TranscriptCore
@testable import TranscriptSummaries

struct SourceExcerptSummaryTests {
    private func source(_ texts: [String]) -> TranscriptSummary {
        TranscriptSummary(
            cues: texts.enumerated().map { index, text in
                TranscriptCue(
                    id: index, startSeconds: Double(index * 600),
                    endSeconds: Double(index * 600 + 1), startLabel: "", endLabel: "",
                    text: text)
            }, inputKind: .tokenSequence)
    }

    @Test func exactUnicodeRangesAndOverviewNeverUseGeneratedProse() async throws {
        let input = source([
            "やばいやばい。フォロワーさん。Aは高い。Bは低い。猫型ではない。",
            "原文だけ。", "次。", "最後。",
        ])
        let result = try await SourceExcerptGenerator { request in
            #expect(request.selectsSource)
            #expect(request.modelInstructions.contains("JSON"))
            #expect(request.prompt.contains("ID"))
            #expect(request.source.utf8.count <= SummaryRequest.inputByteLimit)
            return "[0]"
        }.generate(input, fingerprint: "pin")
        try result.validate(source: input, fingerprint: "pin")
        #expect(result.sections.count == 4)
        #expect(result.sections[0].excerpts.first?.text == "やばいやばい。")
        #expect(!result.overview.map(\.text).joined().contains("という猫型ロボット"))
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try result.validate(source: input, fingerprint: "other")
        }
        let changed = SourceExcerptSummary(
            mode: "invented", fingerprint: result.fingerprint,
            sections: result.sections, overview: result.overview)
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try changed.validate(source: input, fingerprint: "pin")
        }
    }

    @Test(arguments: ["[9]", "[-1]", "[]", "[0,0]", "[0.5]", "猫型ロボットです。", "{\"id\":0}"])
    func rejectsAnythingExceptOneExistingInteger(raw: String) async {
        await #expect(throws: SourceExcerptError.invalidSelection) {
            try await SourceExcerptGenerator { _ in raw }.generate(
                source(["一。二。三。四。"]), fingerprint: "pin")
        }
    }

    @Test func longTextAndBlankBoundaries() async throws {
        let input = source(["\n" + String(repeating: "文章を読みます ", count: 300) + "\n\n末尾"])
        let result = try await SourceExcerptGenerator { _ in "[1]" }.generate(
            input, fingerprint: "p")
        try result.validate(source: input, fingerprint: "p")
        #expect(result.sections[0].excerpts.allSatisfy { $0.text.utf8.count <= 600 })
        await #expect(throws: SummaryGenerationError.emptyTranscript) {
            try await SourceExcerptGenerator { _ in "[0]" }.generate(source([]), fingerprint: "p")
        }
        await #expect(throws: SummaryGenerationError.oversizedCharacter) {
            try await SourceExcerptGenerator { _ in "[0]" }.generate(
                source(["a" + String(repeating: "\u{0301}", count: 601)]), fingerprint: "p")
        }
    }

    @Test func validatesClockTextRangeAndOverviewMembership() async throws {
        let input = source(["原文。"])
        let good = try await SourceExcerptGenerator { _ in "[0]" }.generate(input, fingerprint: "p")
        let badSpans: [SourceExcerptSummary.Span] = [
            .init(sectionStartSeconds: 1, lowerCharacter: 0, upperCharacter: 3, text: "原文。"),
            .init(sectionStartSeconds: 0, lowerCharacter: -1, upperCharacter: 3, text: "原文。"),
            .init(sectionStartSeconds: 0, lowerCharacter: 0, upperCharacter: 4, text: "原文。"),
            .init(sectionStartSeconds: 0, lowerCharacter: 0, upperCharacter: 3, text: "新事実"),
        ]
        for span in badSpans {
            let bad = SourceExcerptSummary(
                mode: good.mode, fingerprint: "p",
                sections: [
                    .init(startSeconds: 0, endSeconds: 600, excerpts: [span])
                ], overview: [span])
            #expect(throws: SourceExcerptError.invalidEvidence) {
                try bad.validate(source: input, fingerprint: "p")
            }
            let badOverview = SourceExcerptSummary(
                mode: good.mode, fingerprint: "p",
                sections: good.sections, overview: [span])
            #expect(throws: SourceExcerptError.invalidEvidence) {
                try badOverview.validate(source: input, fingerprint: "p")
            }
        }
        let badClock = SourceExcerptSummary(
            mode: good.mode, fingerprint: "p",
            sections: [
                .init(startSeconds: 0, endSeconds: 599, excerpts: good.overview)
            ], overview: good.overview)
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try badClock.validate(source: input, fingerprint: "p")
        }
    }

    @Test func commandWritesSeparateArtifactAndInspectsWithoutInference() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let labels = fixture.root.appendingPathComponent("labels.json")
        let output = fixture.root.appendingPathComponent("excerpts.json")
        try Data(#"{"labels":[{"start":0.738,"end":1,"text":"原文。"}]}"#.utf8).write(to: labels)
        let args = [
            "--source-labels", labels.path, "--source-media", fixture.video.movieURL.path,
            "--clock-domain", "source-presentation", "--source-label-layout", "tokens",
            "--excerpts-output", output.path,
        ]
        #expect(
            try await SummaryCommand(arguments: ["generate"] + args).run(
                respond: { _ in "[0]" }, output: { _ in }) == 0)
        #expect(
            try await SummaryCommand(arguments: ["inspect"] + args).run(
                respond: { _ in throw SourceExcerptError.invalidSelection }, output: { _ in }) == 0)
        await #expect(throws: SummaryStoreError.invalidRecord) {
            try await SummaryCommand(arguments: ["generate"] + args).run(output: { _ in })
        }
        #expect(throws: SummaryCommandError.arguments) {
            try SummaryCommand(arguments: ["generate", "--force"] + args)
        }
    }

    @Test func commandTracesSelectionsAndRejectsUnsafeStorage() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let labels = fixture.root.appendingPathComponent("labels.json")
        let output = fixture.root.appendingPathComponent("excerpts.json")
        try Data(#"{"labels":[{"start":0,"end":1,"text":"一。二。三。四。"}]}"#.utf8)
            .write(to: labels)
        let base = [
            "--source-labels", labels.path, "--source-media", fixture.video.movieURL.path,
            "--clock-domain", "source-presentation",
        ]
        #expect(
            try await SummaryCommand(
                arguments: ["generate"] + base + [
                    "--excerpts-output", output.path, "--diagnostics",
                    fixture.root.appendingPathComponent("diagnostics").path,
                ]
            ).run(respond: { _ in "[0]" }, output: { _ in }) == 0)
        let link = fixture.root.appendingPathComponent("linked.json")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: output)
        await #expect(throws: SummaryStoreError.unsafePath) {
            try await SummaryCommand(
                arguments: ["inspect"] + base + ["--excerpts-output", link.path]
            )
            .run(output: { _ in })
        }
        let oversized = fixture.root.appendingPathComponent("oversized.json")
        try Data(repeating: 0, count: 8 * 1024 * 1024 + 1).write(to: oversized)
        #expect(
            try await SummaryCommand(
                arguments: ["inspect"] + base + [
                    "--excerpts-output", oversized.path,
                ]
            ).run(output: { _ in }) == 1)
    }
}
