import Foundation
import Testing

@testable import TranscriptCore
@testable import TranscriptSummaries

struct SummaryQualityRepairTests {
    @Test(arguments: [
        "地域の催しでは参加者が料理の工夫を紹介している。地域の催しでは参加者が料理の工夫を紹介している。",
        "地域の催しでは参加者が料理の工夫を紹介している。\n\n地域の催しでは参加者が料理の工夫を紹介している。",
    ])
    func rejectsRepeatedSentences(_ text: String) {
        #expect(throws: SummaryGenerationError.repeatedResponse) {
            try NarrativeSummaryGenerator.validateResponse(text)
        }
    }

    @Test(arguments: [
        "この作業を完了できる可能性は高いとされていますが、この作業を完了できる可能性は低いとされています。",
        "この町で暮らす人の数は多いと説明されている。この町で暮らす人の数は少ないと説明されている。",
        "この設備を使った作業は可能です。この設備を使った作業は不可能です。",
    ])
    func rejectsIdenticalOpposingClaims(_ text: String) {
        #expect(throws: SummaryGenerationError.contradictoryResponse) {
            try SummaryProseQuality.validate(text)
        }
    }

    @Test(arguments: [
        "当初は成功する可能性が高いと言ったが、後で低いと訂正した。",
        "Aの成功する可能性は高い。Bの成功する可能性は低い。",
        "「成功する可能性は高い」という意見と「成功する可能性は低い」という意見がある。",
        "手順を説明している。作業に必要な材料を紹介している。",
        "はい。はい。",
    ])
    func preservesContextAndDistinctSubjects(_ text: String) throws {
        try SummaryProseQuality.validate(text)
    }

    @Test(arguments: [
        "-1", "nan", "inf", "1,1", "1,", "", "text",
    ])
    func rejectsInvalidRepairArguments(_ value: String) {
        #expect(throws: SummaryCommandError.self) {
            try SummaryCommand(arguments: [
                "generate", "--library", "x", "--repair-sections", value,
            ])
        }
    }

    @Test func repairConflictsAreExplicit() {
        #expect(throws: SummaryCommandError.self) {
            try SummaryCommand(arguments: ["inspect", "--library", "x", "--repair-sections", "0"])
        }
        #expect(throws: SummaryCommandError.self) {
            try SummaryCommand(arguments: [
                "generate", "--library", "x", "--force", "--repair-sections", "0",
            ])
        }
    }

    @Test func repairsOnlySelectedSectionAndOverview() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var video = fixture.video
        video.cues.append(
            TranscriptCue(
                id: 1, startSeconds: 601, endSeconds: 610, startLabel: "", endLabel: "",
                text: "午後は展示を見学した。"))
        let initial = AppleSummaryService(store: SummaryStore()) { _ in "午前は展示の準備について説明している。" }
        _ = try await initial.generate(video: video, force: false) { _, _ in }
        let before = try Data(contentsOf: fixture.current)
        let fingerprint = try SummaryStore.fingerprint(video)
        let recorder = SummaryRecorder()
        let repairing = AppleSummaryService(store: SummaryStore()) { await recorder.respond($0) }
        let result = try await repairing.generate(video: video, force: false, repairing: [600]) {
            _, _ in
        }
        #expect(result.sections[0].text == "午前は展示の準備について説明している。")
        #expect(result.sections[1].text == "この区間では、旅行の予定と準備について話している。")
        #expect(await recorder.requests.map(\.kind) == [.section, .overview])
        #expect(try SummaryStore.fingerprint(video) == fingerprint)
        let files = try FileManager.default.contentsOfDirectory(
            at: fixture.current.deletingLastPathComponent(), includingPropertiesForKeys: nil)
        #expect(
            try files.filter { $0.lastPathComponent.hasPrefix("previous-") }.map {
                try Data(contentsOf: $0)
            } == [before])
    }

    @Test func repairCanReplaceInvalidSelectedTextButNotUnselectedText() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        _ = try await fixture.generate()
        let data = try Data(contentsOf: fixture.current)
        let old = try JSONDecoder().decode(SummaryStore.Record.self, from: data)
        let invalid = SummaryStore.Record(
            version: old.version, fingerprint: old.fingerprint, overview: "",
            sections: [
                .init(
                    startSeconds: 0, endSeconds: 600, text: "同じ作業について何度も説明している。同じ作業について何度も説明している。")
            ])
        try JSONEncoder().encode(invalid).write(to: fixture.current)
        await #expect(throws: SummaryGenerationError.repeatedResponse) {
            try await SummaryStore().load(video: fixture.video)
        }
        await #expect(throws: SummaryGenerationError.invalidRepairSelection) {
            try await SummaryStore().load(video: fixture.video, repairing: [600])
        }
        let command = try SummaryCommand(arguments: [
            "generate", "--library", fixture.root.path, "--repair-sections", "0",
        ])
        #expect(
            try await command.run(respond: { _ in "旅行の準備を順序に沿って説明している。" }, output: { _ in }) == 0)
        #expect(
            try await SummaryStore().load(video: fixture.video)?.sections[0].text
                == "旅行の準備を順序に沿って説明している。")
    }

    @Test func failedRepairRetainsPointerAndMissingPriorIsRejected() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let failing = AppleSummaryService(store: SummaryStore()) { _ in
            throw SummaryGenerationError.emptyResponse
        }
        await #expect(throws: SummaryGenerationError.invalidRepairSelection) {
            try await failing.generate(video: fixture.video, force: false, repairing: [0]) { _, _ in
            }
        }
        _ = try await fixture.generate()
        let before = try Data(contentsOf: fixture.current)
        await #expect(throws: SummaryGenerationError.emptyResponse) {
            try await failing.generate(video: fixture.video, force: false, repairing: [0]) { _, _ in
            }
        }
        #expect(try Data(contentsOf: fixture.current) == before)
    }

    @Test func generatorRejectsMismatchedRepair() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let generator = NarrativeSummaryGenerator { _ in "説明している。" }
        await #expect(throws: SummaryGenerationError.invalidRepairSelection) {
            try await generator.generate(fixture.video.summary, replacing: [0])
        }
        await #expect(throws: SummaryGenerationError.invalidRepairSelection) {
            try await generator.generate(
                fixture.video.summary, reusing: .init(overview: "", sections: []), replacing: [0])
        }
        #expect(SummaryGenerationError.repeatedResponse.diagnosticCode == "repeated_response")
        #expect(
            SummaryGenerationError.contradictoryResponse.diagnosticCode == "contradictory_response")
        #expect(
            SummaryGenerationError.invalidRepairSelection.diagnosticCode
                == "invalid_repair_selection")
        #expect(SummaryGenerationError.repeatedResponse.errorDescription != nil)
        #expect(SummaryGenerationError.contradictoryResponse.errorDescription != nil)
        #expect(SummaryGenerationError.invalidRepairSelection.errorDescription != nil)
    }
}
