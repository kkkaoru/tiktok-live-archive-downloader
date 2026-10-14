import Foundation
import FoundationModels
import Testing
import TranscriptCore

@testable import TranscriptSummaries

struct NativeSummaryBoundaryTests {
    @Test func sdkFailureIsSanitizedWhileCancellationAndUnrelatedFailuresPropagate() async throws {
        guard #available(macOS 26.0, *) else {
            Issue.record("Native summary boundary verification requires macOS 26 or later")
            return
        }
        let context = LanguageModelSession.GenerationError.Context(
            debugDescription: "private diagnostic prompt")
        await #expect(
            throws: SummaryGenerationError.providerFailure(
                code: "rate_limited", message: "Apple Intelligenceが使用中です。少し待って再試行してください。")
        ) {
            try await AppleSummaryService.mappedResponse {
                throw LanguageModelSession.GenerationError.rateLimited(context)
            }
        }
        await #expect(throws: CancellationError.self) {
            try await AppleSummaryService.mappedResponse { throw CancellationError() }
        }
        await #expect(throws: SummaryGenerationError.insufficientCompression) {
            try await AppleSummaryService.mappedResponse {
                throw SummaryGenerationError.insufficientCompression
            }
        }
        #expect(try await AppleSummaryService.mappedResponse { "元の文章。" } == "元の文章。")
    }
}

/// Separate on-device integration: no operational summaries, recordings or model-setting changes.
struct NativeSummaryIntegrationTests {
    @Test(.timeLimit(.minutes(2)))
    func nativeCliGenerationPersistsACompleteResultReusableByTheDefaultService() async throws {
        // Native inference needs actual facts to summarize, not an empty topic label.
        // These are synthetic facts; operational recordings and model settings are untouched.
        let fixture = try SummaryFixture(
            source:
                "京都へ二泊三日の旅行に行く予定を話している。出発は金曜日の朝で、電車を使う。"
                + "週末は雨の予報なので、折り畳み傘と防水の靴を準備する。"
                + "荷物を増やさないため、着替えは二日分だけにする。"
                + "宿泊先は駅から徒歩十分で、チェックインは午後三時。"
                + "乗り換え時刻と予約番号を前日に確認する。")
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        // Retain opt-in synthetic prompt/response evidence even if native quality
        // validation fails; removal of the disposable fixture must not erase it.
        let diagnostics = try transcriptTestWorkRoot().appending(path: "native-summary-diagnostics")
        let command = try SummaryCommand(arguments: [
            "generate", "--library", fixture.root.path,
            "--diagnostics", diagnostics.path, "--include-text",
        ])
        #expect(try await command.run(output: { print($0) }) == 0)
        let result = try #require(try await SummaryStore().load(video: fixture.video))
        #expect(!result.overview.isEmpty)
        #expect(result.sections.count == 1)
        #expect(result.sections.map { $0.source.startSeconds } == [0])
        #expect(result.sections.map { $0.source.endSeconds } == [600])
        let supported = result.overview + result.sections.map(\.text).joined()
        #expect(supported.contains("旅行") || supported.contains("京都"))
        #expect(["雨", "傘", "荷物", "準備", "着替え"].contains { supported.contains($0) })
        let pointer = try Data(contentsOf: fixture.current)
        let cached = try await AppleSummaryService().generate(video: fixture.video, force: false) {
            _, _ in
        }
        #expect(cached == result)
        #expect(try Data(contentsOf: fixture.current) == pointer)
    }
}
