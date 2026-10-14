import Foundation
import FoundationModels
import Testing
import TranscriptCore

@testable import TranscriptSummaries

struct SummaryContextTests {
    @Test func exactContextBudgetFits() throws {
        try AppleSummaryService.validateContext(
            instructions: 100, input: 2868, response: 1000, capacity: 4096)
    }

    @Test(arguments: [
        (-1, 0, 1, 4096), (0, -1, 1, 4096), (0, 0, 0, 4096),
        (0, 0, -1, 4096), (0, 0, 1, 127), (0, 0, 1, -1),
        (0, 0, 3969, 4096), (3969, 0, 1, 4096), (100, 2869, 1000, 4096),
        (Int.max, 1, 1, Int.max), (1, Int.max, 1, Int.max), (1, 1, Int.max, Int.max),
    ])
    func rejectsInvalidOrOverflowingContext(_ counts: (Int, Int, Int, Int)) {
        #expect(throws: SummaryGenerationError.insufficientCompression) {
            try AppleSummaryService.validateContext(
                instructions: counts.0, input: counts.1, response: counts.2, capacity: counts.3)
        }
    }
}

struct AppleSummaryPolicyTests {
    private func verifyOlderOSRejection() async {
        let video = VideoTranscript(
            title: "synthetic", movieURL: URL(fileURLWithPath: "/synthetic.mp4"),
            transcriptURL: URL(fileURLWithPath: "/synthetic.srt"),
            cues: [
                TranscriptCue(
                    id: 0, startSeconds: 0, endSeconds: 1,
                    startLabel: "", endLabel: "", text: "合成資料。")
            ])
        await #expect(
            throws: SummaryGenerationError.unavailable("Apple Intelligence要約にはmacOS 26以降が必要です。")
        ) {
            _ = try await AppleSummaryService(respond: AppleSummaryService.respond).generate(
                video: video, force: false
            ) { _, _ in }
        }
    }

    @Test func availabilityReasonsAreActionable() async {
        guard #available(macOS 26.0, *) else {
            await verifyOlderOSRejection()
            return
        }
        #expect(AppleSummaryService.availabilityMessage(.available) == nil)
        #expect(
            AppleSummaryService.availabilityMessage(.unavailable(.deviceNotEligible))
                == "このMacはApple Intelligenceに対応していません。")
        #expect(
            AppleSummaryService.availabilityMessage(.unavailable(.appleIntelligenceNotEnabled))
                == "システム設定でApple Intelligenceを有効にしてください。")
        #expect(
            AppleSummaryService.availabilityMessage(.unavailable(.modelNotReady))
                == "Apple Intelligenceのモデルを準備中です。準備完了後に再試行してください。")
    }

    @Test func nativeGenerationErrorsDoNotExposeTheirDebugContext() async {
        guard #available(macOS 26.0, *) else {
            await verifyOlderOSRejection()
            return
        }
        let context = LanguageModelSession.GenerationError.Context(
            debugDescription: "synthetic prompt text must not appear in the message")
        #expect(
            AppleSummaryService.generationCode(.exceededContextWindowSize(context))
                == "context_window_exceeded")
        #expect(
            AppleSummaryService.generationCode(.assetsUnavailable(context)) == "assets_unavailable")
        #expect(
            AppleSummaryService.generationCode(.guardrailViolation(context))
                == "guardrail_violation")
        #expect(
            AppleSummaryService.generationCode(.unsupportedLanguageOrLocale(context))
                == "unsupported_language")
        #expect(AppleSummaryService.generationCode(.rateLimited(context)) == "rate_limited")
        #expect(
            AppleSummaryService.generationCode(.concurrentRequests(context))
                == "concurrent_requests")
        #expect(AppleSummaryService.generationCode(.decodingFailure(context)) == "decoding_failure")
        #expect(
            AppleSummaryService.generationCode(.unsupportedGuide(context)) == "unsupported_guide")
        #expect(
            AppleSummaryService.generationMessage(.exceededContextWindowSize(context))
                == "モデルの入力上限を超えました。要約は作成されていません。")
        #expect(
            AppleSummaryService.generationMessage(.assetsUnavailable(context))
                == "Apple Intelligenceのモデルを利用できません。準備完了後に再試行してください。")
        #expect(
            AppleSummaryService.generationMessage(.guardrailViolation(context))
                == "Apple Intelligenceがこの内容の要約を作成できませんでした。")
        #expect(
            AppleSummaryService.generationMessage(.unsupportedLanguageOrLocale(context))
                == "Apple Intelligenceが文字起こしの言語を処理できませんでした。")
        #expect(
            AppleSummaryService.generationMessage(.rateLimited(context))
                == "Apple Intelligenceが使用中です。少し待って再試行してください。")
        #expect(
            AppleSummaryService.generationMessage(.concurrentRequests(context))
                == "Apple Intelligenceが使用中です。少し待って再試行してください。")
        #expect(
            AppleSummaryService.generationMessage(.decodingFailure(context))
                == "Apple Intelligenceの応答を読み取れませんでした。再試行してください。")
        #expect(
            AppleSummaryService.generationMessage(.unsupportedGuide(context))
                == "Apple Intelligenceの応答を読み取れませんでした。再試行してください。")
    }
}
