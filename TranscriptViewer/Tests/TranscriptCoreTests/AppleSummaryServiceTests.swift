import Foundation
import Testing
import TranscriptCore
import TranscriptSummaries

@testable import TranscriptViewer

private actor DelayedSummaryResponse {
    private var response: CheckedContinuation<String, Never>?
    private var arrival: CheckedContinuation<Void, Never>?

    func respond(_ request: SummaryRequest) async -> String {
        if request.source == "old" {
            return await withCheckedContinuation { continuation in
                response = continuation
                arrival?.resume()
                arrival = nil
            }
        }
        return "新しい動画の要約です。"
    }

    func waitUntilRequested() async {
        if response != nil { return }
        await withCheckedContinuation { arrival = $0 }
    }

    func release() {
        response?.resume(returning: "古い動画の要約です。")
        response = nil
    }
}

@MainActor
struct AppleSummaryServiceTests {
    private func video(_ text: String, name: String = "synthetic") -> VideoTranscript {
        VideoTranscript(
            title: name, movieURL: URL(fileURLWithPath: "/\(name).mp4"),
            transcriptURL: URL(fileURLWithPath: "/\(name).srt"),
            cues: [
                TranscriptCue(
                    id: 0, startSeconds: 0, endSeconds: 10,
                    startLabel: "", endLabel: "", text: text)
            ])
    }

    @Test func cachesOnlyUnchangedInputAndSupportsRegeneration() async throws {
        let recorder = SummaryRecorder()
        let service = AppleSummaryService { await recorder.respond($0) }
        let first = video("原文")
        let original = try await service.generate(video: first, force: false) { _, _ in }
        let cached = try await service.generate(video: first, force: false) { _, _ in }
        #expect(cached == original)
        #expect(await recorder.requests.count == 2)
        _ = try await service.generate(video: first, force: true) { _, _ in }
        #expect(await recorder.requests.count == 4)
        _ = try await service.generate(video: video("変更後"), force: false) { _, _ in }
        #expect(await recorder.requests.count == 6)
        _ = try await service.generate(video: video("変更後", name: "different"), force: false) {
            _, _ in
        }
        #expect(await recorder.requests.count == 8)
    }

    @Test func modelPublishesCompletedNarrative() async {
        let recorder = SummaryRecorder()
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { await recorder.respond($0) })
        #expect(model.state == .idle)
        #expect(model.video == nil)
        let input = video("原文")
        await model.load(video: input)
        #expect(model.video == input)
        guard case .ready(let summary) = model.state else {
            Issue.record("Expected completed narrative")
            return
        }
        #expect(summary.overview == "動画では旅行の計画を紹介し、準備の手順を説明している。")
    }

    @Test func modelShowsFailureWithoutExtractiveFallback() async {
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { _ in
                throw SummaryGenerationError.unavailable("Apple Intelligenceを有効にしてください。")
            })
        await model.load(video: video("原文の抜粋を表示しない"))
        #expect(model.state == .failed("Apple Intelligenceを有効にしてください。"))
    }

    @Test(.timeLimit(.minutes(1))) func staleCompletionCannotReplaceCurrentVideo() async {
        let delayed = DelayedSummaryResponse()
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { await delayed.respond($0) })
        // A cached selection must remain available while another video owns inference.
        await model.load(video: video("new", name: "new"))
        async let old: Void = model.load(video: video("old", name: "old"))
        await delayed.waitUntilRequested()
        #expect(model.state == .generating(completed: 0, total: 1))
        await model.load(video: video("new", name: "new"))
        let currentState = model.state
        await delayed.release()
        await old
        #expect(model.state == currentState)
        #expect(model.video?.title == "new")
    }

    @Test(.timeLimit(.minutes(1))) func cancellationDoesNotBecomeAProviderFailure() async {
        let delayed = DelayedSummaryResponse()
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { request in
                _ = await delayed.respond(request)
                throw SummaryGenerationError.unavailable(
                    "Provider finished with an error after cancellation")
            })
        let input = video("old")
        let task = Task { await model.load(video: input) }
        await delayed.waitUntilRequested()
        task.cancel()
        await delayed.release()
        await task.value
        #expect(model.state == .idle)
    }

    @Test(.timeLimit(.minutes(1))) func cancellationDiscardsLateResponseAndAllowsRetry() async {
        let delayed = DelayedSummaryResponse()
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { await delayed.respond($0) })
        let input = video("old")
        let task = Task { await model.load(video: input) }
        await delayed.waitUntilRequested()
        task.cancel()
        await delayed.release()
        await task.value
        #expect(model.state == .idle)
        await model.load(video: video("new"))
        guard case .ready = model.state else {
            Issue.record("Expected a successful retry after cancellation")
            return
        }
    }
}
