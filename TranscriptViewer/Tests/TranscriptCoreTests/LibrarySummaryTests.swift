import Foundation
import Testing
import TranscriptCore
import TranscriptSummaries

@testable import TranscriptViewer

private actor LibraryResponses {
    var sections: [String] = []
    func respond(_ request: SummaryRequest) throws -> String {
        if request.kind == .section {
            sections.append(request.source)
            if request.source == "failed" {
                throw SummaryGenerationError.unavailable("synthetic failure")
            }
        }
        return "動画の内容をまとめた文章です。"
    }
}

private actor PausedResponse {
    private var releaseResponse: CheckedContinuation<String, Never>?
    private var arrival: CheckedContinuation<Void, Never>?
    private var hasPaused = false
    var sectionRequests = 0

    func respond(_ request: SummaryRequest) async -> String {
        if request.kind == .section { sectionRequests += 1 }
        guard request.source == "blocked", !hasPaused else { return "生成された要約です。" }
        hasPaused = true
        return await withCheckedContinuation { continuation in
            releaseResponse = continuation
            arrival?.resume()
            arrival = nil
        }
    }

    func waitForRequest() async {
        if hasPaused { return }
        await withCheckedContinuation { arrival = $0 }
    }

    func release() {
        releaseResponse?.resume(returning: "生成された要約です。")
        releaseResponse = nil
    }
}

private actor QueueArrival {
    private var arrived = false
    private var continuation: CheckedContinuation<Void, Never>?
    func signal() {
        arrived = true
        continuation?.resume()
        continuation = nil
    }
    func wait() async {
        if arrived { return }
        await withCheckedContinuation { continuation = $0 }
    }
}

@MainActor
struct LibrarySummaryTests {
    private func video(_ text: String, date: String? = nil) -> VideoTranscript {
        VideoTranscript(
            title: text, movieURL: URL(fileURLWithPath: "/\(text).mp4"),
            transcriptURL: URL(fileURLWithPath: "/\(text).srt"),
            cues: [
                TranscriptCue(
                    id: 0, startSeconds: 0, endSeconds: 1,
                    startLabel: "", endLabel: "", text: text)
            ],
            broadcastStartJST: date)
    }

    @Test func generatesLoadedSnapshotNewestFirstWithUnknownDatesLast() async {
        let replies = LibraryResponses()
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { try await replies.respond($0) })
        await model.generateLibrary(videos: [
            video("undated"), video("old", date: "2026-01-01 12:00:00 JST"),
            video("new", date: "2026-02-01 12:00:00 JST"),
        ])
        #expect(await replies.sections == ["new", "old", "undated"])
        #expect(model.libraryProgress?.total == 3)
        #expect(model.libraryProgress?.completed == 3)
        #expect(model.libraryProgress?.succeeded == 3)
        #expect(model.libraryProgress?.failures == [])
        #expect(!model.isBatchRunning)
        #expect(model.libraryProgress?.wasCancelled == false)
        #expect(model.video == nil)
        await model.generateLibrary(videos: [video("undated")])
        #expect(await replies.sections == ["new", "old", "undated"])
    }

    @Test func failureDoesNotStopOtherVideosOrCountAsSuccess() async {
        let replies = LibraryResponses()
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { try await replies.respond($0) })
        let failed = video("failed", date: "2026-02-01 12:00:00 JST")
        let loaded = [video("old", date: "2026-01-01 12:00:00 JST"), failed]
        await model.generateLibrary(videos: loaded)
        #expect(await replies.sections == ["failed", "old"])
        #expect(model.libraryProgress?.completed == 2)
        #expect(model.libraryProgress?.succeeded == 1)
        #expect(model.libraryProgress?.failures.count == 1)
        await model.load(video: failed)
        #expect(model.state == .failed("synthetic failure"))
        #expect(await replies.sections == ["failed", "old"])
        await model.generateLibrary(videos: loaded)
        #expect(await replies.sections == ["failed", "old", "failed"])
        await model.load(video: failed, force: true)
        #expect(await replies.sections == ["failed", "old", "failed", "failed"])
    }

    @Test func batchPreservesSelectedVideoAndReusesItsSummary() async {
        let replies = LibraryResponses()
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { try await replies.respond($0) })
        let selected = video("selected")
        await model.load(video: selected)
        let selectedState = model.state
        await model.generateLibrary(videos: [
            selected, video("new", date: "2026-02-01 12:00:00 JST"),
        ])
        #expect(model.video == selected)
        #expect(model.state == selectedState)
        #expect(await replies.sections == ["selected", "new"])
    }

    @Test(.timeLimit(.minutes(1))) func cancellationIsNotACompletedOrFailedVideo() async {
        let paused = PausedResponse()
        let service = AppleSummaryService { await paused.respond($0) }
        let model = NarrativeSummaryModel(service: service)
        let loaded = [video("blocked", date: "2026-02-01 12:00:00 JST"), video("later")]
        let task = Task { await model.generateLibrary(videos: loaded) }
        await paused.waitForRequest()
        #expect(model.isBatchRunning)
        await model.load(video: loaded[1])
        #expect(model.state == .waitingForLibrary)
        task.cancel()
        await paused.release()
        await task.value
        #expect(model.libraryProgress?.wasCancelled == true)
        #expect(model.libraryProgress?.completed == 0)
        #expect(model.libraryProgress?.failures == [])
        #expect(!model.isBatchRunning)
        #expect(await service.cached(video: loaded[0]) == nil)
    }

    @Test(.timeLimit(.minutes(1))) func cancelledQueuedRequestDoesNotCancelCurrentOwner()
        async throws
    {
        let paused = PausedResponse()
        let arrival = QueueArrival()
        let service = AppleSummaryService { await paused.respond($0) }
        let ownerVideo = video("blocked")
        let otherVideo = video("other")
        let owner = Task { try await service.generate(video: ownerVideo, force: false) { _, _ in } }
        await paused.waitForRequest()
        let queued = Task {
            try await service.generate(video: otherVideo, force: false) { _, _ in
                await arrival.signal()
            }
        }
        await arrival.wait()
        queued.cancel()
        switch await queued.result {
        case .success: Issue.record("A cancelled waiter must not start inference.")
        case .failure(let error): #expect(error is CancellationError)
        }
        #expect(await service.cached(video: otherVideo) == nil)
        await paused.release()
        _ = try await owner.value
        #expect(await service.cached(video: ownerVideo) != nil)
        _ = try await service.generate(video: otherVideo, force: false) { _, _ in }
        #expect(await paused.sectionRequests == 2)
    }

    @Test(.timeLimit(.minutes(1))) func queuedDuplicateReusesOwnersCompletedResult() async throws {
        let paused = PausedResponse()
        let arrival = QueueArrival()
        let service = AppleSummaryService { await paused.respond($0) }
        let input = video("blocked")
        let owner = Task { try await service.generate(video: input, force: false) { _, _ in } }
        await paused.waitForRequest()
        let queued = Task {
            try await service.generate(video: input, force: false) { _, _ in
                await arrival.signal()
            }
        }
        await arrival.wait()
        await paused.release()
        let original = try await owner.value
        let duplicate = try await queued.value
        #expect(duplicate == original)
        #expect(await paused.sectionRequests == 1)
    }

    @Test func emptyLibraryAndEqualDateTieBreakAreDeterministic() async {
        let replies = LibraryResponses()
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { try await replies.respond($0) })
        await model.generateLibrary(videos: [])
        #expect(model.libraryProgress?.total == 0)
        #expect(!model.isBatchRunning)
        await model.generateLibrary(videos: [video("b"), video("a")])
        #expect(await replies.sections == ["a", "b"])
    }
}
