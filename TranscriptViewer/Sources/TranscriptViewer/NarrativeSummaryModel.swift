import Foundation
import Observation
import TranscriptCore
import TranscriptSummaries

@MainActor
@Observable
final class NarrativeSummaryModel {
    enum State: Equatable {
        case idle
        case loadingSaved
        case waitingForLibrary
        case generating(completed: Int, total: Int)
        case ready(NarrativeSummary)
        case failed(String)
    }

    // Request counters belong to the same stable model as the displayed result.
    // SwiftUI's task identity consumes each user action once, including after redraws.
    var requestAttempt = 0
    var handledAttempt = 0
    var requestReload = 0
    var handledReload = 0

    private(set) var state = State.idle
    private(set) var video: VideoTranscript?
    enum ExcerptState: Equatable {
        case absent
        case ready(SourceExcerptCandidate)
        case representative(RepresentativeSummaryCandidate)
        case failed
    }
    private(set) var excerptState = ExcerptState.absent
    private(set) var excerptVideo: VideoTranscript?
    private var excerptGeneration = 0

    /// Excerpts have separate state and never replace a narrative result or its failure.
    func loadExcerpts(video: VideoTranscript) async {
        excerptGeneration += 1
        excerptVideo = video
        let current = excerptGeneration
        excerptState = .absent
        do {
            if let candidate = try await RepresentativeSummaryStore().load(video: video) {
                guard current == excerptGeneration, !Task.isCancelled else { return }
                excerptState = .representative(candidate)
                return
            }
            let result = try await SourceExcerptStore().load(video: video)
            guard current == excerptGeneration, !Task.isCancelled else { return }
            excerptState = result.map(ExcerptState.ready) ?? .absent
        } catch {
            guard current == excerptGeneration, !Task.isCancelled else { return }
            excerptState = .failed
        }
    }
    struct LibraryProgress: Equatable {
        let total: Int
        var completed = 0
        var succeeded = 0
        var failures: [String] = []
        var currentDate: String?
        var isRunning = true
        var wasCancelled = false
    }
    private(set) var libraryProgress: LibraryProgress?
    var isBatchRunning: Bool { libraryProgress?.isRunning == true }
    private var batchGeneration = 0
    private var batchFailures: [URL: (TranscriptSummary, String)] = [:]
    private let service: AppleSummaryService
    private var generation = 0

    init(service: AppleSummaryService = AppleSummaryService()) {
        self.service = service
    }

    /// The SwiftUI task owns cancellation. Identity checks also reject a late
    /// response from a provider that completes after cancellation or navigation.
    func load(video: VideoTranscript, force: Bool = false) async {
        generation += 1
        self.video = video
        let current = generation
        if isBatchRunning {
            let cached = await service.cached(video: video)
            guard current == generation, !Task.isCancelled else { return }
            state = cached.map(State.ready) ?? .waitingForLibrary
            return
        }
        if force { batchFailures.removeValue(forKey: video.id) }
        if !force, let failure = batchFailures[video.id], failure.0 == video.summary {
            state = .failed(failure.1)
            return
        }
        state = .generating(completed: 0, total: video.summary.sections.count)
        do {
            let result = try await service.generate(video: video, force: force) { done, total in
                await self.updateProgress(done: done, total: total, generation: current)
            }
            try Task.checkCancellation()
            guard current == generation else { return }
            state = .ready(result)
        } catch is CancellationError {
            if current == generation { state = .idle }
        } catch {
            guard current == generation else { return }
            state = Task.isCancelled ? .idle : .failed(error.localizedDescription)
        }
    }

    /// Explicitly reload CLI/app output without starting an inference.
    func loadSaved(video: VideoTranscript) async {
        generation += 1
        self.video = video
        let current = generation
        state = .loadingSaved
        do {
            let result = try await SummaryStore().load(video: video)
            try Task.checkCancellation()
            guard current == generation else { return }
            state = result.map(State.ready) ?? .failed("この字幕に一致する保存済み要約がありません。")
        } catch is CancellationError {
            if current == generation { state = .idle }
        } catch {
            if current == generation {
                state = Task.isCancelled ? .idle : .failed(error.localizedDescription)
            }
        }
    }

    /// Uses only the caller's loaded catalog snapshot. Missing dates sort last.
    /// The view owns cancellation; failures are counted, not disguised as successes.
    func generateLibrary(videos: [VideoTranscript]) async {
        batchGeneration += 1
        generation += 1
        let run = batchGeneration
        batchFailures.removeAll()
        let ordered = AppleSummaryService.newestFirst(videos)
        libraryProgress = LibraryProgress(total: ordered.count)
        defer {
            if run == batchGeneration {
                libraryProgress?.isRunning = false
                libraryProgress?.currentDate = nil
            }
        }
        for input in ordered {
            guard run == batchGeneration else { return }
            if Task.isCancelled {
                libraryProgress?.wasCancelled = true
                return
            }
            libraryProgress?.currentDate = input.broadcastStartJST ?? "配信日時不明"
            do {
                let result = try await service.generate(video: input, force: false) { done, total in
                    await self.updateBatchProgress(video: input, done: done, total: total, run: run)
                }
                try Task.checkCancellation()
                guard run == batchGeneration else { return }
                libraryProgress?.succeeded += 1
                if video == input { state = .ready(result) }
            } catch is CancellationError {
                if run == batchGeneration { libraryProgress?.wasCancelled = true }
                return
            } catch {
                guard run == batchGeneration else { return }
                if Task.isCancelled {
                    libraryProgress?.wasCancelled = true
                    return
                }
                let message = error.localizedDescription
                batchFailures[input.id] = (input.summary, message)
                libraryProgress?.failures.append(
                    "\(input.broadcastStartJST ?? input.title): \(message)")
                if video == input { state = .failed(message) }
            }
            libraryProgress?.completed += 1
        }
    }

    private func updateBatchProgress(video: VideoTranscript, done: Int, total: Int, run: Int) {
        guard run == batchGeneration, !Task.isCancelled, self.video == video else { return }
        state = .generating(completed: done, total: total)
    }

    private func updateProgress(done: Int, total: Int, generation: Int) {
        guard generation == self.generation, !Task.isCancelled else { return }
        state = .generating(completed: done, total: total)
    }
}
