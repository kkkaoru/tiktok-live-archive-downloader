import Foundation
import FoundationModels
import TranscriptCore

/// Shared native inference and validated persistence for the GUI and CLI.
/// Injected responders default to memory-only storage for isolated tests.
public actor AppleSummaryService {
    private struct Cached {
        let transcript: TranscriptSummary
        let result: NarrativeSummary
    }
    private var cache: [URL: Cached] = [:]
    private let generator: NarrativeSummaryGenerator
    private let store: SummaryStore?
    private var isGenerating = false
    private var waiters: [UUID: AsyncStream<Void>.Continuation] = [:]

    public init() {
        store = SummaryStore()
        generator = NarrativeSummaryGenerator(respond: Self.respond)
    }

    public init(store: SummaryStore? = nil, respond: @escaping NarrativeSummaryGenerator.Respond) {
        self.store = store
        generator = NarrativeSummaryGenerator(respond: respond)
    }

    public nonisolated static func newestFirst(_ videos: [VideoTranscript]) -> [VideoTranscript] {
        videos.sorted {
            let left = $0.broadcastStartJST ?? ""
            let right = $1.broadcastStartJST ?? ""
            return left == right ? $0.id.path < $1.id.path : left > right
        }
    }

    public func generate(
        video: VideoTranscript, force: Bool,
        repairing starts: Set<Double> = [],
        progress: @Sendable (Int, Int) async -> Void
    ) async throws -> NarrativeSummary {
        try Task.checkCancellation()
        if !force, starts.isEmpty, case nil = store, let result = cached(video: video) {
            return result
        }
        await progress(0, video.summary.sections.count)
        try await waitForTurn()
        isGenerating = true
        defer {
            isGenerating = false
            for waiter in waiters.values { waiter.finish() }
            waiters.removeAll()
        }
        if !force, starts.isEmpty, let store, let result = try await store.loadIfClean(video: video)
        {
            cache[video.id] = Cached(transcript: video.summary, result: result)
            return result
        }
        if !force, starts.isEmpty, case nil = store, let result = cached(video: video) {
            return result
        }
        let lease: SummaryLease?
        if let store {
            lease = try await SummaryLease.acquire(video: video, store: store)
        } else {
            lease = nil
        }
        let outcome: Result<NarrativeSummary, Error>
        do {
            outcome = .success(
                try await produce(video: video, force: force, repairing: starts, progress: progress)
            )
        } catch { outcome = .failure(error) }
        try await lease?.release()
        let result = try outcome.get()
        try Task.checkCancellation()
        cache[video.id] = Cached(transcript: video.summary, result: result)
        return result
    }

    private func produce(
        video: VideoTranscript, force: Bool,
        repairing starts: Set<Double>,
        progress: @Sendable (Int, Int) async -> Void
    ) async throws -> NarrativeSummary {
        // Another process may have completed between the initial read and lock acquisition.
        if !force, starts.isEmpty, let store,
            let result = try await store.loadCleaning(video: video)
        {
            return result
        }
        let previous: NarrativeSummary?
        if starts.isEmpty {
            previous = nil
        } else {
            guard !force, let store,
                let saved = try await store.load(video: video, repairing: starts)
            else {
                throw SummaryGenerationError.invalidRepairSelection
            }
            previous = saved
        }
        let result = try await generator.generate(
            video.summary, reusing: previous, replacing: starts, progress: progress)
        if let store { try await store.save(result, video: video) }
        return result
    }

    public func cached(video: VideoTranscript) -> NarrativeSummary? {
        guard let saved = cache[video.id], saved.transcript == video.summary else { return nil }
        return saved.result
    }

    /// One complete video at a time, including requests from the selected-video UI.
    /// AsyncStream termination removes cancelled waiters without cancelling another owner.
    private func waitForTurn() async throws {
        while isGenerating {
            let id = UUID()
            let signal = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
            waiters[id] = signal.continuation
            defer {
                signal.continuation.finish()
                waiters.removeValue(forKey: id)
            }
            for await _ in signal.stream {}
            try Task.checkCancellation()
        }
        try Task.checkCancellation()
    }

    public static func respond(_ request: SummaryRequest) async throws -> String {
        guard #available(macOS 26.0, *) else {
            throw SummaryGenerationError.unavailable("Apple Intelligence要約にはmacOS 26以降が必要です。")
        }
        // Apple's documented preset for transforming existing text, not open-ended chat.
        // Explicit String refusals are rejected by the shared generator.
        let model = SystemLanguageModel(guardrails: .permissiveContentTransformations)
        if let message = availabilityMessage(model.availability) {
            throw SummaryGenerationError.unavailable(message)
        }
        guard model.supportsLocale(Locale(identifier: "ja_JP")) else {
            throw SummaryGenerationError.unavailable("このApple Intelligenceモデルは日本語に対応していません。")
        }
        try Task.checkCancellation()
        let session = LanguageModelSession(model: model, instructions: request.modelInstructions)
        return try await mappedResponse {
            if #available(macOS 26.4, *) {
                let instructions = try await model.tokenCount(
                    for: Instructions(request.modelInstructions))
                let input = try await model.tokenCount(for: request.prompt)
                try validateContext(
                    instructions: instructions, input: input,
                    response: request.maximumResponseTokens, capacity: model.contextSize)
            }
            try Task.checkCancellation()
            let options = GenerationOptions(
                temperature: 0.3, maximumResponseTokens: request.maximumResponseTokens)
            let response = try await session.respond(to: request.prompt, options: options)
            try Task.checkCancellation()
            return response.content
        }
    }

    /// Keep cancellation and unexpected I/O errors intact; SDK diagnostics may contain prompts.
    @available(macOS 26.0, *)
    static func mappedResponse(_ operation: () async throws -> String) async throws -> String {
        do {
            return try await operation()
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as LanguageModelSession.GenerationError {
            throw SummaryGenerationError.providerFailure(
                code: generationCode(error), message: generationMessage(error))
        }
    }

    /// Reject invalid counts and overflow without starting a native inference.
    static func validateContext(
        instructions: Int, input: Int, response: Int, capacity: Int
    ) throws {
        let framingReserve = 128
        guard instructions >= 0, input >= 0, response > 0,
            capacity >= framingReserve,
            response <= capacity - framingReserve,
            instructions <= capacity - framingReserve - response,
            input <= capacity - framingReserve - response - instructions
        else { throw SummaryGenerationError.insufficientCompression }
    }

    @available(macOS 26.0, *)
    static func availabilityMessage(_ availability: SystemLanguageModel.Availability) -> String? {
        switch availability {
        case .available: nil
        case .unavailable(let reason):
            switch reason {
            case .deviceNotEligible:
                "このMacはApple Intelligenceに対応していません。"
            case .appleIntelligenceNotEnabled:
                "システム設定でApple Intelligenceを有効にしてください。"
            case .modelNotReady:
                "Apple Intelligenceのモデルを準備中です。準備完了後に再試行してください。"
            @unknown default:
                "Apple Intelligenceを現在利用できません。"
            }
        }
    }

    @available(macOS 26.0, *)
    static func generationCode(_ error: LanguageModelSession.GenerationError) -> String {
        switch error {
        case .exceededContextWindowSize: "context_window_exceeded"
        case .assetsUnavailable: "assets_unavailable"
        case .guardrailViolation: "guardrail_violation"
        case .refusal: "model_refusal"
        case .unsupportedLanguageOrLocale: "unsupported_language"
        case .rateLimited: "rate_limited"
        case .concurrentRequests: "concurrent_requests"
        case .decodingFailure: "decoding_failure"
        case .unsupportedGuide: "unsupported_guide"
        @unknown default: "unknown_provider_error"
        }
    }

    /// Fixed messages never echo the SDK's prompt-bearing debug context.
    @available(macOS 26.0, *)
    static func generationMessage(_ error: LanguageModelSession.GenerationError) -> String {
        switch error {
        case .exceededContextWindowSize:
            "モデルの入力上限を超えました。要約は作成されていません。"
        case .assetsUnavailable:
            "Apple Intelligenceのモデルを利用できません。準備完了後に再試行してください。"
        case .guardrailViolation, .refusal:
            "Apple Intelligenceがこの内容の要約を作成できませんでした。"
        case .unsupportedLanguageOrLocale:
            "Apple Intelligenceが文字起こしの言語を処理できませんでした。"
        case .rateLimited, .concurrentRequests:
            "Apple Intelligenceが使用中です。少し待って再試行してください。"
        case .decodingFailure, .unsupportedGuide:
            "Apple Intelligenceの応答を読み取れませんでした。再試行してください。"
        @unknown default:
            "Apple Intelligenceの要約生成に失敗しました。再試行してください。"
        }
    }
}
