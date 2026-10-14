import Foundation
import TranscriptCore

/// Opt-in private, immutable per-request evidence. Text is omitted unless explicitly enabled.
public actor SummaryDiagnostics {
    public nonisolated let directory: URL
    private let includeText: Bool
    private let provider: NarrativeSummaryGenerator.Respond
    private var sequence = 0
    private var videoIndex: Int?

    private struct VideoRecord: Encodable {
        let index: Int
        let transcript: String
        let movie: String
        let fingerprint: String
        let pipelineVersion: Int
        let osVersion: String
    }

    private struct Outcome: Encodable {
        let index: Int
        let errorCode: String?
    }

    struct Event: Codable {
        let phase: String
        let video: Int?
        let request: Int
        let kind: SummaryRequest.Kind
        let sectionStartSeconds: Double?
        let correctsFormat: Bool
        let inputBytes: Int
        let maximumResponseTokens: Int
        let elapsedSeconds: Double?
        let responseBytes: Int?
        let errorCode: String?
        let instructions: String?
        let prompt: String?
        let response: String?
    }

    private init(
        directory: URL, includeText: Bool, provider: @escaping NarrativeSummaryGenerator.Respond
    ) {
        self.directory = directory
        self.includeText = includeText
        self.provider = provider
    }

    public static func create(
        in root: URL, includeText: Bool = false,
        provider: @escaping NarrativeSummaryGenerator.Respond
    ) async throws -> SummaryDiagnostics {
        let directory = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try await SummaryStore.io {
            try SummaryStore.prepareDirectory(root)
            try SummaryStore.prepareDirectory(directory)
        }
        return SummaryDiagnostics(
            directory: directory, includeText: includeText, provider: provider)
    }

    public func beginVideo(_ video: VideoTranscript, index: Int) async throws {
        videoIndex = index
        let record = VideoRecord(
            index: index, transcript: video.transcriptURL.path, movie: video.movieURL.path,
            fingerprint: try SummaryStore.fingerprint(video),
            pipelineVersion: SummaryStore.pipelineVersion,
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString)
        let data = try JSONEncoder().encode(record)
        let url = directory.appendingPathComponent("video-\(index)-input.json")
        try await SummaryStore.io { try SummaryStore.writePrivate(data, to: url, atomic: false) }
    }

    public func finishVideo(index: Int, errorCode: String?) async throws {
        let data = try JSONEncoder().encode(Outcome(index: index, errorCode: errorCode))
        let url = directory.appendingPathComponent("video-\(index)-outcome.json")
        try await SummaryStore.io { try SummaryStore.writePrivate(data, to: url, atomic: false) }
    }

    public func respond(_ request: SummaryRequest) async throws -> String {
        sequence += 1
        let number = sequence
        let video = videoIndex
        let clock = ContinuousClock()
        let start = clock.now
        try await record(
            request, video: video, number: number, phase: "request", elapsed: nil, response: nil,
            error: nil)
        let outcome: Result<String, Error>
        do { outcome = .success(try await provider(request)) } catch { outcome = .failure(error) }
        let elapsed = start.duration(to: clock.now).components
        let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
        switch outcome {
        case .success(let text):
            try await record(
                request, video: video, number: number, phase: "response",
                elapsed: seconds,
                response: text, error: nil)
            return text
        case .failure(let error):
            let code = Self.errorCode(error)
            try await record(
                request, video: video, number: number, phase: "error", elapsed: seconds,
                response: nil, error: code)
            throw error
        }
    }

    public static func errorCode(_ error: any Error) -> String {
        if error is CancellationError { return "cancelled" }
        if let error = error as? SummaryGenerationError { return error.diagnosticCode }
        if let error = error as? SummaryStoreError {
            if case .busy = error { return "summary_busy" }
            return "invalid_storage"
        }
        if error is DecodingError { return "invalid_json" }
        return "io_or_unexpected_error"
    }

    private func record(
        _ request: SummaryRequest, video: Int?, number: Int, phase: String,
        elapsed: Double?, response: String?, error: String?
    ) async throws {
        let event = Event(
            phase: phase, video: video, request: number, kind: request.kind,
            sectionStartSeconds: request.sectionStartSeconds,
            correctsFormat: request.correctsFormat,
            inputBytes: request.source.utf8.count,
            maximumResponseTokens: request.maximumResponseTokens,
            elapsedSeconds: elapsed, responseBytes: response?.utf8.count, errorCode: error,
            instructions: includeText ? request.modelInstructions : nil,
            prompt: includeText ? request.prompt : nil, response: includeText ? response : nil)
        let data = try JSONEncoder().encode(event)
        let url = directory.appendingPathComponent(String(format: "%06d-%@.json", number, phase))
        try await SummaryStore.io { try SummaryStore.writePrivate(data, to: url, atomic: false) }
    }
}
