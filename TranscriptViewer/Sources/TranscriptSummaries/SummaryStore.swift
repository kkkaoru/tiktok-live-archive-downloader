import CryptoKit
import Foundation
import TranscriptCore

public enum SummaryStoreError: Error, LocalizedError, Equatable {
    case invalidRecord
    case unsafePath
    case oversizedRecord
    case busy

    public var errorDescription: String? {
        switch self {
        case .invalidRecord: "保存済み要約が不正です。元ファイルを保持して調査してください。"
        case .unsafePath: "要約の保存先が通常のプライベートディレクトリではありません。"
        case .oversizedRecord: "保存済み要約のサイズが上限を超えています。"
        case .busy: "別のプロセスでこの動画を要約中です。完了後に「保存済みを読み込む」を押してください。"
        }
    }
}

/// Immutable revisions plus an atomic current pointer, beside the selected transcript.
/// All filesystem work is dispatched off the UI/cooperative executor.
public struct SummaryStore: Sendable {
    public static let pipelineVersion = 1
    private static let maximumBytes = 8 * 1024 * 1024
    private static let queue = DispatchQueue(label: "TranscriptViewer.summary-storage")

    struct Record: Codable {
        struct Section: Codable {
            let startSeconds: Double
            let endSeconds: Double
            let text: String
        }
        let version: Int
        let fingerprint: String
        let overview: String
        let sections: [Section]
    }

    public init() {}

    public func directory(for video: VideoTranscript) -> URL {
        video.transcriptURL.appendingPathExtension("summaries")
    }

    public static func fingerprint(_ video: VideoTranscript) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(video.summary)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public func load(video: VideoTranscript, repairing starts: Set<Double> = []) async throws
        -> NarrativeSummary?
    {
        try await loadRecord(video: video, repairing: starts)?.summary
    }

    /// A reusable saved result; when validation removed text from it (sentence loops kept by an
    /// older pipeline), the cleaned result is saved through `save` so readers of the file
    /// (the web viewer's catalog build) get it too. Removal only: nothing is regenerated.
    public func loadCleaning(video: VideoTranscript) async throws -> NarrativeSummary? {
        guard let loaded = try await loadRecord(video: video, repairing: []) else { return nil }
        if loaded.changed { try await save(loaded.summary, video: video) }
        return loaded.summary
    }

    /// A saved result that needs no cleaning; nil also when `loadCleaning` would rewrite it,
    /// so callers clean it only under the generation lease.
    public func loadIfClean(video: VideoTranscript) async throws -> NarrativeSummary? {
        guard let loaded = try await loadRecord(video: video, repairing: []), !loaded.changed
        else { return nil }
        return loaded.summary
    }

    private func loadRecord(video: VideoTranscript, repairing starts: Set<Double>) async throws
        -> (summary: NarrativeSummary, changed: Bool)?
    {
        try await Self.io {
            let directory = directory(for: video)
            guard FileManager.default.fileExists(atPath: directory.path) else { return nil }
            try Self.checkDirectory(directory)
            let url = directory.appendingPathComponent("current.json")
            guard let data = try Self.readRecord(url) else { return nil }
            let record = try JSONDecoder().decode(Record.self, from: data)
            guard record.version == Self.pipelineVersion,
                record.fingerprint == (try Self.fingerprint(video))
            else { return nil }
            guard !record.sections.isEmpty, record.sections.count == video.summary.sections.count
            else {
                throw SummaryStoreError.invalidRecord
            }
            guard starts.isSubset(of: Set(video.summary.sections.map(\.startSeconds))) else {
                throw SummaryGenerationError.invalidRepairSelection
            }
            let sections = try zip(record.sections, video.summary.sections).map { saved, source in
                guard saved.startSeconds == source.startSeconds,
                    saved.endSeconds == source.endSeconds
                else {
                    throw SummaryStoreError.invalidRecord
                }
                return NarrativeSummary.Section(
                    source: source,
                    text: starts.contains(source.startSeconds)
                        ? saved.text : try NarrativeSummaryGenerator.validateResponse(saved.text)
                )
            }
            let summary = NarrativeSummary(
                overview: starts.isEmpty
                    ? try NarrativeSummaryGenerator.validateResponse(record.overview)
                    : record.overview,
                sections: sections)
            let changed =
                summary.overview != record.overview
                || zip(summary.sections, record.sections).contains { $0.text != $1.text }
            return (summary, changed)
        }
    }

    /// Failed generation never calls this method and cannot replace a completed record.
    public func save(_ result: NarrativeSummary, video: VideoTranscript) async throws {
        guard !video.summary.sections.isEmpty,
            result.sections.map(\.source) == video.summary.sections
        else {
            throw SummaryStoreError.invalidRecord
        }
        let record = Record(
            version: Self.pipelineVersion, fingerprint: try Self.fingerprint(video),
            overview: try NarrativeSummaryGenerator.validateResponse(result.overview),
            sections: try result.sections.map {
                Record.Section(
                    startSeconds: $0.source.startSeconds, endSeconds: $0.source.endSeconds,
                    text: try NarrativeSummaryGenerator.validateResponse($0.text))
            })
        let data = try JSONEncoder().encode(record)
        guard data.count <= Self.maximumBytes else { throw SummaryStoreError.oversizedRecord }
        try Task.checkCancellation()
        try await Self.io {
            let directory = directory(for: video)
            try Self.prepareDirectory(directory)
            let current = directory.appendingPathComponent("current.json")
            if let previous = try Self.readRecord(current) {
                // Also preserve a manually changed pointer before explicit regeneration.
                let backup = directory.appendingPathComponent("previous-\(UUID().uuidString).json")
                try Self.writePrivate(previous, to: backup, atomic: false)
            }
            let revision = directory.appendingPathComponent(UUID().uuidString + ".json")
            try Self.writePrivate(data, to: revision, atomic: false)
            try Self.writePrivate(data, to: current, atomic: true)
        }
    }

    private static func readRecord(_ url: URL) throws -> Data? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else {
            throw SummaryStoreError.unsafePath
        }
        let handle = try FileHandle(forReadingFrom: url)
        let read = Result { try handle.read(upToCount: Self.maximumBytes + 1) ?? Data() }
        try handle.close()
        let data = try read.get()
        guard data.count <= Self.maximumBytes else { throw SummaryStoreError.oversizedRecord }
        return data
    }

    static func prepareDirectory(_ url: URL) throws {
        if !FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.createDirectory(
                at: url, withIntermediateDirectories: false,
                attributes: [.posixPermissions: 0o700])
        }
        try checkDirectory(url)
    }

    static func checkDirectory(_ url: URL) throws {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isDirectory == true, values.isSymbolicLink != true,
            url.standardizedFileURL == url.resolvingSymlinksInPath().standardizedFileURL
        else { throw SummaryStoreError.unsafePath }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let permissions = attributes[.posixPermissions] as? NSNumber,
            permissions.intValue & 0o077 == 0
        else { throw SummaryStoreError.unsafePath }
    }

    static func writePrivate(_ data: Data, to url: URL, atomic: Bool) throws {
        guard url.standardizedFileURL == url.resolvingSymlinksInPath().standardizedFileURL else {
            throw SummaryStoreError.unsafePath
        }
        try data.write(to: url, options: atomic ? .atomic : .withoutOverwriting)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    static func io<T: Sendable>(_ operation: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { continuation.resume(with: Result(catching: operation)) }
        }
    }
}
