import CryptoKit
import Foundation
import TranscriptCore

/// A separately selected source-clock excerpt artifact. Exact text is not quality approval.
public struct SourceExcerptCandidate: Equatable, Sendable {
    public let summary: SourceExcerptSummary
    public let canSeekSelectedMovie: Bool
}

public struct SourceExcerptStore: Sendable {
    struct Selection: Codable {
        let version: Int
        let clockDomain: String
        let sourceLabelLayout: String
        let sourceLabels: String
        let sourceLabelsSha256: String
        let sourceMedia: String
        let sourceMediaSize: Int
        let artifact: String
        let artifactSha256: String
    }

    public init() {}

    /// Reads only an explicit sidecar and its pinned inputs, never a source directory scan.
    public func load(video: VideoTranscript) async throws -> SourceExcerptCandidate? {
        try await SummaryStore.io {
            let directory = SummaryStore().directory(for: video)
            guard FileManager.default.fileExists(atPath: directory.path) else { return nil }
            try SummaryStore.checkDirectory(directory)
            let pointer = directory.appending(path: "source-excerpts-selection.json")
            guard
                try FileManager.default.contentsOfDirectory(atPath: directory.path)
                    .contains(pointer.lastPathComponent)
            else { return nil }
            let selection = try JSONDecoder().decode(
                Selection.self, from: Self.read(pointer, maximum: 65_536))
            guard selection.version == 1, selection.clockDomain == "source-presentation",
                selection.sourceLabelLayout == "tokens", selection.sourceMediaSize > 0
            else { throw SummaryStoreError.invalidRecord }
            let labels = try Self.absolute(selection.sourceLabels)
            let media = try Self.absolute(selection.sourceMedia)
            let artifact = try Self.absolute(selection.artifact)
            guard
                try media.resourceValues(forKeys: [.fileSizeKey]).fileSize
                    == selection.sourceMediaSize
            else { throw SummaryStoreError.invalidRecord }
            try Self.checkHash(
                Self.read(labels, maximum: SourceLabelTranscript.maximumBytes),
                expected: selection.sourceLabelsSha256)
            let data = try Self.read(artifact, maximum: 8 * 1_024 * 1_024)
            try Self.checkHash(data, expected: selection.artifactSha256)
            let source = try SourceLabelTranscript(
                labels: labels, media: media, clockDomain: selection.clockDomain,
                inputKind: .tokenSequence
            ).load()
            let summary = try JSONDecoder().decode(SourceExcerptSummary.self, from: data)
            try summary.validate(
                source: source.summary, fingerprint: SummaryStore.fingerprint(source))
            return SourceExcerptCandidate(
                summary: summary,
                canSeekSelectedMovie: video.movieURL.resolvingSymlinksInPath() == media)
        }
    }

    private static func absolute(_ path: String) throws -> URL {
        guard path.hasPrefix("/"), !path.contains("\0") else { throw SummaryStoreError.unsafePath }
        let url = URL(fileURLWithPath: path)
        guard url == url.standardizedFileURL, url == url.resolvingSymlinksInPath(),
            !url.pathComponents.contains(where: { ["private", "downloads"].contains($0) }),
            try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true
        else { throw SummaryStoreError.unsafePath }
        return url
    }

    private static func read(_ url: URL, maximum: Int) throws -> Data {
        _ = try absolute(url.path)
        let handle = try FileHandle(forReadingFrom: url)
        let result = Result { try handle.read(upToCount: maximum + 1) ?? Data() }
        try handle.close()
        let data = try result.get()
        guard data.count <= maximum else { throw SummaryStoreError.oversizedRecord }
        return data
    }

    private static func checkHash(_ data: Data, expected: String) throws {
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard digest == expected else { throw SummaryStoreError.invalidRecord }
    }
}
