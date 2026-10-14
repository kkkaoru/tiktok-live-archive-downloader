import CryptoKit
import Foundation
import TranscriptCore

public struct RepresentativeSummaryCandidate: Equatable, Sendable {
    public let summary: RepresentativeSourceSummary
    public let canSeekSelectedMovie: Bool
    public let topics: SourceTopicSummary?
    public let prose: SourceSupportedProse?

    public init(
        summary: RepresentativeSourceSummary, canSeekSelectedMovie: Bool,
        topics: SourceTopicSummary? = nil, prose: SourceSupportedProse? = nil
    ) {
        self.summary = summary
        self.canSeekSelectedMovie = canSeekSelectedMovie
        self.topics = topics
        self.prose = prose
    }
}

/// Explicit bindings keep canonical tokens separate from progressive display captions.
public struct RepresentativeSummaryStore: Sendable {
    public static let proseSelectionFilename = "source-supported-prose-selection.json"
    struct Pin: Codable {
        let path: String
        let sha256: String
    }
    struct Selection: Codable {
        let version: Int
        let inputKind: String
        let input: Pin
        let artifact: Pin
        let selectedMovie: String
        let selectedMovieSize: Int
        let selectedTranscriptSha256: String
        let sourceMedia: String?
        let candidateSelection: Pin?
        var morphology: Pin? = nil
    }
    private struct TokenBinding: Decodable { let displaySRT: Pin }
    private struct ArtifactHeader: Decodable { let method: String }
    private struct Delivery: Decodable {
        let version: Int
        let status: String
        let movie: String
        let movieSize: Int
        let transcript: String
        let transcriptSha256: String
    }

    public init() {}

    public func load(video: VideoTranscript) async throws -> RepresentativeSummaryCandidate? {
        try await SummaryStore.io {
            let directory = SummaryStore().directory(for: video)
            guard FileManager.default.fileExists(atPath: directory.path) else { return nil }
            try SummaryStore.checkDirectory(directory)
            let entries = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            // Prose is an explicit new immutable binding. A bad new binding must
            // fail validation instead of silently displaying an older topic artifact.
            let isProseBinding = entries.contains(Self.proseSelectionFilename)
            let filename =
                isProseBinding
                ? Self.proseSelectionFilename : "representative-summary-selection.json"
            guard entries.contains(filename) else { return nil }
            let path = directory.appending(path: filename)
            let selection = try JSONDecoder().decode(
                Selection.self, from: Self.read(path, limit: 65_536))
            let movie = video.movieURL.resolvingSymlinksInPath()
            guard selection.version == 1, selection.selectedMovie == movie.path,
                selection.selectedMovieSize > 0,
                try movie.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]).isRegularFile
                    == true,
                try movie.resourceValues(forKeys: [.fileSizeKey]).fileSize
                    == selection.selectedMovieSize,
                try Self.digest(Self.read(video.transcriptURL, limit: 16 << 20))
                    == selection.selectedTranscriptSha256
            else { throw SummaryStoreError.invalidRecord }
            let input = try Self.pinned(selection.input, limit: 64 << 20)
            let source: VideoTranscript
            let domain: RepresentativeSourceSummary.ClockDomain
            switch selection.inputKind {
            case "caption-tokens":
                let binding = try JSONDecoder().decode(TokenBinding.self, from: input)
                guard binding.displaySRT.sha256 == selection.selectedTranscriptSha256,
                    let candidate = selection.candidateSelection,
                    selection.sourceMedia == nil
                else { throw SummaryStoreError.invalidRecord }
                try Self.validateDelivery(candidate, video: video, selection: selection)
                source = try CaptionTokenTranscript(
                    manifest: URL(fileURLWithPath: selection.input.path)
                ).load()
                domain = .editedOutput
            case "source-labels":
                guard let media = selection.sourceMedia, selection.candidateSelection == nil
                else { throw SummaryStoreError.invalidRecord }
                source = try SourceLabelTranscript(
                    labels: URL(fileURLWithPath: selection.input.path),
                    media: URL(fileURLWithPath: media),
                    clockDomain: "source-presentation", inputKind: .tokenSequence
                ).load()
                domain = .sourcePresentation
            default: throw SummaryStoreError.invalidRecord
            }
            let data = try Self.pinned(selection.artifact, limit: 8 << 20)
            let header = try JSONDecoder().decode(ArtifactHeader.self, from: data)
            guard !isProseBinding || header.method == SourceSupportedProse.methodName else {
                throw SummaryStoreError.invalidRecord
            }
            let artifact: RepresentativeSourceSummary
            let topics: SourceTopicSummary?
            let prose: SourceSupportedProse?
            if header.method == SourceSupportedProse.methodName {
                guard let pin = selection.morphology else { throw SummaryStoreError.invalidRecord }
                let morphology = try JSONDecoder().decode(
                    TranscriptMorphology.self, from: Self.pinned(pin, limit: 64 << 20))
                let result = try JSONDecoder().decode(SourceSupportedProse.self, from: data)
                try result.validate(
                    video: source, fingerprint: SummaryStore.fingerprint(source),
                    clockDomain: domain, morphology: morphology)
                artifact = result.representativeEvidence
                prose = result
                topics = nil
            } else if header.method == SourceTopicSummary.methodName {
                guard selection.morphology == nil else { throw SummaryStoreError.invalidRecord }
                let result = try JSONDecoder().decode(SourceTopicSummary.self, from: data)
                try result.validate(
                    video: source, fingerprint: SummaryStore.fingerprint(source),
                    clockDomain: domain)
                artifact = result.representative
                topics = result
                prose = nil
            } else {
                guard selection.morphology == nil else { throw SummaryStoreError.invalidRecord }
                artifact = try JSONDecoder().decode(RepresentativeSourceSummary.self, from: data)
                topics = nil
                prose = nil
            }
            try artifact.validate(
                video: source, fingerprint: SummaryStore.fingerprint(source), clockDomain: domain)
            return RepresentativeSummaryCandidate(
                summary: artifact,
                canSeekSelectedMovie: domain == .editedOutput || source.movieURL == movie,
                topics: topics, prose: prose)
        }
    }

    private static func validateDelivery(_ pin: Pin, video: VideoTranscript, selection: Selection)
        throws
    {
        let record = try JSONDecoder().decode(Delivery.self, from: pinned(pin, limit: 65_536))
        let path = URL(fileURLWithPath: pin.path)
        let folder = path.deletingLastPathComponent().deletingLastPathComponent()
        guard path.lastPathComponent == "candidate-selection.json",
            path.deletingLastPathComponent().lastPathComponent == "evidence",
            record.version == 1, record.status == "unverified-candidate",
            record.movieSize == selection.selectedMovieSize,
            record.transcriptSha256 == selection.selectedTranscriptSha256,
            safeDelivery(record.movie), safeDelivery(record.transcript),
            folder.appending(path: record.movie).standardizedFileURL
                == video.movieURL.standardizedFileURL,
            folder.appending(path: record.transcript).standardizedFileURL
                == video.transcriptURL.standardizedFileURL
        else { throw SummaryStoreError.invalidRecord }
    }

    private static func safeDelivery(_ path: String) -> Bool {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 4 && parts.first == "revisions" && parts[2] == "delivery"
            && !parts.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." })
    }

    private static func pinned(_ pin: Pin, limit: Int) throws -> Data {
        guard pin.path.hasPrefix("/") else { throw SummaryStoreError.unsafePath }
        let data = try read(URL(fileURLWithPath: pin.path), limit: limit)
        guard digest(data) == pin.sha256 else { throw SummaryStoreError.invalidRecord }
        return data
    }

    private static func read(_ url: URL, limit: Int) throws -> Data {
        guard url == url.standardizedFileURL, url == url.resolvingSymlinksInPath(),
            !url.pathComponents.contains(where: { ["private", "downloads"].contains($0) }),
            try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true
        else { throw SummaryStoreError.unsafePath }
        let file = try FileHandle(forReadingFrom: url)
        let result = Result { try file.read(upToCount: limit + 1) ?? Data() }
        try file.close()
        let data = try result.get()
        guard data.count <= limit else { throw SummaryStoreError.oversizedRecord }
        return data
    }

    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
