import CryptoKit
import Foundation
import TranscriptCore

/// Recognition tokens before display-stage expansion, with explicit segment-clock provenance.
/// The pinned SRT is a related display artifact, not an interchangeable summary fingerprint.
public struct CaptionTokenTranscript: Sendable {
    private struct Pin: Decodable {
        let path: String
        let sha256: String
    }
    private struct Manifest: Decodable {
        struct Job: Decodable {
            let job: Int
            let tokens: Pin
        }
        let version: Int
        let clockDomain: String
        let media: String
        let displaySRT: Pin
        let segmentIndex: Pin
        let reflow: Pin
        let jobs: [Job]
        let sectionStarts: [Int]?
    }
    private struct Index: Decodable {
        struct Segment: Decodable {
            let segment: Int
            let offsetFrame: Int
            let frames: Int
            let jobs: [Int]
        }
        let version: Int
        let rate: Int
        let segments: [Segment]
    }
    private struct Reflow: Decodable { let inputs: [String: String] }
    private struct Tokens: Decodable {
        struct Token: Decodable {
            let text: String
            let start: Double
            let end: Double
            let utterance: String
        }
        let version: Int
        let clock: String
        let tokens: [Token]
    }

    public let manifest: URL
    public init(manifest: URL) { self.manifest = manifest }

    public func load() throws -> VideoTranscript {
        let document = try JSONDecoder().decode(
            Manifest.self, from: Self.read(manifest, limit: 4 << 20))
        guard document.version == 1, document.clockDomain == "edited-output",
            !document.jobs.isEmpty, document.jobs.count <= 10000
        else { throw SummaryStoreError.invalidRecord }
        let media = URL(fileURLWithPath: document.media)
        try Self.validateFile(media)
        _ = try Self.read(document.displaySRT, limit: 16 << 20)
        let index = try JSONDecoder().decode(
            Index.self, from: Self.read(document.segmentIndex, limit: 4 << 20))
        let reflow = try JSONDecoder().decode(
            Reflow.self, from: Self.read(document.reflow, limit: 64 << 20))
        guard index.version == 1, (1...240).contains(index.rate),
            !index.segments.isEmpty, index.segments.count <= 512,
            reflow.inputs[document.segmentIndex.path] == document.segmentIndex.sha256
        else { throw SummaryStoreError.invalidRecord }
        var jobs: [Int: Pin] = [:]
        for job in document.jobs {
            guard job.job >= 0, jobs[job.job] == nil,
                reflow.inputs[job.tokens.path] == job.tokens.sha256
            else { throw SummaryStoreError.invalidRecord }
            jobs[job.job] = job.tokens
        }
        var seen: Set<Int> = []
        var segments: Set<Int> = []
        var cues: [TranscriptCue] = []
        for segment in index.segments {
            guard segment.segment >= 0, segments.insert(segment.segment).inserted,
                segment.offsetFrame >= 0, segment.frames > 0, segment.offsetFrame <= 10_000_000,
                segment.frames <= 10_000_000, !segment.jobs.isEmpty
            else { throw SummaryStoreError.invalidRecord }
            let offset = Double(segment.offsetFrame) / Double(index.rate)
            let duration = Double(segment.frames) / Double(index.rate)
            let firstCue = cues.count
            for job in segment.jobs {
                guard seen.insert(job).inserted, let pin = jobs[job] else {
                    throw SummaryStoreError.invalidRecord
                }
                let tokens = try JSONDecoder().decode(
                    Tokens.self, from: Self.read(pin, limit: 4 << 20))
                guard tokens.version == 1, tokens.clock == "output",
                    tokens.tokens.count <= 10000
                else { throw SummaryStoreError.invalidRecord }
                var previousStart = 0.0
                for token in tokens.tokens {
                    guard token.start.isFinite, token.end.isFinite,
                        token.start >= previousStart, token.end >= token.start,
                        token.end <= duration + 0.000_001,
                        !token.utterance.isEmpty, token.utterance.utf8.count <= 512,
                        !token.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                        token.text.utf8.count <= 16384, cues.count < 1_000_000
                    else { throw SummaryStoreError.invalidRecord }
                    previousStart = token.start
                    let start = offset + token.start
                    let end = offset + token.end
                    cues.append(
                        .init(
                            id: cues.count, startSeconds: start, endSeconds: end,
                            startLabel: String(start), endLabel: String(end), text: token.text))
                }
            }
            // A completed recognition window may contain silence. Keep its pin
            // and membership, but every selected speech segment must contribute.
            guard cues.count > firstCue else { throw SummaryStoreError.invalidRecord }
        }
        guard seen == Set(jobs.keys), !cues.isEmpty else { throw SummaryStoreError.invalidRecord }
        // No prefix collapse: repeated recognition tokens represent original speech.
        cues.sort {
            $0.startSeconds == $1.startSeconds ? $0.id < $1.id : $0.startSeconds < $1.startSeconds
        }
        if let requested = document.sectionStarts {
            let selection = Set(requested)
            guard !selection.isEmpty, selection.count == requested.count,
                requested.allSatisfy({ $0 >= 0 && $0 % 600 == 0 })
            else { throw SummaryStoreError.invalidRecord }
            cues = cues.filter { selection.contains(Int(floor($0.startSeconds / 600)) * 600) }
            guard Set(cues.map { Int(floor($0.startSeconds / 600)) * 600 }) == selection else {
                throw SummaryStoreError.invalidRecord
            }
        }
        return VideoTranscript(
            title: media.deletingPathExtension().lastPathComponent,
            movieURL: media, transcriptURL: manifest, cues: cues,
            summaryInputKind: .tokenSequence)
    }

    private static func validateFile(_ url: URL) throws {
        guard url.isFileURL, url.path == url.standardizedFileURL.path,
            url.path == url.resolvingSymlinksInPath().path,
            !url.pathComponents.contains(where: { ["private", "downloads"].contains($0) }),
            try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true
        else { throw SummaryStoreError.unsafePath }
    }

    private static func read(_ url: URL, limit: Int) throws -> Data {
        try validateFile(url)
        let file = try FileHandle(forReadingFrom: url)
        let read = Result { try file.read(upToCount: limit + 1) ?? Data() }
        try file.close()
        let data = try read.get()
        guard data.count <= limit else { throw SummaryStoreError.oversizedRecord }
        return data
    }

    private static func read(_ pin: Pin, limit: Int) throws -> Data {
        let data = try read(URL(fileURLWithPath: pin.path), limit: limit)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard digest == pin.sha256 else { throw SummaryStoreError.invalidRecord }
        return data
    }
}
