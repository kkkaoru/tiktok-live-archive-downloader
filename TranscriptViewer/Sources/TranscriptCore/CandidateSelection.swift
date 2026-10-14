import CryptoKit
import Foundation

/// Explicitly selects an unverified revision without superseding retained evidence.
struct CandidateSelection: Decodable {
    let version: Int
    let status: String
    let movie: String
    let movieSize: Int
    let transcript: String
    let transcriptSha256: String
    let movieTarget: String

    struct Selection {
        let movie: URL
        let transcript: URL
        let identity: RecordingIdentity
    }

    enum Invalid: Error {
        case manifest, path, provenance, content
    }

    static func load(in folder: URL, library: URL, broadcast: String?, fileManager: FileManager)
        throws -> Selection?
    {
        let manifest = folder.appending(path: "evidence/candidate-selection.json")
        let evidence = folder.appending(path: "evidence")
        // Directory names also detect dangling selection symlinks, which must fail closed.
        guard fileManager.fileExists(atPath: evidence.path) else { return nil }
        guard
            try fileManager.contentsOfDirectory(atPath: evidence.path)
                .contains("candidate-selection.json")
        else { return nil }
        guard manifest.resolvingSymlinksInPath() == manifest.standardizedFileURL,
            try regularSize(manifest) <= 65_536
        else { throw Invalid.manifest }
        let record = try JSONDecoder().decode(Self.self, from: Data(contentsOf: manifest))
        guard record.version == 1, record.status == "unverified-candidate", record.movieSize > 0
        else { throw Invalid.manifest }
        let movie = try relativeDelivery(record.movie, in: folder)
        let transcript = try relativeDelivery(record.transcript, in: folder)
        guard ["mp4", "mov", "m4v"].contains(movie.pathExtension),
            transcript == movie.deletingPathExtension().appendingPathExtension("srt"),
            transcript.resolvingSymlinksInPath() == transcript.standardizedFileURL,
            !record.movieTarget.isEmpty, !record.movieTarget.hasPrefix("/"),
            !record.movieTarget.contains("\\"), !record.movieTarget.contains("\0")
        else { throw Invalid.path }
        let target = folder.appending(path: record.movieTarget).resolvingSymlinksInPath()
        let root = library.resolvingSymlinksInPath().pathComponents
        guard target.pathComponents.count > root.count,
            Array(target.pathComponents.prefix(root.count)) == root,
            !target.pathComponents.contains(where: { ["private", "downloads"].contains($0) }),
            movie.resolvingSymlinksInPath() == target,
            try regularSize(target) == record.movieSize
        else { throw Invalid.content }
        let title = folder.lastPathComponent
        guard let identity = RecordingIdentity(movie: movie), identity.recordingID == title,
            title.hasPrefix("youtube-") || broadcast == identity.broadcastStartJST
        else { throw Invalid.provenance }
        guard try regularSize(transcript) <= 32 * 1_024 * 1_024,
            record.transcriptSha256.count == 64,
            record.transcriptSha256.allSatisfy({ "0123456789abcdef".contains($0) })
        else { throw Invalid.content }
        let digest = SHA256.hash(data: try Data(contentsOf: transcript))
            .map { String(format: "%02x", $0) }.joined()
        guard digest == record.transcriptSha256 else { throw Invalid.content }
        return Selection(movie: movie, transcript: transcript, identity: identity)
    }

    private static func relativeDelivery(_ value: String, in folder: URL) throws -> URL {
        let parts = value.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 4, parts[0] == "revisions", parts[2] == "delivery",
            !parts.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }),
            !value.contains("\\"), !value.contains("\0")
        else { throw Invalid.path }
        let url = folder.appending(path: value)
        guard
            url.deletingLastPathComponent().resolvingSymlinksInPath()
                == url.deletingLastPathComponent().standardizedFileURL
        else { throw Invalid.path }
        return url
    }

    private static func regularSize(_ url: URL) throws -> Int {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true, let size = values.fileSize, size > 0
        else { throw Invalid.content }
        return size
    }
}
