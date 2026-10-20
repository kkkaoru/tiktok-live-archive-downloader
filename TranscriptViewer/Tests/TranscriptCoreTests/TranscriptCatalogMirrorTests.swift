import Foundation
import Testing

@testable import TranscriptCore

struct TranscriptCatalogMirrorTests {
    private static let stem = "2026-09-06_18-29-52_JST_replay-7-review"
    private static let delivery = "revisions/render-a/delivery"

    /// A record mirror (`backup/replay-7`) holding the SRT and edit-config, and an
    /// offload store (`deliveries/offload/replay-7`) holding only the movie.
    private static func makeLibraries(storeEvidence: Bool = false, movie: Bool = true) throws
        -> (backup: URL, deliveries: URL, store: URL)
    {
        let base = try transcriptTestWorkRoot().appending(
            path: "mirror-\(UUID().uuidString)", directoryHint: .isDirectory)
        let backup = base.appending(path: "backup", directoryHint: .isDirectory)
        let deliveries = base.appending(path: "deliveries", directoryHint: .isDirectory)
        let store = deliveries.appending(path: "offload/replay-7", directoryHint: .isDirectory)
        let mirror = backup.appending(path: "replay-7", directoryHint: .isDirectory)
        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: mirror.appending(path: delivery), withIntermediateDirectories: true)
        try fileManager.createDirectory(
            at: store.appending(path: delivery), withIntermediateDirectories: true)
        try Data("1\n00:00:00,000 --> 00:00:01,000\nmirrored\n".utf8)
            .write(to: mirror.appending(path: "\(delivery)/\(stem).srt"))
        try Data(
            #"{"source":"/m/2026-09-06_18-29-52_JST_replay-7.mp4","storage_root":"\#(store.path(percentEncoded: false))"}"#
                .utf8
        ).write(to: mirror.appending(path: "edit-config.json"))
        if movie {
            try Data().write(to: store.appending(path: "\(delivery)/\(stem).mp4"))
        }
        if storeEvidence {
            try fileManager.createDirectory(
                at: store.appending(path: "evidence"), withIntermediateDirectories: true)
        }
        return (backup, deliveries, store)
    }

    @Test func mirrorLoadsTheOffloadedMovieDatedByItsEditConfig() throws {
        let libraries = try Self.makeLibraries()
        let scan = try TranscriptCatalog.scan(in: libraries.backup)
        #expect(scan.videos.count == 1)
        #expect(scan.videos[0].title == "replay-7/render-a")
        #expect(
            scan.videos[0].movieURL.path(percentEncoded: false)
                == libraries.store.appending(path: "\(Self.delivery)/\(Self.stem).mp4")
                .path(percentEncoded: false))
        #expect(scan.videos[0].cues.map(\.text) == ["mirrored"])
        #expect(scan.videos[0].broadcastStartJST == "2026-09-06 18:29:52 JST")
        #expect(scan.issues.isEmpty)
        // The store's own library does not report its media as movies without captions.
        #expect(try TranscriptCatalog.scan(in: libraries.deliveries).issues.isEmpty)
        try FileManager.default.removeItem(at: libraries.backup.deletingLastPathComponent())
    }

    @Test func storeWithEvidenceIsLeftToItsOwnLibrary() throws {
        let libraries = try Self.makeLibraries(storeEvidence: true)
        let scan = try TranscriptCatalog.scan(in: libraries.backup)
        #expect(scan.videos.isEmpty)
        #expect(scan.issues.isEmpty)
        try FileManager.default.removeItem(at: libraries.backup.deletingLastPathComponent())
    }

    @Test func missingOffloadedMovieIsNotLoaded() throws {
        let libraries = try Self.makeLibraries(movie: false)
        let scan = try TranscriptCatalog.scan(in: libraries.backup)
        #expect(scan.videos.isEmpty)
        try FileManager.default.removeItem(at: libraries.backup.deletingLastPathComponent())
    }
}
