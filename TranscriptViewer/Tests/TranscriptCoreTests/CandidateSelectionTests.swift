import CryptoKit
import Foundation
import Testing

@testable import TranscriptCore

struct CandidateSelectionTests {
    struct Fixture {
        let root: URL
        let folder: URL
        let movie: URL
        let transcript: URL
        let manifest: URL
        var fields: [String: String]

        init(youtube: Bool = false, ext: String = "mp4", recordID: String? = nil) throws {
            root = try transcriptTestWorkRoot().appending(path: "candidate-\(UUID().uuidString)")
            let title = recordID ?? (youtube ? "youtube-AbC012_-xyz" : "replay-123")
            folder = root.appending(path: title)
            let name = "2026-10-04_14-30-09_JST_\(title)-review"
            movie = folder.appending(path: "revisions/new/delivery/\(name).\(ext)")
            transcript = movie.deletingPathExtension().appendingPathExtension("srt")
            manifest = folder.appending(path: "evidence/candidate-selection.json")
            try FileManager.default.createDirectory(
                at: movie.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.createDirectory(
                at: manifest.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("movie".utf8).write(to: movie)
            let srt = Data("1\n00:00:00,100 --> 00:00:01,200\nhello\n".utf8)
            try srt.write(to: transcript)
            try Data(
                "{\"format\":{\"filename\":\"/media/2026-10-04_14-30-09_JST_\(title).mp4\"}}".utf8
            ).write(to: folder.appending(path: "evidence/source-probe.json"))
            fields = [
                "status": "unverified-candidate",
                "movie": "revisions/new/delivery/\(name).\(ext)",
                "movieTarget": "revisions/new/delivery/\(name).\(ext)",
                "transcript": "revisions/new/delivery/\(name).srt",
                "transcriptSha256": SHA256.hash(data: srt).map { String(format: "%02x", $0) }
                    .joined(),
            ]
            try write()
        }

        func write(version: Int = 1, size: Int = 5) throws {
            let encoder = JSONEncoder()
            let encoded = try fields.sorted(by: { $0.key < $1.key }).map { key, value in
                let data = try encoder.encode(value)
                return "\"\(key)\":\(String(decoding: data, as: UTF8.self))"
            }.joined(separator: ",")
            try Data("{\"version\":\(version),\"movieSize\":\(size),\(encoded)}".utf8)
                .write(to: manifest)
        }

        func remove() throws { try FileManager.default.removeItem(at: root) }
    }

    @Test(arguments: ["mp4", "mov", "m4v"])
    func selectsOneRevision(ext: String) throws {
        let fixture = try Fixture(ext: ext)
        let old = fixture.folder.appending(path: "revisions/old/delivery")
        try FileManager.default.createDirectory(at: old, withIntermediateDirectories: true)
        try Data("old".utf8).write(to: old.appending(path: fixture.movie.lastPathComponent))
        let scan = try TranscriptCatalog.scan(in: fixture.root)
        #expect(scan.videos.count == 1)
        #expect(scan.videos.first?.title == "replay-123/new [未検証候補]")
        #expect(scan.videos.first?.transcriptURL == fixture.transcript)
        #expect(scan.issues.isEmpty)
        try fixture.remove()
    }

    @Test(arguments: ["replay-123", "youtube-AbC012_-xyz"])
    func openingARecordingFolderKeepsOneDatedCandidateAndFailsClosed(recordID: String) throws {
        var fixture = try Fixture(recordID: recordID)
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let old = fixture.folder.appending(path: "revisions/old/delivery")
        try FileManager.default.createDirectory(at: old, withIntermediateDirectories: true)
        try Data("old".utf8).write(to: old.appending(path: fixture.movie.lastPathComponent))
        try Data(contentsOf: fixture.transcript).write(
            to: old.appending(path: fixture.transcript.lastPathComponent))
        let scan = try TranscriptCatalog.scan(in: fixture.folder)
        #expect(scan.videos.count == 1)
        #expect(scan.videos.first?.movieURL == fixture.movie)
        #expect(scan.videos.first?.broadcastStartJST == "2026-10-04 14:30:09 JST")
        #expect(scan.videos.first?.title == "\(recordID)/new [未検証候補]")
        fixture.fields["status"] = "verified"
        try fixture.write()
        let rejected = try TranscriptCatalog.scan(in: fixture.folder)
        #expect(rejected.videos.isEmpty)
        #expect(rejected.invalidCandidateSelectionLabels == [recordID])
    }

    @Test func recordingFolderWithoutASelectionStillUsesProvenanceAndAmbiguityGuards() throws {
        let fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try FileManager.default.removeItem(at: fixture.manifest)
        let scan = try TranscriptCatalog.scan(in: fixture.folder)
        #expect(scan.videos.count == 1)
        #expect(scan.videos.first?.broadcastStartJST == "2026-10-04 14:30:09 JST")
        let old = fixture.folder.appending(path: "revisions/old/delivery")
        try FileManager.default.createDirectory(at: old, withIntermediateDirectories: true)
        try Data("old".utf8).write(to: old.appending(path: fixture.movie.lastPathComponent))
        try Data(contentsOf: fixture.transcript).write(
            to: old.appending(path: fixture.transcript.lastPathComponent))
        let ambiguous = try TranscriptCatalog.scan(in: fixture.folder)
        #expect(ambiguous.videos.isEmpty)
        #expect(ambiguous.ambiguousDeliveryLabels == ["replay-123"])
    }

    @Test(arguments: ["replay-123", "youtube-AbC012_-xyz"])
    func narrowRootsUseTheSamePinnedDatedRecordingWithSiblingMediaTarget(recordID: String) throws {
        var fixture = try Fixture(recordID: recordID)
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let target = fixture.folder.appending(path: "revisions/encoded/movie.mp4")
        try FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: fixture.movie, to: target)
        try FileManager.default.createSymbolicLink(at: fixture.movie, withDestinationURL: target)
        fixture.fields["movieTarget"] = "revisions/encoded/movie.mp4"
        try fixture.write()
        let delivery = fixture.movie.deletingLastPathComponent()
        let revision = delivery.deletingLastPathComponent()
        let full = try TranscriptCatalog.scan(in: fixture.root)
        let record = try TranscriptCatalog.scan(in: fixture.folder)
        let narrowRevision = try TranscriptCatalog.scan(in: revision)
        let narrowDelivery = try TranscriptCatalog.scan(in: delivery)
        #expect(full.videos.count == 1)
        #expect(full.videos.first?.broadcastStartJST == "2026-10-04 14:30:09 JST")
        #expect(full.videos.first?.title == "\(recordID)/new [未検証候補]")
        #expect(record.videos == full.videos)
        #expect(narrowRevision.videos == full.videos)
        #expect(narrowDelivery.videos == full.videos)
        #expect(narrowRevision.issues.isEmpty)
        #expect(narrowDelivery.issues.isEmpty)
    }

    @Test(arguments: ["missing-selection", "different-revision", "conflicting-source-date"])
    func narrowRootsRejectMissingOrConflictingSelection(change: String) throws {
        let fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var opened = fixture.movie.deletingLastPathComponent()
        switch change {
        case "missing-selection": try FileManager.default.removeItem(at: fixture.manifest)
        case "different-revision":
            opened = fixture.folder.appending(path: "revisions/old/delivery")
            try FileManager.default.createDirectory(at: opened, withIntermediateDirectories: true)
            try Data("movie".utf8).write(
                to: opened.appending(path: fixture.movie.lastPathComponent))
            try Data(contentsOf: fixture.transcript).write(
                to: opened.appending(path: fixture.transcript.lastPathComponent))
        case "conflicting-source-date":
            try Data(
                #"{"format":{"filename":"/media/2026-10-05_14-30-09_JST_replay-123.mp4"}}"#.utf8
            ).write(to: fixture.folder.appending(path: "evidence/source-probe.json"))
        default: Issue.record("Unknown fixture change")
        }
        let scan = try TranscriptCatalog.scan(in: opened)
        #expect(scan.videos.isEmpty)
        #expect(scan.invalidCandidateSelectionLabels == ["replay-123"])
    }

    @Test func revisionNamedDeliveryKeepsItsRecordingIdentity() throws {
        var fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let revision = fixture.folder.appending(path: "revisions/delivery")
        try FileManager.default.moveItem(
            at: fixture.movie.deletingLastPathComponent().deletingLastPathComponent(), to: revision)
        fixture.fields["movie"] = "revisions/delivery/delivery/\(fixture.movie.lastPathComponent)"
        fixture.fields["movieTarget"] = fixture.fields["movie"]
        fixture.fields["transcript"] =
            "revisions/delivery/delivery/\(fixture.transcript.lastPathComponent)"
        try fixture.write()
        let scan = try TranscriptCatalog.scan(in: revision)
        let delivery = try TranscriptCatalog.scan(in: revision.appending(path: "delivery"))
        #expect(scan.videos.count == 1)
        #expect(scan.videos.first?.broadcastStartJST == "2026-10-04 14:30:09 JST")
        #expect(scan.videos.first?.title == "replay-123/delivery [未検証候補]")
        #expect(delivery.videos == scan.videos)
        #expect(scan.issues.isEmpty)
        #expect(delivery.issues.isEmpty)
    }

    @Test(arguments: [
        "replay-not-an-id", "replay-" + String(repeating: "1", count: 33),
        "youtube-short", "youtube-AbC012_-xyz\n",
    ])
    func nonRecordNamesRetainGenericChildScanning(name: String) throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let root = fixture.root.appending(path: name)
        let child = root.appending(path: "clip")
        try FileManager.default.createDirectory(at: child, withIntermediateDirectories: true)
        try Data("movie".utf8).write(to: child.appending(path: "sample.mp4"))
        try Data(contentsOf: fixture.video.transcriptURL).write(
            to: child.appending(path: "sample.srt"))
        let scan = try TranscriptCatalog.scan(in: root)
        #expect(scan.videos.count == 1)
        #expect(scan.videos.first?.title == "clip")
    }

    @Test func youtubeAndRelativeMediaTarget() throws {
        let fixture = try Fixture(youtube: true, ext: "m4v")
        let target = fixture.root.appending(path: "encoded.m4v")
        try FileManager.default.moveItem(at: fixture.movie, to: target)
        try FileManager.default.createSymbolicLink(at: fixture.movie, withDestinationURL: target)
        var updated = fixture
        updated.fields["movieTarget"] = "../encoded.m4v"
        try updated.write()
        let scan = try TranscriptCatalog.scan(in: fixture.root)
        #expect(scan.videos.count == 1)
        #expect(scan.videos.first?.broadcastStartJST == "2026-10-04 14:30:09 JST")
        try fixture.remove()
    }

    @Test(arguments: [
        ("movie", "../outside.mp4"), ("movie", "revisions/../delivery/movie.mp4"),
        ("movie", "revisions//delivery/movie.mp4"),
        ("movie", "revisions/a/delivery/evil\\name.mp4"),
        ("transcript", "revisions/new/delivery/other.srt"),
        ("movieTarget", "/outside.mp4"), ("movieTarget", "../../outside.mp4"),
        ("movieTarget", ""), ("transcriptSha256", "BAD"),
        ("transcriptSha256", String(repeating: "0", count: 64)),
        ("status", "verified"),
    ])
    func invalidRecordsFailClosed(field: String, value: String) throws {
        var fixture = try Fixture()
        fixture.fields[field] = value
        try fixture.write()
        let scan = try TranscriptCatalog.scan(in: fixture.root)
        #expect(scan.videos.isEmpty)
        #expect(scan.invalidCandidateSelectionLabels == ["replay-123"])
        #expect(scan.issues.contains("候補の選択が不正: replay-123"))
        try fixture.remove()
    }

    @Test(arguments: [
        "size", "version", "empty", "json", "missing-movie", "missing-srt", "date", "srt-change",
        "manifest-link", "srt-link", "directory-link",
    ])
    func changedEvidenceFailsClosed(change: String) throws {
        let fixture = try Fixture()
        switch change {
        case "size": try fixture.write(size: 6)
        case "version": try fixture.write(version: 2)
        case "empty": try Data().write(to: fixture.manifest)
        case "json": try Data("invalid".utf8).write(to: fixture.manifest)
        case "missing-movie": try FileManager.default.removeItem(at: fixture.movie)
        case "missing-srt": try FileManager.default.removeItem(at: fixture.transcript)
        case "date":
            try FileManager.default.removeItem(
                at: fixture.folder.appending(path: "evidence/source-probe.json"))
        case "srt-change": try Data("changed".utf8).write(to: fixture.transcript)
        case "manifest-link", "srt-link", "directory-link":
            let source =
                change == "manifest-link"
                ? fixture.manifest
                : (change == "srt-link"
                    ? fixture.transcript : fixture.movie.deletingLastPathComponent())
            let target = fixture.root.appending(path: ".retained")
            try FileManager.default.moveItem(at: source, to: target)
            try FileManager.default.createSymbolicLink(at: source, withDestinationURL: target)
        default: Issue.record("Unknown fixture change")
        }
        let scan = try TranscriptCatalog.scan(in: fixture.root)
        #expect(scan.videos.isEmpty)
        #expect(scan.invalidCandidateSelectionLabels == ["replay-123"])
        try fixture.remove()
    }

    @Test func absentSelectionRetainsLegacyBehavior() throws {
        let fixture = try Fixture()
        try FileManager.default.removeItem(at: fixture.manifest)
        let scan = try TranscriptCatalog.scan(in: fixture.root)
        #expect(scan.videos.first?.title == "replay-123/new")
        #expect(scan.invalidCandidateSelectionLabels.isEmpty)
        try fixture.remove()
    }
}
