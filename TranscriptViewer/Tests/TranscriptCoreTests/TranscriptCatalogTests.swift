import Foundation
import Testing

@testable import TranscriptCore

struct TranscriptCatalogTests {
    private static let body = Data(
        "start_seconds,end_seconds,start,end,text\n0.000,1.000,00:00:00.000,00:00:01.000,hi\n".utf8
    )

    private static func makeRoot(_ name: String) throws -> URL {
        let root = try transcriptTestWorkRoot().appending(
            path: "\(name)-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @Test func loadsDatedFolderWithNestedMovie() throws {
        let root = try Self.makeRoot("lib")
        let dated = root.appending(path: "2026-09-14_replay-1", directoryHint: .isDirectory)
        let compressor = dated.appending(path: "compressor-A", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: compressor, withIntermediateDirectories: true)
        try Data().write(to: root.appending(path: "readme.txt"))
        let csv = dated.appending(path: "talk.csv")
        let csvB = dated.appending(path: "talk-b.csv")
        try Self.body.write(to: csv)
        try Self.body.write(to: csvB)
        let olderMovie = compressor.appending(path: "talk.mov")
        let newerMovie = compressor.appending(path: "talk-b.mov")
        try Data().write(to: olderMovie)
        try Data().write(to: newerMovie)
        let scan = try TranscriptCatalog.scan(in: root)
        #expect(scan.videos.count == 2)
        #expect(scan.issues.isEmpty)
        #expect(scan.videos[0].title == "2026-09-14_replay-1")
        #expect(scan.videos[0].id.lastPathComponent == "talk-b.mov")
        #expect(scan.videos[0].movieURL.lastPathComponent == "talk-b.mov")
        #expect(scan.videos[1].movieURL.lastPathComponent == "talk.mov")
        #expect(scan.videos[0].cues[0].text == "hi")
        try FileManager.default.removeItem(at: root)
    }

    @Test @MainActor func backgroundScanReturnsOnMainActorWithTheSameVideos() async throws {
        let root = try Self.makeRoot("lib-background-scan")
        try Self.body.write(to: root.appending(path: "sample.csv"))
        try Data().write(to: root.appending(path: "sample.mp4"))

        let result: Result<CatalogScan, Error> = await withCheckedContinuation { continuation in
            TranscriptCatalog.scan(in: root) { result in
                #expect(Thread.isMainThread)
                continuation.resume(returning: result)
            }
        }

        #expect(try result.get().videos.map(\.title) == [root.lastPathComponent])
        try FileManager.default.removeItem(at: root)
    }

    @Test @MainActor func backgroundScanReportsErrorsWithoutLosingTheCallback() async throws {
        let root = try Self.makeRoot("lib-background-error")
        let result: Result<CatalogScan, Error> = await withCheckedContinuation { continuation in
            TranscriptCatalog.scan(in: root.appending(path: "missing")) { result in
                #expect(Thread.isMainThread)
                continuation.resume(returning: result)
            }
        }

        #expect(throws: TranscriptError.self) { try result.get() }
        try FileManager.default.removeItem(at: root)
    }

    @Test func pendingMovieIsReportedWithoutBlockingOtherFolders() throws {
        let root = try Self.makeRoot("lib-pending")
        let done = root.appending(path: "2026-09-14_replay-1", directoryHint: .isDirectory)
        let encoding = root.appending(path: "2026-09-15_replay-2", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: done, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: encoding, withIntermediateDirectories: true)
        try Self.body.write(to: done.appending(path: "replay-1.csv"))
        try Data().write(to: done.appending(path: "replay-1.mov"))
        try Self.body.write(to: encoding.appending(path: "replay-2.csv"))

        let scan = try TranscriptCatalog.scan(in: root)

        #expect(scan.videos.map(\.title) == ["2026-09-14_replay-1"])
        #expect(scan.pendingMovieLabels == ["2026-09-15_replay-2/replay-2"])
        #expect(scan.issues == ["動画待ち: 2026-09-15_replay-2/replay-2"])
        try FileManager.default.removeItem(at: root)
    }

    @Test func unreadableTranscriptIsReportedWithoutBlockingOtherFolders() throws {
        let root = try Self.makeRoot("lib-broken")
        let done = root.appending(path: "2026-09-14_replay-1", directoryHint: .isDirectory)
        let broken = root.appending(path: "2026-09-15_replay-2", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: done, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: broken, withIntermediateDirectories: true)
        try Self.body.write(to: done.appending(path: "replay-1.csv"))
        try Data().write(to: done.appending(path: "replay-1.mov"))
        try Data("not,a,transcript\n".utf8).write(to: broken.appending(path: "replay-2.csv"))
        try Data().write(to: broken.appending(path: "replay-2.mov"))

        let scan = try TranscriptCatalog.scan(in: root)

        #expect(scan.videos.map(\.title) == ["2026-09-14_replay-1"])
        #expect(
            scan.unreadableTranscriptLabels == ["2026-09-15_replay-2/replay-2.csv (CSVの見出しが不正です)"])
        #expect(scan.issues == ["読み込み不可: 2026-09-15_replay-2/replay-2.csv (CSVの見出しが不正です)"])
        try FileManager.default.removeItem(at: root)
    }

    @Test func undecodableTranscriptFileIsReported() throws {
        let root = try Self.makeRoot("lib-binary")
        let dated = root.appending(path: "2026-09-14_replay-1", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: dated, withIntermediateDirectories: true)
        try Data([0xFF, 0xFE, 0x00, 0x01]).write(to: dated.appending(path: "replay-1.csv"))
        try Data().write(to: dated.appending(path: "replay-1.mov"))

        let scan = try TranscriptCatalog.scan(in: root)

        #expect(scan.videos.isEmpty)
        #expect(scan.unreadableTranscriptLabels.count == 1)
        #expect(scan.unreadableTranscriptLabels[0].hasPrefix("2026-09-14_replay-1/replay-1.csv ("))
        try FileManager.default.removeItem(at: root)
    }

    @Test func loadsProcessedMP4WithNestedSRT() throws {
        let root = try Self.makeRoot("lib-processed")
        let revision = root.appending(path: "processed-revision", directoryHint: .isDirectory)
        let delivery = revision.appending(path: "delivery", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
        let movie = delivery.appending(path: "full-review.mp4")
        let subtitle = delivery.appending(path: "full-review.srt")
        try Data().write(to: movie)
        try Data("1\n00:00:00,000 --> 00:00:01,250\nprocessed dialogue\n".utf8)
            .write(to: subtitle)

        let scan = try TranscriptCatalog.scan(in: root)

        #expect(scan.videos.count == 1)
        let video = try #require(scan.videos.first)
        #expect(video.title == "processed-revision")
        #expect(video.movieURL.resolvingSymlinksInPath() == movie.resolvingSymlinksInPath())
        #expect(video.transcriptURL.resolvingSymlinksInPath() == subtitle.resolvingSymlinksInPath())
        #expect(video.cues.map(\.text) == ["processed dialogue"])
        #expect(scan.issues.isEmpty)
        try FileManager.default.removeItem(at: root)
    }

    @Test func movieWithoutTranscriptIsReported() throws {
        let root = try Self.makeRoot("lib-movie-only")
        let dated = root.appending(path: "2026-09-14_replay-1", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: dated, withIntermediateDirectories: true)
        let movie = dated.appending(path: "replay-1.mov")
        try Data().write(to: movie)

        let scan = try TranscriptCatalog.scan(in: root)

        #expect(scan.videos.isEmpty)
        #expect(scan.moviesWithoutTranscript.map(\.lastPathComponent) == ["replay-1.mov"])
        #expect(scan.issues == ["字幕なし: replay-1.mov"])
        try FileManager.default.removeItem(at: root)
    }

    @Test func folderWithoutAnyDeliverableIsReported() throws {
        let root = try Self.makeRoot("lib-empty")
        let done = root.appending(path: "2026-09-14_replay-1", directoryHint: .isDirectory)
        let empty = root.appending(path: "2026-09-18_replay-2", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: done, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        try Self.body.write(to: done.appending(path: "replay-1.csv"))
        try Data().write(to: done.appending(path: "replay-1.mov"))

        let scan = try TranscriptCatalog.scan(in: root)

        #expect(scan.videos.count == 1)
        #expect(scan.emptyFolders == ["2026-09-18_replay-2"])
        #expect(scan.issues == ["未処理: 2026-09-18_replay-2"])
        try FileManager.default.removeItem(at: root)
    }

    @Test func loadsBothMoviesWhenStemsCollideInSeparateFolders() throws {
        let root = try Self.makeRoot("lib-duplicate")
        let dated = root.appending(path: "2026-09-14_replay-1", directoryHint: .isDirectory)
        let nested = dated.appending(path: "compressor-a", directoryHint: .isDirectory)
        let alsoNested = dated.appending(path: "compressor-b", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: alsoNested, withIntermediateDirectories: true)
        try Self.body.write(to: dated.appending(path: "replay-1.csv"))
        let firstMovie = nested.appending(path: "replay-1.mov")
        let secondMovie = alsoNested.appending(path: "replay-1.mov")
        try Data().write(to: firstMovie)
        try Data().write(to: secondMovie)

        let scan = try TranscriptCatalog.scan(in: root)

        #expect(scan.videos.count == 2)
        #expect(scan.videos[0].movieURL.path(percentEncoded: false).contains("compressor-a/"))
        #expect(scan.videos[1].movieURL.path(percentEncoded: false).contains("compressor-b/"))
        #expect(scan.videos[0].cues == scan.videos[1].cues)
        #expect(scan.issues.isEmpty)
        try FileManager.default.removeItem(at: root)
    }

    @Test func loadsEveryRevisionWithTheSameMovieNameWithoutIntermediates() throws {
        let root = try Self.makeRoot("lib-revisions")
        let replay = root.appending(path: "replay-synthetic", directoryHint: .isDirectory)
        let first = replay.appending(path: "revisions/first/delivery", directoryHint: .isDirectory)
        let second = replay.appending(
            path: "revisions/second/render/delivery", directoryHint: .isDirectory)
        let work = replay.appending(path: "work", directoryHint: .isDirectory)
        let privateFolder = root.appending(path: "private", directoryHint: .isDirectory)
        for folder in [first, second, work, privateFolder] {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try Data().write(to: folder.appending(path: "full-review.mp4"))
            let text = folder == first ? "first" : "second"
            try Data("1\n00:00:00,000 --> 00:00:01,250\n\(text)\n".utf8)
                .write(to: folder.appending(path: "full-review.srt"))
        }
        try FileManager.default.createDirectory(
            at: first.appending(path: "not-a-movie.mp4"), withIntermediateDirectories: true)
        try Data().write(to: first.appending(path: "full-review.incomplete.mp4"))
        try Data("1\n00:00:00,000 --> 00:00:01,250\npartial\n".utf8)
            .write(to: first.appending(path: "full-review.incomplete.srt"))
        let scan = try TranscriptCatalog.scan(in: root)
        #expect(scan.videos.isEmpty)
        #expect(scan.unknownBroadcastDateLabels == ["replay-synthetic"])
        try FileManager.default.removeItem(at: root)
    }

    @Test func processedSubtitleWinsOverStaleCSVForSameMovie() throws {
        let root = try Self.makeRoot("lib-subtitle-preference")
        let delivery = root.appending(path: "revision/delivery", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
        try Data().write(to: delivery.appending(path: "full-review.mp4"))
        try Data("bad,csv\n".utf8).write(to: delivery.appending(path: "full-review.csv"))
        try Data("1\n00:00:00,000 --> 00:00:01,000\nselected subtitle\n".utf8)
            .write(to: delivery.appending(path: "full-review.srt"))
        let scan = try TranscriptCatalog.scan(in: root)
        #expect(scan.videos.count == 1)
        #expect(scan.videos[0].cues.map(\.text) == ["selected subtitle"])
        #expect(scan.issues.isEmpty)
        try FileManager.default.removeItem(at: root)
    }

    @Test func loadsRootLevelAndNestedMoviesWithoutDuplicates() throws {
        let root = try Self.makeRoot("lib-root-movies")
        let nested = root.appending(path: "replay-synthetic", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        let srt = Data("1\n00:00:00,000 --> 00:00:01,000\ncaption\n".utf8)
        let rootMovie = root.appending(path: "full-review.mp4")
        let nestedMovie = nested.appending(path: "full-review.mp4")
        try Data().write(to: rootMovie)
        try srt.write(to: root.appending(path: "full-review.srt"))
        try Data().write(to: nestedMovie)
        try srt.write(to: nested.appending(path: "full-review.srt"))
        let scan = try TranscriptCatalog.scan(in: root)
        #expect(scan.videos.map(\.title) == [root.lastPathComponent])
        #expect(scan.pendingMovieLabels == ["replay-synthetic/full-review"])
        try FileManager.default.removeItem(at: root)
    }

    @Test func rootTranscriptCannotSilentlyDescribeAnotherFolder() throws {
        let root = try Self.makeRoot("lib-root-transcript")
        let nested = root.appending(path: "legacy/delivery", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data().write(to: nested.appending(path: "movie.mov"))
        try Data("1\n00:00:00,000 --> 00:00:01,000\nlegacy subtitle\n".utf8)
            .write(to: root.appending(path: "movie.srt"))
        let scan = try TranscriptCatalog.scan(in: root)
        #expect(scan.videos.isEmpty)
        #expect(scan.moviesWithoutTranscript.map(\.lastPathComponent) == ["movie.mov"])
        #expect(scan.pendingMovieLabels == ["\(root.lastPathComponent)/movie"])
        try FileManager.default.removeItem(at: root)
    }

    @Test func incompleteRevisionCannotBorrowAnotherReplaysSubtitle() throws {
        let root = try Self.makeRoot("lib-unmatched")
        let replay = root.appending(path: "replay-synthetic", directoryHint: .isDirectory)
        let delivery = replay.appending(
            path: "revisions/unfinished/delivery", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
        try Data().write(to: delivery.appending(path: "full-review.mp4"))
        try Data("1\n00:00:00,000 --> 00:00:01,000\nstale\n".utf8)
            .write(to: replay.appending(path: "full-review.srt"))
        let scan = try TranscriptCatalog.scan(in: root)
        #expect(scan.videos.isEmpty)
        #expect(scan.moviesWithoutTranscript.isEmpty)
        try FileManager.default.removeItem(at: root)
    }

    @Test func untrustedBroadcastDateIsReported() {
        var scan = CatalogScan()
        scan.unknownBroadcastDateLabels = ["replay-synthetic"]
        #expect(scan.issues == ["配信日時不明: replay-synthetic"])
    }

    @Test func broadcastDateDescendingIgnoresFolderAndFileModificationOrder() throws {
        let root = try Self.makeRoot("lib-broadcast-order")
        let recent = root.appending(
            path: "replay-z/revisions/new/delivery", directoryHint: .isDirectory)
        let older = root.appending(
            path: "replay-a/revisions/old/delivery", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: recent, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: older, withIntermediateDirectories: true)
        let recentMovie = recent.appending(path: "full-review.mp4")
        let olderMovie = older.appending(path: "full-review.mp4")
        let subtitle = Data("1\n00:00:00,000 --> 00:00:01,000\nsynthetic\n".utf8)
        try Data().write(to: recentMovie)
        try Data().write(to: olderMovie)
        try subtitle.write(to: recent.appending(path: "full-review.srt"))
        try subtitle.write(to: older.appending(path: "full-review.srt"))
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 1_000)],
            ofItemAtPath: recentMovie.path(percentEncoded: false))
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 2_000)],
            ofItemAtPath: olderMovie.path(percentEncoded: false))
        let recentEvidence = root.appending(path: "replay-z/evidence", directoryHint: .isDirectory)
        let olderEvidence = root.appending(path: "replay-a/evidence", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: recentEvidence, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: olderEvidence, withIntermediateDirectories: true)
        try Data(
            #"{"format":{"filename":"/synthetic/2026-09-27_19-30-00_JST_replay-z.mp4"}}"#.utf8
        ).write(to: recentEvidence.appending(path: "source-probe.json"))
        try Data(
            #"{"format":{"filename":"/synthetic/2026-09-21_18-00-00_JST_replay-a.mp4"}}"#.utf8
        ).write(to: olderEvidence.appending(path: "source-probe.json"))

        let scan = try TranscriptCatalog.scan(in: root)

        #expect(scan.videos.map(\.title) == ["replay-z/new", "replay-a/old"])
        #expect(
            scan.videos.map(\.broadcastStartJST) == [
                "2026-09-27 19:30:00 JST", "2026-09-21 18:00:00 JST",
            ])
        #expect(scan.issues.isEmpty)
        try FileManager.default.removeItem(at: root)
    }

    @Test func sameDaySortsByTimeAndInvalidOrMissingDatesSortLast() throws {
        let root = try Self.makeRoot("lib-broadcast-ties")
        let late = root.appending(path: "replay-z/delivery", directoryHint: .isDirectory)
        let early = root.appending(path: "replay-a/delivery", directoryHint: .isDirectory)
        let invalid = root.appending(path: "replay-m/delivery", directoryHint: .isDirectory)
        let missing = root.appending(path: "replay-x/delivery", directoryHint: .isDirectory)
        let subtitle = Data("1\n00:00:00,000 --> 00:00:01,000\nsynthetic\n".utf8)
        for delivery in [late, early, invalid, missing] {
            try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
            try Data().write(to: delivery.appending(path: "full-review.mp4"))
            try subtitle.write(to: delivery.appending(path: "full-review.srt"))
        }
        let lateEvidence = root.appending(path: "replay-z/evidence", directoryHint: .isDirectory)
        let earlyEvidence = root.appending(path: "replay-a/evidence", directoryHint: .isDirectory)
        let invalidEvidence = root.appending(path: "replay-m/evidence", directoryHint: .isDirectory)
        for evidence in [lateEvidence, earlyEvidence, invalidEvidence] {
            try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
        }
        try Data(
            #"{"format":{"filename":"/synthetic/2026-09-27_19-00-00_JST_replay-z.mp4"}}"#.utf8
        ).write(to: lateEvidence.appending(path: "source-probe.json"))
        try Data(
            #"{"format":{"filename":"/synthetic/2026-09-27_08-00-00_JST_replay-a.mp4"}}"#.utf8
        ).write(to: earlyEvidence.appending(path: "source-probe.json"))
        try Data(
            #"{"format":{"filename":"/synthetic/2026-02-30_09-00-00_JST_replay-m.mp4"}}"#.utf8
        ).write(to: invalidEvidence.appending(path: "source-probe.json"))

        let scan = try TranscriptCatalog.scan(in: root)

        #expect(
            scan.videos.map(\.title) == [
                "replay-z", "replay-a",
            ])
        #expect(
            scan.videos.map(\.broadcastStartJST) == [
                "2026-09-27 19:00:00 JST", "2026-09-27 08:00:00 JST",
            ])
        #expect(scan.unknownBroadcastDateLabels == ["replay-m", "replay-x"])
        try FileManager.default.removeItem(at: root)
    }

    @Test func datedReplayShowsOneStableRevisionAndUndatedReplayShowsNone() throws {
        let root = try Self.makeRoot("lib-one-revision")
        let dated = root.appending(path: "replay-a", directoryHint: .isDirectory)
        let undated = root.appending(path: "replay-b", directoryHint: .isDirectory)
        let subtitle = Data("1\n00:00:00,000 --> 00:00:01,000\nsynthetic\n".utf8)
        for replay in [dated, undated] {
            for revision in ["a", "b"] {
                let delivery = replay.appending(path: "revisions/\(revision)/delivery")
                try FileManager.default.createDirectory(
                    at: delivery, withIntermediateDirectories: true)
                try Data().write(to: delivery.appending(path: "full-review.mp4"))
                try subtitle.write(to: delivery.appending(path: "full-review.srt"))
            }
        }
        let evidence = dated.appending(path: "evidence")
        try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
        try Data(#"{"format":{"filename":"/synthetic/2026-09-23_19-30-00_JST_replay-a.mp4"}}"#.utf8)
            .write(to: evidence.appending(path: "source-probe.json"))

        let ambiguous = try TranscriptCatalog.scan(in: root)
        #expect(ambiguous.videos.isEmpty)
        #expect(ambiguous.ambiguousDeliveryLabels == ["replay-a"])
        #expect(ambiguous.issues == ["配信日時不明: replay-b", "成果物が重複: replay-a"])
        let selected = dated.appending(path: "revisions/a/delivery")
        try Data().write(
            to: selected.appending(path: "2026-09-23_19-30-00_JST_replay-a-review.mp4"))
        try subtitle.write(
            to: selected.appending(path: "2026-09-23_19-30-00_JST_replay-a-review.srt"))
        let first = try TranscriptCatalog.scan(in: root)
        let second = try TranscriptCatalog.scan(in: root)
        #expect(first.videos.map(\.title) == ["replay-a/a"])
        #expect(first.videos.map(\.movieURL) == second.videos.map(\.movieURL))
        #expect(first.videos[0].broadcastStartJST == "2026-09-23 19:30:00 JST")
        try FileManager.default.removeItem(at: root)
    }

    @Test func supersededDeliveryDoesNotHideItsCompleteSuccessor() throws {
        let root = try Self.makeRoot("lib-superseded")
        let replay = root.appending(path: "replay-a", directoryHint: .isDirectory)
        let name = "2026-09-23_19-30-00_JST_replay-a-review"
        let subtitle = Data("1\n00:00:00,000 --> 00:00:01,000\nsynthetic\n".utf8)
        for revision in ["old", "new"] {
            let delivery = replay.appending(path: "revisions/\(revision)/delivery")
            try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
            try Data(revision.utf8).write(to: delivery.appending(path: "\(name).mp4"))
            try subtitle.write(to: delivery.appending(path: "\(name).srt"))
        }
        let evidence = replay.appending(path: "evidence")
        let superseded = evidence.appending(path: "superseded")
        try FileManager.default.createDirectory(at: superseded, withIntermediateDirectories: true)
        try Data(#"{"format":{"filename":"/synthetic/2026-09-23_19-30-00_JST_replay-a.mp4"}}"#.utf8)
            .write(to: evidence.appending(path: "source-probe.json"))

        let before = try TranscriptCatalog.scan(in: root)
        #expect(before.videos.isEmpty)
        #expect(before.issues == ["成果物が重複: replay-a"])

        try Data(
            #"{"delivery":"revisions/old/delivery/2026-09-23_19-30-00_JST_replay-a-review.mp4","size":3}"#
                .utf8
        ).write(to: superseded.appending(path: "0123456789abcdef.json"))
        let after = try TranscriptCatalog.scan(in: root)
        #expect(after.videos.map(\.title) == ["replay-a/new"])
        #expect(after.issues.isEmpty)
        try FileManager.default.removeItem(at: root)
    }

    @Test func changedSupersededDeliveryIsNotSilentlyHidden() throws {
        let root = try Self.makeRoot("lib-superseded-changed")
        let replay = root.appending(path: "replay-a", directoryHint: .isDirectory)
        let name = "2026-09-23_19-30-00_JST_replay-a-review"
        let subtitle = Data("1\n00:00:00,000 --> 00:00:01,000\nsynthetic\n".utf8)
        for revision in ["old", "new"] {
            let delivery = replay.appending(path: "revisions/\(revision)/delivery")
            try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
            try Data(revision.utf8).write(to: delivery.appending(path: "\(name).mp4"))
            try subtitle.write(to: delivery.appending(path: "\(name).srt"))
        }
        let evidence = replay.appending(path: "evidence")
        let superseded = evidence.appending(path: "superseded")
        try FileManager.default.createDirectory(at: superseded, withIntermediateDirectories: true)
        try Data(#"{"format":{"filename":"/synthetic/2026-09-23_19-30-00_JST_replay-a.mp4"}}"#.utf8)
            .write(to: evidence.appending(path: "source-probe.json"))
        try Data(
            #"{"delivery":"revisions/old/delivery/2026-09-23_19-30-00_JST_replay-a-review.mp4","size":99}"#
                .utf8
        ).write(to: superseded.appending(path: "0123456789abcdef.json"))
        try Data("{".utf8).write(to: superseded.appending(path: "broken.json"))
        try Data().write(to: superseded.appending(path: "notes.txt"))

        let scan = try TranscriptCatalog.scan(in: root)
        #expect(scan.videos.isEmpty)
        #expect(scan.issues == ["成果物が重複: replay-a", "置き換え記録が不正: replay-a/broken.json"])
        try FileManager.default.removeItem(at: root)
    }

    @Test func unreadableSupersededDirectoryIsReported() throws {
        let root = try Self.makeRoot("lib-superseded-file")
        let evidence = root.appending(path: "evidence")
        try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
        try Data().write(to: evidence.appending(path: "superseded"))

        let result = TranscriptCatalog.supersededDeliveries(in: root, fileManager: .default)
        #expect(result.invalidRecords == ["evidence/superseded"])
        #expect(result.sizesByRelativePath.isEmpty)
        let missing = root.appending(path: "revisions/x/delivery/a.mp4")
        #expect(!result.matches(missing, in: root, fileManager: .default))
        try FileManager.default.removeItem(at: root)
    }

    @Test func recordedButMissingDeliveryIsNotAMatch() throws {
        let root = try Self.makeRoot("lib-superseded-missing")
        let result = TranscriptCatalog.SupersededSet(
            sizesByRelativePath: ["revisions/x/delivery/a.mp4": 1], invalidRecords: [])
        let missing = root.appending(path: "revisions/x/delivery/a.mp4")
        #expect(!result.matches(missing, in: root, fileManager: .default))
        try FileManager.default.removeItem(at: root)
    }

    @Test func supersededAliasMatchesByTheMovieItNames() throws {
        let root = try Self.makeRoot("lib-superseded-alias")
        let delivery = root.appending(path: "revisions/x/delivery", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
        try Data(count: 7).write(to: root.appending(path: "revisions/x/movie.mp4"))
        let alias = delivery.appending(path: "a.mp4")
        try FileManager.default.createSymbolicLink(
            atPath: alias.path, withDestinationPath: "../movie.mp4")
        let result = TranscriptCatalog.SupersededSet(
            sizesByRelativePath: ["revisions/x/delivery/a.mp4": 7], invalidRecords: [])
        #expect(result.matches(alias, in: root, fileManager: .default))
        try FileManager.default.removeItem(at: root)
    }

    @Test func youtubeEditsCarryTheirPublicationDateAndPreferTheReview() throws {
        let root = try Self.makeRoot("lib-youtube-date")
        let delivery = root.appending(
            path: "youtube-abcdefghijk/delivery", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
        let subtitle = Data("1\n00:00:00,000 --> 00:00:01,000\nsynthetic\n".utf8)
        for stem in [
            "2026-09-28_21-58-42_JST_youtube-abcdefghijk-edit-old",
            "2026-09-28_21-58-42_JST_youtube-abcdefghijk-review",
        ] {
            try Data().write(to: delivery.appending(path: "\(stem).mp4"))
            try subtitle.write(to: delivery.appending(path: "\(stem).srt"))
        }

        let scan = try TranscriptCatalog.scan(in: root)
        #expect(
            scan.videos.map(\.movieURL.lastPathComponent) == [
                "2026-09-28_21-58-42_JST_youtube-abcdefghijk-review.mp4"
            ])
        #expect(scan.videos.map(\.broadcastStartJST) == ["2026-09-28 21:58:42 JST"])
        #expect(scan.issues.isEmpty)
        try FileManager.default.removeItem(at: root)
    }

    @Test func undatedYoutubeMovieIsReported() throws {
        let root = try Self.makeRoot("lib-youtube-undated")
        let folder = root.appending(path: "youtube-abcdefghijk", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data().write(to: folder.appending(path: "edit.mp4"))
        try Data("1\n00:00:00,000 --> 00:00:01,000\nx\n".utf8)
            .write(to: folder.appending(path: "edit.srt"))

        let scan = try TranscriptCatalog.scan(in: root)
        #expect(scan.videos.isEmpty)
        #expect(scan.issues == ["動画待ち: youtube-abcdefghijk/edit", "配信日時不明: youtube-abcdefghijk"])
        try FileManager.default.removeItem(at: root)
    }

    @Test func youtubeDateRequiresTheFolderIdentityAndACanonicalStamp() {
        let title = "youtube-abcdefghijk"
        #expect(
            TranscriptCatalog.youtubeStartJST(
                movie: URL(fileURLWithPath: "/x/2026-09-28_21-58-42_JST_youtube-abcdefghijk.mp4"),
                title: title) == "2026-09-28 21:58:42 JST")
        #expect(
            TranscriptCatalog.youtubeStartJST(
                movie: URL(fileURLWithPath: "/x/2026-09-28_21-58-42_JST_youtube-abcdefghijkX.mp4"),
                title: title) == nil)
        #expect(
            TranscriptCatalog.youtubeStartJST(
                movie: URL(fileURLWithPath: "/x/2026-13-28_21-58-42_JST_youtube-abcdefghijk.mp4"),
                title: title) == nil)
        #expect(
            TranscriptCatalog.youtubeStartJST(
                movie: URL(fileURLWithPath: "/x/2026-09-28_21-58-42_JST_youtube-abcdefghijk.mp4"),
                title: "replay-1") == nil)
    }

    @Test func checkoutScanIgnoresScratchAndAcceptsExactHistoricalRelativeSource() throws {
        let parent = try Self.makeRoot("lib-scoped-checkout")
        let root = parent.appending(path: "edits")
        let scratch = root.appending(path: "synthetic-work/delivery")
        try FileManager.default.createDirectory(
            at: root.appending(path: "replay-retained-plans/revisions"),
            withIntermediateDirectories: true
        )
        let evidence = root.appending(path: "replay-123/evidence")
        let delivery = root.appending(path: "replay-123/revisions/selected/delivery")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
        try Data().write(to: scratch.appending(path: "candidate.mp4"))
        try Data().write(
            to: delivery.appending(path: "2024-01-01_09-00-00_JST_replay-123-review.mp4"))
        try Data("1\n00:00:00,000 --> 00:00:01,000\nhello\n".utf8)
            .write(to: delivery.appending(path: "2024-01-01_09-00-00_JST_replay-123-review.srt"))
        try Data(
            #"{"format":{"filename":"downloads/2024-01-01_09-00-00_JST_replay-123.mp4"}}"#.utf8
        )
        .write(to: evidence.appending(path: "source-probe.json"))
        let scan = try TranscriptCatalog.scan(in: root)
        #expect(scan.videos.map(\.broadcastStartJST) == ["2024-01-01 09:00:00 JST"])
        #expect(scan.issues.isEmpty)
        try FileManager.default.removeItem(at: parent)
    }

    @Test func symlinkedWorkCannotHideTheFollowingDelivery() throws {
        let root = try Self.makeRoot("lib-linked-work")
        let revision = root.appending(path: "revision")
        let scratch = root.appending(path: ".scratch")
        try FileManager.default.createDirectory(at: revision, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: revision.appending(path: "work"), withDestinationURL: scratch)
        let delivery = revision.appending(path: "delivery")
        try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
        try Data().write(to: delivery.appending(path: "movie.mp4"))
        try Data("1\n00:00:00,000 --> 00:00:01,000\nvisible\n".utf8)
            .write(to: delivery.appending(path: "movie.srt"))
        let scan = try TranscriptCatalog.scan(in: root)
        #expect(scan.videos.map { $0.cues[0].text } == ["visible"])
        #expect(scan.issues.isEmpty)
        try FileManager.default.removeItem(at: root)
    }

    @Test func sourceDateFromAnotherReplayCannotBeUsed() throws {
        let root = try Self.makeRoot("lib-wrong-source")
        let delivery = root.appending(path: "replay-a/delivery", directoryHint: .isDirectory)
        let evidence = root.appending(path: "replay-a/evidence", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
        try Data().write(to: delivery.appending(path: "full-review.mp4"))
        try Data("1\n00:00:00,000 --> 00:00:01,000\nsynthetic\n".utf8)
            .write(to: delivery.appending(path: "full-review.srt"))
        try Data(
            #"{"format":{"filename":"/synthetic/2026-09-27_19-00-00_JST_replay-b.mp4"}}"#.utf8
        ).write(to: evidence.appending(path: "source-probe.json"))

        let scan = try TranscriptCatalog.scan(in: root)

        #expect(scan.videos.isEmpty)
        #expect(scan.unknownBroadcastDateLabels == ["replay-a"])
        try FileManager.default.removeItem(at: root)
    }

    @Test func findsCheckoutLibraryOnlyForItsBuiltApp() throws {
        let root = try Self.makeRoot("lib-checkout")
        let edits = root.appending(path: "edits", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: edits, withIntermediateDirectories: true)
        let app = root.appending(
            path: "TranscriptViewer/dist/TranscriptViewer.app", directoryHint: .isDirectory)
        #expect(TranscriptCatalog.checkoutLibrary(near: app) == edits)
        #expect(
            TranscriptCatalog.checkoutLibrary(
                near: root.appending(path: "TranscriptViewer/dist/TranscriptViewer-new.app"))
                == edits)
        #expect(
            TranscriptCatalog.checkoutLibrary(near: root.appending(path: "dist/Other.app")) == nil)
        #expect(TranscriptCatalog.checkoutLibrary(near: root.appending(path: "other.app")) == nil)
        try FileManager.default.removeItem(at: edits)
        #expect(TranscriptCatalog.checkoutLibrary(near: app) == nil)
        try FileManager.default.removeItem(at: root)
    }

    @Test func privateFolderCannotBeSelectedAsLibrary() throws {
        let root = try Self.makeRoot("lib-restricted")
        let restricted = root.appending(path: "private", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: restricted, withIntermediateDirectories: true)
        try Data().write(to: restricted.appending(path: "full-review.mp4"))
        try Data("1\n00:00:00,000 --> 00:00:01,000\nprivate\n".utf8)
            .write(to: restricted.appending(path: "full-review.srt"))
        try FileManager.default.createSymbolicLink(
            at: root.appending(path: "replay-link"), withDestinationURL: restricted)
        let delivery = root.appending(
            path: "replay-synthetic/delivery", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: delivery.appending(path: "linked.mp4"),
            withDestinationURL: restricted.appending(path: "full-review.mp4"))
        try Data("1\n00:00:00,000 --> 00:00:01,000\nlink\n".utf8)
            .write(to: delivery.appending(path: "linked.srt"))
        #expect(throws: TranscriptError.unreadableDirectory(restricted.path(percentEncoded: false)))
        {
            _ = try TranscriptCatalog.scan(in: restricted)
        }
        #expect(try TranscriptCatalog.scan(in: root).videos.isEmpty)
        try FileManager.default.removeItem(at: root)
    }

    @Test func unreadableDirectoryFails() {
        let missing = URL(
            fileURLWithPath: "/tmp/transcript-viewer-does-not-exist-\(UUID().uuidString)")
        #expect(throws: TranscriptError.unreadableDirectory(missing.path(percentEncoded: false))) {
            _ = try TranscriptCatalog.scan(in: missing)
        }
    }
}
