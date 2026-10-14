import Foundation
import Testing

@testable import TranscriptCore
@testable import TranscriptViewer

@MainActor
private final class CatalogPublicationState {
    var published = false
}

@MainActor
struct DefaultCatalogSessionTests {
    @Test(.timeLimit(.minutes(1)))
    func savedLibraryAddsCurrentCandidateAndLaterPublicationWithoutResettingPlayer()
        async throws
    {
        let existing = try SummaryFixture()
        let published = try CandidateSelectionTests.Fixture()
        defer {
            #expect(throws: Never.self) { try existing.remove() }
            #expect(throws: Never.self) { try published.remove() }
        }
        let manifest = existing.root.appending(path: DefaultLibraryManifest.filename)
        try JSONEncoder().encode(
            DefaultLibraryManifest(version: 1, libraries: [published.root.path])
        )
        .write(to: manifest)
        // Before selection publication, two retained revisions are ambiguous and stay unlisted.
        try FileManager.default.removeItem(at: published.manifest)
        let old = published.folder.appending(path: "revisions/old/delivery")
        try FileManager.default.createDirectory(at: old, withIntermediateDirectories: true)
        try Data("older".utf8).write(to: old.appending(path: published.movie.lastPathComponent))
        try Data(contentsOf: published.transcript).write(
            to: old.appending(path: published.transcript.lastPathComponent))
        let session = PlaybackSession(
            readLibraryRoot: { existing.root },
            saveLibraryRoot: { _ in Issue.record("Default startup must preserve preferences") },
            defaultLibrary: existing.root)
        defer { session.stopObserving() }
        session.loadDefaultLibrary()
        try await finishScan(session)
        #expect(session.videos.map(\.id) == [existing.video.id])
        let previousItem = try #require(session.player.currentItem)
        session.updateCue(at: 0.5)
        try published.write()
        session.refreshLibrary()
        try await finishScan(session)
        #expect(Set(session.videos.map(\.id)) == Set([existing.video.id, published.movie]))
        #expect(session.selectedVideoID == existing.video.id)
        #expect(session.player.currentItem === previousItem)
        #expect(session.currentSeconds == 0.5)
        let added = try #require(session.videos.first { $0.movieURL == published.movie })
        #expect(added.broadcastStartJST == "2026-10-04 14:30:09 JST")
        #expect(added.recordingID == "replay-123")
        #expect(added.title == "replay-123/new [未検証候補]")
        #expect(added.transcriptURL == published.transcript)
        #expect(added.cues.first?.startSeconds == 0.1)
        #expect(added.cues.first?.text == "hello")
        #expect(session.scanIssues.isEmpty)
    }

    @Test func explicitLibraryBypassesDefaultManifestAndInvalidConfigKeepsOrdinaryLibrary() throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var manifestReads = 0
        let session = PlaybackSession(
            readLibraryRoot: { fixture.root }, defaultLibrary: fixture.root,
            readManifest: { _, completion in
                manifestReads += 1
                completion(.failure(DefaultLibraryManifest.Invalid.manifest))
            },
            scanCatalog: { _, completion in completion(.success(CatalogScan())) })
        session.loadDefaultLibrary()
        #expect(manifestReads == 1)
        #expect(session.catalogRevision == 1)
        #expect(session.scanIssues.first?.contains("viewer-default-libraries.json") == true)
        session.openLibrary(url: fixture.root)
        session.refreshLibrary()
        #expect(manifestReads == 1)
        #expect(!session.usesDefaultLibraries)
        #expect(session.scanIssues.isEmpty)
    }

    @Test func aNewRevisionKeepsTheSelectedRecordingButDoesNotReuseAnUnprovenClock() throws {
        let previous = try CandidateSelectionTests.Fixture()
        let current = try CandidateSelectionTests.Fixture()
        defer {
            #expect(throws: Never.self) { try previous.remove() }
            #expect(throws: Never.self) { try current.remove() }
        }
        let first = try TranscriptCatalog.scan(in: previous.root)
        let second = try TranscriptCatalog.scan(in: current.root)
        let publication = CatalogPublicationState()
        let session = PlaybackSession(
            readLibraryRoot: { nil }, defaultLibrary: previous.root,
            readManifest: { _, completion in
                completion(.success(publication.published ? [current.root] : []))
            },
            scanCatalog: { _, completion in
                completion(.success(publication.published ? second : first))
            })
        defer { session.stopObserving() }
        session.loadDefaultLibrary()
        #expect(session.selectedVideoID == previous.movie)
        let oldItem = session.player.currentItem
        session.updateCue(at: 0.5)
        publication.published = true
        session.refreshLibrary()
        #expect(session.videos.map(\.id) == [current.movie])
        #expect(session.selectedVideoID == current.movie)
        #expect(session.player.currentItem !== oldItem)
        #expect(session.currentSeconds == 0)
        #expect(session.player.rate == 0)
    }

    @Test func staleManifestCallbackDoesNotOverwriteExplicitNewSelection() throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var pending: (@MainActor @Sendable (Result<[URL], Error>) -> Void)?
        var roots: [URL] = []
        let session = PlaybackSession(
            readLibraryRoot: { nil }, defaultLibrary: fixture.root,
            readManifest: { _, completion in pending = completion },
            scanCatalog: { root, completion in
                roots.append(root)
                completion(.success(CatalogScan()))
            })
        session.loadDefaultLibrary()
        #expect(session.isScanning)
        session.load(root: fixture.root)
        let oldCompletion = try #require(pending)
        oldCompletion(.success([fixture.root.appending(path: "late")]))
        #expect(roots == [fixture.root])
        #expect(session.catalogRevision == 1)
        #expect(!session.isScanning)
    }

    @Test func periodicRefreshReloadsConfigAndStopsOnCancellationOrExplicitSelection() async {
        var reads = 0
        var scans: [URL] = []
        let root = URL(fileURLWithPath: "/synthetic-library")
        let later = URL(fileURLWithPath: "/synthetic-published-library")
        let session = PlaybackSession(
            readLibraryRoot: { nil }, defaultLibrary: root,
            defaultAdditionalLibraries: [root],
            readManifest: { _, completion in
                reads += 1
                completion(.success(reads == 1 ? [] : [later]))
            },
            scanCatalog: { directory, completion in
                scans.append(directory)
                completion(.success(CatalogScan()))
            })
        session.loadDefaultLibrary()
        var ticks = 0
        await session.observeDefaultLibraryUpdates {
            ticks += 1
            if ticks == 2 { throw CancellationError() }
        }
        #expect(reads == 2)
        #expect(scans == [root, root, later])
        #expect(session.catalogRevision == 2)
        #expect(!session.showsError)
        session.load(root: root)
        await session.observeDefaultLibraryUpdates {
            Issue.record("An explicit library must not run the default refresh loop")
        }
        session.loadDefaultLibrary()
        #expect(PlaybackSession.catalogRefreshInterval == .seconds(30))
    }

    @Test func refreshWaitFailureIsReportedAndObserverCanBeReused() async {
        let session = PlaybackSession(
            readLibraryRoot: { nil },
            readManifest: { _, completion in completion(.success([])) },
            scanCatalog: { _, completion in completion(.success(CatalogScan())) })
        session.loadDefaultLibrary()
        await session.observeDefaultLibraryUpdates { throw DefaultLibraryManifest.Invalid.manifest }
        #expect(session.showsError)
        #expect(session.errorText == "追加ライブラリの設定が不正です。")
        await session.observeDefaultLibraryUpdates { throw CancellationError() }
    }

    @Test func missingSavedLibraryDoesNotHideConfiguredDeliveries() {
        let missing = URL(fileURLWithPath: "/synthetic-missing-saved-library")
        let added = URL(fileURLWithPath: "/synthetic-delivery-library")
        var scanned: [URL] = []
        let session = PlaybackSession(
            readLibraryRoot: { missing },
            readManifest: { _, completion in completion(.success([added])) },
            scanCatalog: { directory, completion in
                scanned.append(directory)
                if directory == missing {
                    completion(.failure(TranscriptError.unreadableDirectory("saved")))
                } else {
                    completion(.success(CatalogScan()))
                }
            })
        session.loadDefaultLibrary()
        #expect(scanned == [missing, added])
        #expect(session.catalogRevision == 1)
        #expect(session.scanIssues.count == 1)
        #expect(!session.showsError)
    }

    @Test func leavingDefaultModeDuringWaitStopsRefreshAndRealWaitCancelsPromptly() async {
        let root = URL(fileURLWithPath: "/synthetic-library")
        let session = PlaybackSession(
            readLibraryRoot: { nil }, defaultLibrary: root,
            readManifest: { _, completion in completion(.success([])) },
            scanCatalog: { _, completion in completion(.success(CatalogScan())) })
        session.loadDefaultLibrary()
        await session.observeDefaultLibraryUpdates { session.load(root: root) }
        #expect(session.catalogRevision == 2)
        let another = PlaybackSession(
            readLibraryRoot: { nil }, defaultLibrary: root,
            readManifest: { _, completion in completion(.success([])) },
            scanCatalog: { _, completion in completion(.success(CatalogScan())) })
        another.loadDefaultLibrary()
        let observer = Task { await another.observeDefaultLibraryUpdates() }
        await Task.yield()
        observer.cancel()
        await observer.value
        #expect(another.catalogRevision == 1)
        #expect(!another.showsError)
    }

    private func finishScan(_ session: PlaybackSession) async throws {
        while session.isScanning {
            try Task.checkCancellation()
            await Task.yield()
        }
    }
}
