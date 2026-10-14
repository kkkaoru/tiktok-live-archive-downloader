import AVFoundation
import AppKit
import Foundation
import Testing
import TranscriptCore

@testable import TranscriptViewer

private enum CatalogTestFailure: Error {
    case unavailable
}

@MainActor
final class TestDirectoryPanel: NSOpenPanel {
    var chosenURL: URL?
    var response = NSApplication.ModalResponse.cancel
    override var url: URL? { chosenURL }
    override func runModal() -> NSApplication.ModalResponse { response }
}

@MainActor
struct PlaybackSessionTests {
    @Test func nonLocalizedScanFailureHasAnActionableFallbackAndClearsBusyState() throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var requestedRoot: URL?
        let session = PlaybackSession(scanCatalog: { root, completion in
            requestedRoot = root
            completion(.failure(CatalogTestFailure.unavailable))
        })
        session.load(root: fixture.root)
        #expect(requestedRoot == fixture.root)
        #expect(!session.isScanning)
        #expect(session.errorText == "読み込みに失敗しました")
        #expect(session.showsError)
        #expect(session.catalogRevision == 0)
        #expect(session.selectedVideoID == nil)
    }

    @Test func utteranceWindowIsConfiguredAndReusedWithoutTakingFocus() throws {
        let application = NSApplication.shared
        let previousKey = application.keyWindow
        var presented: [NSWindow] = []
        let session = PlaybackSession(presentWindow: { window in
            presented.append(window)
            window.alphaValue = 0
            window.orderBack(nil)
        })
        defer {
            for window in presented {
                window.orderOut(nil)
                window.close()
            }
        }
        session.showUtteranceWindow()
        let first = try #require(presented.first)
        #expect(first.title == "Transcripts")
        #expect(first.minSize == NSSize(width: 400, height: 300))
        #expect(first.contentView?.frame.size == NSSize(width: 560, height: 720))
        #expect(first.styleMask.contains(.resizable))
        #expect(!first.isReleasedWhenClosed)
        session.showUtteranceWindow()
        #expect(presented.count == 2)
        #expect(presented.last === first)
        first.orderOut(nil)
        session.showUtteranceWindow()
        #expect(presented.count == 3)
        #expect(presented.last !== first)
        #expect(application.keyWindow === previousKey)
    }

    @Test(.timeLimit(.minutes(1)), arguments: [false, true])
    func defaultLibraryRespectsSavedChoiceWithoutWritingPreferences(_ hasSavedChoice: Bool)
        async throws
    {
        let fallback = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fallback.remove() } }
        let saved = try SummaryFixture()
        defer { #expect(throws: Never.self) { try saved.remove() } }
        let session = PlaybackSession(
            readLibraryRoot: { hasSavedChoice ? saved.root : nil },
            saveLibraryRoot: { _ in Issue.record("Loading must not write preferences") },
            defaultLibrary: fallback.root, defaultAdditionalLibraries: [])
        defer { session.stopObserving() }
        session.loadDefaultLibrary()
        try await finishScan(session)
        await session.legibleSelection?.value
        #expect(session.selectedVideoID == (hasSavedChoice ? saved.video.id : fallback.video.id))
    }

    @Test(.timeLimit(.minutes(1)), arguments: [false, true])
    func ordinaryLaunchAddsDeliveriesAlongsideTheSavedOrDefaultLibrary(_ hasSavedChoice: Bool)
        async throws
    {
        let primary = try SummaryFixture()
        defer { #expect(throws: Never.self) { try primary.remove() } }
        let delivery = try SummaryFixture(source: "配信の納品物を説明する。")
        defer { #expect(throws: Never.self) { try delivery.remove() } }
        let session = PlaybackSession(
            readLibraryRoot: { hasSavedChoice ? primary.root : nil },
            saveLibraryRoot: { _ in Issue.record("Loading must not write preferences") },
            defaultLibrary: primary.root, defaultAdditionalLibraries: [delivery.root])
        defer { session.stopObserving() }
        session.loadDefaultLibrary()
        try await finishScan(session)
        await session.legibleSelection?.value
        let ids = Set(session.videos.map(\.id))
        #expect(ids.contains(primary.video.id))
        #expect(ids.contains(delivery.video.id))
        #expect(session.usesDefaultLibraries)
    }

    @Test(.timeLimit(.minutes(1)))
    func anUnreadableAdditionalLibraryIsReportedWithoutHidingTheDefaultOne() async throws {
        let primary = try SummaryFixture()
        defer { #expect(throws: Never.self) { try primary.remove() } }
        let missing = primary.root.appendingPathComponent("missing-delivery-library")
        let session = PlaybackSession(
            readLibraryRoot: { nil }, saveLibraryRoot: { _ in },
            defaultLibrary: primary.root,
            defaultAdditionalLibraries: [missing],
            scanCatalog: { root, completion in
                if root == missing {
                    completion(.failure(CatalogTestFailure.unavailable))
                } else {
                    TranscriptCatalog.scan(in: root, completion: completion)
                }
            })
        defer { session.stopObserving() }
        session.loadDefaultLibrary()
        try await finishScan(session)
        await session.legibleSelection?.value
        #expect(session.videos.map(\.id) == [primary.video.id])
        #expect(session.scanIssues.contains { $0.contains("missing-delivery-library") })
    }

    @Test(.timeLimit(.minutes(1)), arguments: [false, true])
    func folderChoiceHonorsCancellationAndPersistsOnlyAcceptedDirectory(_ accepted: Bool)
        async throws
    {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var saved: URL?
        let session = PlaybackSession(
            readLibraryRoot: { nil }, saveLibraryRoot: { saved = $0 },
            defaultLibrary: fixture.root)
        let panel = TestDirectoryPanel()
        panel.chosenURL = fixture.root
        panel.response = accepted ? .OK : .cancel
        defer { session.stopObserving() }
        session.chooseFolder(panel: panel)
        try await finishScan(session)
        await session.legibleSelection?.value
        #expect(panel.canChooseDirectories)
        #expect(!panel.canChooseFiles)
        #expect(!panel.allowsMultipleSelection)
        #expect(saved == (accepted ? fixture.root : nil))
        #expect(session.catalogRevision == (accepted ? 1 : 0))
        #expect(session.selectedVideoID == (accepted ? fixture.video.id : nil))
    }

    @Test(.timeLimit(.minutes(1)))
    func refreshKeepsThePlayerAndUpdatesTranscriptThenClearsRemovedSelection() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let session = PlaybackSession()
        defer {
            session.player.pause()
            session.stopObserving()
        }
        session.load(root: fixture.root)
        #expect(session.isScanning)
        session.refreshLibrary()
        try await finishScan(session)
        await session.legibleSelection?.value
        session.stopObserving()
        #expect(session.catalogRevision == 1)
        #expect(session.libraryName == fixture.root.lastPathComponent)
        #expect(session.videos.count == 1)
        #expect(session.selectedVideoID == fixture.video.id)
        let item = try #require(session.player.currentItem)
        session.updateCue(at: 0.5)
        try "1\n00:00:00,000 --> 00:00:07,000\n更新された字幕。\n".write(
            to: fixture.video.transcriptURL, atomically: true, encoding: .utf8)
        session.refreshLibrary()
        try await finishScan(session)
        #expect(session.catalogRevision == 2)
        #expect(session.player.currentItem === item)
        #expect(session.currentSeconds == 0.5)
        #expect(session.currentUtterance?.text == "更新された字幕。")
        #expect(session.duration == 7)
        try FileManager.default.removeItem(at: fixture.video.movieURL)
        session.refreshLibrary()
        try await finishScan(session)
        #expect(session.catalogRevision == 3)
        #expect(session.selectedVideoID == nil)
        #expect(session.player.currentItem == nil)
        #expect(session.utterances.isEmpty)
    }

    @Test(.timeLimit(.minutes(1))) func newLibraryRejectsAnEarlierPendingScan() async throws {
        let first = try SummaryFixture()
        defer { #expect(throws: Never.self) { try first.remove() } }
        let second = try SummaryFixture()
        defer { #expect(throws: Never.self) { try second.remove() } }
        let session = PlaybackSession()
        defer {
            session.player.pause()
            session.stopObserving()
        }
        session.load(root: first.root)
        session.load(root: second.root)
        try await finishScan(session)
        await session.legibleSelection?.value
        #expect(session.catalogRevision == 1)
        #expect(session.libraryName == second.root.lastPathComponent)
        #expect(session.selectedVideoID == second.video.id)
        #expect(session.videos.map(\.id) == [second.video.id])
    }

    @Test(.timeLimit(.minutes(1))) func scanFailureIsVisibleAndDoesNotPublishACatalog() async throws
    {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let session = PlaybackSession()
        let missing = fixture.root.appendingPathComponent("missing")
        session.load(root: missing)
        try await finishScan(session)
        #expect(session.showsError)
        #expect(session.errorText.contains("フォルダ"))
        #expect(session.catalogRevision == 0)
        #expect(session.videos.isEmpty)
    }

    @Test func searchHitSwitchesVideoThenSeeksWithinTheSameVideo() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let session = PlaybackSession()
        defer {
            session.player.pause()
            session.stopObserving()
        }
        session.videos = [fixture.video]
        session.showsSearch = true
        var cue = try #require(fixture.video.cues.first)
        cue.startSeconds = 0.25
        session.play(SearchHit(videoID: fixture.video.id, videoTitle: "synthetic", cue: cue))
        await session.legibleSelection?.value
        session.stopObserving()
        #expect(!session.showsSearch)
        #expect(session.selectedVideoID == fixture.video.id)
        let item = try #require(session.player.currentItem)
        session.handleVideoChange()
        session.handleVideoChange()
        #expect(session.player.currentItem === item)
        #expect(session.currentSeconds == 0.25)
        var next = cue
        next.startSeconds = 0.5
        session.beginTimelineBrowse()
        session.play(SearchHit(videoID: fixture.video.id, videoTitle: "synthetic", cue: next))
        #expect(session.player.currentItem === item)
        #expect(session.currentSeconds == 0.5)
        #expect(session.timelineFollowsPlayback)
    }

    private func finishScan(_ session: PlaybackSession) async throws {
        while session.isScanning {
            try Task.checkCancellation()
            await Task.yield()
        }
    }

    @Test func emptySelectionResetsTimelineAndDoesNotScanAnImplicitLibrary() {
        let session = PlaybackSession()
        #expect(session.libraryName == "未選択")
        #expect(!session.isScanning)
        session.refreshLibrary()
        #expect(session.catalogRevision == 0)
        session.currentSeconds = 40
        session.selectedUtteranceID = 7
        session.browseAnchorID = 7
        session.timelineFollowsPlayback = false
        session.openSelection()
        #expect(session.selectedVideo == nil)
        #expect(session.currentUtterance == nil)
        #expect(session.nearbyUtterances.isEmpty)
        #expect(session.currentSeconds == 0)
        #expect(session.duration == 0)
        #expect(session.sliderUpperBound == 0.001)
        #expect(session.timelineFollowsPlayback)
        #expect(session.browseAnchorID == nil)
        #expect(session.player.currentItem == nil)
        session.seek(toUtterance: 99)
        #expect(session.currentSeconds == 0)
        session.stopObserving()
    }

    @Test func failedMediaStillKeepsTranscriptNavigationAndSelectedSpeed() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let session = PlaybackSession()
        defer {
            session.player.pause()
            session.stopObserving()
        }
        session.videos = [fixture.video]
        session.selectedVideoID = fixture.video.id
        session.playbackSpeed = .double
        session.openSelection()
        await session.legibleSelection?.value
        session.stopObserving()
        #expect(session.showsError)
        #expect(session.errorText.contains("動画"))
        #expect(session.selectedVideo == fixture.video)
        #expect(session.utterances.count == 1)
        #expect(session.duration == 1)
        #expect(session.sliderUpperBound == 1)
        #expect(session.player.currentItem?.audioTimePitchAlgorithm == .timeDomain)
        #expect(session.player.currentItem?.preferredForwardBufferDuration == 2)
        #expect(!session.player.appliesMediaSelectionCriteriaAutomatically)
        session.updateCue(at: 0.5)
        let cue = try #require(session.currentUtterance)
        #expect(cue.text == "旅行の準備を説明する。")
        session.beginTimelineBrowse()
        #expect(!session.timelineFollowsPlayback)
        #expect(session.browseAnchorID == cue.id)
        #expect(session.nearbyUtterances == [cue])
        session.seek(toUtterance: cue.id)
        #expect(session.timelineFollowsPlayback)
        #expect(session.browseAnchorID == nil)
        #expect(session.currentSeconds == 0)
        #expect(session.player.defaultRate == 2)
        session.seek(to: -20)
        #expect(session.currentSeconds == 0)
        session.seek(to: 20)
        #expect(session.currentSeconds == 1)
        session.updateCue(at: 100)
        #expect(session.currentUtterance == nil)
    }

    @Test func deselectionCancelsOldMediaPreparationWithoutAStaleError() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let session = PlaybackSession()
        session.videos = [fixture.video]
        session.selectedVideoID = fixture.video.id
        session.openSelection()
        let oldPreparation = try #require(session.legibleSelection)
        session.selectedVideoID = nil
        session.handleVideoChange()
        await oldPreparation.value
        #expect(oldPreparation.isCancelled)
        #expect(!session.showsError)
        #expect(session.player.currentItem == nil)
        #expect(session.utterances.isEmpty)
    }

    @Test func repeatedWindowNotificationsKeepEachOpenedSelectionAndSeek() async throws {
        let first = try SummaryFixture()
        let second = try SummaryFixture()
        defer {
            #expect(throws: Never.self) { try first.remove() }
            #expect(throws: Never.self) { try second.remove() }
        }
        let session = PlaybackSession()
        defer {
            session.player.pause()
            session.stopObserving()
        }
        session.videos = [first.video, second.video]
        session.selectedVideoID = first.video.id
        session.handleVideoChange()
        await session.legibleSelection?.value
        session.stopObserving()
        let firstItem = try #require(session.player.currentItem)
        session.seek(to: 0.5)
        session.handleVideoChange()
        session.handleVideoChange()
        #expect(session.player.currentItem === firstItem)
        #expect(session.currentSeconds == 0.5)

        session.selectedVideoID = second.video.id
        session.handleVideoChange()
        await session.legibleSelection?.value
        session.stopObserving()
        let secondItem = try #require(session.player.currentItem)
        #expect(secondItem !== firstItem)
        #expect(session.currentSeconds == 0)
        session.seek(to: 0.75)
        session.handleVideoChange()
        session.handleVideoChange()
        #expect(session.player.currentItem === secondItem)
        #expect(session.currentSeconds == 0.75)

        session.selectedVideoID = nil
        session.handleVideoChange()
        session.handleVideoChange()
        #expect(session.player.currentItem == nil)
        #expect(session.currentSeconds == 0)
        #expect(session.utterances.isEmpty)
    }
}
