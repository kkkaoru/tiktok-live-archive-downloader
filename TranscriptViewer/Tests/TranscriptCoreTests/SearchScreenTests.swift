import SwiftUI
import Testing
import TranscriptCore
import ViewInspector

@testable import TranscriptViewer

@MainActor
struct SearchScreenTests {
    @Test func blankQueryClearsPreviousResultsAndClosingUsesTheSheetBinding() async throws {
        let session = PlaybackSession()
        session.searchQuery = "  \n"
        session.isSearching = true
        session.showsSearch = true
        let view = SearchScreen(session: session)
        try await performSearch(view, session: session)
        #expect(session.searchHits.isEmpty)
        #expect(!session.isSearching)
        #expect(
            try view.inspect().find(ViewType.ContentUnavailableView.self).labelView().find(
                text: "文言を入力"
            ).string() == "文言を入力")
        let toolbar = try view.inspect().navigationStack().vStack().toolbar()
        try toolbar.find(button: "閉じる").tap()
        #expect(!session.showsSearch)
    }

    @Test func queryAndScopeBindingsProduceATimestampedResultThatPlaysTheMatchingVideo()
        async throws
    {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let session = PlaybackSession()
        defer {
            session.player.pause()
            session.stopObserving()
        }
        session.videos = [fixture.video]
        session.showsSearch = true
        let view = SearchScreen(session: session)
        try view.inspect().find(ViewType.TextField.self).setInput("旅行")
        try view.inspect().find(ViewType.Picker.self).select(value: SearchScope.allVideos)
        #expect(session.searchQuery == "旅行")
        #expect(session.searchScope == .allVideos)
        try await performSearch(view, session: session)
        #expect(!session.isSearching)
        #expect(session.searchHits.count == 1)
        #expect(try view.inspect().find(text: "synthetic").string() == "synthetic")
        #expect(try view.inspect().find(text: "旅行の準備を説明する。").string() == "旅行の準備を説明する。")
        let list = try view.inspect().find(ViewType.List.self)
        #expect(try list.accessibilityLabel().string() == "検索結果 1件")
        try list.find(ViewType.Button.self).tap()
        await session.legibleSelection?.value
        #expect(!session.showsSearch)
        #expect(session.selectedVideoID == fixture.video.id)
        #expect(session.currentSeconds == 0)
        #expect(session.timelineFollowsPlayback)
    }

    @Test func unmatchedQueryShowsEmptyResultsInsteadOfAnOldHit() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let session = PlaybackSession()
        session.videos = [fixture.video]
        session.selectedVideoID = fixture.video.id
        session.searchQuery = "見つからない文言"
        let cue = try #require(fixture.video.cues.first)
        session.searchHits = [SearchHit(videoID: fixture.video.id, videoTitle: "old", cue: cue)]
        let view = SearchScreen(session: session)
        try await performSearch(view, session: session)
        #expect(session.searchHits.isEmpty)
        #expect(
            try view.inspect().find(ViewType.ContentUnavailableView.self).accessibilityLabel()
                .string() == "検索結果はありません")
        #expect(try view.inspect().findAll(ViewType.List.self).isEmpty)
    }

    @Test(.timeLimit(.minutes(1))) func cancellingDebounceDoesNotPublishResultsOrAnError()
        async throws
    {
        let session = PlaybackSession()
        session.searchQuery = "旅行"
        let view = SearchScreen(session: session)
        let task = Task { try await performSearch(view, session: session) }
        while !session.isSearching {
            try Task.checkCancellation()
            await Task.yield()
        }
        let progress = Result {
            let label = try view.inspect().find(ViewType.ProgressView.self).labelView().text()
                .string()
            #expect(label == "検索中")
        }
        task.cancel()
        try await task.value
        try progress.get()
        #expect(session.searchHits.isEmpty)
        #expect(!session.showsError)
    }

    private func performSearch(_ view: SearchScreen, session: PlaybackSession) async throws {
        let request = SearchRequest(
            query: session.searchQuery, scope: session.searchScope,
            currentVideoID: session.selectedVideoID, catalogRevision: session.catalogRevision)
        try await view.inspect().navigationStack().callTask(id: request)
    }
}
