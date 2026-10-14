import AppKit
import SwiftUI
import Testing
import TranscriptCore
import ViewInspector

@testable import TranscriptSummaries
@testable import TranscriptViewer

@MainActor
struct RootViewTests {
    @Test func fileMenuRevealsTheExactMovieAndOnlyInteractiveScrollingSuspendsFollow() throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var revealed: URL?
        let session = PlaybackSession()
        let view = TranscriptRootView(session: session, revealInFinder: { revealed = $0 })
        try view.fileMenu(for: fixture.video).inspect().find(button: "Reveal Finder").tap()
        #expect(revealed == fixture.video.movieURL)
        view.handleScrollPhase(.idle, .animating)
        #expect(session.timelineFollowsPlayback)
        view.handleScrollPhase(.idle, .interacting)
        #expect(!session.timelineFollowsPlayback)
        view.handleScrollPhase(.interacting, .decelerating)
        #expect(!session.timelineFollowsPlayback)
    }

    @Test func activationRefreshesTheSelectedLibraryWithoutSelectingAnotherFolder() {
        var scanned: URL?
        let root = URL(fileURLWithPath: "/synthetic-library")
        let session = PlaybackSession(scanCatalog: { directory, completion in
            scanned = directory
            completion(.failure(TranscriptError.unreadableDirectory("synthetic")))
        })
        session.load(root: root)
        scanned = nil
        TranscriptRootView(session: session).refreshOnActivation(
            Notification(name: NSApplication.didBecomeActiveNotification))
        #expect(scanned == root)
        #expect(session.libraryName == root.lastPathComponent)
        #expect(session.showsError)
    }

    @Test func summaryTimestampButtonUsesTheRootPlaybackSession() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let model = NarrativeSummaryModel(service: AppleSummaryService { _ in "旅行の準備を説明している。" })
        let session = PlaybackSession(narrativeSummary: model)
        defer {
            session.player.pause()
            session.stopObserving()
        }
        session.videos = [fixture.video]
        session.selectedVideoID = fixture.video.id
        await model.load(video: fixture.video)
        session.currentSeconds = 5
        session.beginTimelineBrowse()
        let summary = try TranscriptRootView(session: session).inspect().find(
            NarrativeSummaryView.self)
        let section = try summary.find(ViewType.Button.self) {
            try $0.labelView().find(ViewType.Label.self).title().text().string().contains("00:00")
        }
        try section.tap()
        #expect(session.currentSeconds == 0)
        #expect(session.timelineFollowsPlayback)
    }

    @Test(.timeLimit(.minutes(1))) func stopBatchClearsItsTaskIdentityAndDisplaysCancellation()
        async throws
    {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let gate = ViewResponseGate()
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { _ in await gate.respond() })
        let session = PlaybackSession(narrativeSummary: model)
        session.videos = [fixture.video]
        let view = TranscriptRootView(session: session)
        try view.inspect().find(button: "新しい順にすべて要約").tap()
        let operation = Task {
            try await view.inspect().navigationSplitView().callTask(id: session.summaryBatchRun)
        }
        await gate.wait()
        let result = Result {
            let stop = try view.inspect().find(button: "一括生成を停止")
            #expect(stop.isDisabled() == false)
            #expect(model.libraryProgress?.isRunning == true)
            #expect(try view.inspect().find(text: "生成中：配信日時不明").string() == "生成中：配信日時不明")
            try stop.tap()
            #expect(session.summaryBatchRun == nil)
            #expect(try view.inspect().find(button: "一括生成を停止").isDisabled())
        }
        operation.cancel()
        await gate.release()
        try await operation.value
        try result.get()
        #expect(model.libraryProgress?.wasCancelled == true)
        #expect(
            try view.inspect().find(text: "一括生成を停止しました。完了した要約は再利用できます。").string()
                == "一括生成を停止しました。完了した要約は再利用できます。")
        #expect(!FileManager.default.fileExists(atPath: fixture.current.path))
    }

    @Test func timelineSliderAndTranscriptButtonResumeFollowAndSeekTheDisplayedCue() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var presented = 0
        let session = PlaybackSession(presentWindow: { window in
            presented += 1
            window.close()
        })
        defer {
            session.player.pause()
            session.stopObserving()
        }
        var video = fixture.video
        video.broadcastStartJST = "2026-01-02_03-04-05"
        session.videos = [video]
        session.selectedVideoID = video.id
        session.openSelection()
        await session.legibleSelection?.value
        session.stopObserving()
        session.showsSummary = false
        let view = TranscriptRootView(session: session)
        #expect(
            try view.inspect().findAll(ViewType.Text.self).contains {
                try $0.string() == "旅行の準備を説明する。"
            })
        session.beginTimelineBrowse()
        let slider = try view.inspect().find(ViewType.Slider.self)
        #expect(try slider.value() == 0)
        try slider.setValue(0.5)
        #expect(session.currentSeconds == 0.5)
        #expect(session.timelineFollowsPlayback)
        #expect(
            try view.inspect().find(ViewType.Slider.self).accessibilityValue().string() == "0時間0分0秒"
        )
        let list = try view.inspect().find(UtteranceListView.self).find(ViewType.List.self)
        try list.find(ViewType.Button.self).tap()
        #expect(session.currentSeconds == 0)
        let nearby = try view.inspect().find(ViewType.ScrollView.self) {
            try $0.accessibilityLabel().string() == "付近の発話"
        }
        try nearby.find(ViewType.Button.self).tap()
        #expect(session.currentSeconds == 0)
        let detached = try view.inspect().find(ViewType.Button.self) {
            try $0.accessibilityLabel().string() == "認識結果を別ウィンドウで開く"
        }
        try detached.tap()
        #expect(presented == 1)
        let separate = UtteranceWindow(session: session)
        #expect(try separate.inspect().findAll(UtteranceListView.self).count == 1)
        session.selectedVideoID = nil
        #expect(
            try separate.inspect().find(ViewType.ContentUnavailableView.self).labelView().find(
                text: "動画が選ばれていません"
            ).string() == "動画が選ばれていません")
    }

    @Test(arguments: [false, true])
    func completedBatchCountsSuccessAndFailureInsteadOfHidingErrors(_ failure: Bool) async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { _ in
                if failure { throw SummaryGenerationError.emptyResponse }
                return "旅行の準備を説明している。"
            })
        let session = PlaybackSession(narrativeSummary: model)
        session.videos = [fixture.video]
        let view = TranscriptRootView(session: session)
        let request: UUID? = nil
        try await view.inspect().navigationSplitView().callTask(id: request)
        #expect(model.libraryProgress == nil)
        try view.inspect().find(button: "新しい順にすべて要約").tap()
        try await view.inspect().navigationSplitView().callTask(id: session.summaryBatchRun)
        #expect(model.libraryProgress?.completed == 1)
        #expect(model.libraryProgress?.succeeded == (failure ? 0 : 1))
        let message = failure ? "失敗した動画があります。再実行で未完了分を試します。" : "読み込み済み動画の要約が完了しました。"
        #expect(try view.inspect().find(text: message).string() == message)
        #expect(try view.inspect().find(button: "新しい順にすべて要約").isDisabled() == false)
    }

    @Test func rootLifecycleLoadsOnlyTheInjectedLibraryAndPresentsErrorsAndSearch() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var scanned: URL?
        let session = PlaybackSession(
            readLibraryRoot: { nil }, defaultLibrary: fixture.root,
            readManifest: { _, completion in completion(.success([])) },
            scanCatalog: { root, completion in
                scanned = root
                completion(.failure(TranscriptError.unreadableDirectory("synthetic")))
            })
        let view = TranscriptRootView(session: session)
        try await view.inspect().navigationSplitView().callTask()
        #expect(scanned == fixture.root)
        let alert = try view.inspect().navigationSplitView().alert()
        #expect(try alert.title().string() == "読み込みエラー")
        #expect(try alert.message().text().string() == "フォルダを読めません: synthetic")
        try alert.find(button: "OK").tap()
        session.showsSearch = true
        #expect(
            try view.inspect().navigationSplitView().sheet().findAll(SearchScreen.self).count == 1)
        session.currentSeconds = 12
        try view.inspect().navigationSplitView().callOnChange(
            oldValue: URL?.none, newValue: URL?.none)
        // Another window reporting the same selection must not reset playback.
        #expect(session.currentSeconds == 12)
        session.scanIssues = ["合成テストの未読込"]
        #expect(try view.inspect().find(text: "未読込 1 件").string() == "未読込 1 件")
    }

    @Test func emptyLibraryShowsSelectionInstructionsAndDisablesBatchGeneration() throws {
        let session = PlaybackSession()
        let inspected = try TranscriptRootView(session: session).inspect()
        #expect(
            try inspected.find(ViewType.ContentUnavailableView.self).labelView().find(text: "動画を選択")
                .string() == "動画を選択")
        #expect(try inspected.find(button: "新しい順にすべて要約").isDisabled())
        #expect(try inspected.find(text: "ライブラリ: 未選択").string() == "ライブラリ: 未選択")
    }

    @Test func ordinaryRootRefreshTaskIsCancelledWithItsView() async throws {
        let root = URL(fileURLWithPath: "/synthetic-default-library")
        let session = PlaybackSession(
            readLibraryRoot: { nil }, defaultLibrary: root,
            readManifest: { _, completion in completion(.success([])) },
            scanCatalog: { _, completion in completion(.success(CatalogScan())) })
        session.loadDefaultLibrary()
        let view = TranscriptRootView(session: session)
        let task = Task {
            try await view.inspect().navigationSplitView().callTask(id: true)
        }
        await Task.yield()
        task.cancel()
        try await task.value
        #expect(session.catalogRevision == 1)
        #expect(!session.showsError)
    }

    @Test func selectedVideoExposesPlaybackSpeedAndSwitchesBetweenSummaryAndTranscript() throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let session = PlaybackSession()
        session.videos = [fixture.video]
        session.selectedVideoID = fixture.video.id
        let view = TranscriptRootView(session: session)
        let speed = try view.inspect().find(ViewType.Picker.self) {
            try $0.labelView().text().string() == "再生速度"
        }
        #expect(try speed.selectedValue(PlaybackSpeed.self) == .normal)
        try speed.select(value: PlaybackSpeed.double)
        #expect(session.playbackSpeed == .double)
        #expect(session.player.defaultRate == 2)
        let content = try view.inspect().find(ViewType.Picker.self) {
            try $0.labelView().text().string() == "表示内容"
        }
        #expect(try content.selectedValue(Bool.self))
        #expect(try view.inspect().findAll(NarrativeSummaryView.self).count == 1)
        try content.select(value: false)
        #expect(!session.showsSummary)
        #expect(try view.inspect().findAll(UtteranceListView.self).count == 1)
        #expect(try view.inspect().find(text: "発話なし").string() == "発話なし")
        try view.inspect().find(button: "新しい順にすべて要約").tap()
        #expect(session.summaryBatchRun != nil)
    }

    @Test func toolbarRoutesSearchAndCancelledFolderChoiceWithoutShowingADialog() throws {
        let session = PlaybackSession(
            readLibraryRoot: { nil },
            saveLibraryRoot: { _ in
                Issue.record("Cancelled folder selection must not persist")
            })
        let panel = TestDirectoryPanel()
        let view = TranscriptRootView(session: session, directoryPanel: { panel })
        let toolbar = try view.inspect().navigationSplitView().sidebarView().vStack().toolbar()
        try toolbar.find(button: "検索").tap()
        #expect(session.showsSearch)
        try toolbar.find(button: "フォルダを開く").tap()
        #expect(session.catalogRevision == 0)
        #expect(!session.isScanning)
        try toolbar.find(button: "再読み込み").tap()
        #expect(session.catalogRevision == 0)
    }
}
