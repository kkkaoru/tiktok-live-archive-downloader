import SwiftUI
import Testing
import TranscriptCore
import ViewInspector

@testable import TranscriptViewer

@MainActor
struct AppSceneTests {
    @Test func mainWindowFactorySharesPlaybackStateAcrossWindows() throws {
        let app = TranscriptViewerApp()
        #expect(
            String(reflecting: type(of: app.body)).contains("WindowGroup<"))
        #expect(String(reflecting: type(of: app.makeWindowContent())).contains("SceneBridgeReader"))
        let first = app.makeRootView()
        let second = app.makeRootView()
        #expect(first.session === second.session)
        #expect(first.session.playbackSpeed == .normal)
        first.session.playbackSpeed = .oneAndHalf
        #expect(second.session.playbackSpeed == .oneAndHalf)
        #expect(second.session.player.defaultRate == 1.5)
        #expect(try second.inspect().find(button: "新しい順にすべて要約").isDisabled())
        #expect(try second.inspect().find(text: "ライブラリ: 未選択").string() == "ライブラリ: 未選択")
    }
}
