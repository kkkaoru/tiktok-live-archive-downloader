import AVFoundation
import AppKit
import SwiftUI
import Testing
import TranscriptCore
import ViewInspector

@testable import TranscriptViewer

@MainActor
struct BGMLibraryViewTests {
    @Test func playbackRevealCopyAndSearchUseTheSelectedTrack() throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.add()
        try fixture.add(id: "HWsOivtPz-o", title: "Amber", channel: "Other")
        var revealed: URL?
        var copied: String?
        let model = BGMLibraryModel(root: fixture.root)
        model.reveal = { revealed = $0 }
        model.copy = { copied = $0 }
        model.reload()
        let track = try #require(model.tracks.last)
        model.query = "night"
        #expect(model.results.map(\.id) == [track.id])
        model.selectedID = track.id
        let view = BGMLibraryView(model: model)

        #expect(model.playingID == track.id)
        #expect(model.playing == track)
        try view.inspect().find(viewWithAccessibilityLabel: "BGMを検索").textField().setInput("amber")
        #expect(model.results.map(\.id) == ["HWsOivtPz-o"])
        #expect(try view.inspect().find(text: "Other").string() == "Other")
        model.query = "night"
        #expect(throws: Never.self) {
            try view.detail(track).inspect().find(viewWithAccessibilityLabel: "再生中のサムネイル")
        }
        #expect(throws: Never.self) { try view.inspect().find(AudioControls.self) }
        try view.detail(track).inspect().find(button: "Reveal Finder").tap()
        #expect(revealed == track.audioURL)
        try view.detail(track).inspect().find(button: "パスをコピー").tap()
        #expect(copied == track.audioURL.path(percentEncoded: false))
        try view.nowPlaying(track).inspect().find(button: "一時停止").tap()
        #expect(model.playingID == nil)
        try view.detail(track).inspect().find(button: "再生").tap()
        try view.detail(track).inspect().find(button: "一時停止").tap()
        #expect(model.playingID == nil)
        try view.detail(track).inspect().find(viewWithAccessibilityLabel: "サムネイル").button().tap()
        #expect(model.playingID == track.id)
        try view.detail(track).inspect().find(viewWithAccessibilityLabel: "再生中のサムネイル").button()
            .tap()
        #expect(model.playingID == nil)
        model.selectedID = track.id
        #expect(model.playingID == nil)
        model.selectedID = nil
        #expect(model.playingID == nil)
        model.selectedID = model.tracks[0].id
        #expect(model.playingID == model.tracks[0].id)
        #expect(model.player.currentTime() == .zero)
        model.player.replaceCurrentItem(with: nil)
    }

    @Test func aLoopTrackRestartsAtItsEndAndAPlainOneDoesNot() throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.add { $0["loop"] = true }
        try fixture.add(id: "HWsOivtPz-o", title: "Amber", channel: "Other")
        let model = BGMLibraryModel(root: fixture.root)
        model.reload()
        let loop = try #require(model.tracks.first { $0.id == "CYzeUrG3p9o" })
        #expect(loop.loops)
        model.selectedID = loop.id
        #expect(
            try BGMLibraryView(model: model).detail(loop).inspect().find(text: "ループ再生").string()
                == "ループ再生")
        NotificationCenter.default.post(
            name: AVPlayerItem.didPlayToEndTimeNotification, object: model.player.currentItem)
        #expect(model.loopRestarts == 1)
        model.selectedID = "HWsOivtPz-o"
        NotificationCenter.default.post(
            name: AVPlayerItem.didPlayToEndTimeNotification, object: model.player.currentItem)
        #expect(model.loopRestarts == 1)
        #expect(throws: (any Error).self) {
            try BGMLibraryView(model: model).detail(model.tracks[0]).inspect().find(text: "ループ再生")
        }
        model.player.replaceCurrentItem(with: nil)
    }

    @Test func oneVolumeAppliesToEveryTrackAndIsStored() async throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.add()
        try fixture.add(id: "HWsOivtPz-o", title: "Amber", channel: "Other")
        var stored: [Float] = []
        let model = BGMLibraryModel(root: fixture.root, volume: 0.4)
        model.storeVolume = { stored.append($0) }
        model.reload()
        #expect(model.player.volume == 0.4)
        let view = BGMLibraryView(model: model)
        #expect(try view.volumeControl.inspect().find(text: "40%").string() == "40%")
        try view.volumeControl.inspect().find(ViewType.Slider.self).setValue(0.25)
        #expect(model.volume == 0.25)
        model.selectedID = "CYzeUrG3p9o"
        model.selectedID = "HWsOivtPz-o"
        #expect(model.player.volume == 0.25)
        model.volume = 3
        #expect(model.player.volume == 1)
        #expect(stored == [0.25, 1])
        model.player.volume = 0.6
        for _ in 0..<100 where model.volume != 0.6 { try await Task.sleep(for: .milliseconds(10)) }
        #expect(model.volume == 0.6)
        #expect(stored == [0.25, 1, 0.6])
        #expect(BGMLibraryModel(root: nil, volume: -1).player.volume == 0)
        #expect(BGMLibraryModel(root: nil).volume >= 0)
        model.player.replaceCurrentItem(with: nil)
    }

    @Test func savingRunsTheDownloaderAndReportsTheOutcome() async throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var requests: [String] = []
        let model = BGMLibraryModel(root: fixture.root)
        model.download = { url, root in
            await MainActor.run { requests.append(url) }
            if url.hasSuffix("bad") { return (1, "403") }
            _ = try? BGMFixture.png().write(to: root.appending(path: "added.txt"))
            return (0, "")
        }
        await model.add()
        #expect(requests.isEmpty)
        model.newURL = " https://www.youtube.com/watch?v=CYzeUrG3p9o "
        await model.add()
        #expect(requests == ["https://www.youtube.com/watch?v=CYzeUrG3p9o"])
        #expect(model.addMessage == "保存しました")
        #expect(model.newURL.isEmpty)
        model.newURL = "https://example/bad"
        let view = BGMLibraryView(model: model)
        try view.addForm.inspect().find(button: "保存").tap()
        for _ in 0..<100 where model.addMessage != "保存できませんでした: 403" {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(model.addMessage == "保存できませんでした: 403")
        #expect(!model.isAdding)
    }

    @Test func emptyAndMissingLibrariesExplainThemselves() throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try FileManager.default.createDirectory(
            at: fixture.root.appending(path: "broken"), withIntermediateDirectories: true)
        let model = BGMLibraryModel(root: fixture.root)
        model.reload()
        let view = BGMLibraryView(model: model)
        #expect(try view.inspect().find(text: "BGMを選択してください").string() == "BGMを選択してください")
        #expect(try view.addForm.inspect().find(text: "読み込めないBGM 1 件").string() == "読み込めないBGM 1 件")
        let missing = BGMLibraryModel(root: nil)
        missing.reload()
        let empty = BGMLibraryView(model: missing)
        #expect(
            try empty.inspect().find(text: "BGMライブラリが見つかりません").string()
                == "BGMライブラリが見つかりません")
        #expect(try empty.addForm.inspect().find(button: "保存").isDisabled())
    }

    @Test func thumbnailFallsBackAndChannelWithoutLinkIsPlainText() throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let folder = try fixture.add { $0["channelURL"] = NSNull() }
        try Data("not an image".utf8).write(
            to: folder.appending(path: "Ｆｒｅｅ BGM ⧸ Night [CYzeUrG3p9o].jpg"))
        let model = BGMLibraryModel(root: fixture.root)
        model.reload()
        let track = try #require(model.tracks.first)
        let view = BGMLibraryView(model: model)
        #expect(throws: Never.self) {
            try view.thumbnail(track).inspect().find(ViewType.Image.self)
        }
        #expect(try view.detail(track).inspect().find(text: "Sharou").string() == "Sharou")
    }

    @Test func audioControlsHostTheSharedPlayer() {
        let player = AVPlayer()
        let host = NSHostingView(
            rootView: AudioControls(player: player).frame(width: 200, height: 44))
        host.layoutSubtreeIfNeeded()
        host.rootView = AudioControls(player: player).frame(width: 220, height: 44)
        host.layoutSubtreeIfNeeded()
        #expect(host.subviews.isEmpty == false)
    }

    @Test func theCheckoutDownloaderRunsTheModuleWithoutAShell() async throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let checkout = fixture.root.deletingLastPathComponent().deletingLastPathComponent()
        let missing = await BGMLibraryModel.runCheckoutDownloader(
            "u", fixture.root, checkout: checkout)
        #expect(missing.status == -1)
        #expect(
            await BGMLibraryModel.runCheckoutDownloader("u", fixture.root, checkout: nil).output
                == "checkout not found")
        let python = checkout.appending(path: ".venv/bin/python")
        try FileManager.default.createDirectory(
            at: python.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("#!/bin/sh\necho \"$@\"\nexit 3\n".utf8).write(to: python)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: python.path)
        let result = await BGMLibraryModel.runCheckoutDownloader(
            "u", fixture.root, checkout: checkout)
        #expect(result.status == 3)
        #expect(result.output == "-B -m replay.bgm_library add u --root \(fixture.root.path)\n")
    }
}
