import AVFoundation
import Foundation
import SwiftUI
import Testing
import TranscriptCore
import ViewInspector

@testable import TranscriptViewer

@MainActor
struct VideoBGMTests {
    static func video(_ name: String) -> VideoTranscript {
        let movie = URL(fileURLWithPath: "/m/\(name).mp4")
        return VideoTranscript(title: name, movieURL: movie, transcriptURL: movie, cues: [])
    }

    static func waitUntil(_ done: () -> Bool) async throws {
        for _ in 0..<1000 where !done() { try await Task.sleep(for: .milliseconds(10)) }
    }

    @Test func eachVideoKeepsItsOwnChoiceAndTheOthersFollowTheDefault() throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.add()
        try fixture.add(id: "HWsOivtPz-o", title: "Amber", channel: "Other")
        let library = BGMLibraryModel(root: fixture.root, volume: 1, defaultID: "CYzeUrG3p9o")
        var defaults: [String?] = []
        library.storeDefault = { defaults.append($0) }
        var saved: [[String: VideoBGMSetting]] = []
        let model = VideoBGMModel(library: library, settings: [:])
        model.store = { saved.append($0) }
        model.loadLevels = { _ in nil }

        model.open(Self.video("first"))
        #expect(library.tracks.count == 2)
        #expect(model.track?.id == "CYzeUrG3p9o")
        #expect(model.playingID == nil)
        model.videoPlaybackChanged(true)
        #expect(model.playingID == "CYzeUrG3p9o")
        model.select(.track("HWsOivtPz-o"))
        #expect(model.playingID == "HWsOivtPz-o")
        #expect(saved.last?["first.mp4"]?.selection == .track("HWsOivtPz-o"))
        model.select(.track("HWsOivtPz-o"))
        #expect(saved.count == 1)

        model.open(Self.video("second"))
        #expect(model.setting == VideoBGMSetting())
        #expect(model.playingID == "CYzeUrG3p9o")
        model.select(.off)
        #expect(model.playingID == nil)
        #expect(model.player.currentItem == nil)

        model.open(Self.video("first"))
        #expect(model.setting.selection == .track("HWsOivtPz-o"))
        model.videoPlaybackChanged(false)
        #expect(model.player.rate == 0)
        model.open(Self.video("third"))
        library.toggleDefault(library.tracks[0])
        #expect(defaults == ["HWsOivtPz-o"])
        #expect(model.track?.id == "HWsOivtPz-o")
        library.toggleDefault(library.tracks[0])
        #expect(model.track == nil)
        #expect(model.defaultTrack == nil)
        model.open(nil)
        #expect(model.key == nil)
        model.player.replaceCurrentItem(with: nil)
        library.player.replaceCurrentItem(with: nil)
    }

    @Test func automaticVolumeFollowsTheVideoAroundThePlayheadAndManualIsFixed() async throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.add { $0["loudness"] = -4.0 }
        let library = BGMLibraryModel(root: fixture.root, volume: 1, defaultID: "CYzeUrG3p9o")
        let model = VideoBGMModel(library: library, settings: [:])
        model.store = { _ in }
        model.loadLevels = { _ in Array(repeating: -26, count: 20) + [-80] }
        model.open(Self.video("first"))
        #expect(model.isLoading)
        #expect(library.tracks.first?.loudness == -4)
        // Until the stored levels are read, a typical video level keeps the BGM quiet.
        let assumed = BGMLevel.gain(video: BGMLevel.typicalVideoDB, bgm: -4, offsetDB: -22)
        #expect(model.volume == assumed)
        #expect(assumed < 0.1)
        try await Self.waitUntil { !model.isLoading }
        #expect(model.videoLevels?.count == 21)
        #expect(VideoBGMControls(model: model).automaticLabel == "声との差 -22dB")
        model.update(at: 100)
        #expect(
            abs(model.player.volume - BGMLevel.gain(video: -26, bgm: -4, offsetDB: -22)) < 1e-6)
        model.setting.offsetDB = 0
        model.update(at: -5)
        #expect(model.player.volume == BGMLevel.gain(video: -26, bgm: -4, offsetDB: 0))
        // The BGM follows the video's own volume and mute.
        let video = AVPlayer()
        model.follow(video)
        video.volume = 0.5
        model.update(at: 0)
        #expect(model.player.volume == BGMLevel.gain(video: -26, bgm: -4, offsetDB: 0) * 0.5)
        video.isMuted = true
        model.update(at: 0)
        #expect(model.player.volume == 0)
        video.isMuted = false
        model.setting.mode = .manual
        model.setting.volume = 0.7
        #expect(model.player.volume == 0.35)

        // A video's stored levels are read once per launch.
        var calls = 0
        model.loadLevels = { _ in
            await MainActor.run { calls += 1 }
            return nil
        }
        model.setting.mode = .automatic
        model.open(Self.video("first"))
        #expect(!model.isLoading)
        #expect(calls == 0)
    }

    @Test func theVideoAndTheLibraryNeverPlayTwoBGMsAndAPlainTrackLoops() async throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.add()
        try fixture.add(id: "HWsOivtPz-o", title: "Amber", channel: "Other")
        let library = BGMLibraryModel(root: fixture.root, volume: 1, defaultID: "CYzeUrG3p9o")
        // The real loader and store: no levels are stored for this video, so typical ones apply.
        let model = VideoBGMModel(library: library, settings: [:])
        model.open(Self.video("first"))
        library.selectedID = "HWsOivtPz-o"
        model.videoPlaybackChanged(true)
        #expect(library.playingID == nil)
        library.selectedID = "CYzeUrG3p9o"
        #expect(model.player.rate == 0)
        NotificationCenter.default.post(
            name: AVPlayerItem.didPlayToEndTimeNotification, object: model.player.currentItem)
        #expect(model.loopRestarts == 1)
        try await Self.waitUntil { !model.isLoading }
        #expect(model.lacksLevels)
        #expect(
            model.volume
                == BGMLevel.gain(
                    video: BGMLevel.typicalVideoDB, bgm: BGMLevel.typicalBGMDB, offsetDB: -22))
        let player = AVPlayer()
        model.follow(player)
        try await Self.waitUntil { !model.videoPlaying }
        #expect(!model.videoPlaying)
        let suite = try #require(UserDefaults(suiteName: "VideoBGMTests"))
        defer { suite.removePersistentDomain(forName: "VideoBGMTests") }
        #expect(VideoBGMModel.saved(suite).isEmpty)
        var setting = VideoBGMSetting()
        setting.selection = .off
        suite.set(try JSONEncoder().encode(["k": setting]), forKey: VideoBGMModel.settingsKey)
        #expect(VideoBGMModel.saved(suite)["k"] == setting)
        model.select(.off)
        UserDefaults.standard.removeObject(forKey: VideoBGMModel.settingsKey)
        let movie = fixture.root.appending(path: "a.mp4")
        try Data("abc".utf8).write(to: movie)
        try Data("[-30]".utf8).write(
            to: fixture.root.appending(path: "levels-2e8e504c5e6b886af70f8c03cbfc31ac.json"))
        #expect(VideoBGMModel.storedLevels(movie, cache: fixture.root) == [-30])
        #expect(VideoBGMModel.storedLevels(movie, cache: nil) == nil)
        model.player.replaceCurrentItem(with: nil)
        library.player.replaceCurrentItem(with: nil)
    }

    @Test func stoppingTheBGMKeepsEveryVideosSettingAndResumesWhereItWas() throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.add()
        let library = BGMLibraryModel(root: fixture.root, volume: 1, defaultID: "CYzeUrG3p9o")
        var saved: [[String: VideoBGMSetting]] = []
        let model = VideoBGMModel(library: library, settings: [:])
        model.store = { saved.append($0) }
        model.loadLevels = { _ in nil }
        model.open(Self.video("first"))
        model.setting.mode = .manual
        model.videoPlaybackChanged(true)
        let item = model.player.currentItem
        #expect(model.playingID == "CYzeUrG3p9o")
        let controls = VideoBGMControls(model: model)
        try controls.inspect().find(button: "BGMを止める").tap()
        #expect(model.stopped)
        #expect(model.player.rate == 0)
        #expect(model.player.currentItem === item)
        #expect(model.setting.mode == .manual)
        #expect(saved.count == 1)
        model.open(Self.video("second"))
        #expect(model.player.rate == 0)
        try controls.inspect().find(button: "BGMを再開").tap()
        #expect(!model.stopped)
        #expect(model.player.currentItem === item)
        #expect(model.player.rate > 0)
        model.player.pause()
        model.player.replaceCurrentItem(with: nil)
    }

    @Test func theChooserSearchesShowsArtworkAndSavesTheChoice() async throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.add()
        try fixture.add(id: "HWsOivtPz-o", title: "Amber", channel: "Other")
        let library = BGMLibraryModel(root: fixture.root, volume: 1, defaultID: nil)
        let model = VideoBGMModel(library: library, settings: [:])
        model.store = { _ in }
        model.loadLevels = { _ in nil }
        model.open(Self.video("first"))
        var closed = 0
        let picker = VideoBGMPicker(model: model) { closed += 1 }
        #expect(throws: Never.self) { try picker.inspect().find(text: "未設定") }
        try picker.inspect().find(viewWithAccessibilityLabel: "BGMを検索").textField()
            .setInput("amber")
        #expect(model.results.map(\.id) == ["HWsOivtPz-o"])
        try picker.inspect().find(viewWithAccessibilityLabel: "BGM候補 Amber").button().tap()
        #expect(model.setting.selection == .track("HWsOivtPz-o"))
        #expect(closed == 1)

        let controls = VideoBGMControls(model: model)
        controls.picker().done()
        #expect(controls.chosenLabel == "Amber")
        try controls.inspect().find(viewWithAccessibilityLabel: "BGMを選択").button().tap()
        #expect(controls.automaticLabel == "声との差 -22dB・読込中")
        try await Self.waitUntil { !model.isLoading }
        #expect(
            try controls.inspect().find(text: "声との差 -22dB・音量データなし").string()
                == "声との差 -22dB・音量データなし")
        try controls.inspect().find(viewWithAccessibilityLabel: "動画の音声に対するBGMの大きさ").slider()
            .setValue(30.0 / 36)
        #expect(model.setting.offsetDB == -6)
        try picker.inspect().find(viewWithAccessibilityLabel: "BGM候補 BGMなし").button().tap()
        #expect(controls.chosenLabel == "BGMなし")
        model.select(.libraryDefault)
        #expect(controls.chosenLabel == "デフォルト（未設定）")
        library.defaultID = "HWsOivtPz-o"
        #expect(controls.chosenLabel == "デフォルト: Amber")
        model.select(.track("gone-111111"))
        #expect(controls.chosenLabel == "見つからないBGM")
        model.setting.mode = .manual
        try controls.inspect().find(viewWithAccessibilityLabel: "BGM音量").slider().setValue(0.5)
        #expect(try controls.inspect().find(text: "50%").string() == "50%")
        model.player.replaceCurrentItem(with: nil)
    }

    @Test func theBGMTabSetsAndClearsTheDefault() throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.add()
        let library = BGMLibraryModel(root: fixture.root, volume: 1, defaultID: nil)
        var stored: [String?] = []
        library.storeDefault = { stored.append($0) }
        library.reload()
        let track = try #require(library.tracks.first)
        let view = BGMLibraryView(model: library)
        try view.detail(track).inspect().find(button: "デフォルトに設定").tap()
        #expect(library.defaultID == track.id)
        #expect(throws: Never.self) {
            try view.inspect().find(viewWithAccessibilityLabel: "デフォルトBGM")
        }
        try view.detail(track).inspect().find(button: "デフォルトを解除").tap()
        #expect(stored == [track.id, nil])
    }
}
