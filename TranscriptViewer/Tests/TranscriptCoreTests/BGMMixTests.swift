import Foundation
import Testing
import TranscriptCore

struct BGMMixTests {
    @Test func aSettingChoosesTheDefaultNoneOrItsOwnTrackAndRoundTrips() throws {
        var setting = VideoBGMSetting()
        #expect(setting.trackID(default: "d") == "d")
        setting.selection = .off
        #expect(setting.trackID(default: "d") == nil)
        setting.selection = .track("t")
        setting.mode = .manual
        #expect(setting.trackID(default: "d") == "t")
        let data = try JSONEncoder().encode(setting)
        #expect(try JSONDecoder().decode(VideoBGMSetting.self, from: data) == setting)
        let movie = URL(
            fileURLWithPath: "/m/2026-10-19_18-30-18_JST_youtube-t__ULrIUydE-review.mp4")
        let video = VideoTranscript(
            title: "t", movieURL: movie, transcriptURL: movie, cues: [])
        #expect(VideoBGMSetting.key(for: video) == "youtube-t__ULrIUydE")
        let plain = VideoTranscript(
            title: "t", movieURL: URL(fileURLWithPath: "/m/clip.mp4"), transcriptURL: movie,
            cues: [])
        #expect(VideoBGMSetting.key(for: plain) == "clip.mp4")
    }

    @Test func levelsAreSmoothedHeldThroughPausesAndMappedToAClampedGain() {
        #expect(BGMLevel.smoothed([]).isEmpty)
        #expect(BGMLevel.smoothed([-90, -80]).isEmpty)
        let levels = [-80.0, -20, -20, -80, -80, -80, -10]
        let smoothed = BGMLevel.smoothed(levels, radius: 1)
        #expect(smoothed.count == levels.count)
        #expect(abs(smoothed[0] + 20) < 1e-9)
        #expect(abs(smoothed[4] + 20) < 1e-9)
        #expect(abs(smoothed[6] + 10) < 1e-9)
        #expect(abs(BGMLevel.gain(video: -20, bgm: -10, offsetDB: -20) - 0.031_623) < 1e-5)
        #expect(BGMLevel.gain(video: -10, bgm: -40, offsetDB: 0) == 1)
    }

    @Test func storedLevelsAreFoundByThePythonMovieKey() throws {
        let root = try transcriptTestWorkRoot().appending(
            path: "bgm-mix-\(UUID().uuidString)", directoryHint: .isDirectory)
        let cache = root.appending(path: "cache", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        defer { #expect(throws: Never.self) { try FileManager.default.removeItem(at: root) } }
        let small = root.appending(path: "a.mp4")
        try Data("abc".utf8).write(to: small)
        let large = root.appending(path: "b.mp4")
        try Data((0..<(3 << 19)).map { UInt8($0 % 251) }).write(to: large)
        let empty = root.appending(path: "e.mp4")
        try Data().write(to: empty)
        // Keys printed by `replay.excitement.movie_key` for the same bytes.
        #expect(try BGMLevel.movieKey(small) == "2e8e504c5e6b886af70f8c03cbfc31ac")
        #expect(try BGMLevel.movieKey(large) == "3d890c891d3c9bf5344b66383deee2a9")
        #expect(try BGMLevel.movieKey(empty) == "5feceb66ffc86f38d952786c6d696c79")

        try Data("[-20.5, -96]".utf8).write(
            to: cache.appending(path: "levels-2e8e504c5e6b886af70f8c03cbfc31ac.json"))
        #expect(BGMLevel.cachedLevels(for: small, in: cache) == [-20.5, -96])
        try Data("\"bad\"".utf8).write(
            to: cache.appending(path: "levels-3d890c891d3c9bf5344b66383deee2a9.json"))
        #expect(BGMLevel.cachedLevels(for: large, in: cache) == nil)
        #expect(BGMLevel.cachedLevels(for: empty, in: cache) == nil)
        #expect(BGMLevel.cachedLevels(for: root.appending(path: "missing.mp4"), in: cache) == nil)
    }
}
