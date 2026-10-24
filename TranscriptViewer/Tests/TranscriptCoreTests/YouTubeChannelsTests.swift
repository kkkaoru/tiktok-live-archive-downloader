import Foundation
import Testing
import TranscriptCore

@testable import TranscriptViewer

struct YouTubeChannelsTests {
    static let kasumi = "UCEmlNV5-edGhNWYDMeGDgWw"

    static func originals() throws -> URL {
        let root = try transcriptTestWorkRoot().appending(
            path: "youtube-channels-\(UUID().uuidString)", directoryHint: .isDirectory)
        let folder = root.appending(
            path: "2026-10-22_00-00-20_JST_youtube-FeR-drWHf4c", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data(#"{"channel_id":"\#(kasumi)","channel":"c","uploader_id":"@c"}"#.utf8).write(
            to: folder.appending(path: "channel.json"))
        return root
    }

    @Test func decodesTheWorkerRecord() throws {
        let data = Data(#"{"version":1,"channels":{"\#(Self.kasumi)":{"label":"熊童子かすみ"}}}"#.utf8)
        #expect(
            try YouTubeChannels.decode(data) == YouTubeChannels(labels: [Self.kasumi: "熊童子かすみ"]))
        #expect(throws: DecodingError.self) { try YouTubeChannels.decode(Data("{}".utf8)) }
    }

    @Test func readsTheChannelBesideTheOriginal() throws {
        let root = try Self.originals()
        defer { #expect(throws: Never.self) { try FileManager.default.removeItem(at: root) } }
        #expect(
            YouTubeChannels.channelID(title: "youtube-FeR-drWHf4c", originals: root) == Self.kasumi)
        #expect(YouTubeChannels.channelID(title: "youtube-missing0000", originals: root) == nil)
        #expect(YouTubeChannels.channelID(title: "replay-1", originals: root) == nil)
        #expect(
            YouTubeChannels.channelID(
                title: "youtube-FeR-drWHf4c", originals: root.appending(path: "none")) == nil)
    }

    @Test func theApiAsksForTheChannelsWithTheBearer() {
        let request = ClipsAPI(base: URL(string: "https://v.example.test")!, token: "t").channels()
        #expect(request.url?.absoluteString == "https://v.example.test/api/channels.json")
        #expect(request.httpMethod == "GET")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer t")
    }

    @MainActor @Test func labelsLocalYouTubeVideosFromTheWorker() async throws {
        let root = try Self.originals()
        defer { #expect(throws: Never.self) { try FileManager.default.removeItem(at: root) } }
        let replies = ClipsModelTests.Replies([
            (#"{"version":1,"channels":{"\#(Self.kasumi)":{"label":"熊童子かすみ"}}}"#, 200),
            ("", 500),
        ])
        let model = ChannelsModel(api: ClipsModelTests.api, originals: root)
        model.fetch = { try replies.next($0) }
        let video = VideoTranscript(
            title: "youtube-FeR-drWHf4c", movieURL: URL(fileURLWithPath: "/m/a.mp4"),
            transcriptURL: URL(fileURLWithPath: "/m/a.srt"), cues: [])
        await model.load(titles: ["youtube-FeR-drWHf4c", "replay-1"])
        #expect(model.label(for: video) == "熊童子かすみ")
        // A failing Worker keeps the labels read before.
        await model.load(titles: ["youtube-FeR-drWHf4c"])
        #expect(model.label(for: video) == "熊童子かすみ")
        #expect(replies.requests == ["GET /api/channels.json?", "GET /api/channels.json?"])
        // No YouTube videos: nothing is fetched.
        await model.load(titles: ["replay-1"])
        #expect(replies.requests.count == 2)
        let none = ChannelsModel(api: nil, originals: root)
        await none.load(titles: ["youtube-FeR-drWHf4c"])
        #expect(none.label(for: video) == nil)
        #expect(ChannelsModel.configuredOriginals?.lastPathComponent != "")
    }
}
