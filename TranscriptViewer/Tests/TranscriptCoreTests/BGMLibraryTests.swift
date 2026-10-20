import AppKit
import Foundation
import Testing

@testable import TranscriptCore

/// A synthetic BGM library; never a downloaded track.
struct BGMFixture {
    let root: URL

    init() throws {
        root = try transcriptTestWorkRoot().appending(
            path: "bgm-\(UUID().uuidString)/edits/bgm-library", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    @discardableResult
    func add(
        id: String = "CYzeUrG3p9o", title: String = "Ｆｒｅｅ BGM ⧸ Night", channel: String = "Sharou",
        record: ((inout [String: Any]) -> Void)? = nil
    ) throws -> URL {
        let folder = root.appending(path: id, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("audio".utf8).write(to: folder.appending(path: "\(title) [\(id)].m4a"))
        try Self.png().write(to: folder.appending(path: "\(title) [\(id)].jpg"))
        var value: [String: Any] = [
            "version": 1, "id": id, "title": title, "channel": channel,
            "channelURL": "https://www.youtube.com/channel/UC1",
            "url": "https://www.youtube.com/watch?v=\(id)", "duration": 60.0,
            "audio": "\(title) [\(id)].m4a", "thumbnail": "\(title) [\(id)].jpg",
        ]
        record?(&value)
        try JSONSerialization.data(withJSONObject: value).write(
            to: folder.appending(path: "track.json"))
        return folder
    }

    func remove() throws {
        try FileManager.default.removeItem(
            at: root.deletingLastPathComponent().deletingLastPathComponent())
    }

    static func png() throws -> Data {
        let bitmap = try #require(
            NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: 2, pixelsHigh: 2, bitsPerSample: 8,
                samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                bytesPerRow: 0, bitsPerPixel: 0))
        return try #require(bitmap.representation(using: .png, properties: [:]))
    }
}

struct BGMLibraryTests {
    @Test func loadsValidTracksAndReportsEachUnreadableFolder() throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.add()
        try fixture.add(id: "HWsOivtPz-o", title: "Amber", channel: "Other") {
            $0["channelURL"] = "https://evil.example/c"
        }
        try fixture.add(id: "bad-version", title: "V") { $0["version"] = 2 }
        try fixture.add(id: "wrong-id-11", title: "W") { $0["id"] = "other" }
        try fixture.add(id: "bad-url-111", title: "U") {
            $0["url"] = "https://www.youtube.com/watch?v=other"
        }
        try fixture.add(id: "nested-1111", title: "N") { $0["audio"] = "../x.m4a" }
        let linked = try fixture.add(id: "linked-1111", title: "L")
        try FileManager.default.removeItem(at: linked.appending(path: "L [linked-1111].m4a"))
        try FileManager.default.createSymbolicLink(
            at: linked.appending(path: "L [linked-1111].m4a"),
            withDestinationURL: fixture.root.appending(path: "CYzeUrG3p9o/track.json"))
        let large = try fixture.add(id: "large-11111", title: "G")
        try Data(repeating: 32, count: 70_000).write(to: large.appending(path: "track.json"))
        try FileManager.default.createDirectory(
            at: fixture.root.appending(path: "empty-folder"), withIntermediateDirectories: true)

        let loaded = BGMLibrary.load(in: fixture.root)
        #expect(loaded.tracks.map(\.id) == ["HWsOivtPz-o", "CYzeUrG3p9o"])
        #expect(loaded.tracks[0].channelURL == nil)
        #expect(
            loaded.tracks[1].channelURL?.absoluteString == "https://www.youtube.com/channel/UC1")
        #expect(
            loaded.tracks[1].pageURL.absoluteString == "https://www.youtube.com/watch?v=CYzeUrG3p9o"
        )
        #expect(loaded.tracks[1].duration == 60)
        #expect(!loaded.tracks[1].loops)
        #expect(
            loaded.issues == [
                "bad-url-111", "bad-version", "empty-folder", "large-11111", "linked-1111",
                "nested-1111", "wrong-id-11",
            ])
        #expect(BGMLibrary.load(in: fixture.root.appending(path: "missing")).tracks.isEmpty)
    }

    @Test func searchMatchesEveryWordInTitleOrChannelIgnoringCaseAndWidth() throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.add()
        try fixture.add(id: "HWsOivtPz-o", title: "Amber", channel: "Other")
        let tracks = BGMLibrary.load(in: fixture.root).tracks
        #expect(BGMLibrary.search(tracks, query: "").count == 2)
        #expect(BGMLibrary.search(tracks, query: "free night").map(\.id) == ["CYzeUrG3p9o"])
        #expect(BGMLibrary.search(tracks, query: "ＳＨＡＲＯＵ").map(\.id) == ["CYzeUrG3p9o"])
        #expect(BGMLibrary.search(tracks, query: "amber other").map(\.id) == ["HWsOivtPz-o"])
        #expect(BGMLibrary.search(tracks, query: "amber night").isEmpty)
    }

    @Test func rootIsTheCheckoutEditFolderOnlyForADistApp() throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let checkout = fixture.root.deletingLastPathComponent().deletingLastPathComponent()
        let app = checkout.appending(path: "TranscriptViewer/dist/TranscriptViewer.app")
        try FileManager.default.createDirectory(at: app, withIntermediateDirectories: true)
        #expect(BGMLibrary.root(near: app)?.standardizedFileURL == fixture.root.standardizedFileURL)
        #expect(BGMLibrary.root(near: checkout.appending(path: "Other.app")) == nil)
    }

    @Test func aConfiguredRootReplacesTheCheckoutFolderAndBadConfigsGiveNone() throws {
        let fixture = try BGMFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let checkout = fixture.root.deletingLastPathComponent().deletingLastPathComponent()
        let app = checkout.appending(path: "TranscriptViewer/dist/TranscriptViewer.app")
        try FileManager.default.createDirectory(at: app, withIntermediateDirectories: true)
        let config = fixture.root.deletingLastPathComponent().appending(path: "bgm-library.json")
        let external = checkout.appending(path: "ssd/bgm-library").standardizedFileURL.path
        func write(_ text: String) throws { try Data(text.utf8).write(to: config) }

        try write(#"{"version":1,"root":"\#(external)"}"#)
        #expect(BGMLibrary.root(near: app)?.path(percentEncoded: false) == external + "/")
        #expect(
            BGMLibrary.load(in: try #require(BGMLibrary.root(near: app))).issues == [external + "/"]
        )
        try write(#"{"version":2,"root":"\#(external)"}"#)
        #expect(BGMLibrary.root(near: app) == nil)
        try write(#"{"version":1,"root":"relative/bgm"}"#)
        #expect(BGMLibrary.root(near: app) == nil)
        try write(#"{"version":1,"root":"\#(external)/../x"}"#)
        #expect(BGMLibrary.root(near: app) == nil)
        try write(#"{"version":1,"root":"/Volumes/x/downloads/bgm"}"#)
        #expect(BGMLibrary.root(near: app) == nil)
        try FileManager.default.removeItem(at: config)
        try FileManager.default.createSymbolicLink(
            at: config, withDestinationURL: fixture.root.appending(path: "none.json"))
        #expect(BGMLibrary.root(near: app) == nil)
    }
}
