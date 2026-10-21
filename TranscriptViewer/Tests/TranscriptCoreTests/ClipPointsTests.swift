import Foundation
import SwiftUI
import Testing
import TranscriptCore
import ViewInspector

@testable import TranscriptViewer

let twoRangeClip = ClipPoint(
    id: "c1", kind: "long", duration: 40,
    ranges: [ClipRange(start: 100, end: 120), ClipRange(start: 300, end: 320)],
    title: "山場", summary: "要約")

struct ClipPointsTests {
    @Test func decodesEveryStatus() throws {
        let ready = Data(
            #"{"status":"ready","record":{"clips":[{"id":"c1","kind":"short","duration":12,"score":1,"ranges":[{"start":1,"end":13}],"title":"t","summary":"s"}]}}"#
                .utf8)
        #expect(
            try ClipStatus.decode(ready)
                == .ready([
                    ClipPoint(
                        id: "c1", kind: "short", duration: 12,
                        ranges: [ClipRange(start: 1, end: 13)], title: "t", summary: "s")
                ]))
        #expect(try ClipStatus.decode(Data(#"{"status":"pending","force":true}"#.utf8)) == .pending)
        #expect(try ClipStatus.decode(Data(#"{"status":"mismatch"}"#.utf8)) == .mismatch)
        #expect(try ClipStatus.decode(Data(#"{"status":"missing"}"#.utf8)) == .missing)
        #expect(try ClipStatus.decode(Data(#"{"status":"ready"}"#.utf8)) == .missing)
        #expect(
            try ClipStatus.decode(
                Data(#"{"status":"ready","record":{"clips":[],"excitement":false}}"#.utf8))
                == .noExcitement)
        #expect(throws: DecodingError.self) { try ClipStatus.decode(Data("{}".utf8)) }
    }

    @Test func labelsLengthsAndElapsedTime() {
        #expect(twoRangeClip.kindLabel == "長尺 30〜90秒")
        #expect(
            ClipPoint(id: "s", kind: "short", duration: 1, ranges: [], title: "", summary: "")
                .kindLabel == "短尺 10〜30秒")
        #expect(ClipPoint.seconds(27.64) == "27.6秒")
        #expect(twoRangeClip.elapsed(range: 1, at: 305) == 25)
        #expect(twoRangeClip.elapsed(range: 0, at: 90) == 0)
        #expect(twoRangeClip.elapsed(range: 2, at: 0) == 40)
    }

    @Test func playsRangesInOrderAndStopsWhenPlaybackLeaves() {
        var cursor = ClipCursor(clip: twoRangeClip)
        #expect(cursor.step(at: 40) == .keep)  // the player still reports the time before the seek
        #expect(cursor.settled == false)
        #expect(cursor.step(at: 100.1) == .keep)
        #expect(cursor.step(at: 120.05) == .jump(to: 300))
        #expect(cursor.index == 1)
        #expect(cursor.step(at: 120.1) == .keep)
        #expect(cursor.step(at: 300) == .keep)
        #expect(cursor.step(at: 320) == .finish)

        var away = ClipCursor(clip: twoRangeClip, index: 1)
        #expect(away.step(at: 301) == .keep)
        #expect(away.step(at: 500) == .leave)
        var empty = ClipCursor(
            clip: ClipPoint(id: "e", kind: "short", duration: 0, ranges: [], title: "", summary: "")
        )
        #expect(empty.range == nil)
        #expect(empty.step(at: 0) == .finish)

        var loop = ClipCursor(clip: twoRangeClip, index: 1, loop: true)
        #expect(loop.step(at: 310) == .keep)
        #expect(loop.step(at: 320) == .jump(to: 100))
        #expect(loop.index == 0)
        #expect(loop.step(at: 100.2) == .keep)
        #expect(loop.step(at: 120) == .jump(to: 300))
        #expect(
            ClipPoint(
                id: "h", kind: "long", pattern: "hook-to-climax", duration: 0, ranges: [],
                title: "",
                summary: ""
            ).patternLabel == "引き→山場")
        #expect(twoRangeClip.patternLabel == nil)
    }

    @Test func readsTheApiFromTheCheckoutAndSendsTheBearer() throws {
        let root = try transcriptTestWorkRoot().appending(
            path: "clips-api-\(UUID().uuidString)", directoryHint: .isDirectory)
        defer { #expect(throws: Never.self) { try FileManager.default.removeItem(at: root) } }
        let folder = root.appending(path: "web-viewer", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        #expect(ClipsAPI.load(checkout: root) == nil)
        try Data("# c\nVIEWER_DOMAIN=\"v.example.test, w.example.test\"\n".utf8).write(
            to: folder.appending(path: "deploy.env"))
        try Data(#"{"CLIPS_API_TOKEN":"short"}"#.utf8).write(
            to: folder.appending(path: "secrets.json"))
        #expect(ClipsAPI.load(checkout: root) == nil)
        let token = String(repeating: "t", count: 40)
        try Data(#"{"CLIPS_API_TOKEN":"\#(token)"}"#.utf8).write(
            to: folder.appending(path: "secrets.json"))
        let api = try #require(ClipsAPI.load(checkout: root))
        #expect(api.base.absoluteString == "https://v.example.test")
        let lookup = api.lookup(videoID: "v 1", movieKey: "abc")
        #expect(lookup.url?.absoluteString == "https://v.example.test/api/clips/v%201.json?key=abc")
        #expect(lookup.value(forHTTPHeaderField: "Authorization") == "Bearer \(token)")
        let reload = api.reload(videoID: "v1")
        #expect(reload.httpMethod == "POST")
        #expect(reload.url?.path() == "/api/clips/v1/reload")
        #expect(ClipsAPI.domain(in: "VIEWER_DOMAIN=") == nil)
        #expect(ClipsAPI.videoID(URL(fileURLWithPath: "/m/2026_a-review.mp4")) == "2026_a-review")
    }
}

@MainActor
struct ClipsModelTests {
    static func waitUntil(_ done: () -> Bool) async throws {
        for _ in 0..<500 where !done() { try await Task.sleep(for: .milliseconds(10)) }
    }

    static let api = ClipsAPI(base: URL(string: "https://v.example.test")!, token: "t")

    /// Answers each request with the next queued reply and records the requests.
    final class Replies: @unchecked Sendable {
        private let lock = NSLock()
        private var replies: [(String, Int)]
        private(set) var requests: [String] = []
        init(_ replies: [(String, Int)]) { self.replies = replies }
        func next(_ request: URLRequest) throws -> (Data, Int) {
            lock.lock()
            defer { lock.unlock() }
            requests.append(
                "\(request.httpMethod ?? "") \(request.url?.path() ?? "")?\(request.url?.query() ?? "")"
            )
            guard !replies.isEmpty else { throw URLError(.notConnectedToInternet) }
            let (body, status) = replies.removeFirst()
            return (Data(body.utf8), status)
        }
    }

    static func model(_ replies: Replies) -> ClipsModel {
        let model = ClipsModel(api: api)
        model.fetch = { try replies.next($0) }
        model.movieKey = { _ in "k" }
        model.pollInterval = .milliseconds(1)
        return model
    }

    static let readyBody =
        #"{"status":"ready","record":{"clips":[{"id":"c1","kind":"long","duration":40,"ranges":[{"start":100,"end":120},{"start":300,"end":320}],"title":"山場","summary":"要約"}]}}"#

    @Test func pollsWhilePendingThenShowsTheClips() async throws {
        let replies = Replies([
            (#"{"status":"pending","force":false}"#, 200), (Self.readyBody, 200),
        ])
        let model = Self.model(replies)
        model.open(URL(fileURLWithPath: "/m/v1.mp4"))
        #expect(model.state == .loading)
        try await Self.waitUntil { model.clips.count == 1 }
        #expect(model.clips == [twoRangeClip])
        #expect(
            replies.requests == ["GET /api/clips/v1.json?key=k", "GET /api/clips/v1.json?key=k"])
        #expect(model.canReload)
    }

    @Test func reportsMissingMismatchAndFailures() async throws {
        let model = Self.model(Replies([(#"{"status":"missing"}"#, 404)]))
        model.open(URL(fileURLWithPath: "/m/a.mp4"))
        try await Self.waitUntil { model.state == .missing }
        #expect(model.state == .missing)

        let mismatch = Self.model(Replies([(#"{"status":"mismatch"}"#, 200)]))
        mismatch.open(URL(fileURLWithPath: "/m/a.mp4"))
        try await Self.waitUntil { mismatch.state == .mismatch }
        #expect(mismatch.state == .mismatch)

        let denied = Self.model(Replies([("", 401)]))
        denied.open(URL(fileURLWithPath: "/m/a.mp4"))
        try await Self.waitUntil { denied.state != .loading }
        #expect(denied.state == .failed("切り抜きを読み込めませんでした（HTTP 401）"))

        let offline = Self.model(Replies([]))
        offline.open(URL(fileURLWithPath: "/m/a.mp4"))
        try await Self.waitUntil { offline.state != .loading }
        #expect(offline.state != .loading)
        #expect(offline.clips.isEmpty)

        let none = ClipsModel(api: nil)
        none.open(URL(fileURLWithPath: "/m/a.mp4"))
        #expect(none.state == .unavailable)
        none.open(nil)
        #expect(none.state == .idle)
        #expect(!none.canReload)
        none.reload()
        #expect(none.state == .idle)

        let early = Self.model(
            Replies([(#"{"status":"ready","record":{"clips":[],"excitement":false}}"#, 200)]))
        early.open(URL(fileURLWithPath: "/m/a.mp4"))
        try await Self.waitUntil { early.state == .noExcitement }
        #expect(early.state == .noExcitement)
    }

    @Test func theDefaultsReadFilesAndTheCheckout() async throws {
        let root = try transcriptTestWorkRoot().appending(
            path: "clips-defaults-\(UUID().uuidString)", directoryHint: .isDirectory)
        defer { #expect(throws: Never.self) { try FileManager.default.removeItem(at: root) } }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let file = root.appending(path: "abc.mp4")
        try Data("abc".utf8).write(to: file)
        let model = ClipsModel(api: ClipsModel.configured)
        #expect(try model.movieKey(file) == "2e8e504c5e6b886af70f8c03cbfc31ac")
        let (data, status) = try await model.fetch(URLRequest(url: file))
        #expect(String(decoding: data, as: UTF8.self) == "abc")
        #expect(status == 0)
    }

    @Test func reloadAsksTheWorkerThenPollsForTheNewClips() async throws {
        let replies = Replies([
            (Self.readyBody, 200), (#"{"status":"pending","force":true}"#, 200),
            (Self.readyBody, 200),
        ])
        let model = Self.model(replies)
        model.open(URL(fileURLWithPath: "/m/v1.mp4"))
        try await Self.waitUntil { model.clips.count == 1 }
        model.reload()
        #expect(model.state == .pending)
        try await Self.waitUntil { model.clips.count == 1 }
        #expect(replies.requests[1] == "POST /api/clips/v1/reload?")
        #expect(replies.requests.count == 3)

        let refused = Self.model(Replies([(Self.readyBody, 200), ("", 403)]))
        refused.open(URL(fileURLWithPath: "/m/v1.mp4"))
        try await Self.waitUntil { refused.clips.count == 1 }
        refused.reload()
        try await Self.waitUntil { refused.state != .pending }
        #expect(refused.state == .failed("再計算を開始できませんでした（HTTP 403）"))

        let offline = Self.model(Replies([(Self.readyBody, 200)]))
        offline.open(URL(fileURLWithPath: "/m/v1.mp4"))
        try await Self.waitUntil { offline.clips.count == 1 }
        offline.reload()
        try await Self.waitUntil { offline.state != .pending }
        #expect(offline.clips.isEmpty)
    }

    @Test func theSessionPlaysEveryRangeOfAClipInOrder() throws {
        let model = ClipsModel(api: nil)
        let session = PlaybackSession(clips: model)
        #expect(model.start(twoRangeClip, range: 5) == 300)
        model.stop()
        #expect(model.cursor == nil)
        #expect(model.step(at: 1) == .keep)
        session.play(twoRangeClip, range: 0)
        #expect(model.cursor?.index == 0)
        session.updateCue(at: 110)
        session.updateCue(at: 120)
        #expect(model.cursor?.index == 1)
        session.updateCue(at: 310)
        session.updateCue(at: 320)
        #expect(model.cursor == nil)
        #expect(session.player.rate == 0)
        session.play(twoRangeClip, range: 1)
        session.updateCue(at: 305)
        session.updateCue(at: 10)
        #expect(model.cursor == nil)
        session.play(twoRangeClip, range: 1, loop: true)
        session.updateCue(at: 310)
        session.updateCue(at: 320)
        #expect(model.cursor?.index == 0)
        #expect(model.cursor?.loop == true)
    }

    @Test func thePanelListsClipsWithTheirLengthsAndRanges() async throws {
        let model = Self.model(Replies([(Self.readyBody, 200)]))
        model.open(URL(fileURLWithPath: "/m/v1.mp4"))
        try await Self.waitUntil { model.clips.count == 1 }
        var played: [String] = []
        let panel = ClipsPanel(model: model) { _, index, loop in played.append("\(index) \(loop)") }
        #expect(panel.note == "1件")
        try panel.inspect().find(viewWithAccessibilityLabel: "山場を再生（40.0秒）").button().tap()
        try panel.inspect().find(viewWithAccessibilityLabel: "山場をループ再生").button().tap()
        try panel.inspect().find(button: "2. 0:05:00〜（20.0秒）").tap()
        #expect(played == ["0 false", "0 true", "1 false"])
        #expect(
            try panel.inspect().find(text: "40.0秒・長尺 30〜90秒・2区間").string() == "40.0秒・長尺 30〜90秒・2区間")
        _ = model.start(twoRangeClip, range: 1)
        #expect(panel.note == "再生中: 山場（40.0秒・区間2/2、20.0秒の区間）")
        _ = model.start(twoRangeClip, range: 0, loop: true)
        #expect(panel.note == "ループ再生中: 山場（40.0秒・区間1/2、20.0秒の区間）")
        model.stop()
        let story = ClipPoint(
            id: "s", kind: "short", pattern: "climax-first", duration: 12,
            ranges: [ClipRange(start: 0, end: 12)], title: "t", summary: "s")
        #expect(
            ClipRow(clip: story, cursor: nil) { _, _, _ in }.lengthLabel
                == "12.0秒・短尺 10〜30秒・山場→経緯")
        let notes = [
            ClipsModel.State.idle, .loading, .ready([]), .pending, .mismatch, .missing,
            .noExcitement, .unavailable,
            .failed("x"),
        ].map { state in
            ClipsPanel.note(for: state)
        }
        #expect(
            notes == [
                "", "読み込み中…", "切り抜きに向く区間が見つかりませんでした", "計算中です（数分かかります）…",
                "Web版の動画と内容が異なるため表示できません", "この動画の切り抜きはまだありません",
                "盛り上がりデータがまだないため作成できません（計算して同期すると自動で作成されます）", "Web版の接続設定がありません",
                "読み込めませんでした: x",
            ])
    }
}
