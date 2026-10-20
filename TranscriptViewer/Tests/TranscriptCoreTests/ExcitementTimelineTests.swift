import AppKit
import CryptoKit
import Foundation
import SwiftUI
import Testing
import TranscriptCore
import ViewInspector

@testable import TranscriptViewer

/// A synthetic transcript with a hand-written estimate; never a real recording.
struct ExcitementFixture {
    let root: URL
    let transcript: URL

    init(transcriptText: String = "1\n00:00:01,000 --> 00:00:02,000\nやあ\n") throws {
        root = try transcriptTestWorkRoot().appending(
            path: "excitement-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        transcript = root.appending(path: "a.srt")
        try Data(transcriptText.utf8).write(to: transcript)
    }

    var sha256: String {
        get throws {
            SHA256.hash(data: try Data(contentsOf: transcript)).map { String(format: "%02x", $0) }
                .joined()
        }
    }

    func write(comments: Bool = true, version: Int = 1, sha: String? = nil, bins: Int = 2) throws {
        var record: [String: Any] = [
            "version": version, "binSeconds": 10, "duration": 20,
            "speech": [
                "score": Array([1.0, 0.25].prefix(bins)), "characters": [12.0, 0.0],
                "loudnessDb": [-20.5, NSNull()], "peaks": [0.0],
            ],
            "comments": NSNull(), "combined": NSNull(),
            "inputs": ["transcriptSha256": try sha ?? sha256],
        ]
        if comments {
            record["comments"] = [
                "score": [0.5, 1.0], "count": [1, 3], "characters": [4, 9], "peaks": [10.0],
            ]
            record["combined"] = ["score": [0.75, 1.0], "peaks": [10.0]]
        }
        try JSONSerialization.data(withJSONObject: record).write(
            to: ExcitementTimeline.url(for: transcript))
    }

    func remove() throws { try FileManager.default.removeItem(at: root) }
}

struct ExcitementTimelineTests {
    @Test func loadsTheEstimateOfTheCurrentTranscript() throws {
        let fixture = try ExcitementFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.write()
        let timeline = try #require(ExcitementTimeline.load(transcriptURL: fixture.transcript))
        #expect(timeline.binCount == 2)
        #expect(timeline.rows(.combined).map(\.label) == ["合算"])
        #expect(timeline.rows(.split).map(\.label) == ["発話", "コメント"])
        #expect(timeline.bin(atFraction: -1) == 0)
        #expect(timeline.bin(atFraction: 0.6) == 1)
        #expect(timeline.bin(atFraction: 2) == 1)
        #expect(timeline.seconds(ofBin: 1) == 10)
        #expect(
            timeline.detail(ofBin: 0)
                == "0:00:00〜 / 合算 75% / 発話 100%（12文字・-20.5dB） / コメント 50%（1件・4文字）")
        #expect(timeline.detail(ofBin: 1) == "0:00:10〜 / 合算 100% / 発話 25%（0文字） / コメント 100%（3件・9文字）")
        #expect(ExcitementTimeline.clock(3725) == "1:02:05")
    }

    @Test func speechOnlyEstimateHasOneRowInEveryMode() throws {
        let fixture = try ExcitementFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.write(comments: false)
        let timeline = try #require(ExcitementTimeline.load(transcriptURL: fixture.transcript))
        #expect(timeline.rows(.split).map(\.label) == ["発話"])
        #expect(timeline.detail(ofBin: 0) == "0:00:00〜 / 発話 100%（12文字・-20.5dB）")
    }

    @Test func ignoresMissingStaleWrongVersionAndInconsistentFiles() throws {
        let fixture = try ExcitementFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        #expect(ExcitementTimeline.load(transcriptURL: fixture.transcript) == nil)
        try fixture.write(sha: "older-transcript-version")
        #expect(ExcitementTimeline.load(transcriptURL: fixture.transcript) == nil)
        try fixture.write(version: 2)
        #expect(ExcitementTimeline.load(transcriptURL: fixture.transcript) == nil)
        try fixture.write(bins: 1)
        #expect(ExcitementTimeline.load(transcriptURL: fixture.transcript) == nil)
    }

    @Test func ignoresASymlinkedEstimate() throws {
        let fixture = try ExcitementFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.write()
        let target = fixture.root.appending(path: "elsewhere.json")
        let link = ExcitementTimeline.url(for: fixture.transcript)
        try FileManager.default.moveItem(at: link, to: target)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        #expect(ExcitementTimeline.load(transcriptURL: fixture.transcript) == nil)
    }
}

@MainActor
struct ExcitementPanelTests {
    @Test func peaksClicksAndHoverSeekAndExplain() throws {
        let fixture = try ExcitementFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.write()
        let timeline = try #require(ExcitementTimeline.load(transcriptURL: fixture.transcript))
        var seeks: [Double] = []
        let panel = ExcitementPanel(timeline: timeline, currentSeconds: 5) { seeks.append($0) }
        UserDefaults.standard.removeObject(forKey: "excitementMode")
        try panel.inspect().find(button: "0:00:10").tap()
        #expect(throws: Never.self) { try panel.inspect().find(text: "合算") }
        panel.jump(fraction: 0.1)
        panel.tapHandler(width: 200)(CGPoint(x: 150, y: 5))
        panel.hoverHandler(width: 200)(.active(CGPoint(x: 10, y: 5)))
        panel.hoverHandler(width: 200)(.ended)
        #expect(seeks == [10, 0, 10])
        #expect(
            panel.peakTitle(row: timeline.rows(.split)[1], start: 10, split: true)
                == "コメント 0:00:10")
        #expect(throws: Never.self) { try panel.inspect().find(ExcitementBars.self) }
        #expect(ExcitementBars.color("コメント") == .green)
        #expect(ExcitementBars.color("合算") == .orange)
        #expect(ExcitementBars.color("発話") == .accentColor)
    }

    @Test func speechOnlyPanelExplainsItsSource() throws {
        let fixture = try ExcitementFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.write(comments: false)
        let timeline = try #require(ExcitementTimeline.load(transcriptURL: fixture.transcript))
        let panel = ExcitementPanel(timeline: timeline, currentSeconds: 0) { _ in }
        #expect(throws: Never.self) { try panel.inspect().find(text: "発話（文字認識量・音量）から推定") }
    }

    @Test func hostedPanelDrawsAndHoverShowsTheDetail() throws {
        let fixture = try ExcitementFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.write()
        let timeline = try #require(ExcitementTimeline.load(transcriptURL: fixture.transcript))
        let section = ExcitementSection(transcriptURL: fixture.transcript, currentSeconds: 3) { _ in
        }
        let host = NSHostingView(
            rootView: VStack {
                section
                ExcitementPanel(timeline: timeline, currentSeconds: 3) { _ in }
            }
        )
        host.frame = CGRect(x: 0, y: 0, width: 600, height: 300)
        host.layoutSubtreeIfNeeded()
        let image = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: image)
        let panel = ExcitementPanel(timeline: timeline, currentSeconds: 0) { _ in }
        panel.show(fraction: 0.9)
        #expect(image.pixelsWide > 0)
    }

    @Test func sectionLoadsTheEstimateForItsTranscript() async throws {
        let fixture = try ExcitementFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.write()
        let section = ExcitementSection(transcriptURL: fixture.transcript, currentSeconds: 0) { _ in
        }
        try await section.inspect().group().callTask(id: fixture.transcript)
        #expect(throws: Never.self) { try section.inspect().group() }
    }
}
