import CryptoKit
import Foundation
import Testing
import ViewInspector

@testable import TranscriptCore
@testable import TranscriptSummaries
@testable import TranscriptViewer

struct ExcerptFixture {
    let root: URL
    let source: VideoTranscript
    let edited: VideoTranscript
    let artifact: URL
    let pointer: URL

    init(text: String = "挨拶だけ。") throws {
        root = try transcriptTestWorkRoot().appending(path: "excerpt-reader-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let labels = root.appending(path: "labels.json")
        let media = root.appending(path: "source.mp4")
        try Data("source".utf8).write(to: media)
        struct Labels: Encodable {
            struct Label: Encodable {
                let start = 600
                let end = 602
                let text: String
            }
            let labels: [Label]
        }
        try JSONEncoder().encode(Labels(labels: [.init(text: text)])).write(to: labels)
        source = try SourceLabelTranscript(
            labels: labels, media: media, clockDomain: "source-presentation",
            inputKind: .tokenSequence
        )
        .load()
        edited = VideoTranscript(
            title: "edited", movieURL: root.appending(path: "edited.mp4"),
            transcriptURL: labels, cues: source.cues)
        let span = SourceExcerptSummary.Span(
            sectionStartSeconds: 600, lowerCharacter: 0, upperCharacter: text.count, text: text)
        let summary = SourceExcerptSummary(
            mode: "source-excerpts-v1", fingerprint: try SummaryStore.fingerprint(source),
            sections: [.init(startSeconds: 600, endSeconds: 1200, excerpts: [span])],
            overview: [span])
        artifact = root.appending(path: "artifact.json")
        try JSONEncoder().encode(summary).write(to: artifact)
        let directory = SummaryStore().directory(for: source)
        try SummaryStore.prepareDirectory(directory)
        pointer = directory.appending(path: "source-excerpts-selection.json")
        try write()
    }

    func write(
        version: Int = 1, clock: String = "source-presentation", labels: String? = nil,
        artifactHash: String? = nil, size: Int = 6
    ) throws {
        let selection = SourceExcerptStore.Selection(
            version: version, clockDomain: clock, sourceLabelLayout: "tokens",
            sourceLabels: labels ?? source.transcriptURL.path,
            sourceLabelsSha256: try Self.hash(source.transcriptURL),
            sourceMedia: source.movieURL.path,
            sourceMediaSize: size, artifact: artifact.path,
            artifactSha256: try artifactHash ?? Self.hash(artifact))
        try JSONEncoder().encode(selection).write(to: pointer)
    }

    static func hash(_ url: URL) throws -> String {
        SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
    }
    func remove() throws { try FileManager.default.removeItem(at: root) }
}

struct SourceExcerptStoreTests {
    @Test func explicitSourceAndEditedClocksRemainSeparate() async throws {
        let fixture = try ExcerptFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let raw = try #require(await SourceExcerptStore().load(video: fixture.source))
        let edited = try #require(await SourceExcerptStore().load(video: fixture.edited))
        #expect(raw.canSeekSelectedMovie)
        #expect(!edited.canSeekSelectedMovie)
        #expect(raw != edited)
        #expect(raw == raw)
        #expect(raw.summary.sections.first?.startSeconds == 600)
        #expect(raw.summary.sections.first?.excerpts.first?.text == "挨拶だけ。")
        #expect(try await SummaryStore().load(video: fixture.edited) == nil)
    }

    @Test(arguments: [
        "version", "clock", "relative", "hash", "size", "labels-changed", "text-changed", "symlink",
        "oversized", "json", "empty",
    ])
    func rejectsChangedEvidence(change: String) async throws {
        let fixture = try ExcerptFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        switch change {
        case "version": try fixture.write(version: 2)
        case "clock": try fixture.write(clock: "edited")
        case "relative": try fixture.write(labels: "labels.json")
        case "hash": try fixture.write(artifactHash: "bad")
        case "size": try fixture.write(size: 7)
        case "labels-changed": try Data("changed".utf8).write(to: fixture.source.transcriptURL)
        case "text-changed":
            let text = try String(contentsOf: fixture.artifact, encoding: .utf8)
                .replacingOccurrences(of: "挨拶だけ。", with: "架空です。")
            try Data(text.utf8).write(to: fixture.artifact)
            try fixture.write()
        case "symlink":
            let saved = fixture.root.appending(path: "saved.json")
            try FileManager.default.moveItem(at: fixture.pointer, to: saved)
            try FileManager.default.createSymbolicLink(
                at: fixture.pointer, withDestinationURL: saved)
        case "oversized": try Data(repeating: 32, count: 65_537).write(to: fixture.pointer)
        case "json": try Data("bad".utf8).write(to: fixture.pointer)
        case "empty": try Data().write(to: fixture.pointer)
        default: Issue.record("Unknown mutation")
        }
        await #expect(throws: (any Error).self) {
            try await SourceExcerptStore().load(video: fixture.source)
        }
    }

    @Test func absentPointerAndAbsentDirectoryReturnNoCandidate() async throws {
        let fixture = try ExcerptFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try FileManager.default.removeItem(at: fixture.pointer)
        #expect(try await SourceExcerptStore().load(video: fixture.source) == nil)
        try FileManager.default.removeItem(at: fixture.pointer.deletingLastPathComponent())
        #expect(try await SourceExcerptStore().load(video: fixture.source) == nil)
    }
}

@MainActor
struct SourceExcerptPanelTests {
    @Test func exactTextIsNotRepresentedAsApprovedSummaryOrEditedSeek() async throws {
        let fixture = try ExcerptFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let model = NarrativeSummaryModel()
        await model.loadExcerpts(video: fixture.edited)
        let panel = SourceExcerptPanel(state: model.excerptState) { _ in
            Issue.record("Edited source clocks cannot seek")
        }
        let inspected = try panel.inspect()
        #expect(try inspected.find(text: "原動画の発話抜粋候補").string() == "原動画の発話抜粋候補")
        #expect(
            try inspected.find(text: "原文の一部を表示しています。話題の代表性・認識精度は未確認で、完成した要約ではありません。").string()
                .contains("未確認"))
        #expect(inspected.findAll(ViewInspector.ViewType.Button.self).isEmpty)
        #expect(model.state == .idle)
        #expect(model.excerptVideo == fixture.edited)
        await model.loadExcerpts(video: fixture.source)
        var sought: Double?
        let raw = SourceExcerptPanel(state: model.excerptState) { sought = $0 }
        try raw.inspect().find(button: "原動画 600秒から").tap()
        #expect(sought == 600)
    }

    @Test func missingInvalidAndCancelledReadsDoNotChangeNarrativeState() async throws {
        let fixture = try ExcerptFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let model = NarrativeSummaryModel()
        try Data("bad".utf8).write(to: fixture.pointer)
        await model.loadExcerpts(video: fixture.source)
        #expect(model.excerptState == .failed)
        #expect(
            try SourceExcerptPanel(state: model.excerptState) { _ in }.inspect()
                .find(text: "原動画の抜粋候補を読み込めません。指定ファイルと照合情報を確認してください。")
                .string().contains("読み込めません"))
        let cancelled = Task { await model.loadExcerpts(video: fixture.source) }
        cancelled.cancel()
        await cancelled.value
        #expect(model.excerptState == .absent)
        try FileManager.default.removeItem(at: fixture.pointer)
        await model.loadExcerpts(video: fixture.source)
        #expect(model.excerptState == .absent)
        #expect(
            try SourceExcerptPanel(state: .absent) { _ in }.inspect().findAll(
                ViewInspector.ViewType.Text.self
            ).isEmpty)
        #expect(model.state == .idle)
    }
}
