import Foundation
import Testing
import TranscriptCore

@testable import TranscriptSummaries
@testable import TranscriptViewer

struct SummaryFixture {
    let root: URL
    let video: VideoTranscript

    init(source: String = "旅行の準備を説明する。") throws {
        root = try transcriptTestWorkRoot().appendingPathComponent("summary-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        let transcript = root.appendingPathComponent("sample.srt")
        let movie = root.appendingPathComponent("sample.mp4")
        try "1\n00:00:00,000 --> 00:00:01,000\n\(source)\n".write(
            to: transcript, atomically: false, encoding: .utf8)
        try Data().write(to: movie)
        video = VideoTranscript(
            title: "synthetic", movieURL: movie, transcriptURL: transcript,
            cues: try TranscriptSRT.cues(contentsOf: transcript))
    }

    func remove() throws { try FileManager.default.removeItem(at: root) }
    var current: URL { SummaryStore().directory(for: video).appendingPathComponent("current.json") }

    func generate(_ text: String = "旅行の準備について説明している。") async throws -> NarrativeSummary {
        let service = AppleSummaryService(store: SummaryStore()) { _ in text }
        return try await service.generate(video: video, force: true) { _, _ in }
    }
}

struct SummaryPersistenceTests {
    @Test @MainActor func cliOutputLoadsInAppWithoutModelAndSurvivesNewService() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let expected = try await fixture.generate()
        let offline = AppleSummaryService(store: SummaryStore()) { _ in
            Issue.record("Saved results must not invoke the provider")
            throw SummaryGenerationError.unavailable("offline")
        }
        let loaded = try await offline.generate(video: fixture.video, force: false) { _, _ in }
        #expect(loaded == expected)
        let model = NarrativeSummaryModel(service: offline)
        await model.loadSaved(video: fixture.video)
        #expect(model.state == .ready(expected))
        #expect(model.video == fixture.video)
        let attributes = try FileManager.default.attributesOfItem(atPath: fixture.current.path)
        #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
    }

    @Test @MainActor func appReloadsCliRevisionWithoutRegeneratingOrKeepingStaleMemory()
        async throws
    {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let command = try SummaryCommand(arguments: [
            "generate", "--library", fixture.root.path, "--force",
        ])
        #expect(
            try await command.run(respond: { _ in "最初の準備手順を説明している。" }, output: { _ in }) == 0)
        let offline = AppleSummaryService(store: SummaryStore()) { _ in
            Issue.record("Loading CLI revisions must not invoke the app provider")
            throw SummaryGenerationError.unavailable("offline")
        }
        let model = NarrativeSummaryModel(service: offline)
        await model.load(video: fixture.video)
        let original = try #require(await offline.cached(video: fixture.video))
        #expect(original.overview == "最初の準備手順を説明している。")
        #expect(model.state == .ready(original))

        #expect(
            try await command.run(respond: { _ in "更新された準備手順を説明している。" }, output: { _ in }) == 0)
        await model.loadSaved(video: fixture.video)
        guard case .ready(let updated) = model.state else {
            Issue.record("The app must display the newly saved CLI revision")
            return
        }
        #expect(updated.overview == "更新された準備手順を説明している。")
        #expect(updated.sections.map(\.text) == ["更新された準備手順を説明している。"])
        #expect(updated.sections.map { $0.source.startSeconds } == [0])
        #expect(updated.sections.map { $0.source.endSeconds } == [600])
        #expect(await offline.cached(video: fixture.video) == original)

        await model.load(video: fixture.video)
        #expect(model.state == .ready(updated))
        #expect(await offline.cached(video: fixture.video) == updated)
        #expect(model.video == fixture.video)
    }

    @Test func forcedFailurePreservesPreviousOutputAndRegenerationKeepsHistory() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        _ = try await fixture.generate()
        let before = try Data(contentsOf: fixture.current)
        let failing = AppleSummaryService(store: SummaryStore()) { _ in
            throw SummaryGenerationError.emptyResponse
        }
        await #expect(throws: SummaryGenerationError.emptyResponse) {
            try await failing.generate(video: fixture.video, force: true) { _, _ in }
        }
        #expect(try Data(contentsOf: fixture.current) == before)
        _ = try await fixture.generate("準備の手順を詳しく説明している。")
        let files = try FileManager.default.contentsOfDirectory(
            at: fixture.current.deletingLastPathComponent(),
            includingPropertiesForKeys: nil)
        #expect(files.filter { $0.pathExtension == "json" }.count == 4)
        #expect(try Data(contentsOf: fixture.current) != before)
    }

    @Test func diskUpdatesOverrideMemoryCacheAndChangedInputInvalidatesResults() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let service = AppleSummaryService(store: SummaryStore()) { _ in "最初の説明文です。" }
        _ = try await service.generate(video: fixture.video, force: false) { _, _ in }
        let updated = try await fixture.generate("更新された説明文です。")
        let loaded = try await service.generate(video: fixture.video, force: false) { _, _ in }
        #expect(loaded == updated)
        var changed = fixture.video
        changed.cues[0].text = "変更された文字起こし。"
        #expect(try await SummaryStore().load(video: changed) == nil)
        #expect(try SummaryStore.fingerprint(changed) != SummaryStore.fingerprint(fixture.video))
    }

    @Test @MainActor func missingAndCorruptFilesNeverBecomeSuccessfulSummaries() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let model = NarrativeSummaryModel()
        await model.loadSaved(video: fixture.video)
        #expect(model.state == .failed("この字幕に一致する保存済み要約がありません。"))
        let store = SummaryStore()
        try SummaryStore.prepareDirectory(store.directory(for: fixture.video))
        #expect(try await store.load(video: fixture.video) == nil)
        try Data("broken json".utf8).write(to: fixture.current)
        await #expect(throws: DecodingError.self) { try await store.load(video: fixture.video) }
        await model.loadSaved(video: fixture.video)
        guard case .failed = model.state else {
            Issue.record("Corrupt JSON must be visible")
            return
        }
    }

    @Test(arguments: ["version", "clock", "count", "refusal", "bullets"])
    func validatesExternalRecordFields(_ field: String) async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let store = SummaryStore()
        try SummaryStore.prepareDirectory(store.directory(for: fixture.video))
        let section = SummaryStore.Record.Section(
            startSeconds: field == "clock" ? 600 : 0,
            endSeconds: 600, text: "準備を説明している。")
        let overview: String
        switch field {
        case "refusal": overview = "申し訳ありません。"
        case "bullets": overview = "- 項目"
        default: overview = "準備を説明している。"
        }
        let record = SummaryStore.Record(
            version: field == "version" ? -1 : 1,
            fingerprint: try SummaryStore.fingerprint(fixture.video),
            overview: overview,
            sections: field == "count" ? [] : [section])
        try JSONEncoder().encode(record).write(to: fixture.current)
        if field == "version" {
            #expect(try await store.load(video: fixture.video) == nil)
        } else if field == "clock" || field == "count" {
            await #expect(throws: SummaryStoreError.self) {
                try await store.load(video: fixture.video)
            }
        } else {
            await #expect(throws: SummaryGenerationError.self) {
                try await store.load(video: fixture.video)
            }
        }
    }

    @Test(arguments: ["public", "symlink", "oversized"])
    func refusesUnsafeStorage(_ kind: String) async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let store = SummaryStore()
        let directory = store.directory(for: fixture.video)
        if kind == "symlink" {
            try FileManager.default.createSymbolicLink(
                at: directory, withDestinationURL: fixture.root)
        } else {
            try SummaryStore.prepareDirectory(directory)
            if kind == "public" {
                try FileManager.default.setAttributes(
                    [.posixPermissions: 0o755], ofItemAtPath: directory.path)
            } else {
                try Data(repeating: 32, count: 8 * 1024 * 1024 + 1).write(to: fixture.current)
            }
        }
        await #expect(throws: SummaryStoreError.self) { try await store.load(video: fixture.video) }
    }

    @Test @MainActor func cancelledSavedReloadDoesNotBecomeAnError() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let model = NarrativeSummaryModel()
        let task = Task { await model.loadSaved(video: fixture.video) }
        task.cancel()
        await task.value
        #expect(model.state == .idle)
    }

    @Test func explicitRegenerationRetainsEvenACorruptedPreviousPointer() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try SummaryStore.prepareDirectory(fixture.current.deletingLastPathComponent())
        let original = Data("damaged record retained for investigation".utf8)
        try original.write(to: fixture.current)
        _ = try await fixture.generate()
        let files = try FileManager.default.contentsOfDirectory(
            at: fixture.current.deletingLastPathComponent(), includingPropertiesForKeys: nil)
        let backup = try #require(files.first { $0.lastPathComponent.hasPrefix("previous-") })
        #expect(try Data(contentsOf: backup) == original)
        #expect(try await SummaryStore().load(video: fixture.video) != nil)
    }

    @Test func refusesEmptyInputsAndOversizedOutputsBeforePublication() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var empty = fixture.video
        empty.cues = []
        await #expect(throws: SummaryStoreError.self) {
            try await SummaryStore().save(
                NarrativeSummary(overview: "説明。", sections: []), video: empty)
        }
        let source = try #require(fixture.video.summary.sections.first)
        let large = NarrativeSummary(
            overview: String(repeating: "文", count: 3_000_000),
            sections: [.init(source: source, text: "説明している。")])
        await #expect(throws: SummaryStoreError.self) {
            try await SummaryStore().save(large, video: fixture.video)
        }
        #expect(!FileManager.default.fileExists(atPath: fixture.current.path))
    }

    @Test func kernelLeaseRejectsAnotherServiceAndCanBeReacquired() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let lease = try await SummaryLease.acquire(video: fixture.video, store: SummaryStore())
        let contender = AppleSummaryService(store: SummaryStore()) { _ in
            Issue.record("A busy transcript must not start inference")
            throw SummaryGenerationError.emptyResponse
        }
        await #expect(throws: SummaryStoreError.busy) {
            try await contender.generate(video: fixture.video, force: false) { _, _ in }
        }
        try await lease.release()
        let next = try await SummaryLease.acquire(video: fixture.video, store: SummaryStore())
        try await next.release()
        #expect(SummaryDiagnostics.errorCode(SummaryStoreError.busy) == "summary_busy")
        #expect(SummaryStoreError.busy.errorDescription != nil)
    }

    @Test(arguments: ["symlink", "readonly"])
    func refusesUnsafeLeaseFiles(_ kind: String) async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let directory = SummaryStore().directory(for: fixture.video)
        try SummaryStore.prepareDirectory(directory)
        let lock = directory.appendingPathComponent(".generation.lock")
        if kind == "symlink" {
            try FileManager.default.createSymbolicLink(
                at: lock, withDestinationURL: fixture.video.transcriptURL)
        } else {
            try Data().write(to: lock)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o400], ofItemAtPath: lock.path)
        }
        await #expect(throws: SummaryStoreError.unsafePath) {
            try await SummaryLease.acquire(video: fixture.video, store: SummaryStore())
        }
    }

    @Test func rejectsWrongSectionsAndNeverFollowsCurrentPointerSymlink() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let store = SummaryStore()
        await #expect(throws: SummaryStoreError.self) {
            try await store.save(
                NarrativeSummary(overview: "説明している。", sections: []), video: fixture.video)
        }
        try SummaryStore.prepareDirectory(store.directory(for: fixture.video))
        try FileManager.default.createSymbolicLink(
            at: fixture.current, withDestinationURL: fixture.video.transcriptURL)
        let before = try Data(contentsOf: fixture.video.transcriptURL)
        await #expect(throws: SummaryStoreError.self) { try await store.load(video: fixture.video) }
        await #expect(throws: SummaryStoreError.self) { try await fixture.generate() }
        #expect(try Data(contentsOf: fixture.video.transcriptURL) == before)
    }
}
