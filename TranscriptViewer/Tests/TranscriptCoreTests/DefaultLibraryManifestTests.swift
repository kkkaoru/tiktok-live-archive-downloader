import Foundation
import Testing

@testable import TranscriptCore

struct DefaultLibraryManifestTests {
    private struct Fixture {
        let root: URL
        let manifest: URL

        init() throws {
            root = try transcriptTestWorkRoot().appending(path: "default-libraries-\(UUID())")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            manifest = root.appending(path: DefaultLibraryManifest.filename)
        }

        func write(_ paths: [String], version: Int = 1) throws {
            try JSONEncoder().encode(DefaultLibraryManifest(version: version, libraries: paths))
                .write(to: manifest)
        }

        func remove() throws { try FileManager.default.removeItem(at: root) }
    }

    @Test func absentConfigAndEmptyListDoNotAddLibraries() throws {
        let fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        #expect(try DefaultLibraryManifest.load(at: fixture.manifest).isEmpty)
        #expect(
            try DefaultLibraryManifest.load(at: fixture.root.appending(path: "absent/config.json"))
                .isEmpty)
        try fixture.write([])
        #expect(try DefaultLibraryManifest.load(at: fixture.manifest).isEmpty)
        #expect(DefaultLibraryManifest.Invalid.manifest.errorDescription == "追加ライブラリの設定が不正です。")
        #expect(
            DefaultLibraryManifest.Invalid.path.errorDescription
                == "追加ライブラリには実体のある絶対パスを指定してください。")
    }

    @Test func missingVolumeRemainsConfiguredAndChangesAreReadOnRefresh() throws {
        let fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let future = fixture.root.appending(path: "later-volume")
        try fixture.write([future.path])
        #expect(try DefaultLibraryManifest.load(at: fixture.manifest).map(\.path) == [future.path])
        let changed = fixture.root.appending(path: "new-library")
        try fixture.write([changed.path])
        #expect(try DefaultLibraryManifest.load(at: fixture.manifest).map(\.path) == [changed.path])
    }

    @Test(arguments: [0, 2]) func unknownVersionsAreRejected(_ version: Int) throws {
        let fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.write([], version: version)
        #expect(throws: DefaultLibraryManifest.Invalid.self) {
            try DefaultLibraryManifest.load(at: fixture.manifest)
        }
    }

    @Test(arguments: [0, 65_537]) func emptyAndOversizedFilesAreRejected(_ size: Int) throws {
        let fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try Data(repeating: 32, count: size).write(to: fixture.manifest)
        #expect(throws: DefaultLibraryManifest.Invalid.self) {
            try DefaultLibraryManifest.load(at: fixture.manifest)
        }
    }

    @Test func malformedJSONAndTooManyRootsAreRejected() throws {
        let fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try Data("not-json".utf8).write(to: fixture.manifest)
        #expect(throws: DecodingError.self) {
            try DefaultLibraryManifest.load(at: fixture.manifest)
        }
        try fixture.write((0..<17).map { fixture.root.appending(path: "library-\($0)").path })
        #expect(throws: DefaultLibraryManifest.Invalid.self) {
            try DefaultLibraryManifest.load(at: fixture.manifest)
        }
    }

    @Test func exactByteLimitIsAcceptedButRegularFileLibraryAndDanglingRootAreRejected() throws {
        let fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let json = Data(#"{"version":1,"libraries":[]}"#.utf8)
        try (json + Data(repeating: 32, count: 65_536 - json.count)).write(to: fixture.manifest)
        #expect(try DefaultLibraryManifest.load(at: fixture.manifest).isEmpty)
        try fixture.write([fixture.manifest.path])
        #expect(throws: DefaultLibraryManifest.Invalid.self) {
            try DefaultLibraryManifest.load(at: fixture.manifest)
        }
        let dangling = fixture.root.appending(path: "dangling-library")
        try FileManager.default.createSymbolicLink(
            at: dangling, withDestinationURL: fixture.root.appending(path: "absent-library"))
        try fixture.write([dangling.path])
        #expect(throws: DefaultLibraryManifest.Invalid.self) {
            try DefaultLibraryManifest.load(at: fixture.manifest)
        }
    }

    @Test func directoryAndLinkedManifestAreRejectedBeforeReading() throws {
        let fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try FileManager.default.createDirectory(
            at: fixture.manifest, withIntermediateDirectories: false)
        #expect(throws: DefaultLibraryManifest.Invalid.self) {
            try DefaultLibraryManifest.load(at: fixture.manifest)
        }
        try FileManager.default.removeItem(at: fixture.manifest)
        let missing = fixture.root.appending(path: "absent.json")
        try FileManager.default.createSymbolicLink(
            at: fixture.manifest, withDestinationURL: missing)
        #expect(throws: DefaultLibraryManifest.Invalid.self) {
            try DefaultLibraryManifest.load(at: fixture.manifest)
        }
    }

    @Test(arguments: [
        "relative", "/private/library", "/downloads/library", "/a/../b", "/a\\b", "/a\0b",
    ])
    func invalidRootsAreRejected(_ path: String) throws {
        let fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.write([path])
        #expect(throws: DefaultLibraryManifest.Invalid.self) {
            try DefaultLibraryManifest.load(at: fixture.manifest)
        }
    }

    @Test func linkedAndDuplicateRootsAreRejected() throws {
        let fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let link = fixture.root.appending(path: "linked")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: fixture.root)
        try fixture.write([link.path])
        #expect(throws: DefaultLibraryManifest.Invalid.self) {
            try DefaultLibraryManifest.load(at: fixture.manifest)
        }
        try fixture.write([fixture.root.path, fixture.root.path])
        #expect(throws: DefaultLibraryManifest.Invalid.self) {
            try DefaultLibraryManifest.load(at: fixture.manifest)
        }
        let remote = try #require(URL(string: "https://example.invalid/config.json"))
        #expect(throws: DefaultLibraryManifest.Invalid.self) {
            try DefaultLibraryManifest.load(at: remote)
        }
        let fileAuthority = try #require(URL(string: "file://remote.invalid/config.json"))
        #expect(throws: DefaultLibraryManifest.Invalid.self) {
            try DefaultLibraryManifest.load(at: fileAuthority)
        }
    }

    @Test @MainActor func asynchronousReaderReturnsValidatedRootsOrError() async throws {
        let fixture = try Fixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try fixture.write([fixture.root.path])
        let loaded: Result<[URL], Error> = await withCheckedContinuation { continuation in
            DefaultLibraryManifest.load(at: fixture.manifest) { continuation.resume(returning: $0) }
        }
        #expect(try loaded.get().map(\.path) == [fixture.root.path])
        try fixture.write([], version: 9)
        let invalid: Result<[URL], Error> = await withCheckedContinuation { continuation in
            DefaultLibraryManifest.load(at: fixture.manifest) { continuation.resume(returning: $0) }
        }
        #expect(throws: DefaultLibraryManifest.Invalid.self) { try invalid.get() }
    }
}
