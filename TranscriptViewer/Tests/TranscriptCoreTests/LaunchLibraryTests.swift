import Foundation
import Testing

@testable import TranscriptCore
@testable import TranscriptViewer

private struct ViewerDocumentRegistration: Decodable {
    struct DocumentType: Decodable, Equatable {
        let role: String
        let rank: String
        let contentTypes: [String]

        private enum CodingKeys: String, CodingKey {
            case role = "CFBundleTypeRole"
            case rank = "LSHandlerRank"
            case contentTypes = "LSItemContentTypes"
        }
    }
    let documentTypes: [DocumentType]

    private enum CodingKeys: String, CodingKey {
        case documentTypes = "CFBundleDocumentTypes"
    }
}

struct LaunchLibraryTests {
    @Test func ordinaryLaunchHasNoOverride() throws {
        #expect(try LaunchLibrary.resolve(arguments: []) == nil)
        #expect(try LaunchLibrary.resolve(arguments: ["-psn_0_12"]) == nil)
    }

    @Test(arguments: [
        ["--library"], ["--library", "relative"], ["--unknown", "/"],
        ["--library", "/", "--library", "/"], ["--library", ""],
    ])
    func rejectsInvalidArguments(_ arguments: [String]) {
        #expect(throws: LaunchLibraryError.self) {
            try LaunchLibrary.resolve(arguments: arguments)
        }
    }

    @Test func acceptsExistingDirectoryAndRejectsMissingOrRegularFiles() throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        #expect(
            try LaunchLibrary.resolve(arguments: ["--library", fixture.root.path])?.path
                == fixture.root.path)
        #expect(throws: LaunchLibraryError.self) {
            try LaunchLibrary.resolve(arguments: [
                "--library", fixture.root.appending(path: "missing").path,
            ])
        }
        #expect(throws: LaunchLibraryError.self) {
            try LaunchLibrary.resolve(arguments: ["--library", fixture.video.transcriptURL.path])
        }
        #expect(LaunchLibraryError.arguments.errorDescription?.contains("--library") == true)
        #expect(
            LaunchLibraryError.unreadableDirectory("/missing").errorDescription
                == "ライブラリを読み込めません: /missing")
    }

    @Test @MainActor func launchOverrideWinsWithoutPersistingAndInvalidInputKeepsCurrentLibrary()
        throws
    {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var requestedRoot: URL?
        let session = PlaybackSession(
            readLibraryRoot: { URL(fileURLWithPath: "/saved") },
            saveLibraryRoot: { _ in Issue.record("Launch override must not save preferences") },
            scanCatalog: { root, completion in
                requestedRoot = root
                completion(.success(CatalogScan()))
            }, launchArguments: ["--library", fixture.root.path])
        session.loadDefaultLibrary()
        #expect(requestedRoot?.path == fixture.root.path)
        #expect(!session.showsError)

        let invalid = PlaybackSession(
            scanCatalog: { _, completion in completion(.success(CatalogScan())) },
            launchArguments: ["--library", "relative"])
        invalid.load(root: fixture.root)
        let revision = invalid.catalogRevision
        invalid.loadDefaultLibrary()
        #expect(invalid.showsError)
        #expect(invalid.errorText.contains("--library"))
        #expect(invalid.libraryName == fixture.root.lastPathComponent)
        #expect(invalid.catalogRevision == revision)
    }

    @Test @MainActor func appPassesItsExplicitLaunchArguments() {
        let app = TranscriptViewerApp(arguments: ["--library", "relative"])
        let session = app.makeRootView().session
        session.loadDefaultLibrary()
        #expect(session.showsError)
        #expect(session.libraryName == "未選択")
    }

    @Test func unreadableDirectoryIsRejected() throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let directory = fixture.root.appending(path: "unreadable", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: directory.path)
        defer {
            #expect(throws: Never.self) {
                try FileManager.default.setAttributes(
                    [.posixPermissions: 0o700], ofItemAtPath: directory.path)
            }
        }
        #expect(throws: LaunchLibraryError.self) {
            try LaunchLibrary.resolve(arguments: ["--library", directory.path])
        }
    }

    @Test(arguments: [
        "https://example.invalid/library", "file://remote.invalid/library",
        "file:///library?mode=all", "file:///library#selection",
    ])
    func openDocumentRejectsUnsupportedURLs(_ text: String) throws {
        let url = try #require(URL(string: text))
        #expect(throws: LaunchLibraryError.self) { try LaunchLibrary.resolve(url: url) }
        #expect(
            LaunchLibraryError.unsupportedURL.errorDescription
                == "ライブラリにはこのMacのフォルダURLを指定してください。")
    }

    @Test @MainActor func explicitOpenDocumentWinsOverLaterStartupWithoutSavingPreferences()
        throws
    {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        var scanned: [URL] = []
        let session = PlaybackSession(
            readLibraryRoot: {
                Issue.record("Explicit open must not read saved selection")
                return nil
            },
            saveLibraryRoot: { _ in Issue.record("Explicit open must not save preferences") },
            scanCatalog: { root, completion in
                scanned.append(root)
                completion(.success(CatalogScan()))
            }, launchArguments: ["--library", fixture.root.path])
        let selected = fixture.root.appending(path: "opened-library", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: selected, withIntermediateDirectories: false)
        session.openLibrary(url: selected)
        session.loadDefaultLibrary()
        #expect(scanned == [selected])
        #expect(session.libraryName == "opened-library")
        #expect(!session.showsError)
    }

    @Test @MainActor func invalidOpenDocumentKeepsLibraryAndSelection() throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let session = PlaybackSession(scanCatalog: { _, completion in
            completion(.success(CatalogScan()))
        })
        session.load(root: fixture.root)
        session.videos = [fixture.video]
        session.selectedVideoID = fixture.video.id
        let revision = session.catalogRevision
        session.openLibrary(url: fixture.video.transcriptURL)
        #expect(session.showsError)
        #expect(session.errorText.contains("ライブラリを読み込めません"))
        #expect(session.libraryName == fixture.root.lastPathComponent)
        #expect(session.catalogRevision == revision)
        #expect(session.selectedVideoID == fixture.video.id)
    }

    @Test func appRegistersOnlyAnAlternateViewerForFolderDocuments() throws {
        let checkout = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: checkout.appending(path: "TranscriptViewer/App/Info.plist"))
        let registration = try PropertyListDecoder().decode(
            ViewerDocumentRegistration.self, from: data)
        #expect(
            registration.documentTypes == [
                .init(role: "Viewer", rank: "Alternate", contentTypes: ["public.folder"])
            ])
    }
}
