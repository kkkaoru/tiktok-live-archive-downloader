import Foundation
import Testing

@testable import ProAppsCore

@MainActor
struct ApplicationTests {
  @Test(arguments: [
    (ProApp.motion, "motn"), (.compressor, "compressor"), (.finalCutPro, "fcpxml"),
    (.logicPro, "mid"), (.mainStage, "concert"),
  ])
  func supportedDocumentTypes(_ app: ProApp, _ expected: String) {
    #expect(app.documentExtensions.contains(expected))
  }

  @Test func logicAudioExtensionsStayScopedToLogic() {
    #expect(
      ProApp.logicPro.documentExtensions
        == ["logicx", "logic", "mid", "midi", "wav", "aif", "aiff", "caf", "m4a", "mp3"])
    #expect(ProApp.motion.documentExtensions == ["motn", "moti", "motr", "moef", "mogen"])
    #expect(ProApp.compressor.documentExtensions == ["compressor"])
    #expect(ProApp.finalCutPro.documentExtensions == ["fcpxml", "fcpxmld", "fcpbundle"])
    #expect(ProApp.mainStage.documentExtensions == ["concert", "patch"])
  }

  @Test(arguments: ["wav", "WAV", "aif", "aiff", "caf", "m4a", "mp3"])
  func logicAudioUsesExactOpenDocumentRecipient(_ extensionName: String) async throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "logic-audio-\(UUID().uuidString)")
    let app = directory.appendingPathComponent("Logic.app")
    let contents = app.appendingPathComponent("Contents")
    try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let info = ["CFBundleIdentifier": "com.apple.mobilelogic", "CFBundlePackageType": "APPL"]
    try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0).write(
      to: contents.appendingPathComponent("Info.plist"))
    let source = directory.appendingPathComponent("audio").appendingPathExtension(extensionName)
    // This tests delivery, not audio decoding; no real application is launched.
    try Data([1, 2, 3]).write(to: source)
    var access = ApplicationAccess()
    access.locate = { $0 == "com.apple.mobilelogic" ? app : nil }
    access.running = { _ in false }
    var calls = 0
    access.open = { document, recipient in
      calls += 1
      #expect(document == source)
      #expect(recipient.path == app.path)
      return "com.apple.mobilelogic"
    }
    let result = try await Applications.openDocument(
      app: .logicPro, bundleID: "com.apple.mobilelogic", path: source.path, access: access)
    #expect(calls == 1)
    #expect(result.contains("Import completion is not verified"))
    #expect(try Data(contentsOf: source) == Data([1, 2, 3]))
    await #expect(throws: (any Error).self) {
      try await Applications.openDocument(
        app: .logicPro, bundleID: nil,
        path: directory.appendingPathComponent("missing.wav").path, access: access)
    }
    let unsupported = directory.appendingPathComponent("movie.mp4")
    try Data([1]).write(to: unsupported)
    await #expect(throws: (any Error).self) {
      try await Applications.openDocument(
        app: .logicPro, bundleID: nil, path: unsupported.path, access: access)
    }
    #expect(calls == 1)
  }

  @Test(arguments: [true, false])
  func resolvesOnlyValidatedEditionsAndNativeRecipients(_ includeVersion: Bool) async throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "pro-apps-bundle-\(UUID().uuidString)")
    let app = directory.appendingPathComponent("Compressor.app")
    let contents = app.appendingPathComponent("Contents")
    try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    var info = [
      "CFBundleIdentifier": "com.apple.CompressorApp", "CFBundleName": "Compressor",
      "CFBundlePackageType": "APPL", "CFBundleShortVersionString": "5.3",
      "CFBundleExecutable": "Compressor",
    ]
    if !includeVersion { info.removeValue(forKey: "CFBundleShortVersionString") }
    try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0).write(
      to: contents.appendingPathComponent("Info.plist"))
    var access = ApplicationAccess()
    access.locate = { $0 == "com.apple.CompressorApp" ? app : nil }
    access.running = { _ in false }
    access.open = { document, recipient in
      #expect(document.pathExtension == "compressor")
      #expect(recipient.path == app.path)
      return "com.apple.CompressorApp"
    }
    let inventory = Applications.inventory(access: access)
    #expect(inventory.count == 1)
    if includeVersion {
      #expect(inventory.first?.version == "5.3")
    } else {
      #expect(inventory.first?.version == "unknown")
    }
    #expect(try Applications.resolve(.compressor, bundleID: nil, access: access).path == app.path)
    #expect(throws: (any Error).self) {
      try Applications.resolve(.motion, bundleID: "com.apple.CompressorApp", access: access)
    }
    #expect(throws: (any Error).self) {
      try Applications.resolve(.motion, bundleID: nil, access: access)
    }
    let document = directory.appendingPathComponent("example.compressor")
    try Data().write(to: document)
    let result = try await Applications.openDocument(
      app: .compressor, bundleID: nil, path: document.path, access: access)
    #expect(result.contains("completion is not verified"))
    await #expect(throws: (any Error).self) {
      try await Applications.openDocument(
        app: .compressor, bundleID: nil,
        path: directory.appendingPathComponent("missing.compressor").path, access: access)
    }
    access.open = { _, _ in "wrong.recipient" }
    await #expect(throws: (any Error).self) {
      try await Applications.openDocument(
        app: .compressor, bundleID: nil, path: document.path, access: access)
    }
    #expect(throws: (any Error).self) { try Compressor.executable(bundleID: nil, access: access) }
    let executable = contents.appendingPathComponent("MacOS/Compressor")
    try FileManager.default.createDirectory(
      at: executable.deletingLastPathComponent(), withIntermediateDirectories: false)
    _ = try Files.writeNew(Data(), to: executable.path, extensions: [""])
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
    #expect(try Compressor.executable(bundleID: nil, access: access) == executable)
    access.locate = { _ in app }
    #expect(Applications.inventory(access: access).count == 1)
  }

  @Test(.timeLimit(.minutes(1)))
  func nativeOpenRejectsANonexistentRecipientWithoutLaunchingAnApp() async throws {
    let missing = try Files.temporaryDirectory().appendingPathComponent(
      "missing-\(UUID().uuidString)")
    await #expect(throws: (any Error).self) {
      try await ApplicationAccess().open(
        missing.appendingPathExtension("fcpxml"), missing.appendingPathExtension("app"))
    }
    await #expect(throws: (any Error).self) {
      try await Applications.openDocument(
        app: .motion, bundleID: "not-an-approved-bundle", path: missing.path)
    }
  }

  @Test func nativeReadOnlyInventoryDoesNotLaunchApps() {
    // Read-only OS integration, no document delivery or launch. Missing apps
    // are an expected result on CI; no assumptions about installed software.
    let inventory = Applications.inventory()
    #expect(inventory.count <= 10)
    #expect(inventory.allSatisfy { $0.app.bundleIDs.contains($0.bundleID) })
  }
}
