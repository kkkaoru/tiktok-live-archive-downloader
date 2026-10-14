import Foundation
import Testing

@testable import ProAppsCore

struct TemporaryDirectoryTests {
  @Test func usesExplicitProcessDirectoryWithoutChangingEnvironment() throws {
    let configured = try #require(ProcessInfo.processInfo.environment["TMPDIR"])
    let expected = URL(fileURLWithPath: configured).resolvingSymlinksInPath().path
    #expect(try Files.temporaryDirectory().path == expected)
    #expect(
      try Files.temporaryDirectory(path: nil).path
        == FileManager.default.temporaryDirectory.resolvingSymlinksInPath().path)
  }

  @Test func rejectsInvalidMissingFileAndLinkedDirectories() throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(
      at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    #expect(try Files.temporaryDirectory(path: root.path).path == root.path)
    #expect(throws: ProAppsError.self) { try Files.temporaryDirectory(path: "") }
    #expect(throws: ProAppsError.self) { try Files.temporaryDirectory(path: "relative") }
    #expect(throws: (any Error).self) {
      try Files.temporaryDirectory(path: root.appendingPathComponent("missing").path)
    }
    let file = root.appendingPathComponent("regular")
    try Data().write(to: file, options: .withoutOverwriting)
    #expect(throws: ProAppsError.self) { try Files.temporaryDirectory(path: file.path) }
    let link = root.appendingPathComponent("link")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: root)
    #expect(throws: ProAppsError.self) { try Files.temporaryDirectory(path: link.path) }
  }
}
