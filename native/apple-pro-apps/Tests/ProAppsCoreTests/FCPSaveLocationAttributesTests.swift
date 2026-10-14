import Foundation
import Testing

@testable import ProAppsCore

@MainActor
struct FCPSaveLocationAttributesTests {
  private func fixture() -> FakeUIBackend {
    let b = FakeUIBackend.standard()
    b.attributes[1]?[.windows] = .elements([10])
    b.attributes[10]?[.children] = .elements([11])
    b.attributes[11] = [
      .role: .string("AXSheet"), .identifier: .string("save-panel"),
      .children: .elements([70, 71, 72]), .document: .string("file:///Volumes/example"),
    ]
    b.attributes[70] = [
      .role: .string("AXPopUpButton"), .identifier: .string("where popup"),
      .children: .elements([]), .value: .string("example"),
    ]
    b.attributes[71] = [
      .role: .string("AXTextField"), .identifier: .string("saveAsNameTextField"),
      .value: .string("same.mov"), .children: .elements([]),
    ]
    b.attributes[72] = [
      .role: .string("AXBrowser"), .identifier: .string("ColumnView"),
      .url: .string("file:///Volumes/example/selected-item"),
    ]
    // Resolving the exact browser never reads its columns, even when inaccessible.
    b.namedFailures["72:AXChildren"] = .accessibility(-25200)
    return b
  }

  @Test(arguments: [
    "stable", "no-urls", "wrong-browser", "ambiguous", "changed-url", "changed-name",
    "changed-handle", "owner-changed", "deadline", "cancelled", "ax-error", "unknown-error",
    "too-many-names", "duplicate-names", "empty-name", "long-name", "invalid-app", "no-title",
    "bad-identifier",
  ])
  func boundedReadHasNoEffectsAndRejectsUnstableEvidence(_ mode: String) async throws {
    let b = fixture()
    if mode == "wrong-browser" { b.attributes[72]?[.role] = .string("AXTable") }
    if mode == "ambiguous" { b.attributes[11]?[.children] = .elements([70, 71, 72, 72]) }
    if mode == "no-urls" {
      b.attributes[11]?[.document] = nil
      b.attributes[72]?[.url] = nil
    }
    var reads = 0
    b.nameObservation = { handle in
      reads += 1
      if mode == "ax-error" { throw UIBackendError.accessibility(-25202) }
      if mode == "unknown-error" { throw CocoaError(.fileReadUnknown) }
      if mode == "cancelled" { throw CancellationError() }
      if mode == "too-many-names" { return (0..<129).map { "AX\($0)" } }
      if mode == "duplicate-names" { return ["AXURL", "AXURL"] }
      if mode == "empty-name" { return [""] }
      if mode == "long-name" { return [String(repeating: "x", count: 129)] }
      if reads == 3 {
        switch mode {
        case "changed-url": b.attributes[11]?[.document] = .string("file:///different")
        case "changed-name": b.attributes[71]?[.value] = .string("different.mov")
        case "changed-handle":
          b.attributes[73] = b.attributes[72]
          b.attributes[11]?[.children] = .elements([70, 71, 73])
        case "bad-identifier": b.attributes[11]?[.identifier] = .bool(true)
        case "owner-changed": b.processes = [:]
        case "deadline": b.clock += 10
        default: break
        }
      }
      return (b.attributes[handle] ?? [:]).keys.map(\.rawValue)
    }
    let request = UIRequest.saveLocationAttributes(
      .init(app: mode == "invalid-app" ? .motion : .finalCutPro),
      .init(window: .init(title: mode == "no-title" ? nil : "Main")))
    #expect(
      try JSONDecoder().decode(UIRequest.self, from: JSONEncoder().encode(request)) == request)
    let response = try await UIAutomation(backend: b).run(request)
    let receipt = try #require(response.saveLocationAttributes)
    #expect(receipt.complete == ["stable", "no-urls"].contains(mode))
    #expect(!receipt.currentDirectoryVerified && !receipt.filesystemIdentityVerified)
    #expect(!receipt.actionsAttempted && response.dispatched == false)
    #expect(b.performed.isEmpty && b.written.isEmpty && b.activated.isEmpty)
    if mode == "stable" {
      #expect(receipt.rows.map(\.scope) == ["savePanel", "wherePopup", "browser"])
      #expect(receipt.rows[0].attributes[1].lexicalFilePath == "/Volumes/example")
      #expect(receipt.rows[1].attributes[2].state == "textOnly")
      #expect(receipt.rows[2].attributes[0].lexicalFilePath == "/Volumes/example/selected-item")
    }
  }

  @Test(arguments: [
    "file:///Volumes/example", "file://localhost/Volumes/example", "https://example.org/x",
    "file://foreign/Volumes/example", "file:///a/../b", "file:///a/%2e%2e/b",
    "file:///a%2fb", "file:///a%5cb", "file:///a?x", "file:///a#x", "file:relative",
    "file:///a//b", "file://user@localhost/a", "file://localhost:12/a", "plain-folder",
  ])
  func fileURLIsOnlyLexicalEvidence(_ text: String) throws {
    let result = try UIAutomation<FakeUIBackend>.locationAttribute(
      .url, advertised: true, value: .string(text))
    let accepted = ["file:///Volumes/example", "file://localhost/Volumes/example"].contains(text)
    #expect((result.lexicalFilePath != nil) == accepted)
    #expect(result.value == text)
  }

  @Test func unsupportedMissingAndOversizedRemainExplicit() throws {
    let missing = try UIAutomation<FakeUIBackend>.locationAttribute(
      .document, advertised: true, value: .missing)
    #expect(missing.state == "noValue" && missing.lexicalFilePath == nil)
    let unsupported = try UIAutomation<FakeUIBackend>.locationAttribute(
      .url, advertised: true, value: .number(5))
    #expect(unsupported.state == "unsupportedType" && unsupported.value == nil)
    #expect(throws: ProAppsError.self) {
      try UIAutomation<FakeUIBackend>.locationAttribute(
        .url, advertised: true, value: .string(String(repeating: "x", count: 4097)))
    }
    #expect(throws: ProAppsError.self) {
      try UIAutomation<FakeUIBackend>.locationAttribute(
        .url, advertised: true, value: .string("file:///a\0b"))
    }
    #expect(throws: UIBackendError.self) {
      try LiveUIBackend().attributeNames(of: -1)
    }
  }
}
