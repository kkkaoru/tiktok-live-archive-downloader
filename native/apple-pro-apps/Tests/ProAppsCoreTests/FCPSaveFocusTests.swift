import Foundation
import Testing

@testable import ProAppsCore

@MainActor
struct FCPSaveFocusTests {
  private func fixture() -> FakeUIBackend {
    let b = FakeUIBackend.standard()
    b.foregroundProcess = b.processes[FakeUIBackend.fcp]
    b.frontmost = [FakeUIBackend.fcp]
    b.attributes[1]?[.role] = .string("AXApplication")
    b.attributes[1]?[.windows] = .elements([10])
    b.attributes[10]?[.children] = .elements([11])
    b.attributes[11] = [
      .role: .string("AXSheet"), .identifier: .string("save-panel"),
      .children: .elements([70, 71]),
    ]
    b.attributes[70] = [
      .role: .string("AXPopUpButton"), .identifier: .string("where popup"),
      .children: .elements([]),
    ]
    b.attributes[71] = [
      .role: .string("AXTextField"), .identifier: .string("saveAsNameTextField"),
      .value: .string("same.mov"), .focused: .bool(false), .children: .elements([]),
    ]
    b.settable.insert("71:AXFocused")
    return b
  }

  @Test(arguments: [
    "preflight", "focus", "already", "missing", "not-settable", "focused-not-settable", "no-effect",
    "wrong-pid",
    "empty-birth", "disallowed", "other-front", "invalid-app", "missing-role", "modal",
    "nested-dialog", "stale-name", "lost-handle", "foreign-owner", "identity-change",
    "set-error", "invalid-bool", "deadline", "window-change",
  ])
  func focusIsAnExplicitSingleGuardedSetter(_ mode: String) async throws {
    let b = fixture()
    if mode == "already" || mode == "focused-not-settable" {
      b.attributes[71]?[.focused] = .bool(true)
    }
    if mode == "missing" { b.attributes[71]?[.focused] = .missing }
    if mode == "not-settable" || mode == "focused-not-settable" { b.settable.removeAll() }
    if mode == "no-effect" { b.ignoresWrites = true }
    if mode == "other-front" { b.frontmost = ["other"] }
    if mode == "missing-role" { b.attributes[71]?[.role] = .missing }
    if mode == "invalid-bool" { b.attributes[71]?[.focused] = .number(0) }
    if mode == "set-error" { b.setFailure = .accessibility(-25205) }
    if mode == "modal" {
      b.attributes[1]?[.windows] = .elements([10, 20])
      b.attributes[20]?[.modal] = .bool(true)
    }
    if mode == "nested-dialog" {
      b.attributes[11]?[.children] = .elements([70, 71, 80])
      b.attributes[80] = [.role: .string("AXSheet"), .children: .elements([])]
    }
    b.identity = { _, _ in
      if mode == "foreign-owner" || (mode == "identity-change" && !b.written.isEmpty) { return 99 }
      return 42
    }
    b.onForegroundRead = { backend in
      if mode == "deadline" { backend.clock = 11 }
      guard !backend.written.isEmpty else { return }
      switch mode {
      case "stale-name": backend.attributes[71]?[.value] = .string("different.mov")
      case "lost-handle":
        backend.attributes[72] = backend.attributes[71]
        backend.attributes[11]?[.children] = .elements([70, 72])
      case "window-change": backend.attributes[10]?[.title] = .string("Other")
      default: break
      }
    }
    let request = UIRequest.saveFocus(
      .init(app: mode == "invalid-app" ? .motion : .finalCutPro),
      .init(
        phase: mode == "preflight" ? .preflight : .focus,
        expectedPID: mode == "wrong-pid" ? 999 : 42,
        expectedBirth: mode == "empty-birth" ? "" : "externally-observed:123",
        allowForeground: mode != "disallowed", window: .init(title: "Main")))
    #expect(
      try JSONDecoder().decode(UIRequest.self, from: JSONEncoder().encode(request)) == request)
    let response = try await UIAutomation(backend: b).run(request)
    let receipt = try #require(response.saveFocus)
    #expect(receipt.effectVerified == (mode == "focus"))
    #expect(response.effectVerified == receipt.effectVerified)
    #expect(
      receipt.complete
        == [
          "preflight", "focus", "already", "missing", "not-settable", "focused-not-settable",
          "no-effect",
        ].contains(mode))
    #expect(b.written.count <= 1 && b.written.allSatisfy { $0 == "AXFocused@71" })
    #expect(b.performed.isEmpty && b.activated.isEmpty)
    #expect(!receipt.nativeBirthVerified && !receipt.activationAttempted)
    #expect(!receipt.filenameAttempted && !receipt.saveAttempted)
    if mode == "focus" {
      #expect(receipt.beforeFocused == false && receipt.afterFocused == true)
      #expect(receipt.setterReturned && receipt.beforeFilename == receipt.afterFilename)
      #expect(receipt.expectedBirth == "externally-observed:123")
    }
    if mode == "preflight" || mode == "already" || mode == "not-settable"
      || mode == "focused-not-settable" || mode == "missing"
    {
      #expect(!receipt.focusAttempted && b.written.isEmpty)
    }
    if mode == "no-effect" { #expect(receipt.status == "noObservedEffect") }
    if mode == "set-error" { #expect(receipt.focusAttempted && !receipt.setterReturned) }
  }
}
