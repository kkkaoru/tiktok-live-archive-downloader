import Foundation
import Testing

@testable import ProAppsCore

@MainActor
struct FCPSaveIdentityTests {
  private func fixture() -> FakeUIBackend {
    let b = FakeUIBackend.standard()
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
      .value: .string("same.mov"), .children: .elements([]),
    ]
    return b
  }

  @Test(arguments: [
    "same", "mixed", "focus", "changed-pid", "changed-focus", "changed-name", "changed-handle",
    "dangling", "zero", "wrong-pid", "invalid-app", "ambiguous", "array-focus", "deadline",
    "cancelled", "unknown", "missing-role", "bad-identifier", "long-identifier",
  ])
  func observesOnlyStableExactObjects(_ mode: String) async throws {
    let b = fixture()
    if mode == "focus" || mode == "changed-focus" {
      b.attributes[1]?[.focusedWindow] = .element(10)
      b.attributes[1]?[.focusedUIElement] = .element(71)
    }
    if mode == "array-focus" { b.attributes[1]?[.focusedWindow] = .elements([10, 11]) }
    if mode == "ambiguous" { b.attributes[11]?[.children] = .elements([70, 71, 71]) }
    if mode == "missing-role" { b.attributes[1]?[.role] = .missing }
    if mode == "bad-identifier" { b.attributes[1]?[.identifier] = .bool(true) }
    if mode == "long-identifier" {
      b.attributes[1]?[.identifier] = .string(String(repeating: "x", count: 513))
    }
    b.identity = { handle, count in
      if mode == "dangling" { throw UIBackendError.accessibility(-25202) }
      if mode == "cancelled" { throw CancellationError() }
      if mode == "unknown" { throw CocoaError(.fileReadUnknown) }
      if mode == "zero" { return 0 }
      if count == 5 {
        switch mode {
        case "changed-name": b.attributes[71]?[.value] = .string("different.mov")
        case "changed-focus": b.attributes[1]?[.focusedUIElement] = .missing
        case "changed-handle":
          b.attributes[72] = b.attributes[71]
          b.attributes[11]?[.children] = .elements([70, 72])
        case "deadline": b.clock += 10
        default: break
        }
      }
      if mode == "changed-pid", count > 5 { return 777 }
      return (mode == "mixed" || mode == "focus") && handle != 1 ? 777 : 42
    }
    let request = UIRequest.saveIdentity(
      .init(app: mode == "invalid-app" ? .motion : .finalCutPro),
      .init(expectedPID: mode == "wrong-pid" ? 99 : 42, window: .init(title: "Main")))
    #expect(
      try JSONDecoder().decode(UIRequest.self, from: JSONEncoder().encode(request)) == request)
    let result = try await UIAutomation(backend: b).run(request)
    let receipt = try #require(result.saveIdentity)
    #expect(receipt.complete == ["same", "mixed", "focus"].contains(mode))
    #expect(receipt.rows.count <= 7)
    #expect(!receipt.routingVerified && !receipt.nativeBirthVerified && !receipt.actionsAttempted)
    #expect(result.dispatched == false && result.effectVerified == false)
    #expect(b.performed.isEmpty && b.written.isEmpty && b.activated.isEmpty)
    if mode == "same" {
      #expect(receipt.rows.count == 5)
      #expect(receipt.focusStates.values.allSatisfy { $0 == "unavailable" })
    }
    if mode == "mixed" { #expect(receipt.rows.map(\.associatedPID) == [42, 777, 777, 777, 777]) }
    if mode == "focus" { #expect(receipt.rows.count == 7) }
  }

  @Test func nativeOwnerLookupOnlyReadsLocalAXReference() throws {
    let backend = LiveUIBackend()
    let pid = ProcessInfo.processInfo.processIdentifier
    #expect(try backend.processID(of: backend.applicationElement(pid: pid)) == pid)
    #expect(throws: UIBackendError.self) { try backend.processID(of: -1) }
  }
}
