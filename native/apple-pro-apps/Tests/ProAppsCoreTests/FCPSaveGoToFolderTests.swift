import CoreGraphics
import Foundation
import Testing

@testable import ProAppsCore

@MainActor
struct FCPSaveGoToFolderTests {
  @Test(arguments: [
    "success", "escape", "missing-focus", "foreign-owner", "no-effect", "name", "handle", "focus",
    "post-error", "early-name", "late-name", "membership-count",
  ])
  func explicitHIDNeverFallsBackAndBoundsChangedFocus(_ mode: String) async throws {
    let b = backend()
    b.attributes[1]?[.role] = .string("AXApplication")
    b.attributes[1]?[.focusedWindow] = .element(11)
    b.attributes[1]?[.focusedUIElement] = .element(71)
    if mode == "missing-focus" { b.attributes[1]?[.focusedUIElement] = .missing }
    if mode == "foreign-owner" { b.identity = { _, _ in 99 } }
    if ["early-name", "late-name", "membership-count"].contains(mode) {
      b.identity = { _, count in
        if (mode == "early-name" && count == 14) || (mode == "late-name" && count == 30) {
          b.attributes[71]?[.value] = .string("changed-at-boundary.mov")
        }
        if mode == "membership-count" && count == 30 {
          b.attributes[90]?[.children] = .elements(Array(repeating: 91, count: 65))
        }
        return 42
      }
    }
    var targetPosts = 0
    var hidPosts = 0
    let events = GoToFolderEventBackend(
      permission: { true }, post: { _ in targetPosts += 1 },
      postForeground: {
        hidPosts += 1
        if mode == "post-error" { throw ProAppsError.unavailable("HID uncertain") }
        if mode == "no-effect" { return }
        if mode == "name" { b.attributes[71]?[.value] = .string("changed.mov") }
        if mode == "focus" { b.frontmost = ["other"] }
        b.attributes[90] = [.role: .string("AXSheet"), .children: .elements([91])]
        b.attributes[91] = [.role: .string("AXTextField"), .children: .elements([])]
        b.attributes[11]?[.children] = .elements([70, 71, 72, 90])
        b.attributes[1]?[.focusedWindow] = .element(90)
        b.attributes[1]?[.focusedUIElement] = .element(91)
        if mode == "escape" {
          b.attributes[92] = [.role: .string("AXTextField"), .children: .elements([])]
          b.attributes[1]?[.focusedUIElement] = .element(92)
        }
        if mode == "handle" {
          b.attributes[73] = b.attributes[71]
          b.attributes[11]?[.children] = .elements([70, 73, 72, 90])
        }
      })
    let result = try await UIAutomation(backend: b, goToFolderEvents: events).run(
      .goToFolder(
        .init(app: .finalCutPro),
        .init(
          phase: .open, expectedPID: 42, expectedBirth: "external:1",
          allowForeground: true, window: .init(title: "Main"), delivery: .foregroundHID)))
    let receipt = try #require(result.goToFolder)
    #expect(targetPosts == 0 && hidPosts <= 1)
    #expect(receipt.delivery == .foregroundHID && !receipt.globalRoutingAtomic)
    #expect(receipt.newDialogObserved == (mode == "success"))
    #expect(!receipt.nameAttempted && !receipt.saveAttempted && !receipt.directoryVerified)
    #expect(b.performed.isEmpty && b.written.isEmpty && b.activated.isEmpty)
    if mode == "success" {
      #expect(receipt.changedFocusPaths["AXFocusedUIElement"] == [0, 3, 0])
      #expect(receipt.beforeIdentity?.complete == true && receipt.afterIdentity?.complete == true)
    }
    if mode == "no-effect" { #expect(receipt.status == "noObservedDialog") }
    if ["missing-focus", "foreign-owner", "early-name"].contains(mode) { #expect(hidPosts == 0) }
  }

  @Test func unreadableWindowFactsRemainUnknown() async throws {
    let backend = backend()
    for attribute in [UIAttributeName.main, .position, .size] {
      backend.namedFailures["10:\(attribute.rawValue)"] = .accessibility(-25200)
    }
    let response = try await UIAutomation(backend: backend).run(.windows(.init(app: .finalCutPro)))
    let window = try #require(response.windows?.first)
    #expect(window.main == nil && window.frame == nil)
    #expect(backend.performed.isEmpty && backend.written.isEmpty)
  }

  private func backend() -> FakeUIBackend {
    let backend = FakeUIBackend.standard()
    backend.frontmost = [FakeUIBackend.fcp]
    backend.foregroundProcess = backend.processes[FakeUIBackend.fcp]
    backend.attributes[1]?[.windows] = .elements([10])
    backend.attributes[10]?[.children] = .elements([11])
    backend.attributes[11] = [
      .role: .string("AXSheet"), .identifier: .string("save-panel"),
      .children: .elements([70, 71, 72]),
    ]
    backend.attributes[70] = [
      .role: .string("AXPopUpButton"), .identifier: .string("where popup"),
      .children: .elements([]),
    ]
    backend.attributes[71] = [
      .role: .string("AXTextField"), .identifier: .string("saveAsNameTextField"),
      .value: .string("unchanged.mov"), .children: .elements([]),
    ]
    backend.attributes[72] = [.role: .string("AXButton"), .children: .elements([])]
    return backend
  }

  @Test(arguments: [
    "observed", "preflight", "denied", "no-effect", "filename", "focus", "wrong-pid",
    "no-consent", "missing-birth", "post-error", "backend-error", "unknown-error",
    "other-modal", "unknown-modal", "clear-other", "deadline", "ambiguous", "too-many",
    "invalid-app", "incomplete", "cancelled",
  ])
  func exactBoundaryNeverTypesOrSaves(_ mode: String) async throws {
    let backend = backend()
    var posts = 0
    let events = GoToFolderEventBackend(
      permission: { mode != "denied" },
      post: { pid in
        #expect(pid == 42)
        posts += 1
        switch mode {
        case "post-error": throw ProAppsError.unavailable("Synthetic post uncertainty")
        case "backend-error": throw UIBackendError.accessibility(-25205)
        case "unknown-error": throw CocoaError(.fileReadUnknown)
        case "cancelled": throw CancellationError()
        case "filename": backend.attributes[71]?[.value] = .string("changed.mov")
        case "focus": backend.frontmost = ["other"]
        case "deadline": backend.clock += 10
        case "no-effect": break
        default:
          backend.attributes[90] = [
            .role: .string("AXSheet"), .identifier: .string("observed-dialog"),
            .children: .elements([91]),
          ]
          backend.attributes[91] = [.role: .string("AXTextField"), .children: .elements([])]
          backend.attributes[11]?[.children] = .elements([70, 71, 72, 90])
          if mode == "ambiguous" {
            backend.attributes[92] = backend.attributes[90]
            backend.attributes[11]?[.children] = .elements([70, 71, 72, 90, 92])
          }
          if mode == "incomplete" {
            backend.attributes[91]?[.children] = .elements(Array(repeating: 72, count: 65))
          }
        }
      })
    if ["other-modal", "unknown-modal", "clear-other"].contains(mode) {
      backend.attributes[1]?[.windows] = .elements([10, 20])
      backend.attributes[20] = [.children: .elements([])]
      if mode != "unknown-modal" {
        backend.attributes[20]?[.modal] = .bool(mode == "other-modal")
      }
    }
    if mode == "too-many" {
      backend.attributes[10]?[.children] = .elements(Array(repeating: 11, count: 65))
    }
    let request = UIRequest.goToFolder(
      .init(app: mode == "invalid-app" ? .motion : .finalCutPro),
      .init(
        phase: mode == "preflight" ? .preflight : .open,
        expectedPID: mode == "wrong-pid" ? 99 : 42,
        expectedBirth: mode == "missing-birth" ? "" : "external-only:1",
        allowForeground: mode != "no-consent", window: .init(title: "Main")))
    #expect(
      try JSONDecoder().decode(UIRequest.self, from: JSONEncoder().encode(request)) == request)
    let result = try await UIAutomation(backend: backend, goToFolderEvents: events).run(request)
    let receipt = try #require(result.goToFolder)
    #expect(receipt.newDialogObserved == (mode == "observed" || mode == "clear-other"))
    #expect(!receipt.directoryVerified && !receipt.nameAttempted && !receipt.saveAttempted)
    #expect(!receipt.nativeBirthVerified && !receipt.goToFolderControlsVerified)
    #expect(posts <= 1 && backend.written.isEmpty && backend.performed.isEmpty)
    if mode == "observed" {
      #expect(receipt.beforeFilename == "unchanged.mov" && receipt.afterFilename == "unchanged.mov")
      #expect(receipt.observations.count == 2 && receipt.shortcutPosted)
    }
    if mode == "preflight" || mode == "denied" { #expect(posts == 0) }
  }
}
