import Foundation
import Testing

@testable import ProAppsCore

@MainActor
struct UIPopupSelectionTests {
  private func fixture() -> FakeUIBackend {
    let backend = FakeUIBackend.standard()
    backend.attributes[13] = [
      .role: .string("AXPopUpButton"), .enabled: .bool(true), .value: .string("old"),
      .children: .elements([]),
    ]
    backend.attributes[50] = [.role: .string("AXMenu"), .children: .elements([51])]
    backend.attributes[51] = [
      .role: .string("AXMenuItem"), .title: .string("Destination"), .enabled: .bool(true),
    ]
    backend.actionNames[13] = ["AXShowMenu"]
    backend.actionNames[51] = ["AXPress"]
    backend.onPerform = { backend, action, _ in
      if action == "AXShowMenu" { backend.attributes[13]?[.children] = .elements([50]) }
      if action == "AXPress" {
        backend.attributes[13]?[.value] = .string("Destination")
        backend.attributes[13]?[.children] = .elements([])
      }
    }
    return backend
  }

  @Test func integratedSuccessAndReceiptRoundTrip() async throws {
    let backend = fixture()
    let result = try await UIAutomation(backend: backend).run(
      .popupSelect(
        UITarget(app: .finalCutPro), window: UIWindowLocator(title: "Main"),
        target: UIElementLocator(path: [2]), title: "Destination"))
    #expect(result.effectVerified == true)
    #expect(result.dispatched == true)
    #expect(result.popupSelection?.beforeValue == "old")
    #expect(result.popupSelection?.afterValue == "Destination")
    #expect(result.popupSelection?.childrenAfter == 0)
    #expect(result.focus.frontmostChanged == false)
    #expect(backend.performed == ["AXShowMenu@13", "AXPress@51"])
    #expect(backend.written.isEmpty)
    #expect(try JSONDecoder().decode(UIResponse.self, from: JSONEncoder().encode(result)) == result)
  }

  @Test(arguments: ["", "\u{0}", String(repeating: "a", count: 2049)])
  func invalidTitle(_ title: String) async throws {
    let backend = fixture()
    await #expect(throws: ProAppsError.self) {
      try await UIPopupSelection(backend: backend).select(13, title: title)
    }
    #expect(backend.performed.isEmpty)
  }

  @Test(arguments: ["role", "enabled", "actions"])
  func invalidPopup(_ field: String) async throws {
    let backend = fixture()
    switch field {
    case "role": backend.attributes[13]?[.role] = .string("AXButton")
    case "enabled": backend.attributes[13]?[.enabled] = .bool(false)
    default: backend.actionNames[13] = []
    }
    await #expect(throws: ProAppsError.self) {
      try await UIPopupSelection(backend: backend).select(13, title: "Destination")
    }
    #expect(backend.performed.isEmpty)
  }

  @Test(arguments: [
    ("missing", UIPopupSelectionStatus.itemMissing), ("duplicate", .itemAmbiguous),
    ("disabled", .itemDisabled), ("unsupported", .itemUnsupported),
  ])
  func rejectedItem(_ mode: String, _ expected: UIPopupSelectionStatus) async throws {
    let backend = fixture()
    switch mode {
    case "missing": backend.attributes[51]?[.title] = .string("Other")
    case "duplicate": backend.attributes[50]?[.children] = .elements([51, 51])
    case "disabled": backend.attributes[51]?[.enabled] = .missing
    default: backend.actionNames[51] = []
    }
    let result = try await UIPopupSelection(backend: backend).select(13, title: "Destination")
    #expect(result.status == expected)
    #expect(!result.pressAttempted)
    #expect(backend.performed == ["AXShowMenu@13"])
  }

  @Test(arguments: [
    (UIAttributeValue.missing, UIPopupSelectionStatus.menuMissing),
    (.elements([50, 50]), .menuAmbiguous), (.unsupported, .invalidChildren),
    (.elements(Array(repeating: 50, count: 257)), .childBudgetExceeded),
  ])
  func rejectedChildren(_ children: UIAttributeValue, _ expected: UIPopupSelectionStatus)
    async throws
  {
    let backend = fixture()
    backend.onPerform = { backend, _, _ in backend.attributes[13]?[.children] = children }
    let result = try await UIPopupSelection(backend: backend).select(13, title: "Destination")
    #expect(result.status == expected)
    #expect(!result.pressAttempted)
    #expect(backend.performed == ["AXShowMenu@13"])
  }

  @Test(arguments: [
    (Int32(-25200), UIPopupSelectionStatus.unreadableChildren), (Int32(-25204), .backendFailure),
  ])
  func unreadableChildren(_ code: Int32, _ expected: UIPopupSelectionStatus) async throws {
    let backend = fixture()
    backend.onPerform = { backend, _, _ in
      backend.namedFailures["13:AXChildren"] = .accessibility(code)
    }
    let result = try await UIPopupSelection(backend: backend).select(13, title: "Destination")
    #expect(result.status == expected)
    #expect(result.showAttempted)
    #expect(!result.pressAttempted)
  }

  @Test func delayedMenuAndNoValueChange() async throws {
    let backend = fixture()
    backend.onPerform = { _, _, _ in }
    backend.onPause = { backend in backend.attributes[13]?[.children] = .elements([50]) }
    let result = try await UIPopupSelection(backend: backend).select(13, title: "Destination")
    #expect(result.status == .valueNotMatched)
    #expect(result.pressAttempted)
    #expect(result.afterValue == "old")
    #expect(backend.clock >= 2.1)
  }

  @Test(arguments: ["show", "press", "post", "value"])
  func failurePreservesAttempt(_ phase: String) async throws {
    let backend = fixture()
    if phase == "show" { backend.performFailure = .notPermitted }
    backend.onPerform = { backend, action, _ in
      if action == "AXShowMenu" {
        backend.attributes[13]?[.children] = .elements([50])
        if phase == "press" { backend.performFailure = .accessibility(-25204) }
      } else if phase == "post" {
        backend.namedFailures["13:AXValue"] = .accessibility(-25202)
      } else {
        backend.attributes[13]?[.value] = .number(1)
      }
    }
    let result = try await UIPopupSelection(backend: backend).select(13, title: "Destination")
    #expect(result.status == .backendFailure)
    #expect(result.showAttempted)
    #expect(result.pressAttempted == (phase != "show"))
    #expect(result.backendError != nil)
  }

  @Test func missingValueAndNonMenuChildren() async throws {
    let backend = fixture()
    backend.attributes[13]?[.value] = .missing
    backend.onPerform = { backend, _, _ in backend.attributes[13]?[.children] = .elements([12]) }
    let result = try await UIPopupSelection(backend: backend).select(13, title: "Destination")
    #expect(result.beforeValue == nil)
    #expect(result.status == .menuMissing)
    #expect(backend.performed == ["AXShowMenu@13"])
  }

  @Test func cancellationDuringPollDoesNotPress() async throws {
    let backend = fixture()
    backend.onPerform = { _, _, _ in }
    let task = Task {
      try await UIPopupSelection(backend: backend).select(13, title: "Destination")
    }
    task.cancel()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(backend.performed.isEmpty)
  }

  @Test func changedPopupDuringWaitIsNotSelected() async throws {
    let backend = fixture()
    backend.onPerform = { _, _, _ in }
    backend.onPause = { backend in
      backend.attributes[13]?[.children] = .elements([50])
      backend.attributes[13]?[.value] = .string("User changed destination")
    }
    let result = try await UIPopupSelection(backend: backend).select(13, title: "Destination")
    #expect(result.status == .popupChanged)
    #expect(!result.pressAttempted)
    #expect(backend.performed == ["AXShowMenu@13"])
  }

  @Test func waitsForDelayedMenuItemsWithoutReopening() async throws {
    let backend = fixture()
    backend.attributes[50]?[.children] = .elements([])
    backend.onPause = { backend in backend.attributes[50]?[.children] = .elements([51]) }
    let result = try await UIPopupSelection(backend: backend).select(13, title: "Destination")
    #expect(result.status == .valueMatched)
    #expect(result.menuChildTitles == ["Destination"])
    #expect(result.menuChildRoles == ["AXMenuItem"])
    #expect(backend.performed == ["AXShowMenu@13", "AXPress@51"])
    #expect(backend.clock == 0.1)
  }

  @Test func nonItemChildrenRemainDiagnosable() async throws {
    let backend = fixture()
    backend.attributes[51]?[.title] = .missing
    backend.attributes[51]?[.role] = .missing
    let result = try await UIPopupSelection(backend: backend).select(13, title: "Destination")
    #expect(result.status == .itemMissing)
    #expect(result.menuChildTitles == [nil])
    #expect(result.menuChildRoles == [nil])
    #expect(!result.pressAttempted)
  }

  @Test func delayedPostValueDoesNotPressTwice() async throws {
    let backend = fixture()
    backend.onPerform = { backend, action, _ in
      if action == "AXShowMenu" { backend.attributes[13]?[.children] = .elements([50]) }
    }
    backend.onPause = { backend in
      backend.attributes[13]?[.value] = .string("Destination")
      backend.attributes[13]?[.children] = .elements([])
    }
    let result = try await UIPopupSelection(backend: backend).select(13, title: "Destination")
    #expect(result.status == .valueMatched)
    #expect(result.childrenAfter == 0)
    #expect(backend.performed == ["AXShowMenu@13", "AXPress@51"])
    #expect(backend.clock == 0.1)
  }
}
