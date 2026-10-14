import CoreGraphics
import Testing

@testable import ProAppsCore

@MainActor
struct GoToFolderEventBackendTests {
  @Test func hidHasCompleteModifierTransitionsAndLeavesTargetUnchanged() throws {
    let events = try GoToFolderEventBackend.foregroundShortcut(permission: { true })
    #expect(
      events.map { $0.getIntegerValueField(.keyboardEventKeycode) } == [55, 56, 5, 5, 56, 55])
    #expect(
      events.map(\.type) == [
        .flagsChanged, .flagsChanged, .keyDown, .keyUp, .flagsChanged, .flagsChanged,
      ])
    #expect(
      events.map(\.flags) == [
        .maskCommand, [.maskCommand, .maskShift], [.maskCommand, .maskShift],
        [.maskCommand, .maskShift], .maskCommand, [],
      ])
    #expect(events.allSatisfy { $0.getIntegerValueField(.keyboardEventAutorepeat) == 0 })
    let target = try GoToFolderEventBackend.shortcut()
    #expect(target.count == 2 && target.map(\.type) == [.keyDown, .keyUp])
    #expect(target.allSatisfy { $0.getIntegerValueField(.keyboardEventKeycode) == 5 })
  }

  @Test(arguments: Array(0..<6))
  func allocationFailureCannotExposeAPartialChord(_ failure: Int) {
    var allocations = 0
    var permissionReads = 0
    var deliveries = 0
    #expect(throws: ProAppsError.self) {
      for _ in try GoToFolderEventBackend.foregroundShortcut(
        make: { key, down in
          defer { allocations += 1 }
          return allocations == failure
            ? nil : CGEvent(keyboardEventSource: nil, virtualKey: key, keyDown: down)
        },
        permission: {
          permissionReads += 1
          return true
        })
      {
        deliveries += 1
      }
    }
    #expect(allocations == failure + 1 && deliveries == 0 && permissionReads == 0)
  }

  @Test func permissionIsRecheckedAfterAllAllocationsBeforeAnyDelivery() {
    var allocations = 0
    var permissionReads = 0
    var deliveries = 0
    #expect(throws: ProAppsError.self) {
      for _ in try GoToFolderEventBackend.foregroundShortcut(
        make: { key, down in
          allocations += 1
          return CGEvent(keyboardEventSource: nil, virtualKey: key, keyDown: down)
        },
        permission: {
          permissionReads += 1
          #expect(allocations == 6)
          return false
        })
      { deliveries += 1 }
    }
    #expect(deliveries == 0 && permissionReads == 1)
  }

  @Test func unprovidedHIDCannotSilentlyUseTargetDelivery() {
    let backend = GoToFolderEventBackend(permission: { true }, post: { _ in })
    #expect(throws: ProAppsError.self) { try backend.postForeground() }
  }
  @Test func constructionDoesNotPostOrTypePath() throws {
    let events = try GoToFolderEventBackend.shortcut()
    #expect(events.count == 2)
    #expect(events[0].type == .keyDown && events[1].type == .keyUp)
    #expect(events.allSatisfy { $0.getIntegerValueField(.keyboardEventKeycode) == 5 })
    #expect(events.allSatisfy { $0.flags == [.maskCommand, .maskShift] })
  }

  @Test func failedConstructionNeverProducesPartialPost() {
    #expect(throws: ProAppsError.self) {
      _ = try GoToFolderEventBackend.shortcut(make: { _ in nil })
    }
  }
}
