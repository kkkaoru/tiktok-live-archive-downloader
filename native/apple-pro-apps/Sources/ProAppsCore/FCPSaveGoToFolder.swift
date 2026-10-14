import Foundation

public struct FCPGoToFolderRequest: Codable, Equatable, Sendable {
  public enum Phase: String, Codable, Sendable { case preflight, open }
  public enum Delivery: String, Codable, Sendable { case target, foregroundHID }
  public let phase: Phase
  public let expectedPID: Int32
  public let expectedBirth: String
  public let allowForeground: Bool
  public let window: UIWindowLocator
  public let delivery: Delivery?

  public init(
    phase: Phase, expectedPID: Int32, expectedBirth: String, allowForeground: Bool,
    window: UIWindowLocator, delivery: Delivery? = nil
  ) {
    self.phase = phase
    self.expectedPID = expectedPID
    self.expectedBirth = expectedBirth
    self.allowForeground = allowForeground
    self.window = window
    self.delivery = delivery
  }
}

/// New dialog appearance is observation, not proof of its unobserved Go controls.
public struct FCPGoToFolderReceipt: Codable, Equatable, Sendable {
  public var status = "refused"
  public var providerPID = ProcessInfo.processInfo.processIdentifier
  public var postEventAccess = false
  public var shortcutAttempted = false
  public var shortcutPosted = false
  public var newDialogObserved = false
  public var beforeFilename: String?
  public var afterFilename: String?
  public var observations: [UIElementSummary] = []
  public var failure: String?
  public var delivery = FCPGoToFolderRequest.Delivery.target
  public var beforeIdentity: FCPSaveIdentityReceipt?
  public var afterIdentity: FCPSaveIdentityReceipt?
  public var changedFocusPaths: [String: [Int]] = [:]
  public private(set) var globalRoutingAtomic = false
  public private(set) var nativeBirthVerified = false
  public private(set) var goToFolderControlsVerified = false
  public private(set) var nameAttempted = false
  public private(set) var saveAttempted = false
  public private(set) var directoryVerified = false
  public init() {}
}

extension UIAutomation {
  func goToFolder(
    _ target: UITarget, _ request: FCPGoToFolderRequest, events: GoToFolderEventBackend
  ) async -> UIResponse {
    var receipt = FCPGoToFolderReceipt()
    receipt.delivery = request.delivery ?? .target
    var response = UIResponse(focus: .init(frontmostBefore: nil, frontmostAfter: nil))
    do {
      guard target.app == .finalCutPro, request.expectedPID > 0, request.allowForeground,
        !request.expectedBirth.isEmpty, request.expectedBirth.utf8.count <= 128,
        let title = request.window.title, !title.isEmpty, request.window.index == nil
      else { throw ProAppsError.invalid("Explicit exact-target foreground Save request required") }
      receipt.postEventAccess = events.permission()
      if request.phase == .preflight {
        receipt.status = "preflightObserved"
      } else {
        guard receipt.postEventAccess else {
          throw ProAppsError.unavailable(
            "Event posting access is not granted; no permission requested")
        }
        let process = try requireProcess(target)
        let boundaryStarted = backend.monotonicSeconds()
        let check = {
          try Task.checkCancellation()
          guard backend.monotonicSeconds() - boundaryStarted < 10 else {
            throw ProAppsError.unavailable("Save shortcut boundary exceeded ten seconds")
          }
          guard process.pid == request.expectedPID,
            try requireProcess(target) == process, backend.frontmostProcess() == process,
            backend.frontmostBundleID() == process.bundleID
          else { throw ProAppsError.unavailable("Exact foreground Save owner changed") }
        }
        try check()
        let window = try resolveWindow(request.window, in: process)
        try requireNoOtherModal(process, window: window, check: check)
        var diagnostic = FCPSaveNavigationReceipt(requestedDirectory: "")
        let before = try readSaveControls(
          target, window: request.window, receipt: &diagnostic, check: check)
        receipt.beforeFilename = before.name
        let baseline = try goDialogs(window: window, check: check)
        guard baseline.count == 1 else {
          throw ProAppsError.unavailable("Save must be the only direct dialog")
        }
        let app = backend.applicationElement(pid: process.pid)
        if receipt.delivery == .foregroundHID {
          receipt.beforeIdentity = try await shortcutIdentity(target, request, check: check)
          _ = try requireShortcutFocus(
            app: app, panel: before.panel.handle,
            name: before.nameElement.handle, allowed: [], expectedPID: process.pid)
          try retainedShortcutControls(target, request, before: before, check: check)
          try check()
        }
        try check()
        receipt.shortcutAttempted = true
        switch receipt.delivery {
        case .target: try events.post(process.pid)
        case .foregroundHID: try events.postForeground()
        }
        receipt.shortcutPosted = true
        let started = backend.monotonicSeconds()
        while true {
          try check()
          let after = try readSaveControls(
            target, window: request.window, receipt: &diagnostic, check: check)
          receipt.afterFilename = after.name
          guard after.name == before.name,
            try backend.sameElement(before.popup.handle, after.popup.handle)
          else {
            throw ProAppsError.unavailable("Save filename changed after shortcut; no retry")
          }
          if receipt.delivery == .foregroundHID {
            guard try backend.sameElement(before.panel.handle, after.panel.handle),
              try backend.sameElement(before.nameElement.handle, after.nameElement.handle),
              try backend.sameElement(window, resolveWindow(request.window, in: process)),
              try backend.sameElement(app, backend.applicationElement(pid: process.pid))
            else { throw ProAppsError.unavailable("Original Save controls changed after HID") }
          }
          let current = try goDialogs(window: window, check: check)
          var added: [UILocatedElement] = []
          for element in current {
            if try !baseline.contains(where: { try backend.sameElement($0.handle, element.handle) })
            {
              added.append(element)
            }
          }
          guard added.count <= 1 else {
            throw ProAppsError.unavailable("New dialogs are ambiguous")
          }
          if let dialog = added.first {
            let inspected = try await run(
              .inspect(
                target, window: request.window, root: .init(path: dialog.path),
                maxDepth: 3, maxElements: 64))
            guard inspected.truncated == false, let elements = inspected.elements else {
              throw ProAppsError.unavailable("New dialog observation is incomplete")
            }
            try check()
            if receipt.delivery == .foregroundHID {
              receipt.afterIdentity = try await shortcutIdentity(target, request, check: check)
              guard receipt.beforeIdentity?.rows.prefix(5) == receipt.afterIdentity?.rows.prefix(5)
              else { throw ProAppsError.unavailable("Original Save identity changed after HID") }
              let allowed = try shortcutMembers(
                dialog, elements: elements, window: window, check: check)
              receipt.changedFocusPaths = try requireShortcutFocus(
                app: app, panel: before.panel.handle,
                name: before.nameElement.handle, allowed: allowed, expectedPID: process.pid)
              try requireNoOtherModal(process, window: window, check: check)
              try retainedShortcutControls(target, request, before: before, check: check)
              try check()
            }
            receipt.observations = elements
            receipt.newDialogObserved = true
            receipt.status = "newDialogObserved"
            break
          }
          guard backend.monotonicSeconds() - started < 2 else {
            if receipt.delivery == .foregroundHID {
              receipt.afterIdentity = try await shortcutIdentity(target, request, check: check)
              guard receipt.beforeIdentity?.rows.prefix(5) == receipt.afterIdentity?.rows.prefix(5)
              else { throw ProAppsError.unavailable("Original Save identity changed after HID") }
              _ = try requireShortcutFocus(
                app: app, panel: before.panel.handle,
                name: before.nameElement.handle, allowed: [], expectedPID: process.pid)
              try retainedShortcutControls(target, request, before: before, check: check)
            }
            receipt.status = "noObservedDialog"
            break
          }
          try await backend.pause(seconds: 0.1)
        }
        response.pid = process.pid
      }
    } catch is CancellationError {
      receipt.status = "cancelled"
    } catch let error as ProAppsError {
      receipt.failure = String(error.description.prefix(512))
    } catch let error as UIBackendError {
      receipt.failure = String(String(describing: error).prefix(512))
    } catch {
      receipt.failure = "Go to Folder observation failed; do not repeat an uncertain shortcut"
    }
    response.goToFolder = receipt
    response.dispatched = receipt.shortcutAttempted
    response.effectVerified = false
    return response
  }

  private func shortcutIdentity(
    _ target: UITarget, _ request: FCPGoToFolderRequest, check: () throws -> Void
  ) async throws -> FCPSaveIdentityReceipt {
    try check()
    let observed = await saveIdentity(
      target,
      .init(expectedPID: request.expectedPID, window: request.window))
    guard let identity = observed.saveIdentity, identity.complete, identity.rows.count == 7,
      identity.rows.allSatisfy({ $0.associatedPID == request.expectedPID })
    else { throw ProAppsError.unavailable("HID requires complete same-owner Save/focus identity") }
    try check()
    return identity
  }

  private func retainedShortcutControls(
    _ target: UITarget, _ request: FCPGoToFolderRequest, before: SaveControlRead,
    check: () throws -> Void
  ) throws {
    var diagnostic = FCPSaveNavigationReceipt(requestedDirectory: "")
    let current = try readSaveControls(
      target, window: request.window, receipt: &diagnostic, check: check)
    guard current.name == before.name,
      try backend.sameElement(current.panel.handle, before.panel.handle),
      try backend.sameElement(current.popup.handle, before.popup.handle),
      try backend.sameElement(current.nameElement.handle, before.nameElement.handle)
    else { throw ProAppsError.unavailable("Save controls changed at the final HID boundary") }
    try check()
  }

  private func requireShortcutFocus(
    app: UIHandle, panel: UIHandle, name: UIHandle, allowed: [UILocatedElement], expectedPID: Int32
  ) throws -> [String: [Int]] {
    var changed: [String: [Int]] = [:]
    for (attribute, original) in [
      (UIAttributeName.focusedWindow, panel), (.focusedUIElement, name),
    ] {
      guard case .element(let focused) = try backend.attribute(attribute, of: app),
        try backend.processID(of: focused) == expectedPID
      else { throw ProAppsError.unavailable("HID focus ownership is unavailable") }
      if try backend.sameElement(focused, original) { continue }
      guard let member = try allowed.first(where: { try backend.sameElement($0.handle, focused) })
      else { throw ProAppsError.unavailable("HID focus escaped the observed Save/Go sheet") }
      changed[attribute.rawValue] = member.path
    }
    return changed
  }

  private func shortcutMembers(
    _ dialog: UILocatedElement, elements: [UIElementSummary], window: UIHandle,
    check: () throws -> Void
  ) throws -> [UILocatedElement] {
    var members: [UILocatedElement] = []
    for element in elements {
      try check()
      guard element.path.starts(with: dialog.path), element.path.count <= dialog.path.count + 3,
        try backend.sameElement(
          dialog.handle, resolveElement(.init(path: dialog.path), in: window).handle)
      else { throw ProAppsError.unavailable("Observed Go sheet membership changed") }
      var parent = dialog.handle
      for offset in element.path.dropFirst(dialog.path.count) {
        let count = try backend.childCount(of: parent)
        guard count > offset, offset >= 0, count <= 64 else {
          throw ProAppsError.unavailable("Go sheet child boundary changed")
        }
        let page = try backend.childPage(of: parent, offset: offset, count: 1)
        guard page.count == 1, let child = page.first,
          try backend.childCount(of: parent) == count,
          try backend.childPage(of: parent, offset: offset, count: 1).first.map({
            try backend.sameElement($0, child)
          }) == true
        else { throw ProAppsError.unavailable("Go sheet child identity changed") }
        parent = child
      }
      let resolved = try resolveElement(
        .init(
          path: element.path, role: element.role,
          identifier: element.identifier), in: window)
      guard try backend.sameElement(parent, resolved.handle) else {
        throw ProAppsError.unavailable("Go sheet observed element changed")
      }
      members.append(resolved)
    }
    return members
  }

  func goDialogs(window: UIHandle, check: () throws -> Void) throws -> [UILocatedElement] {
    var result: [UILocatedElement] = []
    var parents = [UILocatedElement(handle: window, path: [])]
    var cursor = 0
    while cursor < parents.count {
      try check()
      let parent = parents[cursor]
      cursor += 1
      let count = try backend.childCount(of: parent.handle)
      guard count >= 0, count <= 64, parents.count <= 4 else {
        throw ProAppsError.invalid("Direct dialog observation exceeds its bound")
      }
      let children =
        count == 0 ? [] : try backend.childPage(of: parent.handle, offset: 0, count: count)
      guard children.count == count else {
        throw ProAppsError.invalid("Incomplete dialog children")
      }
      for (index, child) in children.enumerated() {
        if try backend.attribute(.role, of: child) == .string("AXSheet") {
          let located = UILocatedElement(handle: child, path: parent.path + [index])
          result.append(located)
          if parent.path.isEmpty { parents.append(located) }
        }
      }
    }
    return result
  }

  func requireNoOtherModal(
    _ process: UIRunningProcess, window: UIHandle, check: () throws -> Void
  ) throws {
    guard
      case .elements(let windows) = try backend.attribute(
        .windows, of: backend.applicationElement(pid: process.pid)), windows.count <= 64
    else { throw ProAppsError.unavailable("Bounded app window inventory is required") }
    for other in windows {
      try check()
      if try backend.sameElement(window, other) { continue }
      let modal = try backend.attribute(.modal, of: other)
      guard modal == .bool(false) || modal == .number(0),
        try goDialogs(window: other, check: check).isEmpty
      else { throw ProAppsError.unavailable("Another modal or sheet is open") }
    }
  }
}
