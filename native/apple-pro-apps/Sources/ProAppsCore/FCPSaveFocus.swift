import Foundation

public struct FCPSaveFocusRequest: Codable, Equatable, Sendable {
  public enum Phase: String, Codable, Sendable { case preflight, focus }
  public let phase: Phase
  public let expectedPID: Int32
  public let expectedBirth: String
  public let allowForeground: Bool
  public let window: UIWindowLocator
  public init(
    phase: Phase, expectedPID: Int32, expectedBirth: String, allowForeground: Bool,
    window: UIWindowLocator
  ) {
    self.phase = phase
    self.expectedPID = expectedPID
    self.expectedBirth = expectedBirth
    self.allowForeground = allowForeground
    self.window = window
  }
}

public struct FCPSaveFocusReceipt: Codable, Equatable, Sendable {
  public var status = "refused"
  public var complete = false
  public var beforeFocused: Bool?
  public var afterFocused: Bool?
  public var focusedSettable = false
  public var focusAttempted = false
  public var setterReturned = false
  public var effectVerified = false
  public var beforeFilename: String?
  public var afterFilename: String?
  public var expectedBirth: String?
  public var beforeIdentity: FCPSaveIdentityReceipt?
  public var afterIdentity: FCPSaveIdentityReceipt?
  public var failure: String?
  public private(set) var nativeBirthVerified = false
  public private(set) var activationAttempted = false
  public private(set) var filenameAttempted = false
  public private(set) var saveAttempted = false
  public init() {}
}

extension UIAutomation {
  func saveFocus(_ target: UITarget, _ request: FCPSaveFocusRequest) async -> UIResponse {
    var receipt = FCPSaveFocusReceipt()
    var response = UIResponse(focus: .init(frontmostBefore: nil, frontmostAfter: nil))
    do {
      guard target.app == .finalCutPro, request.expectedPID > 0, request.allowForeground,
        !request.expectedBirth.isEmpty, request.expectedBirth.utf8.count <= 128,
        let title = request.window.title, !title.isEmpty, request.window.index == nil
      else {
        throw ProAppsError.invalid("Explicit exact-owner foreground Save focus request required")
      }
      receipt.expectedBirth = request.expectedBirth
      let process = try requireProcess(target)
      let started = backend.monotonicSeconds()
      let check = {
        try Task.checkCancellation()
        guard process.pid == request.expectedPID, try requireProcess(target) == process,
          backend.frontmostProcess() == process, backend.frontmostBundleID() == process.bundleID,
          backend.monotonicSeconds() - started < 10
        else {
          throw ProAppsError.unavailable("Exact foreground Save owner changed or deadline exceeded")
        }
      }
      let window = try resolveWindow(request.window, in: process)
      var diagnostic = FCPSaveNavigationReceipt(requestedDirectory: "")
      func controls() throws -> SaveControlRead {
        try check()
        guard try backend.sameElement(window, resolveWindow(request.window, in: process)) else {
          throw ProAppsError.unavailable("Save window changed")
        }
        try requireNoOtherModal(process, window: window, check: check)
        let result = try readSaveControls(
          target, window: request.window, receipt: &diagnostic, check: check)
        let dialogs = try goDialogs(window: window, check: check)
        guard
          result.panel.path.isEmpty
            ? dialogs.isEmpty : (dialogs.count == 1 && dialogs.first?.path == result.panel.path)
        else { throw ProAppsError.unavailable("Save must be the only direct dialog") }
        return result
      }
      func identity() async throws -> FCPSaveIdentityReceipt {
        try check()
        let result = await saveIdentity(
          target, .init(expectedPID: request.expectedPID, window: request.window))
        guard let value = result.saveIdentity, value.complete, value.rows.count >= 5,
          value.rows.prefix(5).allSatisfy({ $0.associatedPID == process.pid })
        else { throw ProAppsError.unavailable("Save associated process identities are unproved") }
        return value
      }
      let before = try controls()
      receipt.beforeFilename = before.name
      receipt.beforeIdentity = try await identity()
      receipt.beforeFocused = try saveNameFocused(before.nameElement.handle)
      receipt.focusedSettable = try backend.isSettable(.focused, of: before.nameElement.handle)
      let ready = try controls()
      try sameFocusControls(before, ready)
      guard try saveNameFocused(ready.nameElement.handle) == receipt.beforeFocused,
        try backend.isSettable(.focused, of: ready.nameElement.handle) == receipt.focusedSettable
      else { throw ProAppsError.unavailable("Save focus capability changed before action") }
      if receipt.beforeFocused == nil || !receipt.focusedSettable {
        receipt.status = "unsupported"
      } else if request.phase == .preflight {
        receipt.status = "preflightObserved"
      } else if receipt.beforeFocused == true {
        receipt.status = "alreadyFocused"
      } else {
        try check()
        receipt.focusAttempted = true
        try backend.set(.focused, to: .bool(true), on: ready.nameElement.handle)
        receipt.setterReturned = true
        receipt.status = "noObservedEffect"
      }
      let after = try controls()
      receipt.afterFilename = after.name
      try sameFocusControls(before, after)
      receipt.afterIdentity = try await identity()
      guard receipt.beforeIdentity?.rows.prefix(5) == receipt.afterIdentity?.rows.prefix(5) else {
        throw ProAppsError.unavailable("Original Save identities changed after focus boundary")
      }
      receipt.afterFocused = try saveNameFocused(after.nameElement.handle)
      try check()
      receipt.effectVerified =
        receipt.focusAttempted && receipt.beforeFocused == false
        && receipt.afterFocused == true
      if receipt.effectVerified { receipt.status = "focusObserved" }
      receipt.complete = true
      response.pid = process.pid
    } catch is CancellationError {
      receipt.failure = "Save focus boundary cancelled"
    } catch let error as ProAppsError {
      receipt.failure = String(error.description.prefix(512))
    } catch let error as UIBackendError {
      receipt.failure = String(String(describing: error).prefix(512))
    } catch {
      receipt.failure = "Save focus boundary failed"
    }
    response.saveFocus = receipt
    response.dispatched = receipt.focusAttempted
    response.effectVerified = receipt.effectVerified
    return response
  }

  private func saveNameFocused(_ handle: UIHandle) throws -> Bool? {
    switch try backend.attribute(.focused, of: handle) {
    case .bool(let value): return value
    case .missing: return nil
    default: throw ProAppsError.unavailable("Save filename focus is not a boolean")
    }
  }

  private func sameFocusControls(_ before: SaveControlRead, _ after: SaveControlRead) throws {
    guard before.name == after.name,
      try backend.sameElement(before.panel.handle, after.panel.handle),
      try backend.sameElement(before.popup.handle, after.popup.handle),
      try backend.sameElement(before.nameElement.handle, after.nameElement.handle)
    else { throw ProAppsError.unavailable("Original Save controls or filename changed") }
  }
}
