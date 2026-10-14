import Foundation

public struct FCPSaveIdentityRequest: Codable, Equatable, Sendable {
  public let expectedPID: Int32
  public let window: UIWindowLocator
  public init(expectedPID: Int32, window: UIWindowLocator) {
    self.expectedPID = expectedPID
    self.window = window
  }
}

public struct FCPSaveIdentityRow: Codable, Equatable, Sendable {
  public let label: String
  public let path: [Int]?
  public let role: String
  public let identifier: String?
  public let associatedPID: Int32
}

public struct FCPSaveIdentityReceipt: Codable, Equatable, Sendable {
  public var complete = false
  public var rows: [FCPSaveIdentityRow] = []
  public var focusStates: [String: String] = [:]
  public var failure: String?
  public private(set) var routingVerified = false
  public private(set) var nativeBirthVerified = false
  public private(set) var actionsAttempted = false
  public init() {}
}

private struct SaveIdentityItem {
  let label: String
  let handle: UIHandle
  let path: [Int]?
}

extension UIAutomation {
  func saveIdentity(_ target: UITarget, _ request: FCPSaveIdentityRequest) async -> UIResponse {
    var receipt = FCPSaveIdentityReceipt()
    var response = UIResponse(focus: .init(frontmostBefore: nil, frontmostAfter: nil))
    do {
      guard target.app == .finalCutPro, request.expectedPID > 0,
        let title = request.window.title, !title.isEmpty, request.window.index == nil
      else { throw ProAppsError.invalid("Exact FCP Save identity request required") }
      let process = try requireProcess(target)
      let started = backend.monotonicSeconds()
      let check = {
        try Task.checkCancellation()
        guard process.pid == request.expectedPID, try requireProcess(target) == process,
          backend.monotonicSeconds() - started < 10
        else { throw ProAppsError.unavailable("Save identity owner changed or deadline exceeded") }
      }
      var diagnostic = FCPSaveNavigationReceipt(requestedDirectory: "")
      func items() throws -> ([SaveIdentityItem], [String: String], String) {
        try check()
        let app = backend.applicationElement(pid: process.pid)
        let window = try resolveWindow(request.window, in: process)
        let controls = try readSaveControls(
          target, window: request.window, receipt: &diagnostic, check: check)
        var result: [SaveIdentityItem] = [
          .init(label: "application", handle: app, path: nil),
          .init(label: "window", handle: window, path: []),
          .init(label: "savePanel", handle: controls.panel.handle, path: controls.panel.path),
          .init(label: "wherePopup", handle: controls.popup.handle, path: controls.popup.path),
          .init(
            label: "filename", handle: controls.nameElement.handle, path: controls.nameElement.path),
        ]
        var states: [String: String] = [:]
        for attribute in [UIAttributeName.focusedWindow, .focusedUIElement] {
          try check()
          switch try backend.attribute(attribute, of: app) {
          case .element(let handle):
            states[attribute.rawValue] = "element"
            result.append(.init(label: attribute.rawValue, handle: handle, path: nil))
          case .missing: states[attribute.rawValue] = "unavailable"
          default: throw ProAppsError.unavailable("Focus reference is not one direct AX element")
          }
        }
        return (result, states, controls.name)
      }
      let before = try items()
      receipt.focusStates = before.1
      for item in before.0 {
        try check()
        receipt.rows.append(try identityRow(item))
      }
      let after = try items()
      guard before.0.count == after.0.count, before.1 == after.1, before.2 == after.2 else {
        throw ProAppsError.unavailable("Save controls or focus references changed")
      }
      for (index, item) in after.0.enumerated() {
        try check()
        guard try backend.sameElement(before.0[index].handle, item.handle),
          try identityRow(item) == receipt.rows[index]
        else { throw ProAppsError.unavailable("AX identity or associated process changed") }
      }
      try check()
      receipt.complete = true
      response.pid = process.pid
    } catch is CancellationError {
      receipt.failure = "Save identity observation cancelled"
    } catch let error as ProAppsError {
      receipt.failure = String(error.description.prefix(512))
    } catch let error as UIBackendError {
      receipt.failure = String(String(describing: error).prefix(512))
    } catch {
      receipt.failure = "Save identity observation failed"
    }
    response.saveIdentity = receipt
    response.dispatched = false
    response.effectVerified = false
    return response
  }

  private func identityRow(_ item: SaveIdentityItem) throws -> FCPSaveIdentityRow {
    let pid = try backend.processID(of: item.handle)
    guard pid > 0, case .string(let role) = try backend.attribute(.role, of: item.handle),
      !role.isEmpty, role.utf8.count <= 128
    else { throw ProAppsError.unavailable("AX associated PID or role is unavailable") }
    let identifier: String?
    switch try backend.attribute(.identifier, of: item.handle) {
    case .string(let value):
      guard value.utf8.count <= 512 else { throw ProAppsError.invalid("AX identity exceeds bound") }
      identifier = value
    case .missing: identifier = nil
    default: throw ProAppsError.unavailable("AX identifier type is invalid")
    }
    return .init(
      label: item.label, path: item.path, role: role, identifier: identifier, associatedPID: pid)
  }
}
