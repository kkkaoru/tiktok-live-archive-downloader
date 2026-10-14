import Foundation

public enum UIActivationStatus: String, Codable, Sendable {
  case activationRefused, foregroundTimeout, processChanged, foregroundVerified
  case menuDispatched, menuFailed
}

/// Foreground preparation is separate from Share, Save and export completion.
/// Failure after either dispatch retains its attempted-action evidence.
public struct UIActivationReceipt: Codable, Equatable, Sendable {
  public let expectedPID: Int32
  public var activationAttempted = false
  public var activationAccepted = false
  public var foregroundVerified = false
  public var menuAttempted = false
  public var menuDispatched = false
  public var observedFrontmostPID: Int32?
  public var observedFrontmostBundleID: String?
  public var status: UIActivationStatus = .foregroundTimeout
  public var failure: String?

  public init(expectedPID: Int32) { self.expectedPID = expectedPID }
}

extension UIAutomation {
  static var activationWaitSeconds: Double { 2 }

  /// Does not launch, restore focus, write a filename or press Save. The caller
  /// owns the single foreground lane and restores focus after its Save workflow.
  func activate(
    _ target: UITarget, expectedPID: Int32, allowForeground: Bool, menuPath: [String]?
  ) async throws -> UIResponse {
    guard allowForeground, expectedPID > 0 else {
      throw ProAppsError.invalid(
        "Activation requires allowForeground: true and a positive expectedPID")
    }
    if let menuPath {
      guard !menuPath.isEmpty, menuPath.count <= 6,
        menuPath.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 256 })
      else { throw ProAppsError.invalid("A bounded nonempty exact menu path is required") }
    }
    let process = try requireProcess(target)
    guard process.pid == expectedPID else {
      throw ProAppsError.invalid("Running application PID differs from expectedPID")
    }
    try Task.checkCancellation()
    var receipt = UIActivationReceipt(expectedPID: expectedPID)
    let started = backend.monotonicSeconds()
    if backend.frontmostProcess() != process {
      receipt.activationAttempted = true
      receipt.activationAccepted = backend.activate(process: process)
      if !receipt.activationAccepted {
        receipt.status = .activationRefused
        return foregroundResponse(process, receipt: receipt)
      }
    }
    while true {
      try Task.checkCancellation()
      guard backend.runningProcess(bundleID: process.bundleID) == process else {
        receipt.status = .processChanged
        return foregroundResponse(process, receipt: receipt)
      }
      if backend.frontmostProcess() == process { break }
      guard backend.monotonicSeconds() - started < Self.activationWaitSeconds else {
        receipt.status = .foregroundTimeout
        return foregroundResponse(process, receipt: receipt)
      }
      try await backend.pause(seconds: Self.pollInterval)
    }
    receipt.foregroundVerified = true
    receipt.status = .foregroundVerified
    if let menuPath {
      do {
        try dispatchForegroundMenu(menuPath, process: process, receipt: &receipt)
      } catch ProAppsError.invalid(let reason) {
        receipt.status = .menuFailed
        receipt.failure = String(reason.prefix(Self.maximumTextLength))
      } catch ProAppsError.unavailable(let reason) {
        receipt.status = .menuFailed
        receipt.failure = String(reason.prefix(Self.maximumTextLength))
      }
    }
    return foregroundResponse(process, receipt: receipt)
  }

  private func dispatchForegroundMenu(
    _ path: [String], process: UIRunningProcess, receipt: inout UIActivationReceipt
  ) throws {
    guard let title = path.last else { throw ProAppsError.invalid("Menu path is empty") }
    let container = try menuContainer(path: Array(path.dropLast()), process: process)
    let item = try uniqueMenuItem(titled: title, in: container)
    guard try bool(.enabled, of: item) == true else {
      throw ProAppsError.unavailable("Foreground menu item is not explicitly enabled")
    }
    guard backend.runningProcess(bundleID: process.bundleID) == process,
      backend.frontmostProcess() == process
    else { throw ProAppsError.unavailable("Foreground application changed before menu dispatch") }
    receipt.menuAttempted = true
    try perform(.press, on: item)
    receipt.menuDispatched = true
    receipt.status = .menuDispatched
  }

  private func foregroundResponse(_ process: UIRunningProcess, receipt: UIActivationReceipt)
    -> UIResponse
  {
    var receipt = receipt
    let frontmost = backend.frontmostProcess()
    receipt.observedFrontmostPID = frontmost?.pid
    receipt.observedFrontmostBundleID = frontmost?.bundleID
    var response = UIResponse(focus: UIFocusEvidence(frontmostBefore: nil, frontmostAfter: nil))
    response.pid = process.pid
    response.running = backend.runningProcess(bundleID: process.bundleID) == process
    response.dispatched = receipt.activationAttempted || receipt.menuAttempted
    response.effectVerified = receipt.foregroundVerified && frontmost == process
    response.activation = receipt
    return response
  }
}
