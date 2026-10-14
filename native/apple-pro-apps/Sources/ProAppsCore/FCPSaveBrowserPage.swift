import Foundation

/// One explicit page, relative to the uniquely observed Save ColumnView. Never searches pages.
public struct FCPSaveBrowserPageRequest: Codable, Equatable, Sendable {
  public let expectedPID: Int32
  public let window: UIWindowLocator
  public let relativePath: [Int]
  public let offset: Int
  public let limit: Int
  public init(
    expectedPID: Int32, window: UIWindowLocator, relativePath: [Int], offset: Int, limit: Int
  ) {
    self.expectedPID = expectedPID
    self.window = window
    self.relativePath = relativePath
    self.offset = offset
    self.limit = limit
  }
}

public struct FCPSaveBrowserPageRow: Codable, Equatable, Sendable {
  public let relativePath: [Int]
  public let role: String
  public let identifier: String?
  public let attributeNames: [String]
  public let url: FCPSaveLocationAttribute
  public let value: FCPSaveLocationAttribute
  public let selected: Bool?
}

public struct FCPSaveBrowserPageReceipt: Codable, Equatable, Sendable {
  public var complete = false
  public var parentCount: Int?
  public var offset: Int?
  public var truncated = false
  public var rows: [FCPSaveBrowserPageRow] = []
  public var failure: String?
  public private(set) var actionsAttempted = false
  public private(set) var currentDirectoryVerified = false
  public init() {}
}

extension UIAutomation {
  func saveBrowserPage(_ target: UITarget, _ request: FCPSaveBrowserPageRequest) async -> UIResponse
  {
    var receipt = FCPSaveBrowserPageReceipt()
    var response = UIResponse(focus: .init(frontmostBefore: nil, frontmostAfter: nil))
    do {
      // Same row bound as Save navigation; no enlarged global tree budget.
      guard target.app == .finalCutPro, request.expectedPID > 0,
        let title = request.window.title, !title.isEmpty, request.window.index == nil,
        request.relativePath.count <= 6,
        request.relativePath.allSatisfy({ (0..<4096).contains($0) }),
        (0...4096).contains(request.offset), (1...16).contains(request.limit)
      else { throw ProAppsError.invalid("Invalid bounded Save browser page request") }
      let process = try requireProcess(target)
      let start = backend.monotonicSeconds()
      let check = {
        try Task.checkCancellation()
        guard process.pid == request.expectedPID, try requireProcess(target) == process,
          backend.monotonicSeconds() - start < 10
        else { throw ProAppsError.unavailable("Save page owner changed or deadline exceeded") }
      }
      let window = try resolveWindow(request.window, in: process)
      var diagnostic = FCPSaveNavigationReceipt(requestedDirectory: "")
      func origin() throws -> (SaveControlRead, UIHandle) {
        try check()
        guard try backend.sameElement(window, resolveWindow(request.window, in: process)) else {
          throw ProAppsError.unavailable("Save page window changed")
        }
        let controls = try readSaveControls(
          target, window: request.window, receipt: &diagnostic, check: check)
        let browsers = controls.elements.filter { $0.role == "AXBrowser" }
        guard browsers.count == 1, let browser = browsers.first,
          browser.identifier == "ColumnView",
          try backend.processID(of: controls.panel.handle) == process.pid,
          try backend.processID(of: controls.nameElement.handle) == process.pid,
          try backend.processID(of: controls.popup.handle) == process.pid
        else {
          throw ProAppsError.unavailable("Unique Save ColumnView required")
        }
        // Resolve only the fixed control origin, never a browser descendant.
        let handle = try pagePath(
          window, path: browser.path, expectedPID: process.pid, check: check)
        return (controls, handle)
      }
      let before = try origin()
      let parent = try pagePath(
        before.1, path: request.relativePath, expectedPID: process.pid, check: check)
      let total = try pageCount(parent)
      guard request.offset <= total else {
        throw ProAppsError.invalid("Save page offset outside parent")
      }
      let size = min(request.limit, total - request.offset)
      let page =
        size == 0 ? [] : try backend.childPage(of: parent, offset: request.offset, count: size)
      guard page.count == size else { throw ProAppsError.unavailable("Incomplete Save page") }
      receipt.parentCount = total
      receipt.offset = request.offset
      receipt.truncated = request.offset > 0 || size < total
      for (index, handle) in page.enumerated() {
        try check()
        receipt.rows.append(
          try browserPageRow(
            handle, path: request.relativePath + [request.offset + index], pid: process.pid))
      }
      let after = try origin()
      guard before.0.name == after.0.name,
        try backend.sameElement(before.0.panel.handle, after.0.panel.handle),
        try backend.sameElement(before.0.nameElement.handle, after.0.nameElement.handle),
        try backend.sameElement(before.1, after.1),
        try backend.sameElement(
          parent,
          pagePath(after.1, path: request.relativePath, expectedPID: process.pid, check: check)),
        try pageCount(parent) == total
      else { throw ProAppsError.unavailable("Save page origin or parent changed") }
      let again =
        size == 0 ? [] : try backend.childPage(of: parent, offset: request.offset, count: size)
      guard again.count == size else { throw ProAppsError.unavailable("Save page changed size") }
      for (index, handle) in again.enumerated() {
        try check()
        guard try backend.sameElement(handle, page[index]),
          try browserPageRow(handle, path: receipt.rows[index].relativePath, pid: process.pid)
            == receipt.rows[index]
        else { throw ProAppsError.unavailable("Save page row changed") }
      }
      guard try pageCount(parent) == total else {
        throw ProAppsError.unavailable("Save page count changed")
      }
      try check()
      response.pid = process.pid
      receipt.complete = true
    } catch is CancellationError {
      receipt.failure = "Save page observation cancelled"
    } catch let error as ProAppsError {
      receipt.failure = String(error.description.prefix(512))
    } catch let error as UIBackendError {
      receipt.failure = String(String(describing: error).prefix(512))
    } catch {
      receipt.failure = "Save page observation failed"
    }
    response.saveBrowserPage = receipt
    response.dispatched = false
    response.effectVerified = false
    return response
  }

  private func pageCount(_ handle: UIHandle) throws -> Int {
    let count = try backend.childCount(of: handle)
    guard (0...4096).contains(count) else {
      throw ProAppsError.invalid("Save page parent exceeds row bound")
    }
    return count
  }

  private func pagePath(
    _ origin: UIHandle, path: [Int], expectedPID: Int32, check: () throws -> Void
  ) throws -> UIHandle {
    var parent = origin
    guard try backend.processID(of: parent) == expectedPID else {
      throw ProAppsError.unavailable("Foreign Save origin")
    }
    for offset in path {
      try check()
      let count = try pageCount(parent)
      guard offset >= 0, offset < count else {
        throw ProAppsError.unavailable("Save path outside parent")
      }
      let page = try backend.childPage(of: parent, offset: offset, count: 1)
      guard page.count == 1, let child = page.first,
        try backend.processID(of: child) == expectedPID, try pageCount(parent) == count,
        try backend.childPage(of: parent, offset: offset, count: 1).first.map({
          try backend.sameElement($0, child)
        }) == true
      else { throw ProAppsError.unavailable("Save path identity changed") }
      parent = child
    }
    return parent
  }

  private func browserPageRow(_ handle: UIHandle, path: [Int], pid: Int32) throws
    -> FCPSaveBrowserPageRow
  {
    guard try backend.processID(of: handle) == pid,
      case .string(let role) = try backend.attribute(.role, of: handle), !role.isEmpty,
      role.utf8.count <= 128
    else { throw ProAppsError.unavailable("Save row owner or role missing") }
    let identifier: String?
    switch try backend.attribute(.identifier, of: handle) {
    case .missing: identifier = nil
    case .string(let value) where value.utf8.count <= 512: identifier = value
    default: throw ProAppsError.unavailable("Invalid Save row identifier")
    }
    let names = try backend.attributeNames(of: handle)
    guard names.count <= 128, Set(names).count == names.count,
      names.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 128 })
    else { throw ProAppsError.invalid("Save row attribute names exceed bound") }
    let selected: Bool?
    switch try backend.attribute(.selected, of: handle) {
    case .missing: selected = nil
    case .bool(let value): selected = value
    default: throw ProAppsError.unavailable("Invalid Save row selected value")
    }
    return .init(
      relativePath: path, role: role, identifier: identifier, attributeNames: names.sorted(),
      url: try Self.locationAttribute(
        .url, advertised: names.contains("AXURL"), value: backend.attribute(.url, of: handle)),
      value: try Self.locationAttribute(
        .value, advertised: names.contains("AXValue"), value: backend.attribute(.value, of: handle)),
      selected: selected)
  }
}
