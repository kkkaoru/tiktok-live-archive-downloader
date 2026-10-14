import Foundation

/// An observational request only: no supplied process ID, path or action.
public struct FCPSaveLocationAttributesRequest: Codable, Equatable, Sendable {
  public let window: UIWindowLocator
  public init(window: UIWindowLocator) { self.window = window }
}

public struct FCPSaveLocationAttribute: Codable, Equatable, Sendable {
  public let attribute: String
  public let state: String
  public let value: String?
  public let lexicalFilePath: String?
}

public struct FCPSaveLocationRow: Codable, Equatable, Sendable {
  public let scope: String
  public let path: [Int]
  public let role: String
  public let identifier: String
  public let attributeNames: [String]
  public let attributes: [FCPSaveLocationAttribute]
}

public struct FCPSaveLocationAttributesReceipt: Codable, Equatable, Sendable {
  public var complete = false
  public var rows: [FCPSaveLocationRow] = []
  public var failure: String?
  public private(set) var currentDirectoryVerified = false
  public private(set) var filesystemIdentityVerified = false
  public private(set) var actionsAttempted = false
  public init() {}
}

extension UIAutomation {
  func saveLocationAttributes(
    _ target: UITarget, _ request: FCPSaveLocationAttributesRequest
  ) async -> UIResponse {
    var receipt = FCPSaveLocationAttributesReceipt()
    var response = UIResponse(focus: .init(frontmostBefore: nil, frontmostAfter: nil))
    do {
      guard target.app == .finalCutPro, let title = request.window.title, !title.isEmpty,
        request.window.index == nil
      else { throw ProAppsError.invalid("An exact FCP Save window title is required") }
      let process = try requireProcess(target)
      let started = backend.monotonicSeconds()
      let check = {
        try Task.checkCancellation()
        guard try requireProcess(target) == process, backend.monotonicSeconds() - started < 10
        else { throw ProAppsError.unavailable("Save location owner changed or deadline exceeded") }
      }
      var diagnostic = FCPSaveNavigationReceipt(requestedDirectory: "")
      func controls() throws -> (SaveControlRead, UILocatedElement) {
        let control = try readSaveControls(
          target, window: request.window, receipt: &diagnostic, check: check)
        let browsers = control.elements.filter { $0.role == "AXBrowser" }
        guard browsers.count == 1, let browser = browsers.first, browser.identifier == "ColumnView"
        else { throw ProAppsError.unavailable("A unique Save ColumnView browser is required") }
        let window = try resolveWindow(request.window, in: process)
        return (
          control,
          try resolveElement(
            .init(path: browser.path, role: "AXBrowser", identifier: "ColumnView"), in: window)
        )
      }
      let before = try controls()
      let handles = [before.0.panel, before.0.popup, before.1]
      let scopes = ["savePanel", "wherePopup", "browser"]
      for (index, item) in handles.enumerated() {
        try check()
        receipt.rows.append(try saveLocationRow(item, scope: scopes[index]))
      }
      let after = try controls()
      guard before.0.name == after.0.name else {
        throw ProAppsError.unavailable("Save filename changed during observation")
      }
      for (index, item) in [after.0.panel, after.0.popup, after.1].enumerated() {
        try check()
        guard try backend.sameElement(handles[index].handle, item.handle),
          try saveLocationRow(item, scope: scopes[index]) == receipt.rows[index]
        else {
          throw ProAppsError.unavailable("Save location attributes changed during observation")
        }
      }
      try check()
      response.pid = process.pid
      receipt.complete = true
    } catch is CancellationError {
      receipt.failure = "Save location observation cancelled"
    } catch let error as ProAppsError {
      receipt.failure = String(error.description.prefix(512))
    } catch let error as UIBackendError {
      receipt.failure = String(String(describing: error).prefix(512))
    } catch {
      receipt.failure = "Save location observation failed"
    }
    response.saveLocationAttributes = receipt
    response.dispatched = false
    response.effectVerified = false
    return response
  }

  private func saveLocationRow(_ item: UILocatedElement, scope: String) throws -> FCPSaveLocationRow
  {
    guard case .string(let role) = try backend.attribute(.role, of: item.handle),
      ["AXSheet", "AXWindow", "AXPopUpButton", "AXBrowser"].contains(role),
      case .string(let identifier) = try backend.attribute(.identifier, of: item.handle),
      identifier.utf8.count <= 512
    else { throw ProAppsError.unavailable("Save location role or identity is unavailable") }
    let names = try backend.attributeNames(of: item.handle)
    guard names.count <= 128, Set(names).count == names.count,
      names.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 128 })
    else { throw ProAppsError.unavailable("AX attribute names exceed the typed metadata bound") }
    var attributes: [FCPSaveLocationAttribute] = []
    for name in [UIAttributeName.url, .document, .value] {
      let value =
        names.contains(name.rawValue) ? try backend.attribute(name, of: item.handle) : .missing
      attributes.append(
        try Self.locationAttribute(name, advertised: names.contains(name.rawValue), value: value))
    }
    return .init(
      scope: scope, path: item.path, role: role, identifier: identifier,
      attributeNames: names.sorted(), attributes: attributes)
  }

  /// This is lexical validation, not stat, symlink resolution or directory proof.
  static func locationAttribute(
    _ name: UIAttributeName, advertised: Bool, value: UIAttributeValue
  ) throws -> FCPSaveLocationAttribute {
    let state: String
    let text: String?
    var path: String?
    switch value {
    case .missing:
      state = advertised ? "noValue" : "notAdvertised"
      text = nil
    case .string(let string):
      guard string.utf8.count <= 4096, !string.contains("\0") else {
        throw ProAppsError.invalid("Save location string exceeds the metadata bound")
      }
      text = string
      if name == .value, !string.hasPrefix("file:") {
        state = "textOnly"
      } else if let url = URL(string: string), url.isFileURL,
        url.host == nil || url.host == "" || url.host == "localhost",
        url.user == nil, url.password == nil, url.port == nil, url.query == nil,
        url.fragment == nil,
        url.path.hasPrefix("/"), url.standardizedFileURL.path == url.path,
        !url.path.contains("\0"), !url.path.contains("//"),
        !string.lowercased().contains("%2f"), !string.lowercased().contains("%5c"),
        !url.path.split(separator: "/").contains(where: { $0 == "." || $0 == ".." })
      {
        state = "lexicalFileURLOnly"
        path = url.path
      } else {
        state = "notCanonicalLocalFileURL"
      }
    default:
      state = "unsupportedType"
      text = nil
    }
    return .init(attribute: name.rawValue, state: state, value: text, lexicalFilePath: path)
  }
}
