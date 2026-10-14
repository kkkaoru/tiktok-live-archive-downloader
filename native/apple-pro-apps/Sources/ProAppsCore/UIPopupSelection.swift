import Foundation

/// A value match is observation, not a path identity or successful application transaction.
public enum UIPopupSelectionStatus: String, Codable, Sendable {
  case valueMatched, valueNotMatched, menuMissing, menuAmbiguous, itemMissing, itemAmbiguous
  case itemDisabled, itemUnsupported, backendFailure, unreadableChildren, invalidChildren
  case childBudgetExceeded, popupChanged
}

/// Retains attempted side effects even when Accessibility fails after dispatch.
public struct UIPopupSelectionReceipt: Codable, Equatable, Sendable {
  public var status: UIPopupSelectionStatus = .menuMissing
  public var showAttempted = false
  public var pressAttempted = false
  public var beforeValue: String?
  public var afterValue: String?
  public var menuCount = 0
  public var matchingItemCount = 0
  public var menuChildTitles: [String?] = []
  public var menuChildRoles: [String?] = []
  public var childrenAfter: Int?
  public var backendError: String?
}

/// Keeps the resolved popup handle across opening, bounded child discovery and one selection.
@MainActor
struct UIPopupSelection<Backend: UIBackend> {
  static var timeout: Double { 2 }
  static var interval: Double { 0.1 }
  static var maximumChildren: Int { 256 }
  let backend: Backend

  func select(_ popup: UIHandle, title: String) async throws -> UIPopupSelectionReceipt {
    guard !title.isEmpty, title.utf8.count <= 2048, !title.contains("\u{0}") else {
      throw ProAppsError.invalid(
        "Popup title must be nonempty, NUL-free and at most 2048 UTF-8 bytes")
    }
    guard case .string("AXPopUpButton") = try backend.attribute(.role, of: popup),
      try backend.attribute(.enabled, of: popup) == .bool(true),
      try backend.actions(of: popup).contains("AXShowMenu")
    else {
      throw ProAppsError.invalid("Selection requires an enabled AXPopUpButton with AXShowMenu")
    }
    var receipt = UIPopupSelectionReceipt()
    do {
      receipt.beforeValue = try textValue(popup)
      try Task.checkCancellation()
      receipt.showAttempted = true
      try backend.perform("AXShowMenu", on: popup)
      let started = backend.monotonicSeconds()
      while true {
        try Task.checkCancellation()
        let menus = try children(popup).filter {
          try backend.attribute(.role, of: $0) == .string("AXMenu")
        }
        receipt.menuCount = menus.count
        if let menu = menus.first {
          guard menus.count == 1 else {
            receipt.status = .menuAmbiguous
            return receipt
          }
          guard try textValue(popup) == receipt.beforeValue else {
            receipt.status = .popupChanged
            return receipt
          }
          try selectItem(menu, title: title, receipt: &receipt)
          if receipt.pressAttempted {
            try await observeSelection(popup, title: title, receipt: &receipt)
          }
          if receipt.status != .itemMissing { return receipt }
        } else {
          receipt.status = .menuMissing
        }
        guard backend.monotonicSeconds() - started < Self.timeout else { return receipt }
        try await backend.pause(seconds: Self.interval)
      }
    } catch let error as UIBackendError {
      receipt.status = .backendFailure
      receipt.backendError = String(describing: error)
      return receipt
    } catch let error as PopupReadError {
      receipt.status = error.status
      return receipt
    }
  }

  private func observeSelection(
    _ popup: UIHandle, title: String, receipt: inout UIPopupSelectionReceipt
  ) async throws {
    let started = backend.monotonicSeconds()
    while true {
      try Task.checkCancellation()
      receipt.afterValue = try textValue(popup)
      receipt.childrenAfter = try children(popup).count
      receipt.status = receipt.afterValue == title ? .valueMatched : .valueNotMatched
      if receipt.status == .valueMatched || backend.monotonicSeconds() - started >= Self.timeout {
        return
      }
      try await backend.pause(seconds: Self.interval)
    }
  }

  private func selectItem(
    _ menu: UIHandle, title: String, receipt: inout UIPopupSelectionReceipt
  ) throws {
    let menuChildren = try children(menu)
    receipt.menuChildTitles = try menuChildren.map { child in
      guard case .string(let title) = try backend.attribute(.title, of: child) else { return nil }
      return String(title.prefix(512))
    }
    receipt.menuChildRoles = try menuChildren.map { child in
      guard case .string(let role) = try backend.attribute(.role, of: child) else { return nil }
      return String(role.prefix(128))
    }
    let matches = try menuChildren.filter {
      try backend.attribute(.role, of: $0) == .string("AXMenuItem")
        && backend.attribute(.title, of: $0) == .string(title)
    }
    receipt.matchingItemCount = matches.count
    guard let item = matches.first else {
      receipt.status = .itemMissing
      return
    }
    guard matches.count == 1 else {
      receipt.status = .itemAmbiguous
      return
    }
    guard try backend.attribute(.enabled, of: item) == .bool(true) else {
      receipt.status = .itemDisabled
      return
    }
    guard try backend.actions(of: item).contains("AXPress") else {
      receipt.status = .itemUnsupported
      return
    }
    try Task.checkCancellation()
    receipt.pressAttempted = true
    try backend.perform("AXPress", on: item)
  }

  private func children(_ handle: UIHandle) throws -> [UIHandle] {
    let value: UIAttributeValue
    do { value = try backend.attribute(.children, of: handle) } catch UIBackendError.accessibility(
      let code) where code != -25204
    {
      throw PopupReadError(status: .unreadableChildren)
    }
    switch value {
    case .missing: return []
    case .elements(let values):
      guard values.count <= Self.maximumChildren else {
        throw PopupReadError(status: .childBudgetExceeded)
      }
      return values
    default: throw PopupReadError(status: .invalidChildren)
    }
  }

  private func textValue(_ handle: UIHandle) throws -> String? {
    switch try backend.attribute(.value, of: handle) {
    case .string(let value): return value
    case .missing: return nil
    default: throw UIBackendError.unavailable("Popup value is not text")
    }
  }
}

private struct PopupReadError: Error {
  let status: UIPopupSelectionStatus
}
