import CoreGraphics
import Foundation

/// Failures reported by a UI backend. Accessibility codes are AXError raw values.
public enum UIBackendError: Error, Equatable, Sendable {
  case accessibility(Int32)
  case notPermitted
  case unavailable(String)
}

/// One element resolved inside a window, with its child-index path.
public struct UILocatedElement: Equatable, Sendable {
  public let handle: UIHandle
  public let path: [Int]
  public init(handle: UIHandle, path: [Int]) {
    self.handle = handle
    self.path = path
  }
}

/// Captured window pixels encoded as PNG.
public struct UICapturedImage: Sendable {
  public let png: Data
  public let width: Int
  public let height: Int
  public init(png: Data, width: Int, height: Int) {
    self.png = png
    self.width = width
    self.height = height
  }
}

/// Framework boundary for native UI automation. Production uses Accessibility,
/// Text Input Sources, ScreenCaptureKit and AppKit; tests supply a synthetic tree.
/// Handles are only valid inside the backend instance that issued them.
@MainActor
public protocol UIBackend {
  func runningProcess(bundleID: String) -> UIRunningProcess?
  func frontmostBundleID() -> String?
  func frontmostProcess() -> UIRunningProcess?
  func activate(process: UIRunningProcess) -> Bool
  func applicationElement(pid: Int32) -> UIHandle
  func processID(of handle: UIHandle) throws -> Int32
  func attribute(_ name: UIAttributeName, of handle: UIHandle) throws -> UIAttributeValue
  func attributeNames(of handle: UIHandle) throws -> [String]
  func childCount(of handle: UIHandle) throws -> Int
  func childPage(of handle: UIHandle, offset: Int, count: Int) throws -> [UIHandle]
  func sameElement(_ left: UIHandle, _ right: UIHandle) throws -> Bool
  func actions(of handle: UIHandle) throws -> [String]
  func isSettable(_ name: UIAttributeName, of handle: UIHandle) throws -> Bool
  func perform(_ action: String, on handle: UIHandle) throws
  func set(_ name: UIAttributeName, to value: UISettableValue, on handle: UIHandle) throws
  func selectChild(of parent: UIHandle, at index: Int, expected: UIHandle) throws
  func pause(seconds: Double) async throws
  func monotonicSeconds() -> Double
  func captureWindow(pid: Int32, title: String?, frame: UIFrame?) async throws -> UICapturedImage
  func currentInputSource() throws -> UIInputSource
  func asciiCapableInputSource() throws -> UIInputSource
  func inputSources() throws -> [UIInputSource]
  func selectInputSource(id: String) throws
  func applicationURL(bundleID: String) -> URL?
  func launch(applicationAt url: URL, activate: Bool) async throws -> Int32
  func terminate(pid: Int32, force: Bool) -> Bool
}

extension UIBackend {
  public func attributeNames(of handle: UIHandle) throws -> [String] {
    throw UIBackendError.unavailable("AX attribute-name observation is unsupported")
  }
  public func processID(of handle: UIHandle) throws -> Int32 {
    throw UIBackendError.unavailable("AX object process identity is unsupported")
  }
  public func selectChild(of parent: UIHandle, at index: Int, expected: UIHandle) throws {
    throw UIBackendError.unavailable("Verified single-child selection is unsupported")
  }
  public func childCount(of handle: UIHandle) throws -> Int {
    guard case .elements(let children) = try attribute(.children, of: handle) else {
      throw UIBackendError.unavailable("Children count is unavailable")
    }
    return children.count
  }

  public func childPage(of handle: UIHandle, offset: Int, count: Int) throws -> [UIHandle] {
    guard offset >= 0, count > 0, count <= 64,
      case .elements(let children) = try attribute(.children, of: handle),
      offset <= children.count, count <= children.count - offset
    else { throw UIBackendError.unavailable("Invalid bounded children page") }
    return Array(children[offset..<(offset + count)])
  }

  public func sameElement(_ left: UIHandle, _ right: UIHandle) throws -> Bool { left == right }
}

/// Native, background-first UI automation limited to the supported Pro apps.
/// Only an explicit foreground request activates an already running app.
/// It never synthesizes keyboard/pointer input; every result reports focus changes.
@MainActor
public struct UIAutomation<Backend: UIBackend> {
  public static var maximumSearchDepth: Int { 24 }
  public static var maximumSearchNodes: Int { 5000 }
  public static var maximumTextLength: Int { 512 }
  public static var textSearchDepth: Int { 4 }
  public static var pollInterval: Double { 0.2 }
  public static var maximumWaitSeconds: Double { 30 }
  public static var quitWaitSeconds: Double { 10 }

  let backend: Backend
  let saveDirectoryEnvironment: FCPSaveDirectoryEnvironment
  let goToFolderEvents: GoToFolderEventBackend?

  public init(
    backend: Backend, saveDirectoryEnvironment: FCPSaveDirectoryEnvironment = .live,
    goToFolderEvents: GoToFolderEventBackend? = nil
  ) {
    self.backend = backend
    self.saveDirectoryEnvironment = saveDirectoryEnvironment
    self.goToFolderEvents = goToFolderEvents
  }

  public func run(_ request: UIRequest) async throws -> UIResponse {
    let before = backend.frontmostBundleID()
    var response = try await execute(request)
    response.focus = UIFocusEvidence(
      frontmostBefore: before, frontmostAfter: backend.frontmostBundleID())
    return response
  }

  private func execute(_ request: UIRequest) async throws -> UIResponse {
    var response = UIResponse(focus: UIFocusEvidence(frontmostBefore: nil, frontmostAfter: nil))
    switch request {
    case .windows(let target):
      guard let process = try runningProcess(target) else {
        response.running = false
        response.windows = []
        return response
      }
      response.running = true
      response.pid = process.pid
      response.windows = try windows(of: process).enumerated().map { index, window in
        try windowSummary(window, index: index)
      }
    case .inspect(let target, let window, let root, let maxDepth, let maxElements):
      let process = try requireProcess(target)
      let windowHandle = try resolveWindow(window, in: process)
      let start =
        try root.map { try resolveElement($0, in: windowHandle) }
        ?? UILocatedElement(handle: windowHandle, path: [])
      var elements: [UIElementSummary] = []
      let truncated = try collect(
        start.handle, path: start.path, depth: 0, maxDepth: maxDepth, maxElements: maxElements,
        into: &elements)
      response.pid = process.pid
      response.elements = elements
      response.truncated = truncated
    case .menuList(let target, let path):
      let process = try requireProcess(target)
      let container = try menuContainer(path: path, process: process)
      response.pid = process.pid
      response.menuItems = try menuChildren(of: container).compactMap(menuItem)
    case .menuSelect(let target, let path):
      let process = try requireProcess(target)
      guard let last = path.last else { throw ProAppsError.invalid("Menu path is empty") }
      let container = try menuContainer(path: Array(path.dropLast()), process: process)
      let item = try uniqueMenuItem(titled: last, in: container)
      guard try bool(.enabled, of: item) != false else {
        throw ProAppsError.unavailable("Menu item is disabled; inspect the app state first")
      }
      try perform(.press, on: item)
      response.pid = process.pid
      response.dispatched = true
      response.effectVerified = false
    case .popupSelect(let target, let window, let locator, let title):
      let process = try requireProcess(target)
      let windowHandle = try resolveWindow(window, in: process)
      let popup = try resolveElement(locator, in: windowHandle)
      let selection = try await UIPopupSelection(backend: backend).select(
        popup.handle, title: title)
      response.pid = process.pid
      response.popupSelection = selection
      response.dispatched = selection.showAttempted
      response.effectVerified = selection.status == .valueMatched
    case .saveNavigate(let target, let navigation):
      response = await saveNavigate(target, request: navigation)
    case .goToFolder(let target, let request):
      let events =
        goToFolderEvents
        ?? GoToFolderEventBackend(
          permission: CGPreflightPostEventAccess,
          post: { pid in
            for event in try GoToFolderEventBackend.shortcut() { event.postToPid(pid) }
          },
          postForeground: {
            for event in try GoToFolderEventBackend.foregroundShortcut() {
              event.post(tap: .cghidEventTap)
            }
          })
      response = await goToFolder(target, request, events: events)
    case .saveIdentity(let target, let request):
      response = await saveIdentity(target, request)
    case .saveLocationAttributes(let target, let request):
      response = await saveLocationAttributes(target, request)
    case .saveBrowserPage(let target, let request):
      response = await saveBrowserPage(target, request)
    case .saveFocus(let target, let request):
      response = await saveFocus(target, request)
    case .perform(let target, let window, let locator, let action):
      let process = try requireProcess(target)
      let windowHandle = try resolveWindow(window, in: process)
      let element = try resolveElement(locator, in: windowHandle)
      let available = try backend.actions(of: element.handle)
      guard available.contains(action.axName) else {
        throw ProAppsError.invalid(
          "Element does not support \(action.rawValue); available: \(available.sorted().joined(separator: ", "))"
        )
      }
      try perform(action, on: element.handle)
      response.pid = process.pid
      response.dispatched = true
      response.effectVerified = false
      try readBack(locator, window: window, process: process, into: &response)
    case .set(let target, let window, let locator, let attribute, let value):
      let process = try requireProcess(target)
      let windowHandle = try resolveWindow(window, in: process)
      let element = try resolveElement(locator, in: windowHandle)
      guard try backend.isSettable(attribute.attribute, of: element.handle) else {
        throw ProAppsError.invalid("The selected attribute is not settable on this element")
      }
      do {
        try backend.set(attribute.attribute, to: value, on: element.handle)
      } catch let error as UIBackendError {
        throw Self.mapped(error)
      }
      response.pid = process.pid
      response.dispatched = true
      let observed = try backend.attribute(attribute.attribute, of: element.handle)
      response.effectVerified = Self.equals(observed, value)
      response.element = try summary(element.handle, path: element.path)
    case .wait(let target, let window, let locator, let condition, let timeout):
      guard timeout > 0, timeout <= Self.maximumWaitSeconds else {
        throw ProAppsError.invalid("Wait timeout must be within 0–30 seconds")
      }
      guard window != nil || locator != nil else {
        throw ProAppsError.invalid("Wait requires a window or element locator")
      }
      let started = backend.monotonicSeconds()
      var met = false
      while true {
        try Task.checkCancellation()
        let present: Bool
        do {
          present = try isPresent(target: target, window: window, element: locator)
        } catch ProAppsError.unavailable(Self.busyReason) {
          // A launching or busy app cannot answer yet; keep polling within the bound.
          present = false
        }
        met = (condition == .exists) == present
        if met || backend.monotonicSeconds() - started >= timeout { break }
        try await backend.pause(seconds: Self.pollInterval)
      }
      response.conditionMet = met
      response.elapsedSeconds = backend.monotonicSeconds() - started
    case .capture(let target, let window, let outputPath):
      let process = try requireProcess(target)
      let windowHandle = try resolveWindow(window, in: process)
      let image: UICapturedImage
      do {
        image = try await backend.captureWindow(
          pid: process.pid, title: try string(.title, of: windowHandle),
          frame: try frame(of: windowHandle))
      } catch let error as UIBackendError {
        throw Self.mapped(error)
      }
      let url = try Files.writeNew(image.png, to: outputPath, extensions: ["png"])
      response.pid = process.pid
      response.capture = UICaptureResult(
        outputPath: url.path, width: image.width, height: image.height)
    case .inputSourceStatus:
      response.inputSource = try mappedBackend { try backend.currentInputSource() }
    case .inputSourceSelect(let id, let asciiCapable):
      try selectInputSource(id: id, asciiCapable: asciiCapable, into: &response)
    case .launch(let target, let activate):
      let bundleID = try target.resolvedBundleID()
      if let process = backend.runningProcess(bundleID: bundleID) {
        response.pid = process.pid
        response.running = true
        response.dispatched = false
        return response
      }
      guard let url = backend.applicationURL(bundleID: bundleID) else {
        throw ProAppsError.unavailable("Selected app edition is not installed")
      }
      response.pid = try await mappedAsyncBackend {
        try await backend.launch(applicationAt: url, activate: activate)
      }
      response.running = true
      response.dispatched = true
    case .quit(let target, let force):
      guard let process = try runningProcess(target) else {
        response.running = false
        response.terminated = true
        response.dispatched = false
        return response
      }
      response.pid = process.pid
      response.dispatched = backend.terminate(pid: process.pid, force: force)
      let started = backend.monotonicSeconds()
      var gone = backend.runningProcess(bundleID: process.bundleID) == nil
      while response.dispatched == true, !gone,
        backend.monotonicSeconds() - started < Self.quitWaitSeconds
      {
        try await backend.pause(seconds: Self.pollInterval)
        gone = backend.runningProcess(bundleID: process.bundleID) == nil
      }
      response.terminated = gone
      response.running = !gone
    case .activate(let target, let expectedPID, let allowForeground, let menuPath):
      response = try await activate(
        target, expectedPID: expectedPID, allowForeground: allowForeground, menuPath: menuPath)
    }
    return response
  }

  // MARK: Process and windows

  private func runningProcess(_ target: UITarget) throws -> UIRunningProcess? {
    backend.runningProcess(bundleID: try target.resolvedBundleID())
  }

  func requireProcess(_ target: UITarget) throws -> UIRunningProcess {
    guard let process = try runningProcess(target) else {
      throw ProAppsError.unavailable("Selected app edition is not running")
    }
    return process
  }

  private func windows(of process: UIRunningProcess) throws -> [UIHandle] {
    let app = backend.applicationElement(pid: process.pid)
    return try handles(.windows, of: app)
  }

  func resolveWindow(_ locator: UIWindowLocator, in process: UIRunningProcess) throws
    -> UIHandle
  {
    guard let window = try findWindow(locator, in: process) else {
      throw ProAppsError.unavailable("No matching window; list windows first")
    }
    return window
  }

  private func findWindow(_ locator: UIWindowLocator, in process: UIRunningProcess) throws
    -> UIHandle?
  {
    let all = try windows(of: process)
    let candidates = try all.filter { window in
      if let title = locator.title, try string(.title, of: window) != title { return false }
      if let subrole = locator.subrole, try string(.subrole, of: window) != subrole {
        return false
      }
      return true
    }
    if let index = locator.index {
      guard index >= 0, index < candidates.count else { return nil }
      return candidates[index]
    }
    guard candidates.count <= 1 else {
      throw ProAppsError.invalid("Window locator is ambiguous; add title or index")
    }
    return candidates.first
  }

  private func windowSummary(_ window: UIHandle, index: Int) throws -> UIWindowSummary {
    let sheets = try children(of: window).enumerated().compactMap {
      offset, child -> UIElementSummary? in
      let role = try string(.role, of: child)
      guard role == "AXSheet" || role == "AXWindow" else { return nil }
      return try summary(child, path: [offset])
    }
    return UIWindowSummary(
      index: index, title: try string(.title, of: window), role: try string(.role, of: window),
      subrole: try string(.subrole, of: window), main: try bool(.main, of: window),
      minimized: try bool(.minimized, of: window), modal: try bool(.modal, of: window),
      frame: try frame(of: window), sheets: sheets)
  }

  // MARK: Elements

  func resolveElement(_ locator: UIElementLocator, in window: UIHandle) throws
    -> UILocatedElement
  {
    guard locator.hasCriteria else {
      throw ProAppsError.invalid("Element locator needs a path or at least one attribute")
    }
    let matches = try findElements(locator, in: window)
    if let index = locator.index {
      guard index >= 0, index < matches.count else {
        throw ProAppsError.unavailable("Element index is outside the \(matches.count) matches")
      }
      return matches[index]
    }
    guard let first = matches.first else {
      throw ProAppsError.unavailable("No matching element; inspect the window first")
    }
    guard matches.count == 1 else {
      throw ProAppsError.invalid(
        "Element locator matched \(matches.count) elements; refine it or add index")
    }
    return first
  }

  private func findElements(_ locator: UIElementLocator, in window: UIHandle) throws
    -> [UILocatedElement]
  {
    if let path = locator.path {
      var current = window
      for step in path {
        let next = try children(of: current)
        guard step >= 0, step < next.count else { return [] }
        current = next[step]
      }
      return try matches(current, locator) ? [UILocatedElement(handle: current, path: path)] : []
    }
    var found: [UILocatedElement] = []
    var queue: [UILocatedElement] = [UILocatedElement(handle: window, path: [])]
    var visited = 0
    while !queue.isEmpty, visited < Self.maximumSearchNodes {
      let node = queue.removeFirst()
      visited += 1
      if try matches(node.handle, locator) { found.append(node) }
      guard node.path.count < Self.maximumSearchDepth else { continue }
      for (offset, child) in try children(of: node.handle).enumerated() {
        queue.append(UILocatedElement(handle: child, path: node.path + [offset]))
      }
    }
    return found
  }

  private func matches(_ handle: UIHandle, _ locator: UIElementLocator) throws -> Bool {
    let checks: [(String?, UIAttributeName)] = [
      (locator.role, .role), (locator.subrole, .subrole), (locator.identifier, .identifier),
      (locator.title, .title), (locator.description, .description),
    ]
    for (expected, name) in checks {
      if let expected, try string(name, of: handle) != expected { return false }
    }
    if let expected = locator.value, try text(.value, of: handle) != expected { return false }
    if let needle = locator.containsText {
      return try containsText(needle, in: handle, depth: 0)
    }
    return true
  }

  private func containsText(_ needle: String, in handle: UIHandle, depth: Int) throws -> Bool {
    for name in [UIAttributeName.title, .description, .value] {
      if let text = try text(name, of: handle), text.contains(needle) { return true }
    }
    guard depth < Self.textSearchDepth else { return false }
    for child in try children(of: handle)
    where try containsText(needle, in: child, depth: depth + 1) {
      return true
    }
    return false
  }

  private func collect(
    _ handle: UIHandle, path: [Int], depth: Int, maxDepth: Int, maxElements: Int,
    into elements: inout [UIElementSummary]
  ) throws -> Bool {
    guard elements.count < maxElements else { return true }
    elements.append(try summary(handle, path: path))
    guard depth < maxDepth else { return !(try children(of: handle)).isEmpty }
    var truncated = false
    for (offset, child) in try children(of: handle).enumerated() {
      if try collect(
        child, path: path + [offset], depth: depth + 1, maxDepth: maxDepth,
        maxElements: maxElements, into: &elements)
      {
        truncated = true
      }
    }
    return truncated
  }

  func summary(_ handle: UIHandle, path: [Int]) throws -> UIElementSummary {
    var unreadable: [String] = []
    func value(_ name: UIAttributeName) throws -> UIAttributeValue {
      guard let value = try read(name, of: handle) else {
        unreadable.append(name.rawValue)
        return .missing
      }
      return value
    }
    let role = Self.string(try value(.role))
    let subrole = Self.string(try value(.subrole))
    let identifier = Self.string(try value(.identifier))
    let title = Self.string(try value(.title))
    let description = Self.string(try value(.description))
    let text = Self.text(try value(.value))
    let enabled = Self.bool(try value(.enabled))
    let selected = Self.bool(try value(.selected))
    let focused = Self.bool(try value(.focused))
    let frame = Self.frame(position: try value(.position), size: try value(.size))
    let actions = try tolerant { try backend.actions(of: handle) }
    if actions == nil { unreadable.append("actions") }
    let settable = try tolerant { try backend.isSettable(.value, of: handle) }
    if settable == nil { unreadable.append("settable") }
    let selectedSettable = try tolerant { try backend.isSettable(.selected, of: handle) }
    let selectedChildrenSettable = try tolerant {
      try backend.isSettable(.selectedChildren, of: handle)
    }
    if selectedSettable == nil { unreadable.append("AXSelectedSettable") }
    if selectedChildrenSettable == nil { unreadable.append("AXSelectedChildrenSettable") }
    var result = UIElementSummary(
      path: path, role: role, subrole: subrole, identifier: identifier, title: title,
      description: description, value: text, enabled: enabled, selected: selected,
      focused: focused, actions: actions ?? [], valueSettable: settable ?? false, frame: frame,
      unreadable: unreadable)
    result.selectedSettable = selectedSettable
    result.selectedChildrenSettable = selectedChildrenSettable
    return result
  }

  private func readBack(
    _ locator: UIElementLocator, window: UIWindowLocator, process: UIRunningProcess,
    into response: inout UIResponse
  ) throws {
    guard let windowHandle = try findWindow(window, in: process) else {
      response.elementExists = false
      return
    }
    let found = try findElements(locator, in: windowHandle)
    let selected: UILocatedElement?
    if let index = locator.index {
      selected = index >= 0 && index < found.count ? found[index] : nil
    } else {
      selected = found.count == 1 ? found[0] : nil
    }
    response.elementExists = selected != nil
    response.element = try selected.map { try summary($0.handle, path: $0.path) }
  }

  private func isPresent(target: UITarget, window: UIWindowLocator?, element: UIElementLocator?)
    throws -> Bool
  {
    guard let process = try runningProcess(target) else { return false }
    let windowHandle: UIHandle
    if let window {
      guard let found = try findWindow(window, in: process) else { return false }
      windowHandle = found
    } else {
      windowHandle = backend.applicationElement(pid: process.pid)
    }
    guard let element else { return true }
    return !(try findElements(element, in: windowHandle)).isEmpty
  }

  // MARK: Menus

  func menuContainer(path: [String], process: UIRunningProcess) throws -> UIHandle {
    let app = backend.applicationElement(pid: process.pid)
    guard case .element(let bar) = try mappedBackend({ try backend.attribute(.menuBar, of: app) })
    else {
      throw ProAppsError.unavailable("The app exposes no menu bar")
    }
    var container = bar
    for title in path {
      let item = try uniqueMenuItem(titled: title, in: container)
      guard let submenu = try children(of: item).first else {
        throw ProAppsError.invalid("Menu item has no submenu: \(title)")
      }
      container = submenu
    }
    return container
  }

  private func menuChildren(of container: UIHandle) throws -> [UIHandle] {
    try children(of: container)
  }

  func uniqueMenuItem(titled title: String, in container: UIHandle) throws -> UIHandle {
    let items = try menuChildren(of: container).filter { try string(.title, of: $0) == title }
    guard let first = items.first else {
      throw ProAppsError.unavailable("Menu item not found: \(title)")
    }
    guard items.count == 1 else {
      throw ProAppsError.invalid("Menu title is ambiguous: \(title)")
    }
    return first
  }

  private func menuItem(_ handle: UIHandle) throws -> UIMenuItem? {
    guard let title = try string(.title, of: handle), !title.isEmpty else { return nil }
    return UIMenuItem(
      title: title, enabled: try bool(.enabled, of: handle),
      hasSubmenu: !(try children(of: handle)).isEmpty)
  }

  // MARK: Input sources

  private func selectInputSource(id: String?, asciiCapable: Bool, into response: inout UIResponse)
    throws
  {
    guard (id == nil) == asciiCapable else {
      throw ProAppsError.invalid("Supply exactly one of id or asciiCapable: true")
    }
    let previous = try mappedBackend { try backend.currentInputSource() }
    let target: UIInputSource
    if let id {
      let available = try mappedBackend { try backend.inputSources() }
      guard let match = available.first(where: { $0.id == id }), match.selectable else {
        throw ProAppsError.unavailable("Input source is not installed or not selectable")
      }
      target = match
    } else if previous.asciiCapable {
      target = previous
    } else {
      target = try mappedBackend { try backend.asciiCapableInputSource() }
    }
    if target.id != previous.id {
      try mappedBackend { try backend.selectInputSource(id: target.id) }
      response.dispatched = true
    } else {
      response.dispatched = false
    }
    let current = try mappedBackend { try backend.currentInputSource() }
    response.previousInputSource = previous
    response.inputSource = current
    response.effectVerified = current.id == target.id
  }

  // MARK: Attribute helpers

  func perform(_ action: UIAction, on handle: UIHandle) throws {
    try mappedBackend { try backend.perform(action.axName, on: handle) }
  }

  private func children(of handle: UIHandle) throws -> [UIHandle] {
    Self.handles(try read(.children, of: handle) ?? .missing)
  }

  private func handles(_ name: UIAttributeName, of handle: UIHandle) throws -> [UIHandle] {
    Self.handles(try mappedBackend { try backend.attribute(name, of: handle) })
  }

  /// Element-specific Accessibility failures (for example AXError -25200 from a
  /// custom view) return nil so traversal continues and summaries can report the
  /// attribute as unreadable. Permission and busy failures still propagate.
  private func tolerant<T>(_ body: () throws -> T) throws -> T? {
    do { return try body() } catch UIBackendError.accessibility(let code)
      where code != Self.cannotComplete
    {
      return nil
    } catch let error as UIBackendError {
      throw Self.mapped(error)
    }
  }

  private func read(_ name: UIAttributeName, of handle: UIHandle) throws -> UIAttributeValue? {
    try tolerant { try backend.attribute(name, of: handle) }
  }

  // An unreadable attribute is treated as absent for matching; summaries list it.
  private func string(_ name: UIAttributeName, of handle: UIHandle) throws -> String? {
    Self.string(try read(name, of: handle) ?? .missing)
  }

  private func text(_ name: UIAttributeName, of handle: UIHandle) throws -> String? {
    Self.text(try read(name, of: handle) ?? .missing)
  }

  func bool(_ name: UIAttributeName, of handle: UIHandle) throws -> Bool? {
    Self.bool(try read(name, of: handle) ?? .missing)
  }

  private func frame(of handle: UIHandle) throws -> UIFrame? {
    Self.frame(
      position: try read(.position, of: handle) ?? .missing,
      size: try read(.size, of: handle) ?? .missing)
  }

  static func handles(_ value: UIAttributeValue) -> [UIHandle] {
    switch value {
    case .elements(let handles): return handles
    case .element(let handle): return [handle]
    default: return []
    }
  }

  static func string(_ value: UIAttributeValue) -> String? {
    guard case .string(let text) = value else { return nil }
    return String(text.prefix(maximumTextLength))
  }

  static func text(_ value: UIAttributeValue) -> String? {
    switch value {
    case .string(let text): return String(text.prefix(maximumTextLength))
    case .number(let number): return String(number)
    case .bool(let flag): return String(flag)
    default: return nil
    }
  }

  static func bool(_ value: UIAttributeValue) -> Bool? {
    switch value {
    case .bool(let flag): return flag
    case .number(let number): return number != 0
    default: return nil
    }
  }

  static func frame(position: UIAttributeValue, size: UIAttributeValue) -> UIFrame? {
    guard case .point(let origin) = position, case .size(let size) = size else { return nil }
    return UIFrame(x: origin.x, y: origin.y, width: size.width, height: size.height)
  }

  private func mappedBackend<T>(_ body: () throws -> T) throws -> T {
    do { return try body() } catch let error as UIBackendError { throw Self.mapped(error) }
  }

  private func mappedAsyncBackend<T>(_ body: () async throws -> T) async throws -> T {
    do { return try await body() } catch let error as UIBackendError { throw Self.mapped(error) }
  }

  static var busyReason: String {
    "The app is busy or still launching (AXError -25204); wait or observe before retrying"
  }
  static var cannotComplete: Int32 { -25204 }

  static func mapped(_ error: UIBackendError) -> ProAppsError {
    switch error {
    case .accessibility(let code) where code == cannotComplete: return .unavailable(busyReason)
    case .accessibility(let code):
      return .unavailable(
        "Accessibility operation failed (AXError \(code)); observe the app before retrying")
    case .notPermitted:
      return .unavailable(
        "The Executor host lacks the required Accessibility or Screen Recording permission")
    case .unavailable(let reason): return .unavailable(reason)
    }
  }

  static func equals(_ observed: UIAttributeValue, _ requested: UISettableValue) -> Bool {
    switch (observed, requested) {
    case (.string(let actual), .string(let expected)): return actual == expected
    case (.bool(let actual), .bool(let expected)): return actual == expected
    case (.number(let actual), .bool(let expected)): return (actual != 0) == expected
    default: return false
    }
  }
}

/// Child-process result. Exactly one field is present; reasons are developer-facing
/// English and never include project contents.
public struct UIChildOutcome: Codable, Equatable, Sendable {
  public let response: UIResponse?
  public let error: String?
  public let invalid: Bool?
  public init(response: UIResponse?, error: String?, invalid: Bool? = nil) {
    self.response = response
    self.error = error
    self.invalid = invalid
  }

  /// The parent rethrows the same error kind without re-prefixing the reason.
  public static func failure(_ error: ProAppsError) -> UIChildOutcome {
    switch error {
    case .invalid(let reason): return UIChildOutcome(response: nil, error: reason, invalid: true)
    case .unavailable(let reason):
      return UIChildOutcome(response: nil, error: reason, invalid: false)
    case .commandFailed, .timedOut, .outputLimit:
      return UIChildOutcome(response: nil, error: error.description, invalid: false)
    }
  }

  public func rethrown() -> ProAppsError? {
    guard let error else { return nil }
    return invalid == true ? .invalid(error) : .unavailable(error)
  }
}

extension UIAutomation {
  /// Decode one child request, run it and encode the bounded outcome.
  public func runEncoded(_ request: String) async throws -> String {
    let outcome: UIChildOutcome
    do {
      let decoded = try JSONDecoder().decode(UIRequest.self, from: Data(request.utf8))
      outcome = UIChildOutcome(response: try await run(decoded), error: nil)
    } catch let error as ProAppsError {
      outcome = UIChildOutcome.failure(error)
    } catch is DecodingError {
      outcome = UIChildOutcome.failure(.invalid("malformed UI request"))
    }
    return String(decoding: try JSONEncoder().encode(outcome), as: UTF8.self)
  }
}
