import AppKit
import ApplicationServices
import Carbon
import CoreGraphics
import Foundation
import ImageIO
import ScreenCaptureKit

/// Production UI backend: the single approved Accessibility / Text Input Source /
/// ScreenCaptureKit interoperability boundary (see NATIVE-BOUNDARIES.md).
///
/// Invariants:
/// - Every CFTypeRef is checked with CFGetTypeID before conversion; no forced cast.
/// - TIS property pointers are borrowed (get rule) and consumed immediately with
///   takeUnretainedValue; copied sources use the create/copy rule (takeRetainedValue).
/// - AXUIElement references are retained by `elements` for this backend's lifetime,
///   which is one child process handling one request.
/// - No operation activates an app or synthesizes keyboard/pointer input.
@MainActor
public final class LiveUIBackend: UIBackend {
  static let messagingTimeoutSeconds: Float = 3
  private var elements: [AXUIElement] = []

  /// Touching NSApplication establishes the window-server connection that
  /// ScreenCaptureKit and CoreGraphics require in a command-line child process.
  public init() { _ = NSApplication.shared }

  public func runningProcess(bundleID: String) -> UIRunningProcess? {
    NSRunningApplication.runningApplications(withBundleIdentifier: bundleID)
      .first { !$0.isTerminated }
      .map { UIRunningProcess(pid: $0.processIdentifier, bundleID: bundleID) }
  }

  public func frontmostBundleID() -> String? {
    NSWorkspace.shared.frontmostApplication?.bundleIdentifier
  }

  public func applicationElement(pid: Int32) -> UIHandle {
    let element = AXUIElementCreateApplication(pid)
    // A bounded messaging timeout prevents an unresponsive app from stalling the child.
    AXUIElementSetMessagingTimeout(element, Self.messagingTimeoutSeconds)
    return register(element)
  }

  public func attribute(_ name: UIAttributeName, of handle: UIHandle) throws -> UIAttributeValue {
    let element = try lookup(handle)
    var raw: CFTypeRef?
    let status = AXUIElementCopyAttributeValue(element, name.rawValue as CFString, &raw)
    switch status {
    case .success: break
    case .noValue, .attributeUnsupported, .invalidUIElement: return .missing
    default: throw Self.error(status)
    }
    guard let raw else { return .missing }
    return convert(raw)
  }

  public func processID(of handle: UIHandle) throws -> Int32 {
    var pid: pid_t = 0
    let status = AXUIElementGetPid(try lookup(handle), &pid)
    guard status == .success else { throw Self.error(status) }
    guard pid > 0 else { throw UIBackendError.unavailable("AX object PID is not positive") }
    return pid
  }

  public func attributeNames(of handle: UIHandle) throws -> [String] {
    var raw: CFArray?
    let status = AXUIElementCopyAttributeNames(try lookup(handle), &raw)
    guard status == .success else { throw Self.error(status) }
    guard let raw, CFArrayGetCount(raw) <= 128, let names = raw as? [String],
      Set(names).count == names.count,
      names.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 128 })
    else { throw UIBackendError.unavailable("Invalid bounded AX attribute-name array") }
    return names
  }

  public func actions(of handle: UIHandle) throws -> [String] {
    var raw: CFArray?
    let status = AXUIElementCopyActionNames(try lookup(handle), &raw)
    switch status {
    case .success: return (raw as? [String]) ?? []
    case .noValue, .actionUnsupported, .invalidUIElement: return []
    default: throw Self.error(status)
    }
  }

  public func childCount(of handle: UIHandle) throws -> Int {
    var count = 0
    let status = AXUIElementGetAttributeValueCount(
      try lookup(handle), kAXChildrenAttribute as CFString, &count)
    guard status == .success else { throw Self.error(status) }
    return count
  }

  public func childPage(of handle: UIHandle, offset: Int, count: Int) throws -> [UIHandle] {
    guard offset >= 0, count > 0, count <= 64 else {
      throw UIBackendError.unavailable("Invalid bounded children page")
    }
    var raw: CFArray?
    let status = AXUIElementCopyAttributeValues(
      try lookup(handle), kAXChildrenAttribute as CFString, offset, count, &raw)
    guard status == .success else { throw Self.error(status) }
    guard let raw, case .elements(let elements) = convert(raw), elements.count == count else {
      throw UIBackendError.unavailable("Incomplete children page")
    }
    return elements
  }

  public func sameElement(_ left: UIHandle, _ right: UIHandle) throws -> Bool {
    try CFEqual(lookup(left), lookup(right))
  }

  public func isSettable(_ name: UIAttributeName, of handle: UIHandle) throws -> Bool {
    var settable = DarwinBoolean(false)
    let status = AXUIElementIsAttributeSettable(
      try lookup(handle), name.rawValue as CFString, &settable)
    switch status {
    case .success: return settable.boolValue
    case .noValue, .attributeUnsupported, .invalidUIElement: return false
    default: throw Self.error(status)
    }
  }

  public func perform(_ action: String, on handle: UIHandle) throws {
    let status = AXUIElementPerformAction(try lookup(handle), action as CFString)
    guard status == .success else { throw Self.error(status) }
  }

  public func set(_ name: UIAttributeName, to value: UISettableValue, on handle: UIHandle) throws {
    let raw: CFTypeRef
    switch value {
    case .string(let text): raw = text as CFString
    case .bool(let flag): raw = flag ? kCFBooleanTrue : kCFBooleanFalse
    }
    let status = AXUIElementSetAttributeValue(try lookup(handle), name.rawValue as CFString, raw)
    guard status == .success else { throw Self.error(status) }
  }

  public func pause(seconds: Double) async throws {
    try await Task.sleep(for: .milliseconds(Int(seconds * 1000)))
  }

  public func selectChild(of parent: UIHandle, at index: Int, expected: UIHandle) throws {
    let count = try childCount(of: parent)
    guard index >= 0, index < count, count <= 4096,
      try isSettable(.selectedChildren, of: parent)
    else { throw UIBackendError.unavailable("Verified child selection is unavailable") }
    let page = try childPage(of: parent, offset: index, count: 1)
    guard let child = page.first, try sameElement(child, expected),
      try childCount(of: parent) == count
    else { throw UIBackendError.unavailable("Verified child selection identity changed") }
    let selected = [try lookup(child)] as CFArray
    let status = AXUIElementSetAttributeValue(
      try lookup(parent), kAXSelectedChildrenAttribute as CFString, selected)
    guard status == .success else { throw Self.error(status) }
  }

  public func monotonicSeconds() -> Double { ProcessInfo.processInfo.systemUptime }

  public func captureWindow(pid: Int32, title: String?, frame: UIFrame?) async throws
    -> UICapturedImage
  {
    guard CGPreflightScreenCaptureAccess() else { throw UIBackendError.notPermitted }
    let content = try await SCShareableContent.excludingDesktopWindows(
      false, onScreenWindowsOnly: false)
    let owned = content.windows.filter { $0.owningApplication?.processID == pid }
    guard
      let window = Self.matchWindow(owned.map(WindowCandidate.init), title: title, frame: frame)
        .flatMap({ index in owned.indices.contains(index) ? owned[index] : nil })
    else {
      throw UIBackendError.unavailable("The window is not uniquely available for capture")
    }
    let filter = SCContentFilter(desktopIndependentWindow: window)
    let configuration = SCStreamConfiguration()
    let scale = Double(filter.pointPixelScale)
    configuration.width = max(1, Int(window.frame.width * scale))
    configuration.height = max(1, Int(window.frame.height * scale))
    configuration.showsCursor = false
    let image = try await SCScreenshotManager.captureImage(
      contentFilter: filter, configuration: configuration)
    return UICapturedImage(
      png: try Self.png(image), width: image.width, height: image.height)
  }

  public func currentInputSource() throws -> UIInputSource {
    try Self.describe(TISCopyCurrentKeyboardInputSource().takeRetainedValue())
  }

  public func asciiCapableInputSource() throws -> UIInputSource {
    try Self.describe(TISCopyCurrentASCIICapableKeyboardInputSource().takeRetainedValue())
  }

  public func inputSources() throws -> [UIInputSource] {
    let filter =
      [kTISPropertyInputSourceCategory as String: kTISCategoryKeyboardInputSource as String]
      as CFDictionary
    guard
      let list = TISCreateInputSourceList(filter, false)?.takeRetainedValue() as? [TISInputSource]
    else { throw UIBackendError.unavailable("Cannot enumerate keyboard input sources") }
    return try list.map(Self.describe)
  }

  public func selectInputSource(id: String) throws {
    let filter = [kTISPropertyInputSourceID as String: id] as CFDictionary
    guard
      let list = TISCreateInputSourceList(filter, false)?.takeRetainedValue() as? [TISInputSource],
      let source = list.first
    else { throw UIBackendError.unavailable("Input source is not installed") }
    let status = TISSelectInputSource(source)
    guard status == noErr else {
      throw UIBackendError.unavailable("Input source selection failed (\(status))")
    }
  }

  public func applicationURL(bundleID: String) -> URL? {
    NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID)
  }

  public func launch(applicationAt url: URL, activate: Bool) async throws -> Int32 {
    let configuration = NSWorkspace.OpenConfiguration()
    configuration.activates = activate
    configuration.promptsUserIfNeeded = false
    return try await NSWorkspace.shared.openApplication(at: url, configuration: configuration)
      .processIdentifier
  }

  public func terminate(pid: Int32, force: Bool) -> Bool {
    guard let app = NSRunningApplication(processIdentifier: pid) else { return false }
    return force ? app.forceTerminate() : app.terminate()
  }

  // MARK: Boundary conversion

  private func register(_ element: AXUIElement) -> UIHandle {
    elements.append(element)
    return elements.count - 1
  }

  private func lookup(_ handle: UIHandle) throws -> AXUIElement {
    guard elements.indices.contains(handle) else {
      throw UIBackendError.unavailable("Unknown accessibility handle")
    }
    return elements[handle]
  }

  func convert(_ raw: CFTypeRef) -> UIAttributeValue {
    let type = CFGetTypeID(raw)
    if type == CFStringGetTypeID(), let text = raw as? String { return .string(text) }
    if type == CFURLGetTypeID(), let url = raw as? URL { return .string(url.absoluteString) }
    if type == CFBooleanGetTypeID(), let flag = raw as? Bool { return .bool(flag) }
    if type == CFNumberGetTypeID(), let number = raw as? Double { return .number(number) }
    if type == AXUIElementGetTypeID(), let element = Self.element(raw) {
      return .element(register(element))
    }
    if type == AXValueGetTypeID(), let value = Self.axValue(raw) { return Self.geometry(value) }
    if type == CFArrayGetTypeID(), let items = raw as? [AnyObject] {
      let converted = items.compactMap { item -> AXUIElement? in
        CFGetTypeID(item) == AXUIElementGetTypeID() ? Self.element(item) : nil
      }
      guard converted.count == items.count else { return .unsupported }
      return .elements(converted.map(register))
    }
    return .unsupported
  }

  static func element(_ raw: CFTypeRef) -> AXUIElement? {
    (([raw] as NSArray) as? [AXUIElement])?.first
  }

  static func axValue(_ raw: CFTypeRef) -> AXValue? {
    (([raw] as NSArray) as? [AXValue])?.first
  }

  static func geometry(_ value: AXValue) -> UIAttributeValue {
    switch AXValueGetType(value) {
    case .cgPoint:
      var point = CGPoint.zero
      guard AXValueGetValue(value, .cgPoint, &point) else { return .unsupported }
      return .point(UIPoint(x: point.x, y: point.y))
    case .cgSize:
      var size = CGSize.zero
      guard AXValueGetValue(value, .cgSize, &size) else { return .unsupported }
      return .size(UISize(width: size.width, height: size.height))
    default: return .unsupported
    }
  }

  static func error(_ status: AXError) -> UIBackendError {
    status == .apiDisabled ? .notPermitted : .accessibility(status.rawValue)
  }

  static func describe(_ source: TISInputSource) throws -> UIInputSource {
    guard let id = string(source, kTISPropertyInputSourceID) else {
      throw UIBackendError.unavailable("Input source has no identifier")
    }
    return UIInputSource(
      id: id, asciiCapable: flag(source, kTISPropertyInputSourceIsASCIICapable),
      selectable: flag(source, kTISPropertyInputSourceIsSelectCapable))
  }

  static func string(_ source: TISInputSource, _ key: CFString) -> String? {
    guard let pointer = TISGetInputSourceProperty(source, key) else { return nil }
    return Unmanaged<CFString>.fromOpaque(pointer).takeUnretainedValue() as String
  }

  static func flag(_ source: TISInputSource, _ key: CFString) -> Bool {
    guard let pointer = TISGetInputSourceProperty(source, key) else { return false }
    return CFBooleanGetValue(Unmanaged<CFBoolean>.fromOpaque(pointer).takeUnretainedValue())
  }

  /// Window identity reduced to plain values so matching is testable without capture.
  struct WindowCandidate: Equatable, Sendable {
    let title: String?
    let frame: CGRect

    init(title: String?, frame: CGRect) {
      self.title = title
      self.frame = frame
    }

    init(_ window: SCWindow) { self.init(title: window.title, frame: window.frame) }
  }

  static let frameTolerance = 2.0

  /// Prefer an exact title and accessibility frame match; fall back to a unique
  /// frame or a unique title. Ambiguity returns nil instead of guessing.
  static func matchWindow(_ candidates: [WindowCandidate], title: String?, frame: UIFrame?)
    -> Int?
  {
    func near(_ candidate: WindowCandidate) -> Bool {
      guard let frame else { return false }
      return abs(candidate.frame.origin.x - frame.x) <= frameTolerance
        && abs(candidate.frame.origin.y - frame.y) <= frameTolerance
        && abs(candidate.frame.width - frame.width) <= frameTolerance
        && abs(candidate.frame.height - frame.height) <= frameTolerance
    }
    let indexed = Array(candidates.enumerated())
    let both = indexed.filter { near($0.element) && title != nil && $0.element.title == title }
    if both.count == 1 { return both[0].offset }
    let byFrame = indexed.filter { near($0.element) }
    if byFrame.count == 1 { return byFrame[0].offset }
    let byTitle = indexed.filter { title != nil && $0.element.title == title }
    return byTitle.count == 1 ? byTitle[0].offset : nil
  }

  static func png(_ image: CGImage) throws -> Data {
    let data = NSMutableData()
    guard
      let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil)
    else { throw UIBackendError.unavailable("Cannot create PNG encoder") }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw UIBackendError.unavailable("PNG encoding failed")
    }
    return data as Data
  }
}
