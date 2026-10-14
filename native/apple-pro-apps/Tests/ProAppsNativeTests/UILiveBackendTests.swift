import AppKit
import ApplicationServices
import Foundation
import Testing

@testable import ProAppsCore

/// Exercises the real Accessibility / TIS / ScreenCaptureKit boundary against the
/// synthetic accessory fixture only. No Apple production app is launched and the
/// keyboard input source is only re-selected to its current value (a no-op).
@MainActor
struct UILiveBackendTests {
  struct Fixture {
    let backend: LiveUIBackend
    let bundleID: String
    let bundle: URL
    let pid: Int32
    let app: UIHandle
    let window: UIHandle
  }

  /// Minimal accessory (LSUIElement) bundle metadata for the synthetic fixture.
  struct BundleInfo: Encodable {
    let identifier: String
    let executable = "UIFixture"
    let packageType = "APPL"
    let accessory = true

    enum CodingKeys: String, CodingKey {
      case identifier = "CFBundleIdentifier"
      case executable = "CFBundleExecutable"
      case packageType = "CFBundlePackageType"
      case accessory = "LSUIElement"
    }
  }

  func launchFixture(explicitScratch: Bool = false) async throws -> Fixture {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "ui-fixture-\(UUID().uuidString)")
    let bundle = root.appendingPathComponent("UIFixture.app")
    let macOS = bundle.appendingPathComponent("Contents/MacOS")
    try FileManager.default.createDirectory(at: macOS, withIntermediateDirectories: true)
    try FileManager.default.copyItem(
      at: Bundle(for: TestBundle.self).bundleURL.deletingLastPathComponent()
        .appendingPathComponent("apple-pro-apps-ui-fixture"),
      to: macOS.appendingPathComponent("UIFixture"))
    let bundleID = "dev.apple-pro-apps.ui-fixture.\(UUID().uuidString.lowercased())"
    let encoder = PropertyListEncoder()
    encoder.outputFormat = .xml
    try encoder.encode(BundleInfo(identifier: bundleID))
      .write(to: bundle.appendingPathComponent("Contents/Info.plist"))
    let backend = LiveUIBackend()
    let pid: Int32
    if explicitScratch {
      let configuration = NSWorkspace.OpenConfiguration()
      configuration.activates = false
      configuration.promptsUserIfNeeded = false
      var environment = ProcessInfo.processInfo.environment
      environment["UI_FIXTURE_SCRATCH"] = root.path
      environment["TMPDIR"] = root.path
      environment["TMP"] = root.path
      environment["TEMP"] = root.path
      configuration.environment = environment
      pid = try await NSWorkspace.shared.openApplication(at: bundle, configuration: configuration)
        .processIdentifier
    } else {
      pid = try await backend.launch(applicationAt: bundle, activate: false)
    }
    let app = backend.applicationElement(pid: pid)
    var window: UIHandle?
    for _ in 0..<50 where window == nil {
      do {
        if case .elements(let windows) = try backend.attribute(.windows, of: app) {
          window = try windows.first {
            try backend.attribute(.title, of: $0) == .string("UI Fixture")
          }
        }
      } catch UIBackendError.accessibility(AXError.cannotComplete.rawValue) {
        // The fixture is still launching and cannot answer Accessibility yet.
      }
      if window == nil { try await backend.pause(seconds: 0.1) }
    }
    return Fixture(
      backend: backend, bundleID: bundleID, bundle: root, pid: pid, app: app,
      window: try #require(window))
  }

  @Test func pagesSyntheticFixtureChildrenWithoutForegroundChanges() async throws {
    let frontmost = LiveUIBackend().frontmostBundleID()
    let fixture = try await launchFixture(explicitScratch: true)
    do {
      let backend = fixture.backend
      #expect(backend.frontmostBundleID() == frontmost)
      let scratch = try element(fixture, identifier: "fixture-scratch")
      let value = try backend.attribute(.value, of: scratch)
      guard case .string(let observation) = value else {
        throw UIBackendError.unavailable("Fixture scratch observation missing")
      }
      let fields = observation.components(separatedBy: "\n")
      try #require(fields.count == 4)
      #expect(fields[0] == fixture.bundle.path)
      #expect(fields[1] == fixture.bundle.path)
      #expect(fields[3] == ProcessInfo.processInfo.environment["LLVM_PROFILE_FILE"])
      print("Fixture actual scratch/cwd/Foundation/profile: \(observation)")
      let descriptors = try await Runner.run(
        URL(fileURLWithPath: "/usr/sbin/lsof"), ["-a", "-p", String(fixture.pid), "-Fn"],
        timeout: .seconds(10))
      try #require(descriptors.status == 0)
      #expect(descriptors.stdout.contains("n\(fixture.bundle.path)/fixture-open-descriptor.txt"))
      #expect(descriptors.stdout.contains("n\(fixture.bundle.path)\n"))
      let table = try element(fixture, identifier: "fixture-table")
      let count = try backend.childCount(of: table)
      try #require(count >= 2 && count <= 64)
      let first = try backend.childPage(of: table, offset: 0, count: count)
      let second = try backend.childPage(of: table, offset: 0, count: count)
      let identities = try zip(first, second).map { try backend.sameElement($0.0, $0.1) }
      #expect(identities.allSatisfy { $0 })
      #expect(try !backend.sameElement(first[0], first[1]))
      #expect(try backend.childCount(of: table) == count)
      print("Fixture selected children: \(try backend.attribute(.selectedChildren, of: table))")
      print("Fixture first row URL: \(try backend.attribute(.url, of: first[0]))")
      #expect(backend.frontmostBundleID() == frontmost)
    } catch {
      await #expect(throws: Never.self) { try await stop(fixture) }
      throw error
    }
    try await stop(fixture)
  }

  func stop(_ fixture: Fixture, force: Bool = false) async throws {
    #expect(fixture.backend.terminate(pid: fixture.pid, force: force))
    for _ in 0..<50 where fixture.backend.runningProcess(bundleID: fixture.bundleID) != nil {
      try await fixture.backend.pause(seconds: 0.1)
    }
    #expect(fixture.backend.runningProcess(bundleID: fixture.bundleID) == nil)
    try FileManager.default.removeItem(at: fixture.bundle)
  }

  func element(_ fixture: Fixture, identifier: String) throws -> UIHandle {
    var queue = [fixture.window]
    while !queue.isEmpty {
      let node = queue.removeFirst()
      if try fixture.backend.attribute(.identifier, of: node) == .string(identifier) {
        return node
      }
      if case .elements(let children) = try fixture.backend.attribute(.children, of: node) {
        queue += children
      }
    }
    throw UIBackendError.unavailable("fixture element missing: \(identifier)")
  }

  func waitForStatus(_ fixture: Fixture, _ expected: String) async throws -> Bool {
    let status = try element(fixture, identifier: "fixture-status")
    for _ in 0..<50 {
      if try fixture.backend.attribute(.value, of: status) == .string(expected) { return true }
      try await fixture.backend.pause(seconds: 0.1)
    }
    return false
  }

  @Test func readsTypedAttributesAndPerformsBackgroundActions() async throws {
    let fixture = try await launchFixture()
    let backend = fixture.backend
    #expect(backend.runningProcess(bundleID: fixture.bundleID)?.pid == fixture.pid)
    #expect(backend.runningProcess(bundleID: "dev.apple-pro-apps.absent") == nil)
    #expect(try backend.attribute(.role, of: fixture.window) == .string("AXWindow"))
    guard case .point = try backend.attribute(.position, of: fixture.window) else {
      Issue.record("expected a point")
      return
    }
    guard case .size = try backend.attribute(.size, of: fixture.window) else {
      Issue.record("expected a size")
      return
    }
    guard case .element = try backend.attribute(.menuBar, of: fixture.app) else {
      Issue.record("expected a menu bar element")
      return
    }
    let button = try element(fixture, identifier: "fixture-button")
    #expect(try backend.attribute(.enabled, of: button) == .bool(true))
    #expect(try backend.attribute(.selected, of: button) == .missing)
    #expect(try backend.actions(of: button).contains("AXPress"))
    #expect(try backend.isSettable(.value, of: button) == false)
    try backend.perform("AXPress", on: button)
    #expect(try await waitForStatus(fixture, "button-pressed"))
    #expect(try backend.attribute(.title, of: button) == .string("Pressed"))
    let checkbox = try element(fixture, identifier: "fixture-checkbox")
    #expect(try backend.attribute(.value, of: checkbox) == .number(0))
    let field = try element(fixture, identifier: "fixture-field")
    #expect(try backend.isSettable(.value, of: field))
    try backend.set(.value, to: .string("/Volumes/ascii/パス"), on: field)
    #expect(try backend.attribute(.value, of: field) == .string("/Volumes/ascii/パス"))
    #expect(throws: UIBackendError.self) { try backend.perform("AXPress", on: fixture.window) }
    #expect(throws: UIBackendError.self) {
      try backend.set(.selected, to: .bool(true), on: fixture.window)
    }
    #expect(throws: UIBackendError.unavailable("Unknown accessibility handle")) {
      try backend.attribute(.role, of: 9_999_999)
    }
    #expect(try backend.actions(of: fixture.app).isEmpty)
    try await stop(fixture)
  }

  @Test func selectsTableRowsAndMenuItemsWithoutActivation() async throws {
    let fixture = try await launchFixture()
    let backend = fixture.backend
    let table = try element(fixture, identifier: "fixture-table")
    guard case .elements(let rows) = try backend.attribute(.children, of: table),
      let row = rows.first
    else {
      Issue.record("expected table rows")
      return
    }
    #expect(try backend.isSettable(.selected, of: row))
    try backend.set(.selected, to: .bool(true), on: row)
    #expect(try await waitForStatus(fixture, "selected-alpha.fcpbundle"))
    guard case .element(let bar) = try backend.attribute(.menuBar, of: fixture.app),
      case .elements(let barItems) = try backend.attribute(.children, of: bar)
    else {
      Issue.record("expected menu bar items")
      return
    }
    let file = try #require(
      try barItems.first { try backend.attribute(.title, of: $0) == .string("File") })
    guard case .elements(let fileMenus) = try backend.attribute(.children, of: file),
      let fileMenu = fileMenus.first,
      case .elements(let fileItems) = try backend.attribute(.children, of: fileMenu)
    else {
      Issue.record("expected File menu")
      return
    }
    let new = try #require(
      try fileItems.first { try backend.attribute(.title, of: $0) == .string("New") })
    guard case .elements(let newMenus) = try backend.attribute(.children, of: new),
      let newMenu = newMenus.first,
      case .elements(let newItems) = try backend.attribute(.children, of: newMenu)
    else {
      Issue.record("expected New submenu")
      return
    }
    let thing = try #require(
      try newItems.first { try backend.attribute(.title, of: $0) == .string("Thing") })
    try backend.perform("AXPress", on: thing)
    #expect(try await waitForStatus(fixture, "menu-selected"))
    try await stop(fixture, force: true)
  }

  @Test func capturesABackgroundWindowAsPNG() async throws {
    let fixture = try await launchFixture()
    let backend = fixture.backend
    guard case .point(let origin) = try backend.attribute(.position, of: fixture.window),
      case .size(let size) = try backend.attribute(.size, of: fixture.window)
    else {
      Issue.record("expected geometry")
      return
    }
    let image = try await backend.captureWindow(
      pid: fixture.pid, title: "UI Fixture",
      frame: UIFrame(x: origin.x, y: origin.y, width: size.width, height: size.height))
    #expect(image.width > 0 && image.height > 0)
    #expect(image.png.prefix(4) == Data([0x89, 0x50, 0x4E, 0x47]))
    await #expect(throws: UIBackendError.self) {
      try await backend.captureWindow(pid: 1, title: "absent", frame: nil)
    }
    try await stop(fixture)
  }

  @Test func readsAndReselectsTheCurrentInputSourceOnly() throws {
    let backend = LiveUIBackend()
    let current = try backend.currentInputSource()
    #expect(!current.id.isEmpty)
    #expect(try backend.asciiCapableInputSource().asciiCapable)
    #expect(try backend.inputSources().contains { $0.id == current.id })
    try backend.selectInputSource(id: current.id)
    #expect(try backend.currentInputSource().id == current.id)
    #expect(throws: UIBackendError.unavailable("Input source is not installed")) {
      try backend.selectInputSource(id: "dev.apple-pro-apps.absent-input-source")
    }
  }

  @Test func reportsProcessClockAndLaunchServicesState() async throws {
    let backend = LiveUIBackend()
    #expect(backend.applicationURL(bundleID: "dev.apple-pro-apps.absent") == nil)
    #expect(backend.terminate(pid: -1, force: false) == false)
    _ = backend.frontmostBundleID()
    let start = backend.monotonicSeconds()
    try await backend.pause(seconds: 0.01)
    #expect(backend.monotonicSeconds() >= start)
  }

  @Test func convertsBoundaryValuesConservatively() {
    let backend = LiveUIBackend()
    #expect(backend.convert("text" as CFString) == .string("text"))
    #expect(
      backend.convert(URL(fileURLWithPath: "/synthetic-folder") as CFURL)
        == .string("file:///synthetic-folder"))
    #expect(backend.convert(kCFBooleanTrue) == .bool(true))
    #expect(backend.convert(NSNumber(value: 2.5)) == .number(2.5))
    #expect(backend.convert(Data([1]) as CFData) == .unsupported)
    #expect(backend.convert(["mixed"] as CFArray) == .unsupported)
    #expect(backend.convert([] as CFArray) == .elements([]))
    var range = CFRange(location: 1, length: 2)
    let rangeValue = AXValueCreate(.cfRange, &range)
    #expect(rangeValue.map(LiveUIBackend.geometry) == .unsupported)
    #expect(LiveUIBackend.error(.apiDisabled) == .notPermitted)
    #expect(LiveUIBackend.error(.cannotComplete) == .accessibility(AXError.cannotComplete.rawValue))
  }

  @Test func boundedChildReadsRejectInvalidHandlesAndRanges() throws {
    let backend = LiveUIBackend()
    #expect(throws: UIBackendError.self) { try backend.childCount(of: -1) }
    #expect(throws: UIBackendError.self) { try backend.childPage(of: -1, offset: 0, count: 1) }
    #expect(throws: UIBackendError.self) { try backend.childPage(of: -1, offset: -1, count: 1) }
    #expect(throws: UIBackendError.self) { try backend.childPage(of: -1, offset: 0, count: 65) }
    #expect(throws: UIBackendError.self) { try backend.sameElement(-1, -1) }
  }

  @Test func matchesCaptureWindowsWithoutGuessing() {
    let a = LiveUIBackend.WindowCandidate(
      title: "Main", frame: CGRect(x: 0, y: 0, width: 10, height: 10))
    let b = LiveUIBackend.WindowCandidate(
      title: "Main", frame: CGRect(x: 50, y: 0, width: 10, height: 10))
    let c = LiveUIBackend.WindowCandidate(
      title: "Other", frame: CGRect(x: 50, y: 0, width: 10, height: 10))
    let frame = UIFrame(x: 50, y: 1, width: 10, height: 10)
    #expect(LiveUIBackend.matchWindow([a, b, c], title: "Main", frame: frame) == 1)
    #expect(LiveUIBackend.matchWindow([a, b], title: nil, frame: frame) == 1)
    #expect(LiveUIBackend.matchWindow([a, c], title: "Other", frame: nil) == 1)
    #expect(LiveUIBackend.matchWindow([a, b], title: "Main", frame: nil) == nil)
    #expect(LiveUIBackend.matchWindow([b, c], title: nil, frame: frame) == nil)
  }
}
