import Foundation
import Testing

@testable import ProAppsCore

/// Synthetic accessibility tree. Handles are fixed literals so expectations stay explicit.
@MainActor
final class FakeUIBackend: UIBackend {
  var nameObservation: ((UIHandle) throws -> [String])?
  func attributeNames(of handle: UIHandle) throws -> [String] {
    if let nameObservation { return try nameObservation(handle) }
    return (attributes[handle] ?? [:]).keys.map(\.rawValue)
  }
  var identityReads = 0
  var identity: (UIHandle, Int) throws -> Int32 = { _, _ in 42 }
  func processID(of handle: UIHandle) throws -> Int32 {
    identityReads += 1
    return try identity(handle, identityReads)
  }
  var attributes: [UIHandle: [UIAttributeName: UIAttributeValue]] = [:]
  var actionNames: [UIHandle: [String]] = [:]
  var settable: Set<String> = []
  var attributeFailures: [UIHandle: UIBackendError] = [:]
  var namedFailures: [String: UIBackendError] = [:]
  var actionFailures: [UIHandle: UIBackendError] = [:]
  var settableFailures: [UIHandle: UIBackendError] = [:]
  var performFailure: UIBackendError?
  var setFailure: UIBackendError?
  var ignoresWrites = false
  var processes: [String: UIRunningProcess] = [:]
  var frontmost: [String?] = ["dev.terminal"]
  var clock = 0.0
  var onPause: (FakeUIBackend) -> Void = { _ in }
  var onPerform: (FakeUIBackend, String, UIHandle) -> Void = { _, _, _ in }
  var performed: [String] = []
  var written: [String] = []
  var capture: Result<UICapturedImage, UIBackendError> = .success(
    UICapturedImage(png: Data([0x89, 0x50]), width: 2, height: 1))
  var captureRequest: (Int32, String?, UIFrame?)?
  var current = UIInputSource(id: "jp.kana", asciiCapable: false, selectable: true)
  var ascii = UIInputSource(id: "abc", asciiCapable: true, selectable: true)
  var sources: [UIInputSource] = [
    UIInputSource(id: "jp.kana", asciiCapable: false, selectable: true),
    UIInputSource(id: "abc", asciiCapable: true, selectable: true),
    UIInputSource(id: "locked", asciiCapable: true, selectable: false),
  ]
  var inputFailure: UIBackendError?
  var installed: [String: URL] = [:]
  var launchResult: Result<Int32, UIBackendError> = .success(77)
  var launched: [(URL, Bool)] = []
  var terminateAccepted = true
  var exitAfterPauses: Int?
  var terminated: [(Int32, Bool)] = []
  var foregroundProcess: UIRunningProcess?
  var foregroundReads = 0
  var onForegroundRead: (FakeUIBackend) -> Void = { _ in }
  var pauseFailure: (any Error)?
  var activationAccepted = true
  var activated: [UIRunningProcess] = []
  var onActivate: (FakeUIBackend, UIRunningProcess) -> Void = { backend, process in
    backend.foregroundProcess = process
    backend.frontmost = [process.bundleID]
  }

  static let fcp = "com.apple.FinalCutApp"

  init() {}

  /// App 1 with a main window (10), a dialog (20) and a menu bar (30).
  static func standard() -> FakeUIBackend {
    let backend = FakeUIBackend()
    backend.processes[fcp] = UIRunningProcess(pid: 42, bundleID: fcp)
    backend.attributes = [
      1: [.windows: .elements([10, 20]), .menuBar: .element(30)],
      10: [
        .role: .string("AXWindow"), .subrole: .string("AXStandardWindow"), .title: .string("Main"),
        .main: .bool(true), .minimized: .bool(false), .modal: .number(0),
        .position: .point(UIPoint(x: 1, y: 2)), .size: .size(UISize(width: 30, height: 40)),
        .children: .elements([11, 12, 13]),
      ],
      11: [.role: .string("AXSheet"), .children: .elements([14])],
      12: [
        .role: .string("AXButton"), .identifier: .string("ok"), .title: .string("OK"),
        .enabled: .bool(true),
      ],
      13: [.role: .string("AXTextField"), .identifier: .string("name"), .value: .string("old")],
      14: [.role: .string("AXButton"), .title: .string("Cancel")],
      20: [
        .role: .string("AXWindow"), .subrole: .string("AXDialog"),
        .title: .string("ライブラリを開く"), .children: .elements([21]),
      ],
      21: [.role: .string("AXTable"), .children: .elements([22, 23])],
      22: [.role: .string("AXRow"), .selected: .bool(false), .children: .elements([24])],
      23: [.role: .string("AXRow"), .selected: .bool(false), .children: .elements([25])],
      24: [.role: .string("AXStaticText"), .value: .string("alpha.fcpbundle")],
      25: [.role: .string("AXStaticText"), .value: .string("beta.fcpbundle")],
      30: [.children: .elements([31])],
      31: [.title: .string("File"), .children: .elements([32])],
      32: [.children: .elements([33, 35, 36, 37, 38])],
      33: [.title: .string("New"), .enabled: .bool(true), .children: .elements([34])],
      34: [.children: .elements([39])],
      35: [.title: .string("Disabled"), .enabled: .bool(false)],
      36: [.title: .string("")],
      37: [.title: .string("Dup")],
      38: [.title: .string("Dup")],
      39: [.title: .string("Library…"), .enabled: .number(1)],
    ]
    backend.actionNames = [12: ["AXPress"], 14: ["AXPress"], 39: ["AXPress"]]
    backend.settable = ["13:AXValue", "22:AXSelected", "23:AXSelected"]
    return backend
  }

  func runningProcess(bundleID: String) -> UIRunningProcess? { processes[bundleID] }
  func frontmostProcess() -> UIRunningProcess? {
    foregroundReads += 1
    onForegroundRead(self)
    return foregroundProcess
  }
  func activate(process: UIRunningProcess) -> Bool {
    activated.append(process)
    if activationAccepted { onActivate(self, process) }
    return activationAccepted
  }

  func frontmostBundleID() -> String? {
    frontmost.count > 1 ? frontmost.removeFirst() : frontmost.first ?? nil
  }

  func applicationElement(pid: Int32) -> UIHandle { 1 }

  func attribute(_ name: UIAttributeName, of handle: UIHandle) throws -> UIAttributeValue {
    if let failure = attributeFailures[handle] { throw failure }
    if let failure = namedFailures["\(handle):\(name.rawValue)"] { throw failure }
    return attributes[handle]?[name] ?? .missing
  }

  func actions(of handle: UIHandle) throws -> [String] {
    if let failure = actionFailures[handle] { throw failure }
    return actionNames[handle] ?? []
  }

  func isSettable(_ name: UIAttributeName, of handle: UIHandle) throws -> Bool {
    if let failure = settableFailures[handle] { throw failure }
    return settable.contains("\(handle):\(name.rawValue)")
  }

  func perform(_ action: String, on handle: UIHandle) throws {
    if let performFailure { throw performFailure }
    performed.append("\(action)@\(handle)")
    onPerform(self, action, handle)
  }

  func set(_ name: UIAttributeName, to value: UISettableValue, on handle: UIHandle) throws {
    if let setFailure { throw setFailure }
    written.append("\(name.rawValue)@\(handle)")
    guard !ignoresWrites else { return }
    switch value {
    case .string(let text): attributes[handle, default: [:]][name] = .string(text)
    case .bool(let flag): attributes[handle, default: [:]][name] = .bool(flag)
    }
  }

  func pause(seconds: Double) async throws {
    if let pauseFailure { throw pauseFailure }
    clock += seconds
    if let remaining = exitAfterPauses {
      if remaining <= 1 { processes[Self.fcp] = nil }
      exitAfterPauses = remaining - 1
    }
    onPause(self)
  }

  func monotonicSeconds() -> Double { clock }

  func captureWindow(pid: Int32, title: String?, frame: UIFrame?) async throws -> UICapturedImage {
    captureRequest = (pid, title, frame)
    return try capture.get()
  }

  func currentInputSource() throws -> UIInputSource {
    if let inputFailure { throw inputFailure }
    return current
  }

  func asciiCapableInputSource() throws -> UIInputSource { ascii }

  func inputSources() throws -> [UIInputSource] { sources }

  func selectInputSource(id: String) throws {
    guard let match = sources.first(where: { $0.id == id }) else {
      throw UIBackendError.unavailable("absent")
    }
    current = match
  }

  func applicationURL(bundleID: String) -> URL? { installed[bundleID] }

  func launch(applicationAt url: URL, activate: Bool) async throws -> Int32 {
    launched.append((url, activate))
    let pid = try launchResult.get()
    processes[Self.fcp] = UIRunningProcess(pid: pid, bundleID: Self.fcp)
    return pid
  }

  func terminate(pid: Int32, force: Bool) -> Bool {
    terminated.append((pid, force))
    return terminateAccepted
  }
}

@MainActor
struct UIAutomationTests {
  @Test func selectionCapabilitiesAreReadOnlyAndPreserveUnknown() throws {
    let backend = FakeUIBackend.standard()
    backend.settable = ["12:AXSelected", "13:AXSelectedChildren"]
    let automation = UIAutomation(backend: backend)
    let selected = try automation.summary(12, path: [1])
    let children = try automation.summary(13, path: [2])
    #expect(selected.selectedSettable == true && selected.selectedChildrenSettable == false)
    #expect(children.selectedSettable == false && children.selectedChildrenSettable == true)
    backend.settableFailures[12] = .accessibility(-25205)
    let unavailable = try automation.summary(12, path: [1])
    #expect(unavailable.selectedSettable == nil && unavailable.selectedChildrenSettable == nil)
    #expect(unavailable.unreadable.contains("AXSelectedChildrenSettable"))
    #expect(backend.written.isEmpty && backend.performed.isEmpty)
    #expect(
      try JSONDecoder().decode(UIElementSummary.self, from: JSONEncoder().encode(selected))
        == selected)
  }

  let target = UITarget(app: .finalCutPro)
  let main = UIWindowLocator(title: "Main")
  let dialog = UIWindowLocator(title: "ライブラリを開く")

  func run(_ backend: FakeUIBackend, _ request: UIRequest) async throws -> UIResponse {
    try await UIAutomation(backend: backend).run(request)
  }

  @Test func listsWindowsWithSheetsAndFocusEvidence() async throws {
    let backend = FakeUIBackend.standard()
    backend.frontmost = ["dev.terminal", "dev.terminal"]
    let response = try await run(backend, .windows(target))
    #expect(response.running == true)
    #expect(response.pid == 42)
    let windows = try #require(response.windows)
    #expect(windows.count == 2)
    #expect(windows[0].title == "Main")
    #expect(windows[0].main == true)
    #expect(windows[0].modal == false)
    #expect(windows[0].frame == UIFrame(x: 1, y: 2, width: 30, height: 40))
    #expect(windows[0].sheets.map(\.path) == [[0]])
    #expect(windows[1].subrole == "AXDialog")
    #expect(windows[1].frame == nil)
    #expect(response.focus.frontmostChanged == false)
  }

  @Test func reportsANotRunningAppWithoutWindows() async throws {
    let backend = FakeUIBackend()
    let response = try await run(backend, .windows(target))
    #expect(response.running == false)
    #expect(response.windows == [])
  }

  @Test func reportsFrontmostChanges() async throws {
    let backend = FakeUIBackend.standard()
    backend.frontmost = ["dev.terminal", "com.apple.FinalCutApp"]
    let response = try await run(backend, .windows(target))
    #expect(response.focus.frontmostChanged)
    #expect(response.focus.frontmostAfter == "com.apple.FinalCutApp")
  }

  @Test func inspectsBoundedSubtrees() async throws {
    let backend = FakeUIBackend.standard()
    let shallow = try await run(
      backend, .inspect(target, window: main, root: nil, maxDepth: 1, maxElements: 50))
    #expect(shallow.elements?.map(\.path) == [[], [0], [1], [2]])
    #expect(shallow.truncated == true)
    let bounded = try await run(
      backend, .inspect(target, window: main, root: nil, maxDepth: 5, maxElements: 2))
    #expect(bounded.elements?.count == 2)
    #expect(bounded.truncated == true)
    let full = try await run(
      backend, .inspect(target, window: main, root: nil, maxDepth: 5, maxElements: 50))
    #expect(full.truncated == false)
    let field = try #require(full.elements?.first { $0.identifier == "name" })
    #expect(field.valueSettable)
    #expect(field.value == "old")
    let rooted = try await run(
      backend,
      .inspect(
        target, window: main, root: UIElementLocator(role: "AXSheet"), maxDepth: 3,
        maxElements: 10))
    #expect(rooted.elements?.map(\.path) == [[0], [0, 0]])
    #expect(rooted.elements?.last?.actions == ["AXPress"])
  }

  @Test(arguments: [
    (UIWindowLocator(subrole: "AXDialog"), true),
    (UIWindowLocator(index: 1), true),
    (UIWindowLocator(index: 0, title: "Main"), true),
    (UIWindowLocator(index: 5), false),
    (UIWindowLocator(title: "Absent"), false),
  ])
  func resolvesWindowLocators(_ locator: UIWindowLocator, _ found: Bool) async throws {
    let backend = FakeUIBackend.standard()
    let request = UIRequest.inspect(target, window: locator, root: nil, maxDepth: 1, maxElements: 1)
    if found {
      #expect(try await run(backend, request).elements?.count == 1)
    } else {
      await #expect(throws: ProAppsError.self) { try await run(backend, request) }
    }
  }

  @Test func rejectsAmbiguousWindowsAndStoppedApps() async throws {
    let backend = FakeUIBackend.standard()
    let ambiguous = UIRequest.inspect(
      target, window: UIWindowLocator(), root: nil, maxDepth: 1, maxElements: 1)
    await #expect(throws: ProAppsError.invalid("Window locator is ambiguous; add title or index")) {
      try await run(backend, ambiguous)
    }
    let stopped = FakeUIBackend()
    await #expect(throws: ProAppsError.unavailable("Selected app edition is not running")) {
      try await run(stopped, .inspect(target, window: main, root: nil, maxDepth: 1, maxElements: 1))
    }
    await #expect(throws: ProAppsError.invalid("Bundle ID does not match the selected app")) {
      try await run(
        backend, .windows(UITarget(app: .finalCutPro, bundleID: "com.apple.Compressor")))
    }
  }

  @Test func selectsRowsByContainedTextAndVerifiesReadBack() async throws {
    let backend = FakeUIBackend.standard()
    let response = try await run(
      backend,
      .set(
        target, window: dialog,
        target: UIElementLocator(role: "AXRow", containsText: "alpha.fcpbundle"),
        attribute: .selected, value: .bool(true)))
    #expect(response.effectVerified == true)
    #expect(response.element?.path == [0, 0])
    #expect(response.element?.selected == true)
    #expect(backend.written == ["AXSelected@22"])
  }

  @Test func reportsUnverifiedWritesAndRejectsUnsettableTargets() async throws {
    let backend = FakeUIBackend.standard()
    backend.ignoresWrites = true
    let unverified = try await run(
      backend,
      .set(
        target, window: main, target: UIElementLocator(identifier: "name"), attribute: .value,
        value: .string("new")))
    #expect(unverified.effectVerified == false)
    await #expect(
      throws: ProAppsError.invalid("The selected attribute is not settable on this element")
    ) {
      try await run(
        backend,
        .set(
          target, window: main, target: UIElementLocator(identifier: "ok"), attribute: .value,
          value: .string("x")))
    }
    backend.setFailure = .accessibility(-25205)
    await #expect(throws: ProAppsError.self) {
      try await run(
        backend,
        .set(
          target, window: main, target: UIElementLocator(identifier: "name"), attribute: .value,
          value: .string("new")))
    }
  }

  @Test func writesTextDirectly() async throws {
    let backend = FakeUIBackend.standard()
    let response = try await run(
      backend,
      .set(
        target, window: main, target: UIElementLocator(path: [2], identifier: "name"),
        attribute: .value, value: .string("/Volumes/ascii/パス")))
    #expect(response.effectVerified == true)
    #expect(response.element?.value == "/Volumes/ascii/パス")
  }

  @Test(arguments: [
    (UIElementLocator(), "Element locator needs a path or at least one attribute"),
    (UIElementLocator(role: "AXRow"), "Element locator matched 2 elements; refine it or add index"),
  ])
  func rejectsInvalidElementLocators(_ locator: UIElementLocator, _ reason: String) async throws {
    let backend = FakeUIBackend.standard()
    await #expect(throws: ProAppsError.invalid(reason)) {
      try await run(backend, .perform(target, window: dialog, target: locator, action: .press))
    }
  }

  @Test(arguments: [
    (UIElementLocator(path: [9]), "No matching element; inspect the window first"),
    (
      UIElementLocator(path: [1], role: "AXTextField"),
      "No matching element; inspect the window first"
    ),
    (UIElementLocator(role: "AXButton", index: 4), "Element index is outside the 2 matches"),
    (UIElementLocator(value: "absent"), "No matching element; inspect the window first"),
  ])
  func reportsMissingElements(_ locator: UIElementLocator, _ reason: String) async throws {
    let backend = FakeUIBackend.standard()
    await #expect(throws: ProAppsError.unavailable(reason)) {
      try await run(backend, .perform(target, window: main, target: locator, action: .press))
    }
  }

  @Test func performsActionsAndReadsBackSurvivingElements() async throws {
    let backend = FakeUIBackend.standard()
    let response = try await run(
      backend,
      .perform(target, window: main, target: UIElementLocator(identifier: "ok"), action: .press))
    #expect(backend.performed == ["AXPress@12"])
    #expect(response.dispatched == true)
    #expect(response.effectVerified == false)
    #expect(response.elementExists == true)
    #expect(response.element?.title == "OK")
    let indexed = try await run(
      backend,
      .perform(
        target, window: main, target: UIElementLocator(role: "AXButton", index: 1), action: .press))
    #expect(indexed.element?.title == "Cancel")
  }

  @Test func reportsElementsAndWindowsThatDisappearAfterActions() async throws {
    let backend = FakeUIBackend.standard()
    backend.actionNames[21] = ["AXPress"]
    backend.onPerform = { backend, _, _ in backend.attributes[1]?[.windows] = .elements([10]) }
    let closed = try await run(
      backend,
      .perform(
        target, window: dialog, target: UIElementLocator(role: "AXTable"), action: .press))
    #expect(closed.elementExists == false)
    #expect(closed.element == nil)
    let fresh = FakeUIBackend.standard()
    fresh.actionNames[21] = ["AXPress"]
    fresh.onPerform = { backend, _, _ in backend.attributes[21]?[.role] = .string("AXList") }
    let changed = try await run(
      fresh,
      .perform(target, window: dialog, target: UIElementLocator(role: "AXTable"), action: .press))
    #expect(changed.elementExists == false)
    let row = FakeUIBackend.standard()
    row.actionNames[23] = ["AXPress"]
    row.onPerform = { backend, _, _ in backend.attributes[21]?[.children] = .elements([22]) }
    let shifted = try await run(
      row,
      .perform(
        target, window: dialog, target: UIElementLocator(role: "AXRow", index: 1), action: .press))
    #expect(shifted.elementExists == false)
  }

  @Test func rejectsUnsupportedActionsAndMapsBackendFailures() async throws {
    let backend = FakeUIBackend.standard()
    await #expect(
      throws: ProAppsError.invalid("Element does not support confirm; available: AXPress")
    ) {
      try await run(
        backend,
        .perform(target, window: main, target: UIElementLocator(identifier: "ok"), action: .confirm)
      )
    }
    backend.performFailure = .notPermitted
    await #expect(throws: UIAutomation<FakeUIBackend>.mapped(.notPermitted)) {
      try await run(
        backend,
        .perform(target, window: main, target: UIElementLocator(identifier: "ok"), action: .press))
    }
    let failing = FakeUIBackend.standard()
    failing.attributeFailures[1] = .accessibility(-25202)
    await #expect(throws: ProAppsError.self) { try await run(failing, .windows(target)) }
  }

  @Test func listsAndSelectsMenus() async throws {
    let backend = FakeUIBackend.standard()
    let top = try await run(backend, .menuList(target, path: []))
    #expect(top.menuItems == [UIMenuItem(title: "File", enabled: nil, hasSubmenu: true)])
    let file = try await run(backend, .menuList(target, path: ["File"]))
    #expect(file.menuItems?.map(\.title) == ["New", "Disabled", "Dup", "Dup"])
    #expect(file.menuItems?.first?.hasSubmenu == true)
    let selected = try await run(backend, .menuSelect(target, path: ["File", "New", "Library…"]))
    #expect(selected.dispatched == true)
    #expect(backend.performed == ["AXPress@39"])
  }

  @Test(arguments: [
    (
      ["File", "Disabled"],
      ProAppsError.unavailable("Menu item is disabled; inspect the app state first")
    ),
    (["File", "Absent"], ProAppsError.unavailable("Menu item not found: Absent")),
    (["File", "Dup"], ProAppsError.invalid("Menu title is ambiguous: Dup")),
    (["File", "Disabled", "Child"], ProAppsError.invalid("Menu item has no submenu: Disabled")),
    ([], ProAppsError.invalid("Menu path is empty")),
  ])
  func rejectsInvalidMenuPaths(_ path: [String], _ error: ProAppsError) async throws {
    let backend = FakeUIBackend.standard()
    await #expect(throws: error) { try await run(backend, .menuSelect(target, path: path)) }
  }

  @Test func requiresAMenuBar() async throws {
    let backend = FakeUIBackend.standard()
    backend.attributes[1]?[.menuBar] = nil
    await #expect(throws: ProAppsError.unavailable("The app exposes no menu bar")) {
      try await run(backend, .menuList(target, path: []))
    }
  }

  @Test func waitsForWindowsAndElementsWithinBounds() async throws {
    let backend = FakeUIBackend.standard()
    let present = try await run(
      backend, .wait(target, window: dialog, target: nil, condition: .exists, timeoutSeconds: 1))
    #expect(present.conditionMet == true)
    #expect(present.elapsedSeconds == 0)
    backend.onPause = { backend in
      if backend.clock >= 0.6 { backend.attributes[1]?[.windows] = .elements([10]) }
    }
    let closed = try await run(
      backend, .wait(target, window: dialog, target: nil, condition: .absent, timeoutSeconds: 5))
    #expect(closed.conditionMet == true)
    let missing = try await run(
      backend,
      .wait(
        target, window: nil, target: UIElementLocator(identifier: "absent"), condition: .exists,
        timeoutSeconds: 0.5))
    #expect(missing.conditionMet == false)
    let fromApp = try await run(
      backend,
      .wait(
        target, window: nil, target: UIElementLocator(identifier: "ok"), condition: .exists,
        timeoutSeconds: 0.5))
    #expect(fromApp.conditionMet == false)
  }

  @Test func treatsBusyAndStoppedAppsAsNotYetPresent() async throws {
    let backend = FakeUIBackend.standard()
    backend.attributeFailures[1] = .accessibility(-25204)
    backend.onPause = { backend in backend.attributeFailures[1] = nil }
    let busy = try await run(
      backend, .wait(target, window: main, target: nil, condition: .exists, timeoutSeconds: 2))
    #expect(busy.conditionMet == true)
    let stopped = try await run(
      FakeUIBackend(),
      .wait(target, window: main, target: nil, condition: .absent, timeoutSeconds: 1))
    #expect(stopped.conditionMet == true)
    let failing = FakeUIBackend.standard()
    failing.attributeFailures[1] = .notPermitted
    await #expect(throws: UIAutomation<FakeUIBackend>.mapped(.notPermitted)) {
      try await run(
        failing, .wait(target, window: main, target: nil, condition: .exists, timeoutSeconds: 1))
    }
  }

  @Test(arguments: [0.0, 31.0])
  func rejectsUnboundedWaits(_ timeout: Double) async throws {
    await #expect(throws: ProAppsError.invalid("Wait timeout must be within 0–30 seconds")) {
      try await run(
        FakeUIBackend.standard(),
        .wait(target, window: nil, target: nil, condition: .exists, timeoutSeconds: timeout))
    }
  }

  @Test func requiresAWaitTarget() async throws {
    await #expect(throws: ProAppsError.invalid("Wait requires a window or element locator")) {
      try await run(
        FakeUIBackend.standard(),
        .wait(target, window: nil, target: nil, condition: .exists, timeoutSeconds: 1))
    }
  }

  @Test func capturesWindowsIntoNewPrivateFiles() async throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "ui-capture-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let output = directory.appendingPathComponent("window.png").path
    let backend = FakeUIBackend.standard()
    let response = try await run(backend, .capture(target, window: main, outputPath: output))
    #expect(response.capture == UICaptureResult(outputPath: output, width: 2, height: 1))
    #expect(backend.captureRequest?.0 == 42)
    #expect(backend.captureRequest?.1 == "Main")
    #expect(backend.captureRequest?.2 == UIFrame(x: 1, y: 2, width: 30, height: 40))
    backend.capture = .success(UICapturedImage(png: Data([0x00]), width: 1, height: 1))
    await #expect(throws: (any Error).self) {
      try await run(backend, .capture(target, window: main, outputPath: output))
    }
    #expect(try Data(contentsOf: URL(fileURLWithPath: output)) == Data([0x89, 0x50]))
    backend.capture = .failure(.notPermitted)
    await #expect(throws: UIAutomation<FakeUIBackend>.mapped(.notPermitted)) {
      try await run(
        backend,
        .capture(
          target, window: main, outputPath: directory.appendingPathComponent("other.png").path))
    }
  }

  @Test func readsAndSelectsInputSources() async throws {
    let backend = FakeUIBackend()
    #expect(try await run(backend, .inputSourceStatus).inputSource?.id == "jp.kana")
    let ascii = try await run(backend, .inputSourceSelect(id: nil, asciiCapable: true))
    #expect(ascii.previousInputSource?.id == "jp.kana")
    #expect(ascii.inputSource?.id == "abc")
    #expect(ascii.dispatched == true)
    #expect(ascii.effectVerified == true)
    let again = try await run(backend, .inputSourceSelect(id: nil, asciiCapable: true))
    #expect(again.dispatched == false)
    let restored = try await run(backend, .inputSourceSelect(id: "jp.kana", asciiCapable: false))
    #expect(restored.inputSource?.id == "jp.kana")
    let same = try await run(backend, .inputSourceSelect(id: "jp.kana", asciiCapable: false))
    #expect(same.dispatched == false)
  }

  @Test(arguments: [
    ("locked", false, ProAppsError.unavailable("Input source is not installed or not selectable")),
    ("absent", false, ProAppsError.unavailable("Input source is not installed or not selectable")),
    ("abc", true, ProAppsError.invalid("Supply exactly one of id or asciiCapable: true")),
  ])
  func rejectsInvalidInputSourceRequests(_ id: String, _ ascii: Bool, _ error: ProAppsError)
    async throws
  {
    await #expect(throws: error) {
      try await run(FakeUIBackend(), .inputSourceSelect(id: id, asciiCapable: ascii))
    }
  }

  @Test func rejectsEmptyInputSourceRequestsAndMapsFailures() async throws {
    await #expect(throws: ProAppsError.invalid("Supply exactly one of id or asciiCapable: true")) {
      try await run(FakeUIBackend(), .inputSourceSelect(id: nil, asciiCapable: false))
    }
    let failing = FakeUIBackend()
    failing.inputFailure = .unavailable("no input sources")
    await #expect(throws: ProAppsError.unavailable("no input sources")) {
      try await run(failing, .inputSourceStatus)
    }
  }

  @Test func launchesOnlyWhenNotRunning() async throws {
    let running = FakeUIBackend.standard()
    let noop = try await run(running, .launch(target, activate: false))
    #expect(noop.dispatched == false)
    #expect(noop.pid == 42)
    let stopped = FakeUIBackend()
    await #expect(throws: ProAppsError.unavailable("Selected app edition is not installed")) {
      try await run(stopped, .launch(target, activate: false))
    }
    let url = URL(fileURLWithPath: "/Applications/Synthetic.app")
    stopped.installed[FakeUIBackend.fcp] = url
    let launched = try await run(stopped, .launch(target, activate: false))
    #expect(launched.pid == 77)
    #expect(launched.dispatched == true)
    #expect(stopped.launched.map(\.1) == [false])
    let failing = FakeUIBackend()
    failing.installed[FakeUIBackend.fcp] = url
    failing.launchResult = .failure(.unavailable("launch refused"))
    await #expect(throws: ProAppsError.unavailable("launch refused")) {
      try await run(failing, .launch(target, activate: true))
    }
  }

  @Test func quitsAndWaitsForExit() async throws {
    let stopped = try await run(FakeUIBackend(), .quit(target, force: false))
    #expect(stopped.terminated == true)
    #expect(stopped.dispatched == false)
    let backend = FakeUIBackend.standard()
    backend.exitAfterPauses = 2
    let quit = try await run(backend, .quit(target, force: true))
    #expect(quit.terminated == true)
    #expect(quit.running == false)
    #expect(backend.terminated.map(\.1) == [true])
    let refused = FakeUIBackend.standard()
    refused.terminateAccepted = false
    let response = try await run(refused, .quit(target, force: false))
    #expect(response.dispatched == false)
    #expect(response.terminated == false)
    let stubborn = FakeUIBackend.standard()
    let waited = try await run(stubborn, .quit(target, force: false))
    #expect(waited.terminated == false)
    #expect(stubborn.clock >= 10)
  }

  @Test func inspectionReportsUnreadableAttributesWithoutAborting() async throws {
    let backend = FakeUIBackend.standard()
    backend.namedFailures["12:AXSubrole"] = .accessibility(-25200)
    backend.namedFailures["11:AXChildren"] = .accessibility(-25200)
    backend.actionFailures[12] = .accessibility(-25200)
    backend.settableFailures[12] = .accessibility(-25200)
    let response = try await run(
      backend, .inspect(target, window: main, root: nil, maxDepth: 5, maxElements: 50))
    let elements = try #require(response.elements)
    #expect(elements.map(\.path) == [[], [0], [1], [2]])
    let button = try #require(elements.first { $0.identifier == "ok" })
    #expect(
      button.unreadable == [
        "AXSubrole", "actions", "settable", "AXSelectedSettable", "AXSelectedChildrenSettable",
      ])
    #expect(button.actions == [])
    #expect(button.valueSettable == false)
    #expect(elements.first?.unreadable == [])
  }

  @Test func matchingTreatsUnreadableAttributesAsAbsent() async throws {
    let backend = FakeUIBackend.standard()
    backend.namedFailures["12:AXIdentifier"] = .accessibility(-25200)
    let response = try await run(
      backend,
      .set(
        target, window: main, target: UIElementLocator(identifier: "name"), attribute: .value,
        value: .string("ok")))
    #expect(response.effectVerified == true)
  }

  @Test func busyAndPermissionFailuresStillPropagateFromSummaries() async throws {
    let busy = FakeUIBackend.standard()
    busy.namedFailures["12:AXTitle"] = .accessibility(-25204)
    await #expect(throws: ProAppsError.unavailable(UIAutomation<FakeUIBackend>.busyReason)) {
      try await run(busy, .inspect(target, window: main, root: nil, maxDepth: 2, maxElements: 9))
    }
    let denied = FakeUIBackend.standard()
    denied.actionFailures[12] = .notPermitted
    await #expect(throws: UIAutomation<FakeUIBackend>.mapped(.notPermitted)) {
      try await run(denied, .inspect(target, window: main, root: nil, maxDepth: 2, maxElements: 9))
    }
  }

  @Test(arguments: [
    (ProAppsError.invalid("bad"), "bad", true),
    (ProAppsError.unavailable("gone"), "gone", false),
    (ProAppsError.timedOut, ProAppsError.timedOut.description, false),
    (ProAppsError.commandFailed(3), ProAppsError.commandFailed(3).description, false),
    (ProAppsError.outputLimit, ProAppsError.outputLimit.description, false),
  ])
  func childFailuresKeepTheirKind(_ error: ProAppsError, _ reason: String, _ invalid: Bool) {
    let outcome = UIChildOutcome.failure(error)
    #expect(outcome.error == reason)
    #expect(outcome.invalid == invalid)
    #expect(outcome.rethrown() == (invalid ? .invalid(reason) : .unavailable(reason)))
    #expect(UIChildOutcome(response: nil, error: nil).rethrown() == nil)
  }

  @Test func mapsBackendErrors() {
    #expect(
      UIAutomation<FakeUIBackend>.mapped(.accessibility(-25204))
        == .unavailable(UIAutomation<FakeUIBackend>.busyReason))
    #expect(
      UIAutomation<FakeUIBackend>.mapped(.accessibility(-25201))
        == .unavailable(
          "Accessibility operation failed (AXError -25201); observe the app before retrying"))
    #expect(UIAutomation<FakeUIBackend>.mapped(.unavailable("x")) == .unavailable("x"))
  }

  @Test(arguments: [
    (UIAttributeValue.string("a"), UISettableValue.string("a"), true),
    (UIAttributeValue.bool(true), UISettableValue.bool(true), true),
    (UIAttributeValue.number(1), UISettableValue.bool(true), true),
    (UIAttributeValue.number(0), UISettableValue.bool(true), false),
    (UIAttributeValue.missing, UISettableValue.string("a"), false),
  ])
  func comparesReadBackValues(
    _ observed: UIAttributeValue, _ requested: UISettableValue, _ equal: Bool
  ) {
    #expect(UIAutomation<FakeUIBackend>.equals(observed, requested) == equal)
  }

  @Test func encodesChildOutcomes() async throws {
    let automation = UIAutomation(backend: FakeUIBackend.standard())
    let ok = try await automation.runEncoded(#"{"operation":"inputSourceStatus"}"#)
    let decoded = try JSONDecoder().decode(UIChildOutcome.self, from: Data(ok.utf8))
    #expect(decoded.response?.inputSource?.id == "jp.kana")
    #expect(decoded.error == nil)
    let failed = try await automation.runEncoded(
      #"{"operation":"windows","target":{"app":"motion"}}"#)
    #expect(
      try JSONDecoder().decode(UIChildOutcome.self, from: Data(failed.utf8)).response?.running
        == false)
    let invalid = try await automation.runEncoded(
      #"{"operation":"inspect","target":{"app":"finalCutPro"},"window":{"title":"Absent"},"maxDepth":1,"maxElements":1}"#
    )
    let missing = try JSONDecoder().decode(UIChildOutcome.self, from: Data(invalid.utf8))
    #expect(missing.error == "No matching window; list windows first")
    #expect(missing.rethrown() == .unavailable("No matching window; list windows first"))
    let malformed = try await automation.runEncoded(#"{"operation":"unknown"}"#)
    let rejected = try JSONDecoder().decode(UIChildOutcome.self, from: Data(malformed.utf8))
    #expect(rejected.rethrown() == .invalid("malformed UI request"))
  }
}

struct UIModelTests {
  static let target = UITarget(app: .logicPro, bundleID: "com.apple.logic10")
  static let window = UIWindowLocator(index: 1, title: "Tracks", subrole: "AXStandardWindow")
  static let element = UIElementLocator(
    path: [0, 2], role: "AXButton", subrole: "AXCloseButton", identifier: "id", title: "t",
    description: "d", value: "v", containsText: "c", index: 3)

  @Test(arguments: [
    UIRequest.windows(target),
    .inspect(target, window: window, root: element, maxDepth: 4, maxElements: 9),
    .inspect(target, window: window, root: nil, maxDepth: 1, maxElements: 1),
    .menuList(target, path: ["File"]),
    .menuSelect(target, path: ["File", "New"]),
    .popupSelect(target, window: window, target: element, title: "Destination"),
    .perform(target, window: window, target: element, action: .showMenu),
    .set(target, window: window, target: element, attribute: .value, value: .string("x")),
    .set(target, window: window, target: element, attribute: .selected, value: .bool(false)),
    .wait(target, window: window, target: element, condition: .absent, timeoutSeconds: 2.5),
    .wait(target, window: nil, target: nil, condition: .exists, timeoutSeconds: 1),
    .capture(target, window: window, outputPath: "/tmp/out.png"),
    .inputSourceStatus,
    .inputSourceSelect(id: "abc", asciiCapable: false),
    .inputSourceSelect(id: nil, asciiCapable: true),
    .launch(target, activate: true),
    .activate(target, expectedPID: 42, allowForeground: true, menuPath: nil),
    .activate(target, expectedPID: 42, allowForeground: true, menuPath: ["File", "Share"]),
    .quit(target, force: false),
  ])
  func requestsRoundTrip(_ request: UIRequest) throws {
    let data = try JSONEncoder().encode(request)
    #expect(try JSONDecoder().decode(UIRequest.self, from: data) == request)
  }

  @Test(arguments: [
    (UIAction.press, "AXPress"), (.confirm, "AXConfirm"), (.cancel, "AXCancel"),
    (.increment, "AXIncrement"), (.decrement, "AXDecrement"), (.showMenu, "AXShowMenu"),
    (.pick, "AXPick"), (.raise, "AXRaise"), (.open, "AXOpen"),
  ])
  func mapsActionsToAccessibilityNames(_ action: UIAction, _ name: String) {
    #expect(action.axName == name)
  }

  @Test @MainActor func opensAdvertisedCellWithoutKeyboardOrPointer() async throws {
    let backend = FakeUIBackend.standard()
    backend.attributes[12]?[.role] = .string("AXCell")
    backend.actionNames[12] = ["AXOpen"]
    let result = try await UIAutomation(backend: backend).run(
      .perform(
        .init(app: .finalCutPro), window: .init(title: "Main"),
        target: UIElementLocator(identifier: "ok"),
        action: .open))
    #expect(backend.performed == ["AXOpen@12"])
    #expect(result.dispatched == true)
    #expect(result.effectVerified == false)
    #expect(!result.focus.frontmostChanged)
  }

  @Test func mapsSettableAttributesAndLocatorCriteria() {
    #expect(UISettableAttribute.value.attribute == .value)
    #expect(UISettableAttribute.selected.attribute == .selected)
    #expect(UIElementLocator().hasCriteria == false)
    #expect(UIElementLocator(index: 1).hasCriteria == false)
    #expect(UIElementLocator(containsText: "x").hasCriteria)
    #expect(Self.element.hasCriteria)
    #expect(UIPoint(x: 1, y: 2) == UIPoint(x: 1, y: 2))
    #expect(UISize(width: 1, height: 2).height == 2)
    #expect(UIRunningProcess(pid: 1, bundleID: "b").pid == 1)
  }

  @Test func encodesFocusEvidenceWithDerivedChange() throws {
    let evidence = UIFocusEvidence(frontmostBefore: "a", frontmostAfter: "b")
    let text = String(decoding: try JSONEncoder().encode(evidence), as: UTF8.self)
    #expect(text.contains(#""frontmostChanged":true"#))
    let decoded = try JSONDecoder().decode(UIFocusEvidence.self, from: Data(text.utf8))
    #expect(decoded == evidence)
    let empty = try JSONDecoder().decode(UIFocusEvidence.self, from: Data("{}".utf8))
    #expect(empty.frontmostChanged == false)
  }
}
