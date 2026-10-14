import Foundation
import Testing

@testable import ProAppsCore

/// The adapter injects an AX failure AFTER a synthetic action took effect.
@MainActor
private final class SaveNavigationBackend: UIBackend {
  let fake = FakeUIBackend.standard()
  let paths: [URL]
  var stage = 0
  var ignoreOpen = false
  var errorAfterOpen = false
  var errorAfterCancel = false
  var ignoreCancel = false
  var failShow = false
  var separateVolumes = false
  var ignoreVolumeSwitch = false
  var errorAfterVolumeSwitch = false
  var duplicateVolume = false
  var onOpen: () -> Void = {}
  var onPage: ((Int) -> Void)?
  var onChildPage: ((UIHandle) -> Void)?
  var selectionAllowed = true
  var selectionFailure = false
  var ignoreSelection = false
  var selectedChildren: [UIHandle] = []
  var onCancel: () -> Void = {}
  var issueFreshHandles = false
  var transitionDelay = 0.0
  var displayTitles: [Int: [String]] = [:]
  var language = "en"
  private var pendingStage: (index: Int, due: Double)?
  var onMenuRecheck: (() -> Void)?
  private var menuChildrenReads = 0
  private var didRecheckMutation = false
  private var aliases: [UIHandle: UIHandle] = [:]
  private var nextHandle = 10_000

  init(paths: [URL]) {
    self.paths = paths
    fake.attributes[11] = [
      .role: .string("AXSheet"), .identifier: .string("save-panel"),
      .children: .elements([13, 70, 73, 74]),
    ]
    fake.attributes[10]?[.children] = .elements([11])
    fake.attributes[13] = [
      .role: .string("AXPopUpButton"), .identifier: .string("where popup"),
      .enabled: .bool(true), .children: .elements([]),
    ]
    fake.attributes[70] = [.role: .string("AXBrowser"), .children: .elements([72])]
    fake.attributes[72] = [.role: .string("AXTextField"), .enabled: .bool(true)]
    fake.attributes[73] = [
      .role: .string("AXTextField"), .identifier: .string("saveAsNameTextField"),
      .value: .string("original.mov"),
    ]
    fake.attributes[74] = [
      .role: .string("AXButton"), .identifier: .string("OKButton"), .enabled: .bool(true),
    ]
    fake.attributes[50] = [.role: .string("AXMenu"), .identifier: .string("save-ancestors")]
    fake.actionNames[13] = ["AXShowMenu"]
    fake.actionNames[50] = ["AXCancel"]
    fake.actionNames[72] = ["AXOpen"]
    fake.actionNames[74] = ["AXPress"]
    update()
    fake.onPerform = { [weak self] backend, action, handle in
      guard let self else { return }
      if action == "AXShowMenu", !self.failShow {
        backend.attributes[13]?[.children] = .elements([50])
      } else if action == "AXCancel", !self.ignoreCancel {
        backend.attributes[13]?[.children] = .elements([])
        self.onCancel()
      } else if action == "AXOpen", !self.ignoreOpen {
        self.transition(to: min(self.stage + 1, self.paths.count - 1))
      } else if action == "AXPress", handle == 95, !self.ignoreVolumeSwitch {
        self.transition(to: 1)
      } else if action == "AXPress",
        backend.attributes[handle]?[.title] == .string("Test volume")
      {
        self.transition(to: 0)
      }
    }
  }

  private func transition(to index: Int) {
    if transitionDelay > 0 {
      pendingStage = (index, fake.clock + transitionDelay)
    } else {
      stage = index
      fake.attributes[13]?[.children] = .elements([])
      update()
      onOpen()
    }
  }

  func useStandaloneDialog() {
    fake.attributes[10]?[.role] = .string("AXWindow")
    fake.attributes[10]?[.subrole] = .string("AXDialog")
    fake.attributes[10]?[.identifier] = .string("save-panel")
    fake.attributes[10]?[.children] = .elements([13, 70, 73, 74, 75])
    fake.attributes[75] = [
      .role: .string("AXButton"), .identifier: .string("CancelButton"), .enabled: .bool(true),
    ]
    fake.actionNames[75] = ["AXPress"]
  }

  func titles(_ index: Int) -> [String] {
    if let supplied = displayTitles[index] { return supplied }
    if separateVolumes {
      switch index {
      case 0: return ["Boot volume"]
      case 1: return ["External volume"]
      default: return [paths[index].lastPathComponent, "External volume"]
      }
    }
    return Array(paths.dropFirst().prefix(index).reversed().map(\.lastPathComponent)) + [
      "Test volume"
    ]
  }

  func update() {
    fake.attributes[13]?[.value] = .string(titles(stage)[0])
    fake.attributes[72]?[.value] = .string(paths[min(stage + 1, paths.count - 1)].lastPathComponent)
    let titles = titles(stage)
    let handles = titles.indices.map { 51 + $0 }
    for (index, title) in titles.enumerated() {
      fake.attributes[51 + index] = [
        .role: .string("AXMenuItem"), .title: .string(title), .enabled: .bool(true),
      ]
      fake.actionNames[51 + index] = ["AXPress"]
    }
    fake.attributes[90] = [.role: .string("AXMenuItem"), .title: .string("")]
    var items = handles + [90]
    if separateVolumes, stage == 0 {
      fake.attributes[95] = [
        .role: .string("AXMenuItem"), .title: .string("External volume"), .enabled: .bool(true),
      ]
      fake.actionNames[95] = ["AXPress"]
      items.append(95)
      if duplicateVolume {
        fake.attributes[96] = fake.attributes[95]
        fake.actionNames[96] = ["AXPress"]
        items.append(96)
      }
    }
    fake.attributes[50]?[.children] = .elements(items)
  }

  // The observed macOS Save shape: direct sheet, split group, sidebar outline,
  // column browser, then the location popup/name/Save controls. Row populations
  // are synthetic and deliberately exceed the old 500-element whole-sheet budget.
  func populateLargeSaveSheet() {
    let siblings = (200..<216).map { UIHandle($0) }
    for handle in siblings { fake.attributes[handle] = [.role: .string("AXGroup")] }
    fake.attributes[10]?[.children] = .elements(siblings + [11])
    fake.attributes[11]?[.children] = .elements([80])
    fake.attributes[80] = [
      .role: .string("AXSplitGroup"), .children: .elements([81, 84, 13, 73, 74]),
    ]
    fake.attributes[81] = [.role: .string("AXScrollArea"), .children: .elements([82])]
    fake.attributes[82] = [.role: .string("AXOutline")]
    fake.attributes[84] = [.role: .string("AXSplitGroup"), .children: .elements([70])]
    let sidebar = (1000..<1700).map { UIHandle($0) }
    fake.attributes[82]?[.children] = .elements(sidebar)
    for handle in sidebar { fake.attributes[handle] = [.role: .string("AXRow")] }
    let columns = (1800..<1804).map { UIHandle($0) }
    fake.attributes[70]?[.children] = .elements(columns)
    for (columnIndex, column) in columns.enumerated() {
      let start = 2000 + columnIndex * 200
      let rows = (start..<(start + 200)).map { UIHandle($0) }
      fake.attributes[column] = [.role: .string("AXScrollArea"), .children: .elements(rows)]
      for handle in rows { fake.attributes[handle] = [.role: .string("AXCell")] }
    }
  }

  var environment: FCPSaveDirectoryEnvironment {
    let candidates = paths.enumerated().map { index, path in
      FCPSaveDirectoryCandidate(
        directory: path.path, volumeURL: paths[separateVolumes && index > 0 ? 1 : 0].path,
        variants: [.init(languages: [language], titles: titles(index))])
    }
    return .init { requested in candidates.filter { requested.contains($0.directory) } }
  }

  func runningProcess(bundleID: String) -> UIRunningProcess? {
    fake.runningProcess(bundleID: bundleID)
  }
  func frontmostBundleID() -> String? { fake.frontmostBundleID() }
  func frontmostProcess() -> UIRunningProcess? { fake.frontmostProcess() }
  func activate(process: UIRunningProcess) -> Bool { fake.activate(process: process) }
  func applicationElement(pid: Int32) -> UIHandle { fake.applicationElement(pid: pid) }
  func attribute(_ name: UIAttributeName, of handle: UIHandle) throws -> UIAttributeValue {
    let original = originalHandle(handle)
    if name == .children, original == 13, menuChildrenReads > 0, !didRecheckMutation,
      let onMenuRecheck
    {
      didRecheckMutation = true
      onMenuRecheck()
    }
    if name == .children, original == 50 { menuChildrenReads += 1 }
    let value = try fake.attribute(name, of: original)
    guard issueFreshHandles else { return value }
    // Mirror LiveUIBackend.register: every native attribute conversion allocates
    // a new storage slot even if the underlying physical AX element is unchanged.
    switch value {
    case .element(let child): return .element(freshHandle(child))
    case .elements(let children): return .elements(children.map(freshHandle))
    default: return value
    }
  }
  func childPage(of handle: UIHandle, offset: Int, count: Int) throws -> [UIHandle] {
    onChildPage?(handle)
    onPage?(offset)
    return try fake.childPage(of: originalHandle(handle), offset: offset, count: count)
  }
  func childCount(of handle: UIHandle) throws -> Int {
    try fake.childCount(of: originalHandle(handle))
  }
  func sameElement(_ left: UIHandle, _ right: UIHandle) throws -> Bool {
    originalHandle(left) == originalHandle(right)
  }
  private func originalHandle(_ handle: UIHandle) -> UIHandle { aliases[handle] ?? handle }
  private func freshHandle(_ handle: UIHandle) -> UIHandle {
    let fresh = nextHandle
    nextHandle += 1
    aliases[fresh] = originalHandle(handle)
    return fresh
  }
  func actions(of handle: UIHandle) throws -> [String] {
    try fake.actions(of: originalHandle(handle))
  }
  func isSettable(_ name: UIAttributeName, of handle: UIHandle) throws -> Bool {
    if name == .selectedChildren,
      fake.attributes[originalHandle(handle)]?[.role] == .string("AXList")
    {
      return selectionAllowed
    }
    return try fake.isSettable(name, of: originalHandle(handle))
  }

  func selectChild(of parent: UIHandle, at index: Int, expected: UIHandle) throws {
    let children = try childPage(of: parent, offset: index, count: 1)
    guard children == [expected], selectionAllowed else {
      throw UIBackendError.unavailable("Changed selection")
    }
    selectedChildren.append(expected)
    if selectionFailure { throw UIBackendError.accessibility(-25205) }
    if !ignoreSelection { transition(to: min(stage + 1, paths.count - 1)) }
  }
  func perform(_ action: String, on handle: UIHandle) throws {
    let handle = originalHandle(handle)
    try fake.perform(action, on: handle)
    if action == "AXPress", handle == 95, errorAfterVolumeSwitch {
      errorAfterVolumeSwitch = false
      throw UIBackendError.accessibility(-25205)
    }
    if action == "AXOpen", errorAfterOpen {
      errorAfterOpen = false
      throw UIBackendError.accessibility(-25205)
    }
    if action == "AXCancel", errorAfterCancel {
      errorAfterCancel = false
      throw UIBackendError.accessibility(-25205)
    }
  }
  func set(_ name: UIAttributeName, to value: UISettableValue, on handle: UIHandle) throws {
    try fake.set(name, to: value, on: originalHandle(handle))
  }
  func pause(seconds: Double) async throws {
    try await fake.pause(seconds: seconds)
    if let pendingStage, fake.clock >= pendingStage.due {
      self.pendingStage = nil
      stage = pendingStage.index
      fake.attributes[13]?[.children] = .elements([])
      update()
      onOpen()
    }
  }
  func monotonicSeconds() -> Double { fake.monotonicSeconds() }
  func captureWindow(pid: Int32, title: String?, frame: UIFrame?) async throws -> UICapturedImage {
    try await fake.captureWindow(pid: pid, title: title, frame: frame)
  }
  func currentInputSource() throws -> UIInputSource { try fake.currentInputSource() }
  func asciiCapableInputSource() throws -> UIInputSource { try fake.asciiCapableInputSource() }
  func inputSources() throws -> [UIInputSource] { try fake.inputSources() }
  func selectInputSource(id: String) throws { try fake.selectInputSource(id: id) }
  func applicationURL(bundleID: String) -> URL? { fake.applicationURL(bundleID: bundleID) }
  func launch(applicationAt url: URL, activate: Bool) async throws -> Int32 {
    try await fake.launch(applicationAt: url, activate: activate)
  }
  func terminate(pid: Int32, force: Bool) -> Bool { fake.terminate(pid: pid, force: force) }
}

@MainActor
struct FCPSaveNavigationTests {
  @Test func boundedBackendPagesPreserveExactOffsetsAndRejectMissingFacts() throws {
    #expect(throws: UIBackendError.self) {
      try SaveNavigationBackend(paths: [URL(fileURLWithPath: "/synthetic")]).attributeNames(of: 1)
    }
    let backend = FakeUIBackend.standard()
    backend.attributes[900] = [.children: .elements([901, 902, 903])]
    #expect(try backend.childCount(of: 900) == 3)
    #expect(try backend.childPage(of: 900, offset: 1, count: 2) == [902, 903])
    #expect(try backend.sameElement(902, 902))
    #expect(try !backend.sameElement(901, 902))
    #expect(throws: UIBackendError.self) { try backend.childCount(of: 999) }
    #expect(throws: UIBackendError.self) { try backend.childPage(of: 900, offset: 3, count: 1) }
    #expect(throws: UIBackendError.self) { try backend.childPage(of: 900, offset: 0, count: 65) }
    #expect(throws: UIBackendError.self) {
      try backend.selectChild(of: 900, at: 0, expected: 901)
    }
  }

  private func withDirectories(_ count: Int, body: ([URL]) async throws -> Void) async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent("save-ui-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    var paths = [root]
    for index in 1..<count {
      let child = paths[index - 1].appendingPathComponent("Folder \(index)")
      try FileManager.default.createDirectory(at: child, withIntermediateDirectories: false)
      paths.append(child)
    }
    try await body(paths)
  }

  private func request(_ paths: [URL], steps: Int = 16, dispatched: Bool = false, pid: Int32 = 42)
    -> FCPSaveNavigationRequest
  {
    .init(
      directory: paths[paths.count - 1].path, fileName: "new.mov",
      candidateDirectories: paths.map(\.path),
      window: .init(title: "Main"), expectedPID: pid, maximumSteps: steps,
      saveDispatched: dispatched)
  }

  private func run(_ backend: SaveNavigationBackend, _ request: FCPSaveNavigationRequest)
    async throws -> UIResponse
  {
    try await UIAutomation(backend: backend, saveDirectoryEnvironment: backend.environment).run(
      .saveNavigate(.init(app: .finalCutPro), request))
  }

  @Test(arguments: [1, 2, 3])
  func wholeProofAndFolderStepsNeverSaveOrWrite(_ count: Int) async throws {
    try await withDirectories(count) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      let result = try await run(backend, request(paths))
      let receipt = try #require(result.saveNavigation)
      #expect(receipt.status == .directoryVerified)
      #expect(receipt.steps.count == count - 1)
      #expect(receipt.observations.last?.directory == paths[count - 1].path)
      #expect(backend.fake.performed.filter { $0 == "AXOpen@72" }.count == count - 1)
      #expect(!backend.fake.performed.contains("AXPress@74"))
      #expect(backend.fake.written.isEmpty)
      #expect(backend.fake.activated.isEmpty && backend.fake.launched.isEmpty)
      #expect(
        try JSONDecoder().decode(UIResponse.self, from: JSONEncoder().encode(result)) == result)
      let value = UIRequest.saveNavigate(.init(app: .finalCutPro), request(paths))
      #expect(try JSONDecoder().decode(UIRequest.self, from: JSONEncoder().encode(value)) == value)
    }
  }

  @Test(arguments: [1, 2, 3])
  func standaloneSaveDialogProvesDirectoryWithoutSaving(_ count: Int) async throws {
    try await withDirectories(count) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.useStandaloneDialog()
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .directoryVerified)
      #expect(receipt.steps.count == count - 1)
      #expect(!backend.fake.performed.contains("AXPress@74"))
      #expect(!backend.fake.performed.contains("AXPress@75"))
      #expect(backend.fake.written.isEmpty)
    }
  }

  @Test(arguments: ["subrole", "identifier", "nested-sheet", "cancel", "save", "browser"])
  func malformedStandaloneDialogRefusesBeforeActions(_ mode: String) async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.useStandaloneDialog()
      switch mode {
      case "subrole": backend.fake.attributes[10]?[.subrole] = .string("AXStandardWindow")
      case "identifier": backend.fake.attributes[10]?[.identifier] = .string("other")
      case "nested-sheet":
        backend.fake.attributes[10]?[.children] = .elements([11, 13, 70, 73, 74, 75])
      case "cancel": backend.fake.attributes[75]?[.enabled] = .bool(false)
      case "save": backend.fake.actionNames[74] = []
      default: backend.fake.attributes[70]?[.role] = .string("AXGroup")
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status != .directoryVerified)
      #expect(backend.fake.performed.isEmpty)
      #expect(backend.fake.written.isEmpty)
    }
  }

  @Test func standaloneCancelIdentityChangeStopsAfterSingleFolderAction() async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.useStandaloneDialog()
      backend.onOpen = { backend.fake.attributes[75]?[.identifier] = .string("changed") }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status != .directoryVerified)
      #expect(backend.fake.performed.filter { $0 == "AXOpen@72" }.count == 1)
      #expect(!backend.fake.performed.contains("AXPress@74"))
    }
  }

  @Test(arguments: [
    "success", "selected-parent", "duplicate", "missing", "unbound", "changed", "deadline",
    "too-many",
  ])
  func pagedCurrentColumnRequiresCompleteUniqueURLIdentity(_ mode: String) async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      let rows = (3000..<3273).map { UIHandle($0) }
      backend.fake.attributes[70]?[.children] = .elements([80])
      backend.fake.attributes[80] = [
        .role: .string("AXList"), .url: .string(paths[0].absoluteString),
        .children: .elements(rows),
      ]
      for row in rows {
        backend.fake.attributes[row] = [.role: .string("AXGroup"), .children: .elements([])]
      }
      backend.fake.attributes[3272] = [
        .role: .string("AXGroup"), .url: .string(paths[1].absoluteString),
        .children: .elements([72]),
      ]
      switch mode {
      case "selected-parent":
        backend.fake.attributes[70]?[.children] = .elements([81, 80])
        backend.fake.attributes[81] = [
          .role: .string("AXList"), .children: .elements([82]), .selectedChildren: .elements([82]),
        ]
        backend.fake.attributes[82] = [
          .role: .string("AXGroup"), .url: .string(paths[0].absoluteString),
          .children: .elements([]),
        ]
        backend.fake.attributes[80]?[.url] = .missing
      case "duplicate": backend.fake.attributes[3000] = backend.fake.attributes[3272]
      case "missing": backend.fake.attributes[3272]?[.url] = .missing
      case "unbound": backend.fake.attributes[80]?[.url] = .missing
      case "changed":
        backend.onPage = { offset in
          if offset == 64 {
            backend.fake.attributes[80]?[.children] = .elements(Array(rows.dropLast()))
          }
        }
      case "deadline": backend.onPage = { _ in backend.fake.clock += 16 }
      case "too-many":
        backend.fake.attributes[80]?[.children] = .elements(Array(repeating: 3000, count: 4097))
      default: break
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(
        (receipt.status == .directoryVerified) == (mode == "success" || mode == "selected-parent"))
      #expect(
        backend.selectedChildren.count
          == (mode == "success" || mode == "selected-parent" ? 1 : 0))
      #expect(!receipt.nameAttempted && !receipt.saveAttempted)
    }
  }

  @Test(arguments: [
    "success", "duplicate", "conflicting-url", "multiple-columns", "missing-selection",
    "duplicate-label", "changed-label", "unsupported", "no-effect", "symlink",
  ])
  func soleColumnUsesProvedParentAndStableImmediateChild(_ mode: String) async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.fake.attributes[70]?[.children] = .elements([80])
      backend.fake.attributes[80] = [
        .role: .string("AXList"), .children: .elements([81]),
        .selectedChildren: .elements([]),
      ]
      backend.fake.attributes[81] = [.role: .string("AXGroup"), .children: .elements([72])]
      switch mode {
      case "duplicate":
        backend.fake.attributes[82] = backend.fake.attributes[81]
        backend.fake.attributes[80]?[.children] = .elements([81, 82])
      case "conflicting-url":
        backend.fake.attributes[81]?[.url] = .string(paths[0].absoluteString)
      case "multiple-columns":
        backend.fake.attributes[82] = [.role: .string("AXList"), .children: .elements([])]
        backend.fake.attributes[70]?[.children] = .elements([80, 82])
      case "missing-selection": backend.fake.attributes[80]?[.selectedChildren] = .missing
      case "duplicate-label":
        backend.fake.attributes[82] = backend.fake.attributes[72]
        backend.fake.attributes[81]?[.children] = .elements([72, 82])
      case "changed-label":
        var reads = 0
        backend.onChildPage = { handle in
          if handle == 81 {
            reads += 1
            if reads == 4 { backend.fake.attributes[72]?[.value] = .string("Changed") }
          }
        }
      case "unsupported": backend.selectionAllowed = false
      case "no-effect": backend.ignoreSelection = true
      case "symlink":
        try FileManager.default.removeItem(at: paths[1])
        try FileManager.default.createSymbolicLink(at: paths[1], withDestinationURL: paths[0])
      default: break
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect((receipt.status == .directoryVerified) == (mode == "success"))
      #expect(backend.selectedChildren.count == (mode == "success" || mode == "no-effect" ? 1 : 0))
      if mode == "success" {
        #expect(receipt.steps.first?.selectionBasis == .verifiedParentChild)
        #expect(receipt.steps.first?.after?.directory == paths[1].path)
      }
      #expect(!receipt.nameAttempted && !receipt.saveAttempted)
    }
  }

  @Test(arguments: [
    "absent", "container-menu", "application-menu", "over-bound", "changed", "standalone",
  ])
  func timeoutRecordsOnlyBoundedDirectMenuAttachments(_ mode: String) async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.failShow = true
      backend.fake.attributes[1]?[.children] = .elements([10])
      backend.fake.attributes[90] = [
        .role: .string("AXMenu"), .identifier: .string("detached-observed-menu"),
        .children: .elements([91]),
      ]
      backend.fake.attributes[91] = [
        .role: .string("AXMenuItem"), .title: .string("Do not traverse"),
      ]
      switch mode {
      case "container-menu":
        backend.fake.attributes[11]?[.children] = .elements([13, 70, 73, 74, 90])
      case "application-menu": backend.fake.attributes[1]?[.children] = .elements([10, 90])
      case "over-bound":
        backend.fake.attributes[1]?[.children] = .elements(Array(repeating: 10, count: 65))
      case "standalone": backend.useStandaloneDialog()
      case "changed":
        var reads = 0
        backend.onChildPage = { handle in
          if handle == 1 {
            reads += 1
            if reads == 2 { backend.fake.attributes[10]?[.identifier] = .string("Changed") }
          }
        }
      default: break
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .navigationUnavailable)
      #expect(receipt.failure == "Unavailable: Save location ancestor menu did not appear")
      #expect(receipt.menuShowsAttempted == 1 && receipt.steps.isEmpty)
      #expect(receipt.menuAttachments.count == 3)
      let application = try #require(receipt.menuAttachments.last)
      #expect(application.scope == .application)
      #expect(application.complete == (mode != "over-bound" && mode != "changed"))
      #expect((receipt.menuAttachmentFailure != nil) == (mode == "over-bound" || mode == "changed"))
      let roles = receipt.menuAttachments.flatMap(\.children).compactMap(\.role)
      #expect(roles.contains("AXMenu") == (mode == "container-menu" || mode == "application-menu"))
      #expect(!roles.contains("AXMenuItem"))
      #expect(!receipt.nameAttempted && !receipt.saveAttempted && !receipt.directoryVerified)
    }
  }

  @Test(arguments: ["missing", "empty", "labels", "unsupported", "too-many", "deep-bound"])
  func selectedColumnDiagnosticsDistinguishMissingFromEmpty(_ mode: String) async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.fake.attributes[70]?[.children] = .elements([80])
      backend.fake.attributes[80] = [
        .role: .string("AXList"), .children: .elements([81]),
      ]
      backend.fake.attributes[81] = [
        .role: .string("AXGroup"), .selected: .bool(true), .children: .elements([82]),
      ]
      backend.fake.attributes[82] = [
        .role: .string("AXTextField"), .value: .string("Observed folder"),
      ]
      switch mode {
      case "empty": backend.fake.attributes[80]?[.selectedChildren] = .elements([])
      case "labels", "deep-bound":
        backend.fake.attributes[80]?[.selectedChildren] = .elements([81])
        if mode == "deep-bound" {
          backend.fake.attributes[81]?[.children] = .elements(Array(repeating: 82, count: 33))
        }
      case "unsupported": backend.fake.attributes[80]?[.selectedChildren] = .string("unknown")
      case "too-many":
        backend.fake.attributes[80]?[.selectedChildren] = .elements(Array(repeating: 81, count: 17))
      default: break
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(!receipt.directoryVerified && receipt.steps.isEmpty)
      #expect(!receipt.nameAttempted && !receipt.saveAttempted)
      if mode == "deep-bound" {
        #expect(receipt.failure?.contains("exceed") == true)
      } else {
        let column = try #require(receipt.columns.first)
        #expect(column.selectedURLs.isEmpty)
        switch mode {
        case "missing":
          #expect(column.selectedChildrenState == .missing)
          #expect(column.selectedChildrenCount == nil)
        case "empty":
          #expect(column.selectedChildrenState == .elements)
          #expect(column.selectedChildrenCount == 0 && column.selectedChildren.isEmpty)
        case "labels":
          #expect(column.selectedChildrenCount == 1)
          #expect(column.selectedChildren.first?.element.selected == true)
          #expect(column.selectedChildren.first?.directChildren.first?.value == "Observed folder")
        case "unsupported": #expect(column.selectedChildrenState == .unsupported)
        default:
          #expect(column.selectedChildrenCount == 17 && column.selectedChildren.isEmpty)
        }
        #expect(
          try JSONDecoder().decode(
            FCPSaveNavigationReceipt.self, from: JSONEncoder().encode(receipt)) == receipt)
      }
    }
  }

  @Test(arguments: [
    "child", "same-url", "conflicting", "changed-url", "changed-count", "unsupported",
    "refusal", "unchanged",
  ])
  func rowURLsUseStableCanonicalDirectChildren(_ mode: String) async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.fake.attributes[70]?[.children] = .elements([80, 83])
      backend.fake.attributes[80] = [
        .role: .string("AXList"), .children: .elements([81]), .selectedChildren: .elements([81]),
      ]
      backend.fake.attributes[81] = [.role: .string("AXGroup"), .children: .elements([82])]
      backend.fake.attributes[82] = [
        .role: .string("AXTextField"), .url: .string(paths[0].absoluteString),
        .value: .string(paths[0].lastPathComponent), .selected: .bool(false),
      ]
      backend.fake.attributes[83] = [.role: .string("AXList"), .children: .elements([84])]
      backend.fake.attributes[84] = [.role: .string("AXGroup"), .children: .elements([72])]
      backend.fake.attributes[72]?[.url] = .string(paths[1].absoluteString)
      backend.selectionAllowed = mode != "unsupported"
      backend.selectionFailure = mode == "refusal"
      backend.ignoreSelection = mode == "unchanged"
      if mode == "same-url" || mode == "conflicting" {
        backend.fake.attributes[84]?[.children] = .elements([72, 85])
        backend.fake.attributes[85] = [
          .role: .string("AXImage"),
          .url: .string((mode == "same-url" ? paths[1] : paths[0]).absoluteString),
        ]
      }
      var targetReads = 0
      backend.onChildPage = { handle in
        if handle == 84 {
          targetReads += 1
          if targetReads == 2 && mode == "changed-url" {
            backend.fake.attributes[72]?[.url] = .string(paths[0].absoluteString)
          }
          if targetReads == 2 && mode == "changed-count" {
            backend.fake.attributes[84]?[.children] = .elements([72, 85])
          }
        }
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.directoryVerified == (mode == "child" || mode == "same-url"))
      #expect(receipt.columns.first?.selectedURLs == [paths[0].path])
      #expect(!receipt.nameAttempted && !receipt.saveAttempted)
      #expect(backend.fake.performed.filter { $0.hasPrefix("AXOpen") }.isEmpty)
      if mode == "child" || mode == "same-url" || mode == "refusal" || mode == "unchanged" {
        #expect(backend.selectedChildren == [84])
        #expect(receipt.steps.first?.action == nil)
        #expect(receipt.steps.first?.selectedChildPath == [0, 1, 1, 0])
      } else {
        #expect(backend.selectedChildren.isEmpty)
      }
      if mode != "child" && mode != "same-url" {
        #expect(backend.fake.performed.filter { $0.hasPrefix("AXOpen") }.isEmpty)
      }
    }
  }

  @Test func anAXErrorAfterEffectIsReadBackWithoutRepeatingOpen() async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.errorAfterOpen = true
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .directoryVerified)
      #expect(receipt.steps[0].actionError != nil)
      #expect(receipt.steps[0].after?.directory == paths[1].path)
      #expect(backend.fake.performed.filter { $0 == "AXOpen@72" }.count == 1)
    }
  }

  @Test func successfulAncestorReadbackSurvivesLaterColumnFailure() async throws {
    try await withDirectories(3) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.onOpen = {
        backend.fake.attributes[70]?[.children] = .elements([80])
        backend.fake.attributes[80] = [
          .role: .string("AXList"), .url: .string(paths[1].absoluteString),
          .children: .elements(Array(repeating: 81, count: 4097)),
        ]
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .navigationUnavailable)
      #expect(receipt.steps.count == 1)
      #expect(receipt.steps.first?.after?.directory == paths[1].path)
      #expect(receipt.steps.first?.actionAttempted == true)
      #expect(receipt.failure?.contains("row budget exceeded") == true)
      #expect(backend.fake.performed.filter { $0 == "AXOpen@72" }.count == 1)
      #expect(!receipt.nameAttempted && !receipt.saveAttempted)
    }
  }

  @Test(arguments: [false, true])
  func delayedDirectoryTransitionIsAwaitedWithoutRepeatingAnUncertainAction(_ axError: Bool)
    async throws
  {
    try await withDirectories(3) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.issueFreshHandles = true
      backend.transitionDelay = 0.3
      backend.errorAfterOpen = axError
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .directoryVerified)
      #expect(receipt.steps.count == 2)
      #expect(receipt.steps[0].after?.directory == paths[1].path)
      #expect(receipt.steps[1].after?.directory == paths[2].path)
      #expect((receipt.steps[0].actionError != nil) == axError)
      #expect(backend.fake.clock >= 0.6 && backend.fake.clock < 2)
      #expect(backend.fake.performed.filter { $0 == "AXOpen@72" }.count == 2)
      #expect(receipt.menuShowsAttempted == 3)
      #expect(backend.fake.written.isEmpty && !backend.fake.performed.contains("AXPress@74"))
    }
  }

  @Test func delayedAncestorSelectionWaitsForTheFullOSChain() async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.stage = 1
      backend.transitionDelay = 0.4
      backend.update()
      let navigation = FCPSaveNavigationRequest(
        directory: paths[0].path, fileName: "new.mov", candidateDirectories: paths.map(\.path),
        window: .init(title: "Main"), expectedPID: 42)
      let receipt = try #require(try await run(backend, navigation).saveNavigation)
      #expect(receipt.status == .directoryVerified)
      #expect(receipt.steps.count == 1)
      #expect(receipt.steps[0].after?.directory == paths[0].path)
      #expect(backend.fake.clock >= 0.4)
      #expect(backend.fake.performed.filter { $0 == "AXPress@52" }.count == 1)
      #expect(backend.fake.performed.filter { $0 == "AXCancel@50" }.count == 1)
    }
  }

  @Test(arguments: [2.5, 0.0])
  func transitionTimeoutKeepsTheOldProofAndNeverRetriesTheFolder(_ delay: Double) async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.transitionDelay = delay
      backend.ignoreOpen = delay == 0
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .uncertainEffect)
      #expect(receipt.steps.count == 1)
      #expect(receipt.steps[0].after?.directory == paths[0].path)
      #expect(backend.fake.clock >= 2 && backend.fake.clock < 2.2)
      #expect(backend.fake.performed.filter { $0 == "AXOpen@72" }.count == 1)
      #expect(backend.fake.written.isEmpty && !backend.fake.performed.contains("AXPress@74"))
    }
  }

  @Test(arguments: [
    "pid", "focus", "deadline", "name", "name-missing", "popup-path", "save", "panel", "cancelled",
  ])
  func asynchronousTransitionCannotChangeOwnershipOrSaveControls(_ mode: String) async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.transitionDelay = 0.4
      let fake = backend.fake
      fake.onPause = { fake in
        switch mode {
        case "pid": fake.processes[FakeUIBackend.fcp] = .init(pid: 999, bundleID: FakeUIBackend.fcp)
        case "focus": fake.frontmost = ["other"]
        case "deadline": fake.clock = 16
        case "name": fake.attributes[73]?[.value] = .string("different.mov")
        case "name-missing": fake.attributes[73]?[.value] = .missing
        case "popup-path": fake.attributes[11]?[.children] = .elements([70, 13, 73, 74])
        case "save": fake.attributes[74]?[.identifier] = .string("changed")
        case "panel": fake.attributes[11]?[.identifier] = .string("another-panel")
        default: break
        }
      }
      if mode == "cancelled" { fake.pauseFailure = CancellationError() }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      let status: FCPSaveNavigationStatus
      switch mode {
      case "pid": status = .processChanged
      case "focus": status = .focusChanged
      case "deadline": status = .deadlineExceeded
      case "cancelled": status = .cancelled
      default: status = .navigationUnavailable
      }
      #expect(receipt.status == status)
      #expect(receipt.steps.count == 1)
      #expect(backend.fake.performed.filter { $0 == "AXOpen@72" }.count == 1)
      #expect(backend.fake.written.isEmpty && !backend.fake.performed.contains("AXPress@74"))
    }
  }

  @Test func anOSDisplayReadinessLabelNeverReplacesFullAncestry() async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      let fake = backend.fake
      backend.onOpen = { fake.attributes[51]?[.title] = .string("Another directory") }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .uncertainEffect)
      #expect(receipt.steps[0].after?.directory == nil)
      #expect(fake.performed.filter { $0 == "AXOpen@72" }.count == 1)
      #expect(fake.written.isEmpty && !fake.performed.contains("AXPress@74"))
    }
  }

  @Test(arguments: ["verified", "name", "closed-menu", "wrong-chain"])
  func popupReadinessRequiresAnIndependentSettledFullChain(_ mode: String) async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      let fake = backend.fake
      backend.onOpen = {
        fake.attributes[51]?[.title] = .string("Test volume")
        fake.attributes[50]?[.children] = .elements([51, 90])
      }
      fake.onPause = { [weak backend] fake in
        guard fake.clock >= 0.3 else { return }
        switch mode {
        case "verified": backend?.update()
        case "name": fake.attributes[73]?[.value] = .string("changed.mov")
        case "closed-menu": fake.attributes[13]?[.children] = .elements([])
        default: break
        }
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      let expected: FCPSaveNavigationStatus
      switch mode {
      case "verified": expected = .directoryVerified
      case "wrong-chain": expected = .uncertainEffect
      default: expected = .navigationUnavailable
      }
      #expect(receipt.status == expected)
      #expect(receipt.steps.count == 1)
      #expect(fake.performed.filter { $0 == "AXOpen@72" }.count == 1)
      #expect(receipt.menuShowsAttempted == 2)
      #expect(fake.written.isEmpty && !fake.performed.contains("AXPress@74"))
      if mode == "verified" {
        #expect(receipt.observations.contains { $0.directory == paths[0].path })
        #expect(receipt.observations.last?.directory == paths[1].path)
        #expect(fake.clock >= 0.3)
      }
    }
  }

  @Test func localizedOSChainReadinessPreservesTheFullHostAndVolumeProof() async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.language = "ja"
      backend.displayTitles = [0: ["起動ディスク", "ホスト"], 1: ["保存場所", "起動ディスク", "ホスト"]]
      backend.transitionDelay = 0.2
      backend.update()
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .directoryVerified)
      #expect(receipt.observations.last?.ancestorTitles == ["保存場所", "起動ディスク", "ホスト"])
      #expect(receipt.observations.last?.languages == ["ja"])
      #expect(backend.fake.performed.filter { $0 == "AXOpen@72" }.count == 1)
      #expect(backend.fake.written.isEmpty)
    }
  }

  @Test(arguments: ["missing-name", "oversized-name", "nul-name", "duplicate-save"])
  func folderDispatchRequiresUniqueObservableUnchangedSaveControls(_ mode: String) async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      switch mode {
      case "missing-name": backend.fake.attributes[73]?[.value] = .missing
      case "oversized-name":
        backend.fake.attributes[73]?[.value] = .string(String(repeating: "x", count: 513))
      case "nul-name": backend.fake.attributes[73]?[.value] = .string("name\0.mov")
      default:
        backend.fake.attributes[75] = backend.fake.attributes[74]
        backend.fake.attributes[11]?[.children] = .elements([13, 70, 73, 74, 75])
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .navigationUnavailable)
      #expect(receipt.steps.isEmpty)
      #expect(!backend.fake.performed.contains("AXOpen@72"))
      #expect(backend.fake.written.isEmpty && !backend.fake.performed.contains("AXPress@74"))
    }
  }

  @Test func freshStorageHandlesPreserveOneMenuPathAndCompleteCanonicalProof() async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.issueFreshHandles = true
      backend.fake.attributes[13]?[.children] = .elements([50])
      let first = try backend.attribute(.children, of: 13)
      let second = try backend.attribute(.children, of: 13)
      guard case .elements(let firstHandles) = first, case .elements(let secondHandles) = second,
        let firstMenu = firstHandles.first, let secondMenu = secondHandles.first
      else {
        Issue.record("Expected synthetic menu handles")
        return
      }
      #expect(firstMenu != secondMenu)
      #expect(try backend.attribute(.identifier, of: firstMenu) == .string("save-ancestors"))
      #expect(try backend.attribute(.identifier, of: secondMenu) == .string("save-ancestors"))
      backend.fake.attributes[13]?[.children] = .elements([])
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .directoryVerified)
      #expect(receipt.steps.count == 1)
      #expect(receipt.steps[0].after?.directory == paths[1].path)
      #expect(receipt.inspections.allSatisfy { $0.complete })
      #expect(backend.fake.performed.filter { $0 == "AXOpen@72" }.count == 1)
      #expect(!backend.fake.performed.contains("AXPress@74"))
      #expect(backend.fake.written.isEmpty)
    }
  }

  @Test(arguments: [
    "reordered", "different-label", "unknown-id", "changed-id", "root-menu", "ambiguous",
    "unreadable-id", "empty-id", "oversized-label", "nul-label", "missing-role", "empty-role",
    "item-id-change", "truncated-fresh", "menu-role", "pid", "focus", "mutation",
  ])
  func aFreshHandleNeverHidesChangedMenuSemanticsOrOwnership(_ mode: String) async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.issueFreshHandles = true
      let fake = backend.fake
      if mode == "unknown-id" {
        fake.attributes[50]?[.identifier] = .missing
      } else if mode == "unreadable-id" {
        fake.attributes[50]?[.identifier] = .bool(true)
      } else if mode == "empty-id" {
        fake.attributes[50]?[.identifier] = .string("")
      } else if mode == "oversized-label" {
        fake.attributes[51]?[.title] = .string(String(repeating: "x", count: 513))
      } else if mode == "nul-label" {
        fake.attributes[51]?[.title] = .string("Volume\0name")
      } else if mode == "missing-role" {
        fake.attributes[51]?[.role] = .missing
      } else if mode == "empty-role" {
        fake.attributes[51]?[.role] = .string("")
      } else {
        backend.onMenuRecheck = {
          switch mode {
          case "reordered": fake.attributes[50]?[.children] = .elements([90, 51])
          case "different-label": fake.attributes[51]?[.title] = .string("Another volume")
          case "changed-id": fake.attributes[50]?[.identifier] = .string("another-menu")
          case "root-menu":
            fake.attributes[99] = [.role: .string("AXGroup")]
            fake.attributes[13]?[.children] = .elements([99, 50])
          case "ambiguous": fake.attributes[13]?[.children] = .elements([50, 50])
          case "item-id-change": fake.attributes[51]?[.identifier] = .string("different-item")
          case "truncated-fresh":
            fake.attributes[50]?[.children] = .elements(Array(repeating: 51, count: 257))
          case "menu-role": fake.attributes[50]?[.role] = .string("AXGroup")
          case "pid":
            fake.processes[FakeUIBackend.fcp] = .init(pid: 999, bundleID: FakeUIBackend.fcp)
          case "focus": fake.frontmost = ["other"]
          default:
            fake.attributes[13]?[.value] = .string("Changed")
            fake.attributes[51]?[.title] = .string("Changed")
          }
        }
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      let expected: FCPSaveNavigationStatus
      switch mode {
      case "pid": expected = .processChanged
      case "focus": expected = .focusChanged
      default: expected = .navigationUnavailable
      }
      #expect(receipt.status == expected)
      #expect(receipt.steps.isEmpty)
      #expect(receipt.menuShowsAttempted == 1)
      #expect(receipt.menuClosesAttempted == 0)
      #expect(receipt.inspections.last?.scope == .ancestorMenu)
      #expect(receipt.inspections.last?.complete == false)
      #expect(backend.fake.performed == ["AXShowMenu@13"])
      #expect(backend.fake.written.isEmpty)
    }
  }

  @Test func actualShapedLargeContentDoesNotBecomeDirectoryProofOrBlockMenuNavigation() async throws
  {
    try await withDirectories(1) { roots in
      let root = try #require(roots.first)
      let boot = root.appendingPathComponent("Boot")
      let volume = root.appendingPathComponent("External")
      let target = volume.appendingPathComponent("Movies")
      try FileManager.default.createDirectory(at: boot, withIntermediateDirectories: false)
      try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
      let paths = [boot, volume, target]
      let backend = SaveNavigationBackend(paths: paths)
      backend.issueFreshHandles = true
      backend.separateVolumes = true
      backend.update()
      backend.populateLargeSaveSheet()
      let old = try await UIAutomation(backend: backend).run(
        .inspect(
          .init(app: .finalCutPro), window: .init(title: "Main"),
          root: .init(role: "AXSheet", identifier: "save-panel"), maxDepth: 20, maxElements: 500))
      #expect(old.truncated == true)
      #expect(old.elements?.count == 500)
      let fake = backend.fake
      backend.onOpen = { fake.attributes[70]?[.children] = .elements([72]) }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .directoryVerified)
      #expect(receipt.steps.count == 2)
      #expect(receipt.steps[0].selectionBasis == .mountedVolumeCandidate)
      #expect(receipt.steps[0].after?.directory == volume.path)
      #expect(receipt.inspections.filter { $0.scope == .columnBrowser }.count == 1)
      #expect(receipt.inspections.allSatisfy { $0.complete })
      #expect(backend.fake.performed.filter { $0 == "AXPress@95" }.count == 1)
      #expect(backend.fake.performed.filter { $0 == "AXOpen@72" }.count == 1)
      #expect(!backend.fake.performed.contains("AXPress@74"))
      #expect(backend.fake.written.isEmpty && backend.fake.activated.isEmpty)
    }
  }

  @Test func alreadySelectedDestinationRequiresNoFileContentInspection() async throws {
    try await withDirectories(1) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.populateLargeSaveSheet()
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .directoryVerified)
      #expect(receipt.steps.isEmpty)
      #expect(receipt.inspections.map(\.scope) == [.controls, .ancestorMenu, .controls])
      #expect(receipt.inspections.allSatisfy { $0.complete })
      #expect(backend.fake.performed == ["AXShowMenu@13", "AXCancel@50"])
    }
  }

  @Test(arguments: ["browser", "controls", "depth", "direct-sheet", "duplicate-popup"])
  func incompleteRequiredScopeRetainsItsBoundAndNeverDispatchesAFolder(_ mode: String) async throws
  {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      switch mode {
      case "browser": backend.populateLargeSaveSheet()
      case "controls":
        let children = (1000..<1130).map { UIHandle($0) }
        backend.fake.attributes[11]?[.children] = .elements(children)
        for child in children { backend.fake.attributes[child] = [.role: .string("AXGroup")] }
      case "depth":
        backend.fake.attributes[11]?[.children] = .elements([1000])
        for handle in 1000..<1014 {
          backend.fake.attributes[UIHandle(handle)] = [
            .role: .string("AXGroup"), .children: .elements([UIHandle(handle + 1)]),
          ]
        }
      case "direct-sheet": backend.fake.attributes[11]?[.identifier] = .string("not-save")
      default:
        backend.fake.attributes[75] = backend.fake.attributes[13]
        backend.fake.attributes[11]?[.children] = .elements([13, 75])
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .navigationUnavailable)
      #expect(receipt.steps.isEmpty)
      #expect(backend.fake.written.isEmpty)
      #expect(!backend.fake.performed.contains("AXOpen@72"))
      #expect(!backend.fake.performed.contains("AXPress@74"))
      if mode == "browser" {
        #expect(receipt.observations.first?.directory == paths[0].path)
        #expect(receipt.inspections.last?.scope == .columnBrowser)
        #expect(receipt.inspections.last?.complete == false)
        #expect(receipt.failure?.contains("columnBrowser") == true)
      }
      if mode == "controls" || mode == "depth" {
        #expect(receipt.inspections.last?.scope == .controls)
        #expect(receipt.inspections.last?.complete == false)
      }
    }
  }

  @Test func unchangedDirectoryAfterAttemptIsUncertainAndNeverRetried() async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.ignoreOpen = true
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == .uncertainEffect)
      #expect(receipt.steps.count == 1)
      #expect(receipt.steps[0].after?.directory == paths[0].path)
      #expect(backend.fake.performed.filter { $0 == "AXOpen@72" }.count == 1)
    }
  }

  @Test func selectingAnObservedAncestorUsesItsWholeVerifiedOSChain() async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.stage = 1
      backend.update()
      let navigation = FCPSaveNavigationRequest(
        directory: paths[0].path, fileName: "new.mov", candidateDirectories: paths.map(\.path),
        window: .init(title: "Main"), expectedPID: 42)
      let receipt = try #require(try await run(backend, navigation).saveNavigation)
      #expect(receipt.status == .directoryVerified)
      #expect(receipt.steps.count == 1)
      #expect(receipt.steps[0].expectedDirectory == paths[0].path)
      #expect(backend.fake.performed.filter { $0 == "AXPress@52" }.count == 1)
      #expect(!backend.fake.performed.contains("AXPress@74"))
    }
  }

  @Test(arguments: [false, true])
  func menuCancellationHasIndependentReadbackEvenAfterAXError(_ ignored: Bool) async throws {
    try await withDirectories(1) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      backend.errorAfterCancel = true
      backend.ignoreCancel = ignored
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status == (ignored ? .uncertainEffect : .directoryVerified))
      #expect(receipt.failure != nil)
      #expect(receipt.menuClosesAttempted == 1)
      #expect(backend.fake.performed.filter { $0 == "AXCancel@50" }.count == 1)
      #expect(receipt.steps.isEmpty)
    }
  }

  @Test(arguments: ["pid", "focus"])
  func finalMenuCancellationCannotApproveChangedOwnership(_ mode: String) async throws {
    try await withDirectories(1) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      let fake = backend.fake
      backend.onCancel = {
        if mode == "pid" {
          fake.processes[FakeUIBackend.fcp] = .init(pid: 999, bundleID: FakeUIBackend.fcp)
        } else {
          fake.frontmost = ["other"]
        }
      }
      let result = try await run(backend, request(paths))
      let receipt = try #require(result.saveNavigation)
      #expect(receipt.status == (mode == "pid" ? .processChanged : .focusChanged))
      #expect(!receipt.directoryVerified)
      #expect(result.effectVerified == false)
      #expect(receipt.menuClosesAttempted == 1)
      #expect(receipt.steps.isEmpty)
      #expect(backend.fake.written.isEmpty)
      #expect(!backend.fake.performed.contains("AXPress@74"))
    }
  }

  @Test(arguments: [
    "verified", "AX-error-after-effect", "unchanged", "same-name", "pid", "focus", "unknown",
  ])
  func mountedVolumeSelectionNeedsFreshWholeProofAndNeverSaves(_ mode: String) async throws {
    try await withDirectories(1) { roots in
      let root = try #require(roots.first)
      let boot = root.appendingPathComponent("Boot")
      let volume = root.appendingPathComponent("External")
      let target = volume.appendingPathComponent("Movies")
      try FileManager.default.createDirectory(at: boot, withIntermediateDirectories: false)
      try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
      let paths = [boot, volume, target]
      let backend = SaveNavigationBackend(paths: paths)
      backend.issueFreshHandles = true
      backend.separateVolumes = true
      backend.ignoreVolumeSwitch = mode == "unchanged"
      backend.errorAfterVolumeSwitch = mode == "AX-error-after-effect"
      backend.duplicateVolume = mode == "same-name"
      backend.update()
      let fake = backend.fake
      switch mode {
      case "pid":
        backend.onOpen = {
          fake.processes[FakeUIBackend.fcp] = .init(pid: 999, bundleID: FakeUIBackend.fcp)
        }
      case "focus": backend.onOpen = { fake.frontmost = ["other"] }
      case "unknown": backend.onOpen = { fake.attributes[51]?[.title] = .string("Unknown") }
      default: break
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(backend.fake.performed.filter { $0 == "AXPress@95" }.count <= 1)
      #expect(!backend.fake.performed.contains("AXPress@74"))
      #expect(backend.fake.written.isEmpty)
      #expect(backend.fake.activated.isEmpty)
      if mode == "verified" || mode == "AX-error-after-effect" {
        #expect(receipt.status == .directoryVerified)
        #expect(receipt.steps.count == 2)
        #expect(receipt.steps[0].selectionBasis == .mountedVolumeCandidate)
        #expect(receipt.steps[0].after?.directory == volume.path)
        #expect(receipt.steps[0].after?.volumeURL == volume.path)
        #expect(receipt.steps[1].selectionBasis == .verifiedChild)
        #expect(receipt.observations.last?.directory == target.path)
        if mode == "AX-error-after-effect" { #expect(receipt.steps[0].actionError != nil) }
      } else {
        #expect(receipt.status != .directoryVerified)
        #expect(receipt.steps.count <= 1)
        if mode == "same-name" { #expect(receipt.steps.isEmpty) }
      }
    }
  }

  @Test(arguments: [
    "wrong-app", "supplier-binding", "metadata-error", "backend-error", "cancelled",
  ])
  func supplierAndBoundaryFailuresHaveTypedReceiptsWithoutActions(_ mode: String) async throws {
    try await withDirectories(1) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      let environment: FCPSaveDirectoryEnvironment
      switch mode {
      case "supplier-binding": environment = .init { _ in [] }
      case "metadata-error": environment = .init { _ in throw CocoaError(.fileReadUnknown) }
      case "cancelled": environment = .init { _ in throw CancellationError() }
      default: environment = backend.environment
      }
      if mode == "backend-error" {
        backend.fake.attributeFailures[10] = .accessibility(-25205)
      }
      let automation = UIAutomation(backend: backend, saveDirectoryEnvironment: environment)
      let result = try await automation.run(
        .saveNavigate(.init(app: mode == "wrong-app" ? .motion : .finalCutPro), request(paths)))
      let receipt = try #require(result.saveNavigation)
      #expect(receipt.status != .directoryVerified)
      #expect(backend.fake.performed.isEmpty)
      #expect(receipt.steps.isEmpty)
      if mode == "metadata-error" {
        #expect(receipt.status == .backendFailure)
      }
      if mode == "backend-error" { #expect(receipt.status == .navigationUnavailable) }
      if mode == "cancelled" { #expect(receipt.status == .cancelled) }
    }
  }

  @Test func priorSaveDispatchOnlyObservesAndNeverOpensAMenu() async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      let receipt = try #require(
        try await run(backend, request(paths, dispatched: true)).saveNavigation)
      #expect(receipt.status == .observeExistingDispatch)
      #expect(backend.fake.performed.isEmpty)
      #expect(receipt.candidates.isEmpty)
    }
  }

  @Test func anOwnedCancelledTaskNeverOpensTheSaveMenu() async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      // The handle is cancelled before suspension and its result is always joined.
      let task = Task { @MainActor in try await run(backend, request(paths)) }
      defer { task.cancel() }
      task.cancel()
      let receipt = try #require(try await task.value.saveNavigation)
      #expect(receipt.status == .cancelled)
      #expect(receipt.steps.isEmpty)
      #expect(backend.fake.performed.isEmpty)
      #expect(backend.fake.activated.isEmpty)
    }
  }

  @Test func stepBudgetStopsAfterVerifiedProgress() async throws {
    try await withDirectories(3) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      let receipt = try #require(try await run(backend, request(paths, steps: 1)).saveNavigation)
      #expect(receipt.status == .stepBudgetExceeded)
      #expect(receipt.steps.count == 1)
      #expect(receipt.steps[0].after?.directory == paths[1].path)
    }
  }

  @Test(arguments: [
    "pid", "focus", "deadline", "missing-popup", "disabled", "unsupported-cancel", "missing-menu",
    "bad-menu", "ambiguous-menu", "unproved", "unsupported-open", "oversized-menu",
    "missing-children", "action-error", "invalid-title", "no-folder-match",
  ])
  func failClosedBeforeAnotherFolderAction(_ mode: String) async throws {
    try await withDirectories(2) { paths in
      let backend = SaveNavigationBackend(paths: paths)
      let fake = backend.fake
      switch mode {
      case "pid":
        backend.fake.processes[FakeUIBackend.fcp] = .init(pid: 999, bundleID: FakeUIBackend.fcp)
      case "focus": backend.onOpen = { fake.frontmost = ["other"] }
      case "deadline": backend.onOpen = { fake.clock = 16 }
      case "missing-popup": backend.fake.attributes[13]?[.identifier] = .string("other")
      case "disabled": backend.fake.attributes[13]?[.enabled] = .bool(false)
      case "unsupported-cancel": backend.fake.actionNames[50] = []
      case "missing-menu": backend.failShow = true
      case "bad-menu": backend.fake.attributes[51]?[.role] = .string("AXTextField")
      case "ambiguous-menu": backend.fake.attributes[13]?[.children] = .elements([50, 50])
      case "unproved": backend.fake.attributes[51]?[.title] = .string("Other volume")
      case "oversized-menu":
        backend.fake.attributes[50]?[.children] = .elements(Array(repeating: 51, count: 257))
      case "missing-children": backend.fake.attributes[50]?[.children] = .missing
      case "action-error": backend.fake.performFailure = .accessibility(-25205)
      case "invalid-title": backend.fake.attributes[51]?[.title] = .bool(true)
      case "no-folder-match": backend.fake.attributes[72]?[.value] = .string("Unrelated")
      default: backend.fake.actionNames[72] = []
      }
      let receipt = try #require(try await run(backend, request(paths)).saveNavigation)
      #expect(receipt.status != .directoryVerified)
      #expect(receipt.steps.count <= 1)
      #expect(backend.fake.written.isEmpty)
      #expect(!backend.fake.performed.contains("AXPress@74"))
    }
  }
}
