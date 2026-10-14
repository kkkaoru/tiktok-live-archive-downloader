import Foundation
import Testing

@testable import ProAppsCore

@MainActor
private final class PageBackend: UIBackend {
  let fake = FakeUIBackend.standard()
  var pages: [UIHandle: [UIHandle]] = [72: [80, 81], 80: [90, 91], 81: []]
  var requests: [(UIHandle, Int, Int)] = []
  var onPage: ((UIHandle, Int) throws -> Void)?
  var counts: [UIHandle: Int] = [:]
  var shorten = false
  init() {
    fake.attributes[1]?[.windows] = .elements([10])
    fake.attributes[10]?[.children] = .elements([11])
    fake.attributes[11] = [
      .role: .string("AXSheet"), .identifier: .string("save-panel"),
      .children: .elements([70, 71, 72]),
    ]
    fake.attributes[70] = [
      .role: .string("AXPopUpButton"), .identifier: .string("where popup"),
      .children: .elements([]),
    ]
    fake.attributes[71] = [
      .role: .string("AXTextField"), .identifier: .string("saveAsNameTextField"),
      .value: .string("original.mov"), .children: .elements([]),
    ]
    fake.attributes[72] = [.role: .string("AXBrowser"), .identifier: .string("ColumnView")]
    fake.attributes[80] = [.role: .string("AXScrollArea")]
    fake.attributes[81] = [.role: .string("AXScrollArea")]
    fake.attributes[90] = [
      .role: .string("AXGroup"), .identifier: .string("row"),
      .url: .string("file:///Volumes/example/movie"), .selected: .bool(false),
      .value: .string("movie"),
    ]
    fake.attributes[91] = [
      .role: .string("AXGroup"), .url: .string("https://example.org"), .selected: .bool(true),
    ]
  }
  func childCount(of h: UIHandle) throws -> Int {
    if let count = counts[h] { return count }
    if let page = pages[h] { return page.count }
    return try fake.childCount(of: h)
  }
  func childPage(of h: UIHandle, offset: Int, count: Int) throws -> [UIHandle] {
    requests.append((h, offset, count))
    try onPage?(h, requests.count)
    if let page = pages[h] {
      guard offset >= 0, count > 0, offset <= page.count, count <= page.count - offset else {
        throw UIBackendError.unavailable("page range")
      }
      if shorten { return [] }
      return Array(page[offset..<(offset + count)])
    }
    return try fake.childPage(of: h, offset: offset, count: count)
  }
  func attribute(_ name: UIAttributeName, of h: UIHandle) throws -> UIAttributeValue {
    if h >= 72, name == .children {
      throw UIBackendError.unavailable("Forbidden full browser children read")
    }
    return try fake.attribute(name, of: h)
  }
  func attributeNames(of h: UIHandle) throws -> [String] { try fake.attributeNames(of: h) }
  func processID(of h: UIHandle) throws -> Int32 { try fake.processID(of: h) }
  func runningProcess(bundleID: String) -> UIRunningProcess? {
    fake.runningProcess(bundleID: bundleID)
  }
  func frontmostBundleID() -> String? { fake.frontmostBundleID() }
  func frontmostProcess() -> UIRunningProcess? { fake.frontmostProcess() }
  func activate(process: UIRunningProcess) -> Bool { fake.activate(process: process) }
  func applicationElement(pid: Int32) -> UIHandle { fake.applicationElement(pid: pid) }
  func actions(of h: UIHandle) throws -> [String] { try fake.actions(of: h) }
  func isSettable(_ name: UIAttributeName, of h: UIHandle) throws -> Bool {
    try fake.isSettable(name, of: h)
  }
  func perform(_ action: String, on h: UIHandle) throws { try fake.perform(action, on: h) }
  func set(_ name: UIAttributeName, to value: UISettableValue, on h: UIHandle) throws {
    try fake.set(name, to: value, on: h)
  }
  func pause(seconds: Double) async throws { try await fake.pause(seconds: seconds) }
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
struct FCPSaveBrowserPageTests {
  @Test(arguments: ["origin", "nested", "empty", "second"])
  func observesOnlyRequestedPage(_ mode: String) async throws {
    let b = PageBackend()
    let path = mode == "origin" ? [] : [mode == "empty" ? 1 : 0]
    let request = UIRequest.saveBrowserPage(
      .init(app: .finalCutPro),
      .init(
        expectedPID: 42, window: .init(title: "Main"), relativePath: path,
        offset: mode == "second" ? 1 : 0, limit: 1))
    #expect(
      try JSONDecoder().decode(UIRequest.self, from: JSONEncoder().encode(request)) == request)
    let response = try await UIAutomation(backend: b).run(request)
    let receipt = try #require(response.saveBrowserPage)
    #expect(receipt.complete)
    #expect(receipt.rows.count == (mode == "empty" ? 0 : 1))
    #expect(!receipt.actionsAttempted && !receipt.currentDirectoryVerified)
    #expect(response.dispatched == false && response.effectVerified == false)
    #expect(b.requests.allSatisfy { $0.2 == 1 })
    #expect(b.fake.performed.isEmpty && b.fake.written.isEmpty && b.fake.activated.isEmpty)
    if mode == "nested" {
      #expect(receipt.rows.first?.url.lexicalFilePath == "/Volumes/example/movie")
    }
    if mode == "second" { #expect(receipt.rows.first?.url.lexicalFilePath == nil) }
    if mode == "origin" { #expect(!b.requests.contains { $0.0 == 80 || $0.0 == 81 }) }
  }

  @Test(arguments: [
    "wrong-pid", "bad-app", "no-title", "deep", "negative", "large-offset", "large-limit",
    "outside", "bad-browser", "large-count", "path-outside", "foreign-origin", "foreign-row",
    "missing-role", "identifier", "selected", "names", "short-page", "changed-page", "changed-name",
    "changed-owner", "changed-count", "changed-row", "ax", "cancel", "unknown", "deadline",
    "foreign-name", "ambiguous-browser", "changed-parent", "changed-window",
  ])
  func refusesInvalidOrChangedEvidence(_ mode: String) async throws {
    let b = PageBackend()
    if mode == "bad-browser" { b.fake.attributes[72]?[.identifier] = .string("other") }
    if mode == "large-count" { b.counts[80] = 4097 }
    if mode == "foreign-origin" { b.fake.identity = { _, _ in 99 } }
    if mode == "foreign-row" { b.fake.identity = { h, _ in h == 90 ? 99 : 42 } }
    if mode == "foreign-name" { b.fake.identity = { h, _ in h == 71 ? 99 : 42 } }
    if mode == "ambiguous-browser" {
      b.fake.attributes[73] = [.role: .string("AXBrowser"), .identifier: .string("other")]
      b.fake.attributes[11]?[.children] = .elements([70, 71, 72, 73])
    }
    if mode == "missing-role" { b.fake.attributes[90]?[.role] = .missing }
    if mode == "identifier" { b.fake.attributes[90]?[.identifier] = .bool(true) }
    if mode == "selected" { b.fake.attributes[90]?[.selected] = .number(1) }
    if mode == "names" { b.fake.nameObservation = { _ in ["AXURL", "AXURL"] } }
    if mode == "short-page" { b.shorten = true }
    var rowReads = 0
    b.fake.nameObservation =
      mode == "names"
      ? b.fake.nameObservation
      : { h in
        if h == 90 {
          rowReads += 1
          if rowReads == (mode == "changed-row" ? 2 : 1) {
            switch mode {
            case "changed-page": b.pages[80] = [91, 90]
            case "changed-name": b.fake.attributes[71]?[.value] = .string("changed.mov")
            case "changed-owner": b.fake.processes = [:]
            case "changed-count": b.pages[80] = [90]
            case "changed-row": b.fake.attributes[90]?[.value] = .string("changed")
            case "changed-parent": b.pages[72] = [81, 80]
            case "changed-window":
              b.fake.attributes[12] = b.fake.attributes[10]
              b.fake.attributes[1]?[.windows] = .elements([12])
            case "ax": throw UIBackendError.accessibility(-25202)
            case "cancel": throw CancellationError()
            case "unknown": throw CocoaError(.fileReadUnknown)
            case "deadline": b.fake.clock = 10
            default: break
            }
          }
        }
        return (b.fake.attributes[h] ?? [:]).keys.map(\.rawValue)
      }
    let path =
      mode == "deep"
      ? [0, 0, 0, 0, 0, 0, 0] : [mode == "negative" ? -1 : (mode == "path-outside" ? 2 : 0)]
    let response = try await UIAutomation(backend: b).run(
      .saveBrowserPage(
        .init(app: mode == "bad-app" ? .motion : .finalCutPro),
        .init(
          expectedPID: mode == "wrong-pid" ? 99 : 42,
          window: .init(title: mode == "no-title" ? nil : "Main"), relativePath: path,
          offset: mode == "large-offset" ? 4097 : (mode == "outside" ? 3 : 0),
          limit: mode == "large-limit" ? 17 : 1)))
    #expect(response.saveBrowserPage?.complete == false)
    #expect(response.saveBrowserPage?.failure != nil)
    #expect(b.fake.performed.isEmpty && b.fake.written.isEmpty && b.fake.activated.isEmpty)
  }
}
