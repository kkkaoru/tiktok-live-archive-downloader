import Foundation
import Testing

@testable import ProAppsCore

struct FCPSavePlannerTests {
  @Test(arguments: ["ユーザ", "Macintosh HD"])
  func internalWholeChainProofSupportsOSDisplayNamesWithoutChangingLegacy(_ title: String) throws {
    let directory = try CoreTests().temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let candidate = FCPSaveDirectoryCandidate(
      directory: directory.path, volumeURL: "/",
      variants: [.init(languages: ["ja"], titles: [title, "OS volume"])])
    let proof = try FCPSaveDirectory.prove([title, "OS volume"], candidates: [candidate])
    var elements = panel(directory)
    elements[0] = element(
      0, id: "where popup", role: "AXPopUpButton", value: title, actions: ["AXShowMenu"])
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(context(directory), elements: elements)
    }
    let plan = try FCPSavePlanner.plan(
      context(directory), elements: elements, verifiedDirectory: proof)
    #expect(plan.directoryBinding == "native-full-ordered-OS-localized-directory-ancestry")
    #expect(plan.state == "set-name-and-observe")
    let unproved = try FCPSaveDirectory.prove([title], candidates: [candidate])
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(context(directory), elements: elements, verifiedDirectory: unproved)
    }
    var mismatch = proof
    mismatch.directory = "/not-current"
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(context(directory), elements: elements, verifiedDirectory: mismatch)
    }
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(
        context(directory), elements: panel(directory), verifiedDirectory: proof)
    }
  }

  @Test func internalAncestorDisplayNamesAreBoundAndNeverAcceptCallerJSON() throws {
    let directory = try CoreTests().temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let child = directory.appendingPathComponent("other")
    try FileManager.default.createDirectory(at: child, withIntermediateDirectories: false)
    let ancestor = FCPSaveDirectoryCandidate(
      directory: directory.path, volumeURL: directory.path,
      variants: [.init(languages: ["ja"], titles: ["Localized volume"])])
    let candidate = FCPSaveDirectoryCandidate(
      directory: child.path, volumeURL: directory.path,
      variants: [.init(languages: ["ja"], titles: ["other", "Localized volume"])])
    let proof = try FCPSaveDirectory.prove(["other", "Localized volume"], candidates: [candidate])
    var request = context(directory, current: child)
    request.navigationAncestor = directory.path
    let elements = panel(child) + [element(3, role: "AXMenuItem", title: "Localized volume")]
    let plan = try FCPSavePlanner.plan(
      request, elements: elements, verifiedDirectory: proof, verifiedNavigationAncestor: ancestor)
    #expect(plan.state == "select-observed-ancestor-and-observe")
    #expect(plan.expectedDirectoryAfterAction == directory.path)
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(
        request, elements: elements, verifiedDirectory: proof, verifiedNavigationAncestor: candidate
      )
    }
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(
        request, elements: elements, verifiedDirectory: nil, verifiedNavigationAncestor: ancestor)
    }
    let recentWithSameTitle = FCPSaveDirectoryCandidate(
      directory: directory.path, volumeURL: directory.path,
      variants: [.init(languages: ["en"], titles: ["Localized volume"])])
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(
        request, elements: elements, verifiedDirectory: proof,
        verifiedNavigationAncestor: recentWithSameTitle)
    }
  }

  private func context(
    _ directory: URL, current: URL? = nil, name: String = "new.mov",
    dispatched: Bool = false
  ) -> FCPSaveContext {
    FCPSaveContext(
      directory: directory.path, fileName: name,
      observedDirectory: (current ?? directory).path,
      window: .init(title: "Save"), bundleID: nil, saveDispatched: dispatched)
  }

  private func element(
    _ index: Int, id: String? = nil, role: String = "AXButton",
    value: String? = nil, actions: [String] = ["AXPress"], title: String? = nil,
    enabled: Bool = true, writable: Bool = false,
    path: [Int]? = nil
  ) -> UIElementSummary {
    .init(
      path: path ?? [index], role: role, subrole: nil, identifier: id,
      title: title, description: nil, value: value, enabled: enabled, selected: nil,
      focused: nil, actions: actions, valueSettable: writable, frame: nil, unreadable: [])
  }

  private func panel(_ directory: URL, name: String = "old.mov") -> [UIElementSummary] {
    [
      element(
        0, id: "where popup", role: "AXPopUpButton", value: directory.lastPathComponent,
        actions: ["AXShowMenu"]),
      element(1, id: "saveAsNameTextField", role: "AXTextField", value: name, writable: true),
      element(2, id: "OKButton"),
    ]
  }

  @Test func namesThenPlansSaveWithoutDispatching() throws {
    let directory = try CoreTests().temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let request = context(directory)
    let name = try FCPSavePlanner.plan(request, elements: panel(directory))
    #expect(name.state == "set-name-and-observe")
    #expect(!name.dispatched && !name.completionVerified)
    let save = try FCPSavePlanner.plan(request, elements: panel(directory, name: "new.mov"))
    #expect(save.state == "ready-to-dispatch-save")
    #expect(save.outputPath == directory.appendingPathComponent("new.mov").path)
    let encoded = try JSONEncoder().encode(save)
    #expect(String(decoding: encoded, as: UTF8.self).contains("completionVerified"))
  }

  @Test func exactChildUsesOpenNotConfirm() throws {
    let directory = try CoreTests().temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let child = directory.appendingPathComponent("日本語 folder")
    try FileManager.default.createDirectory(at: child, withIntermediateDirectories: false)
    let browser = element(3, role: "AXBrowser")
    let cell = element(0, role: "AXCell", actions: ["AXOpen"], path: [3, 0])
    let label = element(0, role: "AXStaticText", value: "日本語 folder", path: [3, 0, 0])
    let elements = panel(directory) + [browser, cell, label]
    let result = try FCPSavePlanner.plan(context(child, current: directory), elements: elements)
    #expect(result.state == "open-child-and-observe")
    guard case .perform(_, _, let locator, let action) = result.next else {
      Issue.record("Expected typed open")
      return
    }
    #expect(action == .open && locator.containsText == "日本語 folder")
    let direct = element(
      0, role: "AXTextField", value: "日本語 folder", actions: ["AXOpen"], path: [3, 0, 0])
    let directPlan = try FCPSavePlanner.plan(
      context(child, current: directory),
      elements: panel(directory) + [browser, cell, direct])
    guard case .perform(_, _, let directLocator, .open) = directPlan.next else {
      Issue.record("Expected advertised text-field open")
      return
    }
    #expect(directLocator.path == [3, 0, 0])
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(context(child, current: directory), elements: panel(directory))
    }
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(
        context(child, current: directory), elements: panel(directory) + [browser])
    }
  }

  @Test func differentAncestorRequestsMenuAndDispatchedNeverRepeats() throws {
    let directory = try CoreTests().temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let other = directory.appendingPathComponent("other")
    try FileManager.default.createDirectory(at: other, withIntermediateDirectories: false)
    let menu = try FCPSavePlanner.plan(context(directory, current: other), elements: panel(other))
    #expect(menu.state == "observe-location-menu")
    try Data().write(to: directory.appendingPathComponent("new.mov"))
    let observed = try FCPSavePlanner.plan(
      context(directory, dispatched: true), elements: panel(directory))
    #expect(observed.state == "observe-existing-dispatch" && observed.next == nil)
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(context(directory), elements: panel(directory))
    }
  }

  @Test func recentAncestorSelectionIsExplicitAndUnique() throws {
    let directory = try CoreTests().temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let other = directory.appendingPathComponent("other")
    try FileManager.default.createDirectory(at: other, withIntermediateDirectories: false)
    var request = context(directory, current: other)
    request.navigationAncestor = directory.path
    let menu = panel(other) + [element(3, role: "AXMenuItem", title: directory.lastPathComponent)]
    let planned = try FCPSavePlanner.plan(request, elements: menu)
    #expect(planned.state == "select-observed-ancestor-and-observe")
    #expect(planned.expectedDirectoryAfterAction == directory.path)
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(
        request,
        elements: menu + [element(4, role: "AXMenuItem", title: directory.lastPathComponent)])
    }
    request.navigationAncestor = other.path
    #expect(throws: ProAppsError.self) { try FCPSavePlanner.plan(request, elements: menu) }
    request.navigationAncestor = directory.path
    let noMenu = try FCPSavePlanner.plan(request, elements: panel(other))
    #expect(noMenu.state == "observe-location-menu")
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(
        request,
        elements: [
          element(
            0, id: "where popup", role: "AXPopUpButton",
            value: other.lastPathComponent, actions: [])
        ])
    }
  }

  @Test func rejectsSymlinksAndInvalidBasenames() throws {
    let directory = try CoreTests().temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let link = directory.appendingPathComponent("link")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: directory)
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.canonicalDirectoryForPlanning(link.path)
    }
    try FileManager.default.createSymbolicLink(
      atPath: directory.appendingPathComponent("new.mov").path,
      withDestinationPath: directory.appendingPathComponent("missing").path)
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(context(directory), elements: panel(directory))
    }
  }

  @Test(arguments: ["", "..", "nested/file.mov", "nul\0.mov", "no-extension"])
  func invalidBasenameFails(_ name: String) throws {
    let directory = try CoreTests().temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(context(directory, name: name), elements: panel(directory))
    }
  }

  @Test(arguments: 0..<7) func changedAmbiguousAndDisabledControlsFail(_ index: Int) throws {
    let directory = try CoreTests().temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let standard = panel(directory, name: "new.mov")
    let cases: [[UIElementSummary]] = [
      [], standard + standard,
      [element(0, id: "where popup", role: "AXPopUpButton", value: "elsewhere")],
      [standard[0]],
      [standard[0], element(1, id: "saveAsNameTextField", role: "AXTextField")],
      [standard[0], standard[1], element(2, id: "OKButton", enabled: false)],
      standard + [element(3, id: "OKButton")],
    ]
    #expect(throws: ProAppsError.self) {
      try FCPSavePlanner.plan(context(directory), elements: cases[index])
    }
  }
}
