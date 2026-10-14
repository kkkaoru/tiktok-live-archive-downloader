import Foundation

private struct SaveNavigationFailure: Error {
  let status: FCPSaveNavigationStatus
}

private struct SaveDirectoryRead {
  let proof: FCPSaveDirectoryObservation
  let elements: [UIElementSummary]
  let menu: UIHandle
  let controls: SaveControlRead
  let childSelection: SaveChildSelection?
}

private struct SaveChildSelection {
  let list: UILocatedElement
  let row: UILocatedElement
  let count: Int
  let directory: String
  let parentBound: Bool
}

struct SaveControlRead {
  let elements: [UIElementSummary]
  let popup: UILocatedElement
  let name: String
  let panel: UILocatedElement
  let nameElement: UILocatedElement
}

private struct SaveControlIdentity: Equatable {
  let path: [Int]
  let role: String
  let identifier: String
  let name: String?
}

private struct SaveMenu {
  let handle: UIHandle
  let offset: Int
}

private struct SaveMenuItemFingerprint: Equatable {
  let role: String
  let identifier: String?
  let title: String?
}

private struct SaveMenuFingerprint: Equatable {
  let identifier: String
  let items: [SaveMenuItemFingerprint]
}

private struct SaveMenuSnapshot {
  let items: [UIHandle]
  let fingerprint: SaveMenuFingerprint
}

extension UIAutomation {
  func saveNavigate(_ target: UITarget, request: FCPSaveNavigationRequest) async -> UIResponse {
    let started = backend.monotonicSeconds()
    var receipt = FCPSaveNavigationReceipt(requestedDirectory: request.directory)
    var response = UIResponse(focus: .init(frontmostBefore: nil, frontmostAfter: nil))
    let foreground = backend.frontmostProcess()
    let foregroundBundle = backend.frontmostBundleID()
    do {
      guard target.app == .finalCutPro else {
        throw ProAppsError.invalid("Save navigation requires FCP")
      }
      let paths = try request.validatedDirectories()
      try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
      response.pid = request.expectedPID
      if request.saveDispatched {
        receipt.status = .observeExistingDispatch
      } else {
        receipt.candidates = try saveDirectoryEnvironment.describe(paths)
        guard Set(receipt.candidates.map(\.directory)) == Set(paths),
          receipt.candidates.count == paths.count,
          try JSONEncoder().encode(receipt.candidates).count <= 384 * 1024
        else {
          throw ProAppsError.invalid("OS directory supplier does not bind explicit candidates")
        }
        try FCPSavePlanner.requireAbsent(
          URL(fileURLWithPath: request.directory).appendingPathComponent(request.fileName))
        try await navigateSaveDirectory(
          target, request, started, foreground, foregroundBundle, receipt: &receipt)
      }
    } catch let error as SaveNavigationFailure {
      receipt.status = error.status
    } catch is CancellationError {
      receipt.status = .cancelled
    } catch let error as ProAppsError {
      receipt.status = .navigationUnavailable
      receipt.failure = String(error.description.prefix(512))
    } catch let error as UIBackendError {
      receipt.status = .backendFailure
      receipt.failure = String(String(describing: error).prefix(512))
    } catch {
      receipt.status = .backendFailure
      receipt.failure = "Directory metadata or native navigation failed; observe before retrying"
    }
    receipt.elapsedSeconds = backend.monotonicSeconds() - started
    response.saveNavigation = receipt
    response.dispatched =
      receipt.menuShowsAttempted > 0 || receipt.steps.contains(where: \.actionAttempted)
    response.effectVerified = receipt.directoryVerified
    return response
  }

  private func checkSaveOwnership(
    _ target: UITarget, _ request: FCPSaveNavigationRequest, _ started: Double,
    _ foreground: UIRunningProcess?, _ foregroundBundle: String?
  ) throws {
    try Task.checkCancellation()
    guard try requireProcess(target).pid == request.expectedPID else {
      throw SaveNavigationFailure(status: .processChanged)
    }
    guard backend.frontmostProcess() == foreground,
      backend.frontmostBundleID() == foregroundBundle
    else { throw SaveNavigationFailure(status: .focusChanged) }
    guard backend.monotonicSeconds() - started < request.maximumSeconds else {
      throw SaveNavigationFailure(status: .deadlineExceeded)
    }
  }

  private func navigateSaveDirectory(
    _ target: UITarget, _ request: FCPSaveNavigationRequest, _ started: Double,
    _ foreground: UIRunningProcess?, _ foregroundBundle: String?,
    receipt: inout FCPSaveNavigationReceipt
  ) async throws {
    let check = {
      try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
    }
    var read = try await readSaveDirectory(
      target, request, started, foreground, foregroundBundle, receipt: &receipt)
    while true {
      guard let current = read.proof.directory else {
        try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
        try closeSaveMenu(
          read.menu, target: target, window: request.window, receipt: &receipt, check: check)
        receipt.status = .unprovedDirectory
        return
      }
      if current == request.directory {
        try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
        try closeSaveMenu(
          read.menu, target: target, window: request.window, receipt: &receipt, check: check)
        receipt.status = .directoryVerified
        return
      }
      guard receipt.steps.count < request.maximumSteps else {
        try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
        try closeSaveMenu(
          read.menu, target: target, window: request.window, receipt: &receipt, check: check)
        receipt.status = .stepBudgetExceeded
        return
      }
      try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
      var context = FCPSaveContext(
        directory: request.directory, fileName: request.fileName, observedDirectory: current,
        window: request.window, bundleID: target.bundleID, saveDispatched: false)
      let ancestor = navigationAncestor(
        request.directory, proof: read.proof, candidates: receipt.candidates,
        elements: read.elements)
      context.navigationAncestor = ancestor?.directory
      let plan = try FCPSavePlanner.plan(
        context, elements: read.elements, verifiedDirectory: read.proof,
        verifiedNavigationAncestor: ancestor)
      guard let action = plan.next, let expected = plan.expectedDirectoryAfterAction,
        receipt.candidates.contains(where: { $0.directory == expected }),
        case .perform(_, _, let element, let kind) = action,
        kind == .open || (kind == .press && element.role == "AXMenuItem")
      else {
        try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
        try closeSaveMenu(
          read.menu, target: target, window: request.window, receipt: &receipt, check: check)
        receipt.status = .navigationUnavailable
        return
      }
      if kind == .open {
        try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
        try closeSaveMenu(
          read.menu, target: target, window: request.window, receipt: &receipt, check: check)
      }
      try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
      let identity = try saveControlIdentity(read.controls)
      let selection = kind == .open ? read.childSelection : nil
      if let selection {
        guard selection.directory == expected,
          try backend.isSettable(.selectedChildren, of: selection.list.handle),
          try backend.childCount(of: selection.list.handle) == selection.count,
          let offset = selection.row.path.last
        else { throw ProAppsError.unavailable("Verified Save list selection is unavailable") }
        let row = try backend.childPage(of: selection.list.handle, offset: offset, count: 1)
        guard let handle = row.first, try backend.sameElement(handle, selection.row.handle),
          try saveChildMatches(
            handle, expected: expected, parentBound: selection.parentBound, check: check)
        else { throw ProAppsError.invalid("Verified Save selection changed before dispatch") }
      }
      let index = receipt.steps.count
      receipt.steps.append(
        .init(
          fromDirectory: current, expectedDirectory: expected,
          action: selection == nil ? action : nil))
      if let selection {
        receipt.steps[index].selectedChildrenPath = selection.list.path
        receipt.steps[index].selectedChildPath = selection.row.path
      }
      if kind == .open {
        receipt.steps[index].selectionBasis =
          selection?.parentBound == true ? .verifiedParentChild : .verifiedChild
      } else {
        receipt.steps[index].selectionBasis =
          ancestor?.ancestorLabels(of: read.proof).isEmpty == false
          ? .verifiedCurrentAncestor : .mountedVolumeCandidate
      }
      // Record the attempt before dispatch: AX can fail after the directory changed.
      receipt.steps[index].actionAttempted = true
      do {
        if let selection, let offset = selection.row.path.last {
          try backend.selectChild(
            of: selection.list.handle, at: offset, expected: selection.row.handle)
        } else {
          _ = try await run(action)
        }
      } catch is CancellationError {
        throw CancellationError()
      } catch let error as ProAppsError {
        receipt.steps[index].actionError = String(error.description.prefix(512))
      } catch let error as UIBackendError {
        receipt.steps[index].actionError = String(String(describing: error).prefix(512))
      }
      // AX dispatch can precede the panel's asynchronous directory transition.
      // Observe it without repeating the action, then require a fresh full chain.
      read = try await awaitSaveDirectory(
        expected, previous: read, identity: identity, target, request, started, foreground,
        foregroundBundle,
        receipt: &receipt)
      receipt.steps[index].after = read.proof
      guard read.proof.directory == expected else {
        try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
        try closeSaveMenu(
          read.menu, target: target, window: request.window, receipt: &receipt, check: check)
        receipt.status = .uncertainEffect
        return
      }
      // Commit the proved effect before discovering another column. A changing
      // next column must not erase the previous action's successful readback.
      if expected != request.directory {
        read = try await readSaveDirectory(
          target, request, started, foreground, foregroundBundle, receipt: &receipt)
      }
    }
  }

  private func saveControlIdentity(_ controls: SaveControlRead) throws -> [SaveControlIdentity] {
    let panels = controls.elements.filter {
      ($0.role == "AXSheet" || $0.role == "AXWindow") && $0.identifier == "save-panel"
    }
    guard panels.count == 1, let panel = panels.first, let panelRole = panel.role else {
      throw ProAppsError.unavailable("A unique Save panel identity is required")
    }
    var required = [
      (panelRole, "save-panel"), ("AXPopUpButton", "where popup"),
      ("AXTextField", "saveAsNameTextField"), ("AXButton", "OKButton"),
    ]
    if panel.role == "AXWindow" { required.append(("AXButton", "CancelButton")) }
    return try required.map { role, identifier in
      let matches = controls.elements.filter { $0.role == role && $0.identifier == identifier }
      guard matches.count == 1, let match = matches.first else {
        throw ProAppsError.unavailable("Save navigation controls changed or are ambiguous")
      }
      let isName = identifier == "saveAsNameTextField"
      return .init(
        path: match.path, role: role, identifier: identifier, name: isName ? controls.name : nil)
    }
  }

  private func awaitSaveDirectory(
    _ expected: String, previous: SaveDirectoryRead, identity: [SaveControlIdentity],
    _ target: UITarget,
    _ request: FCPSaveNavigationRequest, _ started: Double,
    _ foreground: UIRunningProcess?, _ foregroundBundle: String?,
    receipt: inout FCPSaveNavigationReceipt
  ) async throws -> SaveDirectoryRead {
    let labels = Set(
      receipt.candidates.filter { $0.directory == expected }.flatMap { candidate in
        candidate.variants.filter {
          !$0.languages.filter(previous.proof.languages.contains).isEmpty
        }
        .compactMap { $0.titles.first }
      })
    let check = {
      try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
    }
    let waitStarted = backend.monotonicSeconds()
    for attempt in 0..<21 {
      try check()
      let controls = try readSaveControls(
        target, window: request.window, receipt: &receipt, check: check)
      guard try saveControlIdentity(controls) == identity else {
        throw ProAppsError.unavailable(
          "Save navigation controls or filename changed during transition")
      }
      let value = try menuText(.value, of: controls.popup.handle)
      let ready = try saveMenus(controls.popup.handle).isEmpty && value.map(labels.contains) == true
      if ready || attempt == 20 || backend.monotonicSeconds() - waitStarted >= 2 { break }
      try await backend.pause(seconds: 0.1)
    }
    // A popup label is only a readiness hint, never canonical-path authority.
    // The timeout also gets one full read so uncertain effects remain observable.
    var fresh = try await readSaveDirectory(
      target, request, started, foreground, foregroundBundle, discoverChild: false,
      receipt: &receipt)
    for attempt in 0..<21 {
      guard try saveControlIdentity(fresh.controls) == identity else {
        throw ProAppsError.unavailable(
          "Save navigation controls changed during full path verification")
      }
      if fresh.proof.directory == expected || attempt == 20
        || backend.monotonicSeconds() - waitStarted >= 2
      {
        return fresh
      }
      try await backend.pause(seconds: 0.1)
      // Reuse the one opened menu while independently recapturing its full chain.
      // If it closes or changes, fail rather than dispatching another ShowMenu.
      fresh = try await readSaveDirectory(
        target, request, started, foreground, foregroundBundle, showMenuAllowed: false,
        discoverChild: false, receipt: &receipt)
    }
    return fresh
  }

  private func navigationAncestor(
    _ directory: String, proof: FCPSaveDirectoryObservation,
    candidates: [FCPSaveDirectoryCandidate], elements: [UIElementSummary]
  ) -> FCPSaveDirectoryCandidate? {
    let target = URL(fileURLWithPath: directory).pathComponents
    return candidates.filter { candidate in
      target.starts(with: URL(fileURLWithPath: candidate.directory).pathComponents)
        && elements.contains { element in
          element.role == "AXMenuItem"
            && element.title.map(candidate.navigationLabels(of: proof, toward: directory).contains)
              == true
            && element.enabled == true && element.actions.contains("AXPress")
        }
    }.max { $0.directory.count < $1.directory.count }
  }

  private func readSaveDirectory(
    _ target: UITarget, _ request: FCPSaveNavigationRequest, _ started: Double,
    _ foreground: UIRunningProcess?, _ foregroundBundle: String?,
    showMenuAllowed: Bool = true,
    discoverChild: Bool = true,
    receipt: inout FCPSaveNavigationReceipt
  ) async throws -> SaveDirectoryRead {
    try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
    let check = {
      try checkSaveOwnership(target, request, started, foreground, foregroundBundle)
    }
    let controls = try readSaveControls(
      target, window: request.window, receipt: &receipt, check: check)
    var elements = controls.elements
    var childSelection: SaveChildSelection?
    let popup = controls.popup
    guard try backend.attribute(.enabled, of: popup.handle) == .bool(true),
      try backend.actions(of: popup.handle).contains("AXShowMenu")
    else { throw ProAppsError.unavailable("Save location popup is not enabled") }
    var menus = try saveMenus(popup.handle)
    if menus.isEmpty {
      guard showMenuAllowed else {
        throw ProAppsError.unavailable(
          "Save ancestor menu closed during independent settling readback")
      }
      receipt.menuShowsAttempted += 1
      try backend.perform("AXShowMenu", on: popup.handle)
      let waitStarted = backend.monotonicSeconds()
      while menus.isEmpty {
        try check()
        menus = try saveMenus(popup.handle)
        if !menus.isEmpty { break }
        guard backend.monotonicSeconds() - waitStarted < 2 else {
          do {
            try observeSaveMenuAttachments(
              popup, target: target, window: request.window, receipt: &receipt, check: check)
          } catch let error as ProAppsError {
            receipt.menuAttachmentFailure = String(error.description.prefix(512))
          } catch let error as UIBackendError {
            receipt.menuAttachmentFailure = String(String(describing: error).prefix(512))
          }
          throw ProAppsError.unavailable("Save location ancestor menu did not appear")
        }
        try await backend.pause(seconds: 0.1)
      }
    }
    guard menus.count == 1, let menu = menus.first else {
      throw ProAppsError.invalid("Save location ancestor menu is ambiguous")
    }
    let menuIndex = receipt.inspections.count
    receipt.inspections.append(.init(scope: .ancestorMenu))
    let captured = try saveMenuSnapshot(menu, check: check)
    var titles: [String] = []
    var readingAncestors = true
    for (offset, item) in captured.items.enumerated() {
      try check()
      let observed = captured.fingerprint.items[offset]
      if readingAncestors {
        if observed.role == "AXSeparator" || observed.title == "" {
          readingAncestors = false
        } else {
          guard observed.role == "AXMenuItem", let text = observed.title else {
            throw ProAppsError.invalid("Save ancestor menu has unsupported entries")
          }
          titles.append(text)
        }
      }
      elements.append(try summary(item, path: popup.path + [menu.offset, offset]))
      receipt.inspections[menuIndex].elementCount += 1
    }
    // Handles are per-read storage slots, not native AX identity. Keep the offset
    // from the coherent first read and compare one fresh bounded semantic capture.
    let freshMenus = try saveMenus(popup.handle)
    guard freshMenus.count == 1, let freshMenu = freshMenus.first,
      freshMenu.offset == menu.offset,
      try saveMenuSnapshot(freshMenu, check: check).fingerprint == captured.fingerprint
    else {
      throw ProAppsError.unavailable("Save ancestor menu identity or ordered labels changed")
    }
    receipt.inspections[menuIndex].complete = true
    let proof = try FCPSaveDirectory.prove(titles, candidates: receipt.candidates)
    receipt.observations.append(proof)
    if discoverChild, let current = proof.directory, current != request.directory,
      URL(fileURLWithPath: request.directory).pathComponents.starts(
        with: URL(fileURLWithPath: current).pathComponents)
    {
      // Only AXOpen planning needs browser contents. Menu navigation and proof
      // of an already-selected destination do not depend on unrelated file rows.
      let browsers = elements.filter { $0.role == "AXBrowser" }
      guard browsers.count == 1, let browser = browsers.first else {
        throw ProAppsError.unavailable("A unique Save column browser is required")
      }
      let window = try resolveWindow(request.window, in: requireProcess(target))
      let located = try resolveElement(.init(path: browser.path, role: "AXBrowser"), in: window)
      elements.removeAll { $0.path == browser.path }
      let maximumPlanElements = 500
      elements += try inspectSaveColumn(
        located, current: current, destination: request.directory,
        maximumElements: maximumPlanElements - elements.count, receipt: &receipt,
        selection: &childSelection, check: check)
    }
    try check()
    return SaveDirectoryRead(
      proof: proof, elements: elements, menu: freshMenu.handle, controls: controls,
      childSelection: childSelection)
  }

  func readSaveControls(
    _ target: UITarget, window locator: UIWindowLocator,
    receipt: inout FCPSaveNavigationReceipt, check: () throws -> Void
  ) throws -> SaveControlRead {
    try check()
    let window = try resolveWindow(locator, in: requireProcess(target))
    // New Library uses a standalone Save dialog; Share uses a direct sheet.
    // Neither form permits searching unrelated windows or browser contents.
    var panels = try saveChildren(window).enumerated().compactMap { offset, handle in
      guard try backend.attribute(.role, of: handle) == .string("AXSheet"),
        try backend.attribute(.identifier, of: handle) == .string("save-panel")
      else { return nil as UILocatedElement? }
      return UILocatedElement(handle: handle, path: [offset])
    }
    let standalone =
      try backend.attribute(.role, of: window) == .string("AXWindow")
      && backend.attribute(.subrole, of: window) == .string("AXDialog")
      && backend.attribute(.identifier, of: window) == .string("save-panel")
    if standalone { panels.append(UILocatedElement(handle: window, path: [])) }
    guard panels.count == 1, let panel = panels.first else {
      throw ProAppsError.unavailable(
        "A unique direct Save sheet or standalone Save dialog is required")
    }
    let maximumControlElements = 128
    let elements = try inspectSaveScope(
      panel, scope: .controls, maximumElements: maximumControlElements, receipt: &receipt,
      check: check)
    if standalone {
      for identifier in ["OKButton", "CancelButton"] {
        let buttons = elements.filter { $0.role == "AXButton" && $0.identifier == identifier }
        guard buttons.count == 1, let button = buttons.first,
          button.enabled == true, button.actions.contains("AXPress")
        else { throw ProAppsError.unavailable("Standalone Save dialog buttons are unavailable") }
      }
      guard elements.filter({ $0.role == "AXBrowser" }).count == 1 else {
        throw ProAppsError.unavailable("A unique standalone Save browser is required")
      }
    }
    let popups = elements.filter { $0.role == "AXPopUpButton" && $0.identifier == "where popup" }
    guard popups.count == 1, let popupSummary = popups.first else {
      throw ProAppsError.unavailable("A unique Save location popup is required")
    }
    let popup = try resolveElement(
      .init(path: popupSummary.path, role: "AXPopUpButton", identifier: "where popup"), in: window)
    let names = elements.filter {
      $0.role == "AXTextField" && $0.identifier == "saveAsNameTextField"
    }
    guard names.count == 1, let nameSummary = names.first else {
      throw ProAppsError.unavailable("A unique Save filename field is required")
    }
    let nameElement = try resolveElement(
      .init(path: nameSummary.path, role: "AXTextField", identifier: "saveAsNameTextField"),
      in: window)
    guard let name = try menuText(.value, of: nameElement.handle) else {
      throw ProAppsError.unavailable("Save filename cannot be independently observed")
    }
    try check()
    return .init(
      elements: elements, popup: popup, name: name, panel: panel, nameElement: nameElement)
  }

  private func saveURL(_ handle: UIHandle) throws -> String? {
    guard let value = try menuText(.url, of: handle) else { return nil }
    guard let url = URL(string: value), url.isFileURL, url.query == nil, url.fragment == nil,
      url.standardizedFileURL.path == url.path, url.resolvingSymlinksInPath().path == url.path
    else { throw ProAppsError.invalid("Save row URL is not a canonical file identity") }
    return url.path
  }

  private func inspectSaveColumn(
    _ browser: UILocatedElement, current: String, destination: String, maximumElements: Int,
    receipt: inout FCPSaveNavigationReceipt, selection: inout SaveChildSelection?,
    check: () throws -> Void
  ) throws -> [UIElementSummary] {
    let inspection = receipt.inspections.count
    receipt.inspections.append(.init(scope: .columnBrowser))
    var pending = [browser]
    var lists: [UILocatedElement] = []
    var cursor = 0
    while cursor < pending.count {
      try check()
      let node = pending[cursor]
      cursor += 1
      receipt.inspections[inspection].elementCount = cursor
      guard let role = try menuText(.role, of: node.handle) else {
        throw ProAppsError.invalid("Save column role is unreadable")
      }
      if role == "AXList" {
        lists.append(node)
        continue
      }
      // Only column containers are explored, never their file rows or decorative leaves.
      guard ["AXBrowser", "AXScrollArea", "AXSplitGroup"].contains(role) else { continue }
      let children = try saveChildren(node.handle)
      guard pending.count + children.count <= 128, node.path.count - browser.path.count < 12 else {
        throw ProAppsError.invalid("Save columnBrowser structure exceeds its bound")
      }
      pending += children.enumerated().map {
        .init(handle: $0.element, path: node.path + [$0.offset])
      }
    }
    guard !lists.isEmpty else {
      receipt.inspections.remove(at: inspection)
      let result = try inspectSaveScope(
        browser, scope: .columnBrowser, maximumElements: maximumElements,
        receipt: &receipt, check: check)
      return result
    }
    lists.sort { $0.path.lexicographicallyPrecedes($1.path) }
    var candidates: [UILocatedElement] = []
    var priorSelected: [String] = []
    for list in lists {
      try check()
      let count = try backend.childCount(of: list.handle)
      let directory = try saveURL(list.handle)
      var selected: [String] = []
      var observations: [FCPSaveSelectedChild] = []
      let state: FCPSaveSelectionState
      let selectedCount: Int?
      switch try backend.attribute(.selectedChildren, of: list.handle) {
      case .missing:
        state = .missing
        selectedCount = nil
      case .elements(let handles):
        state = .elements
        selectedCount = handles.count
        if handles.count <= 16 {
          for handle in handles {
            try check()
            let label = try saveSelectionLabel(handle)
            if let url = try saveRowURL(handle, check: check) { selected.append(url) }
            var children: [FCPSaveSelectionLabel] = []
            // Read only direct children of selected containers, never all file rows.
            if label.role == "AXGroup" || label.role == "AXRow" {
              let count = try backend.childCount(of: handle)
              guard count >= 0, count <= 32 else {
                throw ProAppsError.invalid("Save selected child labels exceed the native bound")
              }
              if count > 0 {
                let page = try backend.childPage(of: handle, offset: 0, count: count)
                for child in page {
                  try check()
                  children.append(try saveSelectionLabel(child))
                }
              }
            }
            observations.append(.init(element: label, directChildren: children))
          }
        }
      default:
        state = .unsupported
        selectedCount = nil
      }
      receipt.columns.append(
        .init(
          path: list.path, rowCount: count, directoryURL: directory, selectedURLs: selected,
          selectedChildrenState: state, selectedChildrenCount: selectedCount,
          selectedChildren: observations))
      guard state != .unsupported, selectedCount.map({ $0 <= 16 }) != false else {
        throw ProAppsError.invalid("Save selected children are unreadable or over budget")
      }
      if directory == current || priorSelected == [current] { candidates.append(list) }
      priorSelected = selected
    }
    // A single complete browser column belongs to the independently proved Save
    // directory. Do not infer a current column from ordering in a multi-column view.
    let parentBound =
      candidates.isEmpty && lists.count == 1
      && receipt.columns.last?.directoryURL == nil
      && receipt.columns.last?.selectedChildrenState == .elements
      && receipt.columns.last?.selectedChildrenCount == 0
    if parentBound { candidates = lists }
    guard candidates.count == 1, let list = candidates.first else {
      throw ProAppsError.unavailable(
        "Current Save column lacks unique URL-bound directory evidence")
    }
    let total = try backend.childCount(of: list.handle)
    guard total >= 0, total <= 4096 else {
      throw ProAppsError.invalid("Save column row budget exceeded")
    }
    let suffix = URL(fileURLWithPath: destination).pathComponents.dropFirst(
      URL(fileURLWithPath: current).pathComponents.count)
    guard let name = suffix.first else { throw ProAppsError.invalid("Missing next Save folder") }
    let expected = URL(fileURLWithPath: current).appendingPathComponent(name).path
    _ = try FCPSavePlanner.canonicalDirectoryForPlanning(expected)
    var matched: [(row: UILocatedElement, elements: [UIElementSummary])] = []
    var offset = 0
    while offset < total {
      try check()
      let count = min(64, total - offset)
      let page = try backend.childPage(of: list.handle, offset: offset, count: count)
      guard page.count == count else { throw ProAppsError.invalid("Incomplete Save column page") }
      for (index, handle) in page.enumerated() {
        try check()
        let row = UILocatedElement(handle: handle, path: list.path + [offset + index])
        // URL binding avoids matching identical basenames in unrelated columns.
        if try saveChildMatches(
          handle, expected: expected, parentBound: parentBound, check: check)
        {
          let elements = try inspectSaveScope(
            row, scope: .columnBrowser, maximumElements: 32,
            receipt: &receipt, check: check)
          guard elements.contains(where: { $0.value == name && $0.actions.contains("AXOpen") })
          else {
            throw ProAppsError.unavailable("URL-bound child lacks an exact actionable folder label")
          }
          matched.append((row, elements))
        }
      }
      let reread = try backend.childPage(of: list.handle, offset: offset, count: count)
      guard reread.count == page.count else {
        throw ProAppsError.invalid("Save column page changed")
      }
      for (left, right) in zip(page, reread) {
        guard try backend.sameElement(left, right) else {
          throw ProAppsError.invalid("Save column rows changed")
        }
      }
      guard try backend.childCount(of: list.handle) == total else {
        throw ProAppsError.invalid("Save column count changed")
      }
      offset += count
    }
    guard matched.count == 1, let match = matched.first else {
      throw ProAppsError.unavailable("Save column has no unique URL-bound child folder")
    }
    guard let finalOffset = match.row.path.last else {
      throw ProAppsError.invalid("Save row has no index")
    }
    let final = try backend.childPage(of: list.handle, offset: finalOffset, count: 1)
    guard final.count == 1, let finalHandle = final.first,
      try backend.sameElement(match.row.handle, finalHandle),
      try saveChildMatches(
        finalHandle, expected: expected, parentBound: parentBound, check: check)
    else { throw ProAppsError.invalid("Save folder identity changed before planning") }
    let result = [try summary(browser.handle, path: browser.path)] + match.elements
    guard result.count <= maximumElements else {
      throw ProAppsError.invalid("Save plan size exceeded")
    }
    receipt.inspections[inspection].complete = true
    selection = .init(
      list: list, row: match.row, count: total, directory: expected, parentBound: parentBound)
    return result
  }

  private func saveChildMatches(
    _ handle: UIHandle, expected: String, parentBound: Bool, check: () throws -> Void
  ) throws -> Bool {
    if let url = try saveRowURL(handle, check: check) { return url == expected }
    guard parentBound, try menuText(.role, of: handle) == "AXGroup" else { return false }
    _ = try FCPSavePlanner.canonicalDirectoryForPlanning(expected)
    let name = URL(fileURLWithPath: expected).lastPathComponent
    let count = try backend.childCount(of: handle)
    guard count > 0, count <= 32 else { return false }
    let children = try backend.childPage(of: handle, offset: 0, count: count)
    guard children.count == count else {
      throw ProAppsError.invalid("Incomplete Save child labels")
    }
    let labels = try children.map { child in
      try check()
      return try saveSelectionLabel(child)
    }
    let reread = try backend.childPage(of: handle, offset: 0, count: count)
    guard reread.count == count, try backend.childCount(of: handle) == count else {
      throw ProAppsError.invalid("Save child label count changed")
    }
    for (index, child) in reread.enumerated() {
      guard try backend.sameElement(children[index], child),
        try saveSelectionLabel(child) == labels[index]
      else { throw ProAppsError.invalid("Save child label identity changed") }
    }
    let names = labels.filter { $0.role == "AXTextField" }.compactMap(\.value)
    return names == [name]
  }

  private func saveRowURL(_ handle: UIHandle, check: () throws -> Void) throws -> String? {
    try check()
    let direct = try saveURL(handle)
    let role = try menuText(.role, of: handle)
    guard role == "AXGroup" || role == "AXRow" else { return direct }
    let count = try backend.childCount(of: handle)
    guard count >= 0, count <= 32 else {
      throw ProAppsError.invalid("Save row URL children exceed the native bound")
    }
    let children = count == 0 ? [] : try backend.childPage(of: handle, offset: 0, count: count)
    let urls = try children.map { child in
      try check()
      return try saveURL(child)
    }
    let reread = count == 0 ? [] : try backend.childPage(of: handle, offset: 0, count: count)
    guard try backend.childCount(of: handle) == count, reread.count == children.count,
      try saveURL(handle) == direct, try menuText(.role, of: handle) == role
    else { throw ProAppsError.invalid("Save row URL identity changed") }
    for (index, child) in reread.enumerated() {
      try check()
      guard try backend.sameElement(children[index], child), try saveURL(child) == urls[index]
      else {
        throw ProAppsError.invalid("Save row URL children changed")
      }
    }
    let identities = Set(([direct] + urls).compactMap { $0 })
    guard identities.count <= 1 else {
      throw ProAppsError.invalid("Save row has conflicting canonical child URLs")
    }
    return identities.first
  }

  private func saveSelectionLabel(_ handle: UIHandle) throws -> FCPSaveSelectionLabel {
    let selected: Bool?
    switch try backend.attribute(.selected, of: handle) {
    case .bool(let value): selected = value
    case .missing: selected = nil
    default: throw ProAppsError.invalid("Save selected state is not boolean")
    }
    return .init(
      role: try menuText(.role, of: handle), title: try menuText(.title, of: handle),
      value: try menuText(.value, of: handle), selected: selected, url: try saveURL(handle))
  }

  private func inspectSaveScope(
    _ root: UILocatedElement, scope: FCPSaveNavigationInspection.Scope, maximumElements: Int,
    receipt: inout FCPSaveNavigationReceipt, check: () throws -> Void
  ) throws -> [UIElementSummary] {
    let index = receipt.inspections.count
    receipt.inspections.append(.init(scope: scope))
    var elements: [UIElementSummary] = []
    var queue = [root]
    var cursor = 0
    let contentRoles: Set<String> = ["AXBrowser", "AXOutline", "AXTable", "AXMenu"]
    let maximumScopeDepth = 12
    while cursor < queue.count {
      try check()
      guard elements.count < maximumElements else {
        throw ProAppsError.unavailable(
          "Save \(scope.rawValue) inspection exceeds its element budget")
      }
      let node = queue[cursor]
      cursor += 1
      let element = try summary(node.handle, path: node.path)
      elements.append(element)
      receipt.inspections[index].elementCount = elements.count
      // These content subtrees are deliberately outside the controls contract.
      // When needed, the browser gets its own complete bounded inspection below.
      if scope == .controls, element.role.map(contentRoles.contains) == true { continue }
      let children = try saveChildren(node.handle)
      guard children.isEmpty || node.path.count - root.path.count < maximumScopeDepth,
        queue.count + children.count <= maximumElements
      else {
        throw ProAppsError.unavailable(
          "Save \(scope.rawValue) inspection exceeds its structural budget")
      }
      for (offset, handle) in children.enumerated() {
        queue.append(.init(handle: handle, path: node.path + [offset]))
      }
    }
    receipt.inspections[index].complete = true
    return elements
  }

  private func saveChildren(_ handle: UIHandle) throws -> [UIHandle] {
    switch try backend.attribute(.children, of: handle) {
    case .missing: return []
    case .elements(let values) where values.count <= 256: return values
    default:
      throw ProAppsError.invalid("Save menu children exceed the native bound or are unreadable")
    }
  }

  private func saveMenus(_ popup: UIHandle) throws -> [SaveMenu] {
    try saveChildren(popup).enumerated().compactMap { offset, handle in
      guard try backend.attribute(.role, of: handle) == .string("AXMenu") else { return nil }
      return SaveMenu(handle: handle, offset: offset)
    }
  }

  private func observeSaveMenuAttachments(
    _ popup: UILocatedElement, target: UITarget, window: UIWindowLocator,
    receipt: inout FCPSaveNavigationReceipt, check: () throws -> Void
  ) throws {
    try check()
    let process = try requireProcess(target)
    let windowHandle = try resolveWindow(window, in: process)
    let parentPath = Array(popup.path.dropLast())
    let parent =
      parentPath.isEmpty
      ? windowHandle
      : try resolveElement(.init(path: parentPath), in: windowHandle).handle
    let scopes: [(FCPSaveMenuAttachment.Scope, UIHandle, [Int]?)] = [
      (.popup, popup.handle, popup.path), (.saveContainer, parent, parentPath),
      (.application, backend.applicationElement(pid: process.pid), nil),
    ]
    for (scope, handle, path) in scopes {
      try check()
      let count = try backend.childCount(of: handle)
      let role = try menuText(.role, of: handle)
      let identifier = try menuText(.identifier, of: handle)
      let index = receipt.menuAttachments.count
      receipt.menuAttachments.append(
        .init(scope: scope, path: path, role: role, identifier: identifier, childCount: count))
      guard count >= 0, count <= 64 else {
        throw ProAppsError.invalid("Save menu attachment direct children exceed 64")
      }
      let children = count == 0 ? [] : try backend.childPage(of: handle, offset: 0, count: count)
      guard children.count == count else {
        throw ProAppsError.invalid("Save menu attachment page is incomplete")
      }
      for (offset, child) in children.enumerated() {
        try check()
        receipt.menuAttachments[index].children.append(
          .init(
            index: offset, role: try menuText(.role, of: child),
            identifier: try menuText(.identifier, of: child)))
      }
      let fresh = count == 0 ? [] : try backend.childPage(of: handle, offset: 0, count: count)
      guard fresh.count == count, try backend.childCount(of: handle) == count,
        try menuText(.role, of: handle) == role,
        try menuText(.identifier, of: handle) == identifier
      else { throw ProAppsError.invalid("Save menu attachment container changed") }
      for (offset, child) in fresh.enumerated() {
        try check()
        let captured = receipt.menuAttachments[index].children[offset]
        guard try backend.sameElement(children[offset], child),
          try menuText(.role, of: child) == captured.role,
          try menuText(.identifier, of: child) == captured.identifier
        else { throw ProAppsError.invalid("Save menu attachment child changed") }
      }
      try check()
      receipt.menuAttachments[index].complete = true
    }
  }

  private func saveMenuSnapshot(_ menu: SaveMenu, check: () throws -> Void) throws
    -> SaveMenuSnapshot
  {
    try check()
    guard try menuText(.role, of: menu.handle) == "AXMenu",
      let identifier = try menuText(.identifier, of: menu.handle), !identifier.isEmpty
    else {
      throw ProAppsError.unavailable("Save ancestor menu lacks bounded native identity")
    }
    let items = try saveChildren(menu.handle)
    let fingerprints = try items.map { handle in
      try check()
      guard let role = try menuText(.role, of: handle), !role.isEmpty else {
        throw ProAppsError.unavailable("Save ancestor menu item has no native role")
      }
      return SaveMenuItemFingerprint(
        role: role, identifier: try menuText(.identifier, of: handle),
        title: try menuText(.title, of: handle))
    }
    try check()
    return .init(items: items, fingerprint: .init(identifier: identifier, items: fingerprints))
  }

  private func menuText(_ attribute: UIAttributeName, of handle: UIHandle) throws -> String? {
    switch try backend.attribute(attribute, of: handle) {
    case .missing: return nil
    case .string(let text) where text.utf8.count <= Self.maximumTextLength && !text.contains("\0"):
      return text
    default:
      throw ProAppsError.unavailable("Save ancestor menu has unreadable or over-budget labels")
    }
  }

  private func closeSaveMenu(
    _ menu: UIHandle, target: UITarget, window: UIWindowLocator,
    receipt: inout FCPSaveNavigationReceipt, check: () throws -> Void
  ) throws {
    try check()
    guard try backend.actions(of: menu).contains("AXCancel") else {
      throw ProAppsError.unavailable("Ancestor menu does not advertise bounded cancellation")
    }
    receipt.menuClosesAttempted += 1
    do {
      try backend.perform("AXCancel", on: menu)
    } catch let error as UIBackendError {
      receipt.failure =
        "Menu cancellation reported \(String(describing: error).prefix(128)); independently read back"
    }
    let current = try readSaveControls(target, window: window, receipt: &receipt, check: check)
    guard try saveMenus(current.popup.handle).isEmpty else {
      throw SaveNavigationFailure(status: .uncertainEffect)
    }
    try check()
  }
}
