import Foundation

/// Explicit caller evidence is kept separate from a Save panel's basename label.
public struct FCPSaveContext: Codable, Sendable {
  public let directory: String
  public let fileName: String
  public let observedDirectory: String
  public let window: UIWindowLocator
  public let bundleID: String?
  public let saveDispatched: Bool
  public var navigationAncestor: String? = nil
}

public struct FCPSavePlan: Encodable, Sendable {
  public let outputPath: String
  public let next: UIRequest?
  public let state: String
  public let directoryBinding: String
  public var expectedDirectoryAfterAction: String? = nil
  public var dispatched: Bool { false }
  public var completionVerified: Bool { false }

  private enum CodingKeys: String, CodingKey {
    case outputPath, next, state, directoryBinding, expectedDirectoryAfterAction
    case dispatched, completionVerified
  }

  public func encode(to encoder: any Encoder) throws {
    var fields = encoder.container(keyedBy: CodingKeys.self)
    try fields.encode(outputPath, forKey: .outputPath)
    try fields.encode(next, forKey: .next)
    try fields.encode(state, forKey: .state)
    try fields.encode(directoryBinding, forKey: .directoryBinding)
    try fields.encode(expectedDirectoryAfterAction, forKey: .expectedDirectoryAfterAction)
    try fields.encode(dispatched, forKey: .dispatched)
    try fields.encode(completionVerified, forKey: .completionVerified)
  }
}

/// Pure next-step planning over a fresh native AX read. It never operates an app.
/// The caller supplies independently established current-directory provenance;
/// a popup basename is checked for consistency, not treated as full-path proof.
public enum FCPSavePlanner {
  public static func plan(_ context: FCPSaveContext, elements: [UIElementSummary]) throws
    -> FCPSavePlan
  {
    try plan(context, elements: elements, verifiedDirectory: nil)
  }

  // Only the shared native supplier calls this overload after full ancestry proof.
  // It is not a caller JSON field or a replacement for the legacy observed-directory contract.
  static func plan(
    _ context: FCPSaveContext, elements: [UIElementSummary],
    verifiedDirectory: FCPSaveDirectoryObservation?,
    verifiedNavigationAncestor: FCPSaveDirectoryCandidate? = nil
  ) throws -> FCPSavePlan {
    guard let title = context.window.title, !title.isEmpty,
      context.window.index == nil, !elements.isEmpty, elements.count <= 500,
      Set(elements.map(\.path)).count == elements.count
    else {
      throw ProAppsError.invalid("A named Save window and unique bounded AX paths are required")
    }
    let directory = try canonicalDirectoryForPlanning(context.directory)
    let current = try canonicalDirectoryForPlanning(context.observedDirectory)
    let name = context.fileName
    guard !name.isEmpty, name != ".", name != "..", !name.contains("/"),
      !name.contains("\0"), name.utf8.count <= 180,
      !URL(fileURLWithPath: name).pathExtension.isEmpty
    else { throw ProAppsError.invalid("A new output basename with extension is required") }
    let output = directory.appendingPathComponent(name)
    let binding =
      verifiedDirectory == nil
      ? "caller-observed-canonical-directory-plus-AX-popup-basename"
      : "native-full-ordered-OS-localized-directory-ancestry"
    if context.saveDispatched {
      return FCPSavePlan(
        outputPath: output.path, next: nil,
        state: "observe-existing-dispatch", directoryBinding: binding)
    }
    try requireAbsent(output)
    let popup = try unique(elements, identifier: "where popup")
    let expectedDisplayName: String
    if let proof = verifiedDirectory {
      guard proof.directory == current.path, proof.volumeURL != nil,
        !proof.languages.isEmpty, let first = proof.ancestorTitles.first
      else {
        throw ProAppsError.invalid("Verified directory proof does not bind the current directory")
      }
      expectedDisplayName = first
    } else {
      expectedDisplayName = current.lastPathComponent
    }
    guard popup.role == "AXPopUpButton", popup.value == expectedDisplayName else {
      throw ProAppsError.invalid(
        "Save location changed; observe its full directory before planning")
    }
    let target = UITarget(app: .finalCutPro, bundleID: context.bundleID)
    func result(_ next: UIRequest, _ state: String, expectedDirectory: String? = nil) -> FCPSavePlan
    {
      FCPSavePlan(
        outputPath: output.path, next: next, state: state, directoryBinding: binding,
        expectedDirectoryAfterAction: expectedDirectory)
    }
    if current != directory {
      let prefix = current.pathComponents
      guard directory.pathComponents.starts(with: prefix) else {
        if let ancestorPath = context.navigationAncestor {
          let ancestor = try canonicalDirectoryForPlanning(ancestorPath)
          guard directory.pathComponents.starts(with: ancestor.pathComponents) else {
            throw ProAppsError.invalid("Navigation ancestor must contain the requested directory")
          }
          let entries = elements.filter { $0.role == "AXMenuItem" }
          if !entries.isEmpty {
            let names: Set<String>
            if let evidence = verifiedNavigationAncestor {
              guard let proof = verifiedDirectory, evidence.directory == ancestor.path else {
                throw ProAppsError.invalid("OS navigation ancestor evidence does not bind its path")
              }
              names = evidence.navigationLabels(of: proof, toward: directory.path)
              guard !names.isEmpty else {
                throw ProAppsError.invalid(
                  "OS navigation evidence binds neither current ancestry nor target volume")
              }
            } else {
              names = [ancestor.lastPathComponent]
            }
            let matches = entries.filter { $0.title.map(names.contains) == true }
            guard matches.count == 1, let entry = matches.first,
              entry.enabled != false, entry.actions.contains("AXPress")
            else {
              throw ProAppsError.unavailable(
                "Recent ancestor is absent, ambiguous or not pressable")
            }
            return result(
              .perform(target, window: context.window, target: locator(entry), action: .press),
              "select-observed-ancestor-and-observe", expectedDirectory: ancestor.path)
          }
        }
        guard popup.actions.contains("AXShowMenu"), popup.enabled != false else {
          throw ProAppsError.unavailable("Save location menu is not available")
        }
        return result(
          .perform(
            target, window: context.window,
            target: locator(popup), action: .showMenu), "observe-location-menu")
      }
      let childName = directory.pathComponents[prefix.count]
      let browsers = elements.filter { $0.role == "AXBrowser" }
      guard browsers.count == 1, let browser = browsers.first else {
        throw ProAppsError.invalid("A unique Save column browser is required")
      }
      let named = elements.filter {
        $0.role == "AXTextField" && $0.path.starts(with: browser.path)
          && $0.value == childName && $0.actions.contains("AXOpen")
      }
      let cells =
        named.isEmpty
        ? elements.filter { element in
          element.role == "AXCell" && element.path.starts(with: browser.path)
            && elements.contains { label in
              label.path.starts(with: element.path) && label.value == childName
            }
        } : named
      guard cells.count == 1, let cell = cells.first,
        cell.enabled != false, cell.actions.contains("AXOpen")
      else {
        throw ProAppsError.unavailable(
          "Child folder lacks a unique advertised AXOpen; inspect navigation")
      }
      return result(
        .perform(
          target, window: context.window,
          target: UIElementLocator(path: cell.path, role: cell.role, containsText: childName),
          action: .open), "open-child-and-observe",
        expectedDirectory: current.appendingPathComponent(childName).path)
    }
    let field = try unique(elements, identifier: "saveAsNameTextField")
    guard field.role == "AXTextField", field.enabled != false, field.valueSettable else {
      throw ProAppsError.unavailable("Save filename field is not writable")
    }
    if field.value != name {
      return result(
        .set(
          target, window: context.window, target: locator(field),
          attribute: .value, value: .string(name)), "set-name-and-observe")
    }
    let save = try unique(elements, identifier: "OKButton")
    guard save.role == "AXButton", save.enabled == true, save.actions.contains("AXPress") else {
      throw ProAppsError.unavailable("Save is not enabled")
    }
    return result(
      .perform(target, window: context.window, target: locator(save), action: .press),
      "ready-to-dispatch-save")
  }

  public static func canonicalDirectoryForPlanning(_ path: String) throws -> URL {
    let url = try Files.absolute(path)
    guard url.path == path, url.resolvingSymlinksInPath().path == path,
      try url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true
    else { throw ProAppsError.invalid("Directory must be an existing canonical non-symlink path") }
    return url
  }

  static func requireAbsent(_ output: URL) throws {
    do {
      _ = try FileManager.default.attributesOfItem(atPath: output.path)
    } catch CocoaError.fileReadNoSuchFile {
      return
    }
    throw ProAppsError.invalid("Save destination already exists; replacement is forbidden")
  }

  private static func unique(_ elements: [UIElementSummary], identifier: String) throws
    -> UIElementSummary
  {
    let found = elements.filter { $0.identifier == identifier }
    guard found.count == 1, let element = found.first else {
      throw ProAppsError.invalid("Save control is absent or ambiguous: \(identifier)")
    }
    return element
  }

  private static func locator(_ element: UIElementSummary) -> UIElementLocator {
    UIElementLocator(
      path: element.path, role: element.role, identifier: element.identifier,
      title: element.title, value: element.value)
  }
}
