import Foundation
import MCP
import ProAppsCore

extension ToolSpec {
  static let windowLocator = object(
    ["index": integer(0, 63), "title": string(maximum: 512), "subrole": string(maximum: 128)], [])
  static let elementLocator = object(
    [
      "path": array(integer(0, 4096), maximum: 32, minimum: 0),
      "role": string(maximum: 128), "subrole": string(maximum: 128),
      "identifier": string(maximum: 256), "title": string(maximum: 512),
      "description": string(maximum: 512), "value": string(maximum: 512),
      "containsText": string(maximum: 512), "index": integer(0, 999),
    ], [])
  static let menuPath = array(string(maximum: 256), maximum: 6, minimum: 0)
  static let uiTarget: [String: Value] = ["app": app, "bundleID": string()]

  /// Tool properties for one selected app edition plus tool-specific fields.
  static func targeted(_ fields: [String: Value]) -> [String: Value] {
    var properties = uiTarget
    for (key, value) in fields { properties[key] = value }
    return properties
  }

  static let backgroundNote =
    "Background Accessibility only: never activates the app or synthesizes keyboard/pointer input; reports frontmostChanged."

  static let ui: [ToolSpec] = [
    .init(
      name: "fcp_save_browser_page",
      description:
        "Read only one explicit page relative to the unique Save ColumnView. No search, all-column walk, selection, action or current-directory claim. Paths use count-one AX pages; at most 16 rows. File URLs are lexical observations only.",
      properties: targeted([
        "expectedPID": integer(1, Int(Int32.max)), "window": windowLocator,
        "relativePath": array(integer(0, 4095), maximum: 6, minimum: 0),
        "offset": integer(0, 4096), "limit": integer(1, 16),
      ]), required: ["app", "expectedPID", "window", "relativePath", "offset", "limit"],
      readOnly: true),
    .init(
      name: "fcp_save_focus",
      description:
        "Explicit foreground exception for only the unique Save filename field: observe AXFocused/settable, or one focus setter if initially false. Requires already-frontmost exact FCP PID; supplied birth is external evidence only. No activation, typing, path, Go, Save or retry; effect requires actual false-to-true readback.",
      properties: targeted([
        "phase": string(["preflight", "focus"]), "expectedPID": integer(1, Int(Int32.max)),
        "expectedBirth": string(maximum: 128), "allowForeground": boolean, "window": windowLocator,
      ]), required: ["app", "phase", "expectedPID", "expectedBirth", "allowForeground", "window"],
      readOnly: false),
    .init(
      name: "fcp_save_location_attributes",
      description:
        "Read only: bounded attribute names and AXURL/AXDocument/AXValue of the unique Save panel, Where popup and ColumnView browser. No column traversal, file IO, actions or directory-identity claim. A file URL is lexical evidence only, not current-directory proof.",
      properties: targeted(["window": windowLocator]), required: ["app", "window"], readOnly: true),
    .init(
      name: "fcp_save_identity",
      description:
        "Read only: associated AX process identities for one exact FCP Save panel and its controls, plus two direct focus references. No keyboard, focus, value, navigation or Save action. Associated PID is NOT event-routing or birth proof.",
      properties: targeted(["expectedPID": integer(1, Int(Int32.max)), "window": windowLocator]),
      required: ["app", "expectedPID", "window"], readOnly: true),
    .init(
      name: "fcp_save_go_to_folder",
      description:
        "Explicit foreground preflight or ONE Command-Shift-G to the exact already-open FCP Save sheet. Checks existing event access without requesting permission. Open returns bounded new-dialog observations only; no path input, Go, filename, Save, directory proof or retry. Caller supplies externally verified process birth; native birth verification is not provided.",
      properties: targeted([
        "phase": string(["preflight", "open"]), "expectedPID": integer(1, Int(Int32.max)),
        "expectedBirth": string(maximum: 128), "allowForeground": boolean, "window": windowLocator,
        "delivery": string(["target", "foregroundHID"]),
      ]), required: ["app", "phase", "expectedPID", "expectedBirth", "allowForeground", "window"],
      readOnly: false),
    .init(
      name: "ui_windows",
      description:
        "List windows, dialogs and sheets of one running supported Pro app edition through Accessibility. \(backgroundNote)",
      properties: uiTarget, required: ["app"], readOnly: true),
    .init(
      name: "ui_inspect",
      description:
        "Read a bounded Accessibility subtree of one window (optionally from a located root). Returns child-index paths, roles, identifiers, titles, values, actions and frames. \(backgroundNote)",
      properties: targeted([
        "window": windowLocator, "root": elementLocator, "maxDepth": integer(1, 20),
        "maxElements": integer(1, 500),
      ]), required: ["app", "window"], readOnly: true),
    .init(
      name: "ui_menu_list",
      description:
        "List menu-bar items, or items of the submenu at an exact title path, without opening menus. \(backgroundNote)",
      properties: targeted(["menuPath": menuPath]),
      required: ["app"], readOnly: true),
    .init(
      name: "ui_menu_select",
      description:
        "Press one enabled menu item at an exact unique title path (for example File > New > Library). Effects such as dialogs are not verified; observe afterwards. \(backgroundNote)",
      properties: targeted(["menuPath": menuPath]),
      required: ["app", "menuPath"], readOnly: false),
    .init(
      name: "ui_popup_select",
      description:
        "Open one popup and press one unique enabled exact-title direct menu item in one bounded request. Returns stage diagnostics and observed popup value; no Save, filename, keyboard or activation. Value match is not filesystem identity. \(backgroundNote)",
      properties: targeted([
        "window": windowLocator, "element": elementLocator, "title": string(maximum: 512),
      ]), required: ["app", "window", "element", "title"], readOnly: false),
    .init(
      name: "ui_perform",
      description:
        "Perform one typed Accessibility action on exactly one located element and read the element back. The locator must match uniquely unless index is given. \(backgroundNote)",
      properties: targeted([
        "window": windowLocator, "element": elementLocator,
        "action": string(UIAction.allCases.map(\.rawValue)),
      ]), required: ["app", "window", "element", "action"],
      readOnly: false),
    .init(
      name: "ui_set_value",
      description:
        "Set AXValue (text) or AXSelected (row/cell selection) on exactly one located element and verify by reading it back. Text is written directly, so keyboard layouts and input methods do not affect it. \(backgroundNote)",
      properties: targeted([
        "window": windowLocator, "element": elementLocator,
        "attribute": string(UISettableAttribute.allCases.map(\.rawValue)),
        "stringValue": string(maximum: 4096), "boolValue": boolean,
      ]), required: ["app", "window", "element", "attribute"],
      readOnly: false),
    .init(
      name: "ui_wait",
      description:
        "Poll until a window and/or element exists or is absent, up to 30 seconds. Use instead of sleeping in Executor code. \(backgroundNote)",
      properties: targeted([
        "window": windowLocator, "element": elementLocator,
        "condition": string(["exists", "absent"]), "timeoutSeconds": number,
      ]), required: ["app", "condition", "timeoutSeconds"],
      readOnly: true),
    .init(
      name: "ui_capture",
      description:
        "Capture one window of a supported Pro app (including background/occluded windows) with ScreenCaptureKit into a NEW private PNG. Requires Screen Recording for the Executor host. No activation or cursor.",
      properties: targeted(["window": windowLocator, "outputPath": string()]),
      required: ["app", "window", "outputPath"], readOnly: false),
    .init(
      name: "input_source_status",
      description:
        "Read the current keyboard input source ID and whether it is ASCII-capable. Synthetic typing under a Japanese input method can turn '/' into '・'.",
      properties: [:], required: [], readOnly: true),
    .init(
      name: "input_source_select",
      description:
        "Select an installed keyboard input source by exact ID, or the system ASCII-capable source. Global user state: returns previousInputSource; restore it with this tool when finished.",
      properties: ["id": string(maximum: 256), "asciiCapable": boolean], required: [],
      readOnly: false),
    .init(
      name: "app_launch",
      description:
        "Launch the exact supported app edition if it is not running. activate defaults to false (background launch); a no-op returns the existing PID. Launch can show license or first-run UI.",
      properties: targeted(["activate": boolean]),
      required: ["app"], readOnly: false),
    .init(
      name: "app_activate",
      description:
        "Activate only the exact already-running supported edition and expectedPID after explicit allowForeground: true. Verify its foreground PID/bundle for at most two seconds. An optional exact menuPath presses at most one enabled item in the same child after foreground verification; no filename, Save, export completion, keyboard, pointer or focus restoration. Returns attempted-action evidence even on refusal; observe before retrying. Caller owns foreground lane and restoration.",
      properties: targeted([
        "expectedPID": integer(1, Int(Int32.max)), "allowForeground": boolean,
        "menuPath": array(string(maximum: 256), maximum: 6),
      ]), required: ["app", "expectedPID", "allowForeground"], readOnly: false),
    .init(
      name: "app_quit",
      description:
        "Request normal termination of the exact running app edition and wait up to 10 seconds. The app may refuse or ask about unsaved changes. force requires discardUnsavedChanges: true and loses unsaved work.",
      properties: targeted(["force": boolean, "discardUnsavedChanges": boolean]), required: ["app"],
      readOnly: false),
  ]

  static var uiNames: Set<String> { Set(ui.map(\.name)) }
}

extension NativeService {
  private struct UIInput: Decodable {
    let app: ProApp?
    let bundleID: String?
    let window: UIWindowLocator?
    let root: UIElementLocator?
    let element: UIElementLocator?
    let maxDepth: Int?
    let maxElements: Int?
    let menuPath: [String]?
    let action: UIAction?
    let title: String?
    let attribute: UISettableAttribute?
    let stringValue: String?
    let boolValue: Bool?
    let condition: UIWaitCondition?
    let timeoutSeconds: Double?
    let outputPath: String?
    let id: String?
    let asciiCapable: Bool?
    let activate: Bool?
    let force: Bool?
    let discardUnsavedChanges: Bool?
    let expectedPID: Int32?
    let allowForeground: Bool?
  }

  static let defaultInspectDepth = 8
  static let defaultInspectElements = 200

  /// Translate validated MCP arguments into one typed child request.
  func uiRequest(_ name: String, _ arguments: Value) throws -> UIRequest {
    if name == "fcp_save_browser_page" {
      struct Input: Decodable {
        let app: ProApp
        let bundleID: String?
        let expectedPID: Int32
        let window: UIWindowLocator
        let relativePath: [Int]
        let offset: Int
        let limit: Int
      }
      let value = try decode(Input.self, arguments)
      return .saveBrowserPage(
        .init(app: value.app, bundleID: value.bundleID),
        .init(
          expectedPID: value.expectedPID, window: value.window, relativePath: value.relativePath,
          offset: value.offset, limit: value.limit))
    }
    if name == "fcp_save_focus" {
      struct Input: Decodable {
        let app: ProApp
        let bundleID: String?
        let phase: FCPSaveFocusRequest.Phase
        let expectedPID: Int32
        let expectedBirth: String
        let allowForeground: Bool
        let window: UIWindowLocator
      }
      let value = try decode(Input.self, arguments)
      return .saveFocus(
        .init(app: value.app, bundleID: value.bundleID),
        .init(
          phase: value.phase, expectedPID: value.expectedPID, expectedBirth: value.expectedBirth,
          allowForeground: value.allowForeground, window: value.window))
    }
    if name == "fcp_save_location_attributes" {
      struct Input: Decodable {
        let app: ProApp
        let bundleID: String?
        let window: UIWindowLocator
      }
      let value = try decode(Input.self, arguments)
      return .saveLocationAttributes(
        .init(app: value.app, bundleID: value.bundleID), .init(window: value.window))
    }
    if name == "fcp_save_identity" {
      struct Input: Decodable {
        let app: ProApp
        let bundleID: String?
        let expectedPID: Int32
        let window: UIWindowLocator
      }
      let value = try decode(Input.self, arguments)
      return .saveIdentity(
        .init(app: value.app, bundleID: value.bundleID),
        .init(expectedPID: value.expectedPID, window: value.window))
    }
    if name == "fcp_save_go_to_folder" {
      struct Input: Decodable {
        let app: ProApp
        let bundleID: String?
        let phase: FCPGoToFolderRequest.Phase
        let expectedPID: Int32
        let expectedBirth: String
        let allowForeground: Bool
        let delivery: FCPGoToFolderRequest.Delivery?
        let window: UIWindowLocator
      }
      let value = try decode(Input.self, arguments)
      return .goToFolder(
        .init(app: value.app, bundleID: value.bundleID),
        .init(
          phase: value.phase, expectedPID: value.expectedPID, expectedBirth: value.expectedBirth,
          allowForeground: value.allowForeground, window: value.window, delivery: value.delivery))
    }
    let input = try decode(UIInput.self, arguments)
    func target() throws -> UITarget {
      guard let app = input.app else { throw ProAppsError.invalid("Missing app") }
      return UITarget(app: app, bundleID: input.bundleID)
    }
    func window() throws -> UIWindowLocator {
      guard let window = input.window else { throw ProAppsError.invalid("Missing window") }
      return window
    }
    func element() throws -> UIElementLocator {
      guard let element = input.element else { throw ProAppsError.invalid("Missing element") }
      return element
    }
    switch name {
    case "ui_windows": return .windows(try target())
    case "ui_inspect":
      return .inspect(
        try target(), window: try window(), root: input.root,
        maxDepth: input.maxDepth ?? Self.defaultInspectDepth,
        maxElements: input.maxElements ?? Self.defaultInspectElements)
    case "ui_menu_list": return .menuList(try target(), path: input.menuPath ?? [])
    case "ui_menu_select":
      guard let path = input.menuPath, !path.isEmpty else {
        throw ProAppsError.invalid("menuPath must name at least one item")
      }
      return .menuSelect(try target(), path: path)
    case "ui_popup_select":
      guard let title = input.title else { throw ProAppsError.invalid("Missing title") }
      return .popupSelect(try target(), window: try window(), target: try element(), title: title)
    case "ui_perform":
      guard let action = input.action else { throw ProAppsError.invalid("Missing action") }
      return .perform(try target(), window: try window(), target: try element(), action: action)
    case "ui_set_value":
      guard let attribute = input.attribute else { throw ProAppsError.invalid("Missing attribute") }
      let value: UISettableValue
      switch (attribute, input.stringValue, input.boolValue) {
      case (.value, .some(let text), .none): value = .string(text)
      case (.selected, .none, .some(let flag)): value = .bool(flag)
      default:
        throw ProAppsError.invalid("value needs stringValue; selected needs boolValue")
      }
      return .set(
        try target(), window: try window(), target: try element(), attribute: attribute,
        value: value)
    case "ui_wait":
      guard let condition = input.condition, let timeout = input.timeoutSeconds else {
        throw ProAppsError.invalid("Missing condition or timeout")
      }
      return .wait(
        try target(), window: input.window, target: input.element, condition: condition,
        timeoutSeconds: timeout)
    case "ui_capture":
      guard let output = input.outputPath else { throw ProAppsError.invalid("Missing outputPath") }
      return .capture(try target(), window: try window(), outputPath: output)
    case "input_source_status": return .inputSourceStatus
    case "input_source_select":
      return .inputSourceSelect(id: input.id, asciiCapable: input.asciiCapable ?? false)
    case "app_launch": return .launch(try target(), activate: input.activate ?? false)
    case "app_activate":
      guard let pid = input.expectedPID, pid > 0, input.allowForeground == true else {
        throw ProAppsError.invalid(
          "Activation requires allowForeground: true and a positive expectedPID")
      }
      return .activate(
        try target(), expectedPID: pid, allowForeground: true, menuPath: input.menuPath)
    case "app_quit":
      let force = input.force ?? false
      guard !force || input.discardUnsavedChanges == true else {
        throw ProAppsError.invalid("force requires discardUnsavedChanges: true")
      }
      return .quit(try target(), force: force)
    default: throw ProAppsError.invalid("Unknown tool")
    }
  }

  static func uiTimeout(_ request: UIRequest) -> Duration {
    switch request {
    case .wait(_, _, _, _, let seconds): return .seconds(Int(seconds.rounded(.up)) + 15)
    case .launch: return .seconds(90)
    default: return .seconds(45)
    }
  }

  func uiTool(_ name: String, _ arguments: Value) async throws -> CallTool.Result {
    let request = try uiRequest(name, arguments)
    let response = try await interfaces.ui(request, Self.uiTimeout(request))
    guard var fields = try Value(response).objectValue else {
      throw ProAppsError.unavailable("UI result did not encode as an object")
    }
    fields["retrySafe"] = .bool(false)
    if let observation = response.saveBrowserPage {
      fields["completionVerified"] = .bool(false)
      return self.response(fields, failed: !observation.complete)
    }
    if let observation = response.saveFocus {
      fields["completionVerified"] = .bool(false)
      return self.response(
        fields, failed: !observation.complete || observation.status == "unsupported")
    }
    if let observation = response.saveLocationAttributes {
      fields["completionVerified"] = .bool(false)
      return self.response(fields, failed: !observation.complete)
    }
    if let observation = response.saveIdentity {
      fields["completionVerified"] = .bool(false)
      return self.response(fields, failed: !observation.complete)
    }
    if let observation = response.goToFolder {
      fields["completionVerified"] = .bool(false)
      return self.response(
        fields, failed: observation.failure != nil || observation.status == "cancelled")
    }
    if let activation = response.activation {
      fields["completionVerified"] = .bool(false)
      return self.response(
        fields,
        failed: response.effectVerified != true
          || (activation.foregroundVerified && activation.status == .menuFailed))
    }
    return self.response(fields)
  }
}
