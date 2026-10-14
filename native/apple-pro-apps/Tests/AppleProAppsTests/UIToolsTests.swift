import Foundation
import MCP
import ProAppsCore
import Synchronization
import Testing

@testable import AppleProApps

struct UIToolsTests {
  @Test(arguments: [true, false])
  func browserPageIsExplicitReadOnly(_ complete: Bool) async throws {
    let service = Self.service { request, _ in
      #expect(
        request
          == .saveBrowserPage(
            .init(app: .finalCutPro),
            .init(
              expectedPID: 42, window: .init(title: "Save"), relativePath: [], offset: 0, limit: 1))
      )
      var response = Self.response
      var receipt = FCPSaveBrowserPageReceipt()
      receipt.complete = complete
      response.saveBrowserPage = receipt
      return response
    }
    let result = await service.call(
      .init(
        name: "fcp_save_browser_page",
        arguments: [
          "app": .string("finalCutPro"), "expectedPID": .int(42),
          "window": .object(["title": .string("Save")]), "relativePath": .array([]),
          "offset": .int(0), "limit": .int(1),
        ]))
    #expect((result.isError == true) == !complete)
    #expect(result.structuredContent?.objectValue?["completionVerified"] == .bool(false))
    #expect(ToolSpec.ui.first { $0.name == "fcp_save_browser_page" }?.readOnly == true)
  }
  @Test(arguments: ["preflightObserved", "unsupported", "refused"])
  func saveFocusUsesOnlyTheExplicitSeparateTool(_ status: String) async throws {
    let service = Self.service { request, _ in
      #expect(
        request
          == .saveFocus(
            .init(app: .finalCutPro),
            .init(
              phase: .preflight, expectedPID: 42, expectedBirth: "external:1",
              allowForeground: true, window: .init(title: "Save"))))
      var response = Self.response
      var receipt = FCPSaveFocusReceipt()
      receipt.status = status
      receipt.complete = status != "refused"
      response.saveFocus = receipt
      return response
    }
    let result = await service.call(
      .init(
        name: "fcp_save_focus",
        arguments: [
          "app": .string("finalCutPro"), "phase": .string("preflight"), "expectedPID": .int(42),
          "expectedBirth": .string("external:1"), "allowForeground": .bool(true),
          "window": .object(["title": .string("Save")]),
        ]))
    #expect((result.isError == true) == (status != "preflightObserved"))
    #expect(result.structuredContent?.objectValue?["completionVerified"] == .bool(false))
    #expect(!UISettableAttribute.allCases.map(\.rawValue).contains("focused"))
  }
  @Test(arguments: [true, false])
  func saveLocationAttributesAreObservationsOnly(_ complete: Bool) async throws {
    let service = Self.service { request, timeout in
      #expect(timeout == .seconds(45))
      #expect(
        request
          == .saveLocationAttributes(.init(app: .finalCutPro), .init(window: .init(title: "Save"))))
      var response = Self.response
      var receipt = FCPSaveLocationAttributesReceipt()
      receipt.complete = complete
      response.saveLocationAttributes = receipt
      return response
    }
    let result = await service.call(
      .init(
        name: "fcp_save_location_attributes",
        arguments: [
          "app": .string("finalCutPro"), "window": .object(["title": .string("Save")]),
        ]))
    #expect((result.isError == true) == !complete)
    #expect(result.structuredContent?.objectValue?["completionVerified"] == .bool(false))
  }
  @Test func explicitHIDIsPassedToTypedChild() async throws {
    let service = Self.service { request, _ in
      guard case .goToFolder(_, let input) = request else {
        throw ProAppsError.invalid("Unexpected test request")
      }
      #expect(input.delivery == .foregroundHID)
      return Self.response
    }
    let result = await service.call(
      .init(
        name: "fcp_save_go_to_folder",
        arguments: [
          "app": .string("finalCutPro"), "phase": .string("open"), "expectedPID": .int(42),
          "expectedBirth": .string("external:1"), "allowForeground": .bool(true),
          "window": .object(["title": .string("Save")]), "delivery": .string("foregroundHID"),
        ]))
    #expect(result.isError != true)
  }
  @Test func saveIdentityIsReadOnlyAndNeverClaimsRouting() async throws {
    let service = Self.service { request, timeout in
      #expect(timeout == .seconds(45))
      #expect(
        request
          == .saveIdentity(
            .init(app: .finalCutPro), .init(expectedPID: 42, window: .init(title: "Save"))))
      var response = Self.response
      var receipt = FCPSaveIdentityReceipt()
      receipt.complete = true
      response.saveIdentity = receipt
      return response
    }
    let result = await service.call(
      .init(
        name: "fcp_save_identity",
        arguments: [
          "app": .string("finalCutPro"), "expectedPID": .int(42),
          "window": .object(["title": .string("Save")]),
        ]))
    #expect(result.isError != true)
    #expect(result.structuredContent?.objectValue?["completionVerified"] == .bool(false))
  }
  @Test func goToFolderUsesExistingTypedChildAndKeepsObservationSeparate() async throws {
    let fields: [String: Value] = [
      "app": .string("finalCutPro"), "phase": .string("open"), "expectedPID": .int(42),
      "expectedBirth": .string("external:1"), "allowForeground": .bool(true),
      "window": .object(["title": .string("Save")]),
    ]
    let service = Self.service { request, timeout in
      #expect(timeout == .seconds(45))
      #expect(
        request
          == .goToFolder(
            .init(app: .finalCutPro),
            .init(
              phase: .open, expectedPID: 42, expectedBirth: "external:1", allowForeground: true,
              window: .init(title: "Save"))))
      var response = Self.response
      var receipt = FCPGoToFolderReceipt()
      receipt.status = "newDialogObserved"
      receipt.newDialogObserved = true
      response.goToFolder = receipt
      return response
    }
    let result = await service.call(.init(name: "fcp_save_go_to_folder", arguments: fields))
    #expect(result.isError != true)
    #expect(result.structuredContent?.objectValue?["completionVerified"] == .bool(false))
    var invalid = fields
    invalid["phase"] = .string("save")
    #expect(
      await service.call(.init(name: "fcp_save_go_to_folder", arguments: invalid)).isError == true)
  }

  static let target = UITarget(app: .finalCutPro)
  static let window = UIWindowLocator(title: "ライブラリを開く")
  static let row = UIElementLocator(role: "AXRow", containsText: "alpha.fcpbundle")
  static let base: [String: Value] = ["app": .string("finalCutPro")]
  static let windowValue: Value = .object(["title": .string("ライブラリを開く")])
  static let rowValue: Value = .object([
    "role": .string("AXRow"), "containsText": .string("alpha.fcpbundle"),
  ])

  static func service(_ ui: @escaping @Sendable (UIRequest, Duration) async throws -> UIResponse)
    -> NativeService
  {
    var interfaces = NativeInterfaces()
    interfaces.ui = ui
    return NativeService(interfaces: interfaces)
  }

  static let response: UIResponse = {
    var response = UIResponse(
      focus: UIFocusEvidence(frontmostBefore: "dev.terminal", frontmostAfter: "dev.terminal"))
    response.dispatched = true
    return response
  }()

  @Test(arguments: [
    ("ui_windows", base, UIRequest.windows(target)),
    (
      "ui_inspect", base.merging(["window": windowValue]) { a, _ in a },
      .inspect(target, window: window, root: nil, maxDepth: 8, maxElements: 200)
    ),
    (
      "ui_inspect",
      base.merging([
        "window": windowValue, "root": rowValue, "maxDepth": .int(2), "maxElements": .int(5),
      ]) { a, _ in a },
      .inspect(target, window: window, root: row, maxDepth: 2, maxElements: 5)
    ),
    ("ui_menu_list", base, .menuList(target, path: [])),
    (
      "ui_popup_select",
      base.merging(["window": windowValue, "element": rowValue, "title": .string("Destination")]) {
        a, _ in a
      }, .popupSelect(target, window: window, target: row, title: "Destination")
    ),
    (
      "ui_menu_select", base.merging(["menuPath": .array([.string("ファイル")])]) { a, _ in a },
      .menuSelect(target, path: ["ファイル"])
    ),
    (
      "ui_perform",
      base.merging(["window": windowValue, "element": rowValue, "action": .string("press")]) {
        a, _ in a
      }, .perform(target, window: window, target: row, action: .press)
    ),
    (
      "ui_set_value",
      base.merging([
        "window": windowValue, "element": rowValue, "attribute": .string("selected"),
        "boolValue": .bool(true),
      ]) { a, _ in a },
      .set(target, window: window, target: row, attribute: .selected, value: .bool(true))
    ),
    (
      "ui_set_value",
      base.merging([
        "window": windowValue, "element": rowValue, "attribute": .string("value"),
        "stringValue": .string("/Volumes/x"),
      ]) { a, _ in a },
      .set(target, window: window, target: row, attribute: .value, value: .string("/Volumes/x"))
    ),
    (
      "ui_wait",
      base.merging([
        "window": windowValue, "condition": .string("absent"), "timeoutSeconds": .double(2),
      ]) { a, _ in a },
      .wait(target, window: window, target: nil, condition: .absent, timeoutSeconds: 2)
    ),
    (
      "ui_capture",
      base.merging(["window": windowValue, "outputPath": .string("/tmp/w.png")]) { a, _ in a },
      .capture(target, window: window, outputPath: "/tmp/w.png")
    ),
    ("input_source_status", [:], .inputSourceStatus),
    (
      "input_source_select", ["asciiCapable": .bool(true)],
      .inputSourceSelect(id: nil, asciiCapable: true)
    ),
    (
      "input_source_select", ["id": .string("abc")],
      .inputSourceSelect(id: "abc", asciiCapable: false)
    ),
    ("app_launch", base, .launch(target, activate: false)),
    (
      "app_activate",
      base.merging(["expectedPID": .int(42), "allowForeground": .bool(true)]) {
        a, _ in a
      }, .activate(target, expectedPID: 42, allowForeground: true, menuPath: nil)
    ),
    (
      "app_activate",
      base.merging([
        "expectedPID": .int(42), "allowForeground": .bool(true),
        "menuPath": .array([.string("File"), .string("Share")]),
      ]) { a, _ in a },
      .activate(target, expectedPID: 42, allowForeground: true, menuPath: ["File", "Share"])
    ),
    ("app_quit", base, .quit(target, force: false)),
    (
      "app_quit",
      base.merging(["force": .bool(true), "discardUnsavedChanges": .bool(true)]) { a, _ in a },
      .quit(target, force: true)
    ),
  ])
  func mapsToolArgumentsToTypedRequests(
    _ name: String, _ arguments: [String: Value], _ expected: UIRequest
  ) async throws {
    let service = Self.service { _, _ in Self.response }
    #expect(try await service.uiRequest(name, .object(arguments)) == expected)
  }

  @Test(arguments: [
    ("ui_windows", [String: Value](), "Missing app"),
    ("ui_inspect", base, "Missing window"),
    ("ui_popup_select", base, "Missing title"),
    (
      "ui_perform",
      base.merging(["window": windowValue, "action": .string("press")]) { a, _ in a },
      "Missing element"
    ),
    (
      "ui_perform", base.merging(["window": windowValue, "element": rowValue]) { a, _ in a },
      "Missing action"
    ),
    (
      "ui_set_value", base.merging(["window": windowValue, "element": rowValue]) { a, _ in a },
      "Missing attribute"
    ),
    (
      "ui_set_value",
      base.merging([
        "window": windowValue, "element": rowValue, "attribute": .string("selected"),
        "stringValue": .string("x"),
      ]) { a, _ in a }, "value needs stringValue; selected needs boolValue"
    ),
    (
      "ui_menu_select", base.merging(["menuPath": .array([])]) { a, _ in a },
      "menuPath must name at least one item"
    ),
    ("ui_menu_select", base, "menuPath must name at least one item"),
    ("ui_wait", base, "Missing condition or timeout"),
    ("ui_capture", base.merging(["window": windowValue]) { a, _ in a }, "Missing outputPath"),
    (
      "app_quit", base.merging(["force": .bool(true)]) { a, _ in a },
      "force requires discardUnsavedChanges: true"
    ),
    ("ui_unknown", base, "Unknown tool"),
    ("app_activate", base, "Activation requires allowForeground: true and a positive expectedPID"),
    (
      "app_activate",
      base.merging(["expectedPID": .int(42), "allowForeground": .bool(false)]) {
        a, _ in a
      }, "Activation requires allowForeground: true and a positive expectedPID"
    ),
  ])
  func rejectsIncompleteToolArguments(
    _ name: String, _ arguments: [String: Value], _ reason: String
  )
    async throws
  {
    let service = Self.service { _, _ in Self.response }
    await #expect(throws: ProAppsError.invalid(reason)) {
      try await service.uiRequest(name, .object(arguments))
    }
  }

  @Test func boundsChildTimeoutsByRequest() {
    #expect(
      NativeService.uiTimeout(
        .wait(Self.target, window: nil, target: nil, condition: .exists, timeoutSeconds: 2.2))
        == .seconds(18))
    #expect(NativeService.uiTimeout(.launch(Self.target, activate: false)) == .seconds(90))
    #expect(NativeService.uiTimeout(.inputSourceStatus) == .seconds(45))
  }

  @Test func routesUIToolsThroughTheChildInterface() async throws {
    let seen = Mutex<[UIRequest]>([])
    let service = Self.service { request, _ in
      seen.withLock { $0.append(request) }
      return Self.response
    }
    let result = await service.call(
      .init(
        name: "ui_set_value",
        arguments: Self.base.merging([
          "window": Self.windowValue, "element": Self.rowValue, "attribute": .string("selected"),
          "boolValue": .bool(true),
        ]) { a, _ in a }))
    #expect(result.isError == false)
    let fields = try #require(result.structuredContent?.objectValue)
    #expect(fields["retrySafe"] == .bool(false))
    #expect(fields["dispatched"] == .bool(true))
    #expect(fields["focus"]?.objectValue?["frontmostChanged"] == .bool(false))
    #expect(seen.withLock { $0.count } == 1)
  }

  @Test(arguments: [
    (UIActivationStatus.foregroundVerified, true, false),
    (.activationRefused, false, true),
    (.menuFailed, true, true),
  ])
  func activationReturnsItsReceiptAndMarksRefusalAsError(
    _ status: UIActivationStatus, _ foreground: Bool, _ failed: Bool
  ) async throws {
    let service = Self.service { request, timeout in
      #expect(
        request == .activate(Self.target, expectedPID: 42, allowForeground: true, menuPath: nil))
      #expect(timeout == .seconds(45))
      var response = Self.response
      var receipt = UIActivationReceipt(expectedPID: 42)
      receipt.status = status
      receipt.foregroundVerified = foreground
      response.activation = receipt
      response.effectVerified = foreground
      return response
    }
    let result = await service.call(
      .init(
        name: "app_activate",
        arguments: Self.base.merging([
          "expectedPID": .int(42), "allowForeground": .bool(true),
        ]) { a, _ in a }))
    #expect(result.isError == failed)
    #expect(result.structuredContent?.objectValue?["completionVerified"] == .bool(false))
    #expect(result.structuredContent?.objectValue?["retrySafe"] == .bool(false))
    #expect(
      result.structuredContent?.objectValue?["activation"]?.objectValue?["status"]
        == .string(status.rawValue))
  }

  @Test func reportsChildFailuresAndSchemaViolations() async throws {
    let failing = Self.service { _, _ in throw ProAppsError.unavailable("No matching window") }
    let failed = await failing.call(.init(name: "ui_windows", arguments: Self.base))
    #expect(failed.isError == true)
    let strict = Self.service { _, _ in Self.response }
    let unknownKey = await strict.call(
      .init(
        name: "ui_inspect",
        arguments: Self.base.merging(["window": .object(["bogus": .int(1)])]) { a, _ in a }))
    #expect(unknownKey.isError == true)
    let badAction = await strict.call(
      .init(
        name: "ui_perform",
        arguments: Self.base.merging([
          "window": Self.windowValue, "element": Self.rowValue, "action": .string("typeText"),
        ]) { a, _ in a }))
    #expect(badAction.isError == true)
  }

  @Test func advertisesBackgroundAccessibilityTools() async throws {
    #expect(ToolSpec.uiNames.contains("ui_capture"))
    #expect(ToolSpec.all.filter { ToolSpec.uiNames.contains($0.name) }.count == ToolSpec.ui.count)
    #expect(ToolSpec.ui.first { $0.name == "ui_inspect" }?.readOnly == true)
    #expect(ToolSpec.ui.first { $0.name == "ui_perform" }?.readOnly == false)
    var interfaces = NativeInterfaces()
    interfaces.inventory = { [] }
    let result = await NativeService(interfaces: interfaces).call(
      .init(name: "app_capabilities", arguments: [:]))
    #expect(result.structuredContent?.objectValue?["backgroundAccessibility"] == .bool(true))
  }

  @Test func defaultChildInterfaceFailsClosedOutsideTheExecutable() async throws {
    await #expect(throws: (any Error).self) {
      try await NativeInterfaces().ui(.inputSourceStatus, .seconds(10))
    }
  }

  @Test(arguments: [
    (["ui-native", #"{"operation":"inputSourceStatus"}"#], true),
    (["ui-native"], false),
    (["ui-native", "{}", "{}"], false),
    (["ui-native", String(repeating: "x", count: Command.maximumUIRequestBytes + 1)], false),
  ])
  func parsesBoundedChildRequests(_ arguments: [String], _ valid: Bool) throws {
    if valid {
      #expect(try Command.parse(arguments) == .uiNative(arguments[1]))
    } else {
      #expect(throws: ProAppsError.invalid("ui-native requires exactly one bounded JSON request")) {
        try Command.parse(arguments)
      }
    }
  }

  @Test func programRoutesChildRequestsToTheRuntime() async throws {
    var runtime = ProgramRuntime(
      binary: URL(fileURLWithPath: "/tmp/native"), environment: [:], serve: {},
      inventory: { "[]" }, ui: { _ in }, execute: { _, _, _ in "" })
    runtime.uiNative = { request in "handled:\(request)" }
    #expect(try await Program(runtime: runtime).run(.uiNative("{}")) == "handled:{}")
  }
}
