import Foundation
import MCP
import ProAppsCore

extension ToolSpec {
  static let fcpSave: [ToolSpec] = [
    .init(
      name: "fcp_save_plan",
      description:
        "Read one named Final Cut Pro Save window and plan exactly one typed AX action. Existing canonical directory, new basename and independently observed current directory are explicit; no home-only restriction. Does not execute the action, activate apps or synthesize input. Re-observe after each action, refresh this plan immediately before Save, and keep dispatch separate from file completion. Popup basename alone is not full-path evidence.",
      properties: [
        "directory": string(), "fileName": string(maximum: 180),
        "observedDirectory": string(), "bundleID": string(),
        "navigationAncestor": string(),
        "window": object(
          ["title": string(maximum: 512), "subrole": string(maximum: 128)], ["title"]),
        "root": elementLocator, "saveDispatched": boolean,
      ], required: ["directory", "fileName", "observedDirectory", "window", "saveDispatched"],
      readOnly: true),
    .init(
      name: "fcp_save_navigate",
      description:
        "Navigate only one already-open named Final Cut Pro Save panel. Explicit canonical directory candidates (1–16), exact running PID and full ordered OS-localized ancestor chains are required. Bounded 1–16 folder steps and at most 15 seconds of navigation; one observed action per step, independent full readback even after AX error. Never launches/activates apps, opens Share, writes a filename or presses Save. Prior saveDispatched is observation only. Caller owns the GUI lane; uncertain results require observation before retrying.",
      properties: [
        "directory": string(), "fileName": string(maximum: 180),
        "candidateDirectories": array(string(), maximum: 16), "bundleID": string(),
        "expectedPID": integer(1, Int(Int32.max)), "maximumSteps": integer(1, 16),
        "maximumSeconds": number, "saveDispatched": boolean,
        "window": object(
          ["title": string(maximum: 512), "subrole": string(maximum: 128)], ["title"]),
      ],
      required: [
        "directory", "fileName", "candidateDirectories", "expectedPID", "maximumSteps",
        "maximumSeconds", "saveDispatched", "window",
      ], readOnly: false),
  ]
}

extension NativeService {
  func fcpSaveNavigate(_ arguments: Value) async throws -> CallTool.Result {
    struct Input: Decodable { let bundleID: String? }
    let input = try decode(Input.self, arguments)
    let navigation = try decode(FCPSaveNavigationRequest.self, arguments)
    _ = try navigation.validatedDirectories()
    let observed = try await interfaces.ui(
      .saveNavigate(.init(app: .finalCutPro, bundleID: input.bundleID), navigation),
      .seconds(Int(navigation.maximumSeconds.rounded(.up))))
    guard let receipt = observed.saveNavigation else {
      throw ProAppsError.unavailable("Native Save navigation returned no attempt receipt")
    }
    guard var fields = try Value(observed).objectValue else {
      throw ProAppsError.unavailable("Native Save navigation did not encode as an object")
    }
    fields["retrySafe"] = .bool(false)
    fields["completionVerified"] = .bool(false)
    fields["nameAttempted"] = .bool(receipt.nameAttempted)
    fields["saveAttempted"] = .bool(receipt.saveAttempted)
    return response(
      fields,
      failed: receipt.status != .directoryVerified && receipt.status != .observeExistingDispatch)
  }

  func fcpSavePlan(_ arguments: Value) async throws -> CallTool.Result {
    struct Input: Decodable { let root: UIElementLocator? }
    let context = try decode(FCPSaveContext.self, arguments)
    let input = try decode(Input.self, arguments)
    // Validate filesystem destinations before touching the app even for observation.
    _ = try FCPSavePlanner.canonicalDirectoryForPlanning(context.directory)
    let observed = try await interfaces.ui(
      .inspect(
        UITarget(app: .finalCutPro, bundleID: context.bundleID),
        window: context.window, root: input.root, maxDepth: 20, maxElements: 500), .seconds(30))
    guard let elements = observed.elements else {
      throw ProAppsError.unavailable("Save inspection returned no AX elements")
    }
    let plan = try FCPSavePlanner.plan(context, elements: elements)
    return response([
      "plan": try JSONDecoder().decode(Value.self, from: JSONEncoder().encode(plan)),
      "observation": try Value(observed),
      "retrySafe": .bool(false), "completionVerified": .bool(false),
    ])
  }
}
