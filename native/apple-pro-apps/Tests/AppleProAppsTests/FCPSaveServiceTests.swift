import Foundation
import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

struct FCPSaveServiceTests {
  @Test(arguments: ["measure", "edit", "inspect"])
  func childBoundariesFailClosedOutsideNativeExecutable(_ operation: String) async throws {
    let path = try Files.temporaryDirectory().appendingPathComponent(UUID().uuidString).path
    let interfaces = NativeInterfaces()
    await #expect(throws: (any Error).self) {
      switch operation {
      case "measure": _ = try await interfaces.measureMedia("invalid-operation", path)
      case "edit": _ = try await interfaces.editMedia(path)
      default: _ = try await interfaces.inspectMedia(path)
      }
    }
  }

  @Test func offlineCLIUsesDedicatedReadOnlyPlanner() async throws {
    var runtime = ProgramRuntime(
      binary: URL(fileURLWithPath: "/example/native"), environment: [:],
      serve: {}, inventory: { "[]" }, ui: { _ in throw ProAppsError.invalid("Unexpected UI") },
      execute: { _, _, _ in throw ProAppsError.invalid("Unexpected command") })
    let missing = try Files.temporaryDirectory().appendingPathComponent(UUID().uuidString).path
    #expect(throws: (any Error).self) { try runtime.fcpSavePlan(missing) }
    runtime.fcpSavePlan = { path in
      #expect(path == "/example/request.json")
      return "planned"
    }
    #expect(
      try await Program(runtime: runtime).run(.fcpSavePlan("/example/request.json")) == "planned")
  }

  @Test func registeredPlannerInspectsButDoesNotOperate() async throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "save-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    var observed = UIResponse(focus: .init(frontmostBefore: "terminal", frontmostAfter: "terminal"))
    observed.elements = try JSONDecoder().decode(
      [UIElementSummary].self,
      from: Data(
        """
        [{"path":[0],"role":"AXPopUpButton","identifier":"where popup",
          "value":"\(directory.lastPathComponent)","actions":["AXShowMenu"],"valueSettable":false,"unreadable":[]},
         {"path":[1],"role":"AXTextField","identifier":"saveAsNameTextField",
          "value":"old.mov","actions":[],"valueSettable":true,"unreadable":[]}]
        """.utf8))
    let snapshot = observed
    var interfaces = NativeInterfaces()
    interfaces.ui = { request, timeout in
      #expect(
        request
          == .inspect(
            .init(app: .finalCutPro), window: .init(title: "Save"),
            root: nil, maxDepth: 20, maxElements: 500))
      #expect(timeout == .seconds(30))
      return snapshot
    }
    let result = await NativeService(interfaces: interfaces).call(
      .init(
        name: "fcp_save_plan",
        arguments: [
          "directory": .string(directory.path), "observedDirectory": .string(directory.path),
          "fileName": .string("new.mov"), "window": .object(["title": .string("Save")]),
          "saveDispatched": .bool(false),
        ]))
    #expect(result.isError == false)
    let text = String(decoding: try JSONEncoder().encode(result.structuredContent), as: UTF8.self)
    #expect(text.contains("set-name-and-observe"))
    #expect(text.contains("completionVerified"))
  }

  @Test func absentInspectionFailsWithoutMutation() async throws {
    let directory = try Files.temporaryDirectory()
    var interfaces = NativeInterfaces()
    interfaces.ui = { _, _ in UIResponse(focus: .init(frontmostBefore: nil, frontmostAfter: nil)) }
    let result = await NativeService(interfaces: interfaces).call(
      .init(
        name: "fcp_save_plan",
        arguments: [
          "directory": .string(directory.path), "observedDirectory": .string(directory.path),
          "fileName": .string("new.mov"), "window": .object(["title": .string("Save")]),
          "saveDispatched": .bool(false),
        ]))
    #expect(result.isError == true)
  }
}
