import Foundation
import ProAppsCore
import Synchronization
import Testing

@testable import AppleProApps

private actor ProgramRecorder {
  var calls: [[String]] = []
  var environments: [[String: String]] = []
  func execute(_ arguments: [String], environment: [String: String]) -> String {
    calls.append(arguments)
    environments.append(environment)
    if arguments.first == "tools" { return #"{"items":[],"hasMore":false}"# }
    if arguments.contains("list") { return #"{"ok":true,"data":{"connections":[]}}"# }
    return #"{"ok":true}"#
  }
}

struct ProgramTests {
  @Test func routesNativeServerAndInventoryWithoutConfiguration() async throws {
    let invoked = Mutex(false)
    let runtime = ProgramRuntime(
      binary: URL(fileURLWithPath: "/tmp/native"), environment: [:],
      serve: { invoked.withLock { $0 = true } }, inventory: { "[]" },
      ui: { _ in throw CocoaError(.featureUnsupported) },
      execute: { _, _, _ in throw CocoaError(.featureUnsupported) })
    let program = Program(runtime: runtime)
    #expect(try await program.run(.serve) == nil)
    #expect(invoked.withLock { $0 })
    #expect(try await program.run(.capabilities) == "[]")
  }

  @Test(arguments: [false, true])
  func setupUsesExplicitScopeAndOptionalExistingUI(_ withUI: Bool) async throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "pro-apps-program-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: directory.appendingPathComponent("scripts"), withIntermediateDirectories: true)
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let executor = directory.appendingPathComponent("scripts/executor")
    let peekaboo = directory.appendingPathComponent("peekaboo")
    #expect(
      FileManager.default.createFile(
        atPath: executor.path, contents: Data(), attributes: [.posixPermissions: 0o700]))
    #expect(
      FileManager.default.createFile(
        atPath: peekaboo.path, contents: Data(), attributes: [.posixPermissions: 0o700]))
    let recorder = ProgramRecorder()
    let uiInvoked = Mutex(false)
    let runtime = ProgramRuntime(
      binary: directory.appendingPathComponent("native"),
      environment: ["EXECUTOR_SCOPE_DIR": "/wrong"],
      serve: {}, inventory: { "[]" }, ui: { _ in uiInvoked.withLock { $0 = true } },
      execute: { _, args, environment in await recorder.execute(args, environment: environment) })
    let args =
      ["setup", "--repo", directory.path, "--peekaboo", peekaboo.path]
      + (withUI ? ["--with-ui"] : [])
    let program = Program(runtime: runtime)
    let output = try await program.run(Command.parse(args))
    #expect(output?.contains("registration is not application/permission validation") == true)
    #expect(await recorder.calls.count == (withUI ? 8 : 4))
    #expect(await recorder.environments.allSatisfy { $0["EXECUTOR_SCOPE_DIR"] == directory.path })
    let ui = try Command.parse(["serve-ui", "--repo", directory.path, "--peekaboo", peekaboo.path])
    #expect(try await program.run(ui) == nil)
    #expect(uiInvoked.withLock { $0 })
  }

  @Test func mediaCommandUsesTheTypedRuntimeBoundary() async throws {
    let runtime = ProgramRuntime(
      binary: URL(fileURLWithPath: "/tmp/native"), environment: [:], serve: {}, inventory: { "[]" },
      ui: { _ in }, execute: { _, _, _ in "" },
      inspectMedia: { path in
        #expect(path == "/tmp/test.mp4")
        return "synthetic media report"
      })
    #expect(
      try await Program(runtime: runtime).run(Command.parse(["inspect-media", "/tmp/test.mp4"]))
        == "synthetic media report")
    #expect(throws: (any Error).self) { try Command.parse(["inspect-media"]) }
    #expect(throws: (any Error).self) {
      try Command.parse(["inspect-media", "/tmp/a.mp4", "--extra"])
    }
    #expect(throws: (any Error).self) { try Command.parse(["inspect-media", "relative.mp4"]) }
  }

  @Test func editingCommandUsesTheTypedRuntimeBoundary() async throws {
    let runtime = ProgramRuntime(
      binary: URL(fileURLWithPath: "/tmp/native"), environment: [:], serve: {}, inventory: { "[]" },
      ui: { _ in }, execute: { _, _, _ in "" },
      editMedia: { path in
        #expect(path == "/tmp/request.json")
        return "synthetic editing report"
      })
    #expect(
      try await Program(runtime: runtime).run(Command.parse(["edit-media", "/tmp/request.json"]))
        == "synthetic editing report")
    #expect(throws: (any Error).self) { try Command.parse(["edit-media"]) }
    #expect(throws: (any Error).self) { try Command.parse(["edit-media", "relative.json"]) }
    #expect(throws: (any Error).self) {
      try Command.parse(["edit-media", "/tmp/request.json", "extra"])
    }
  }

  @Test func referenceCommandUsesTheTypedRuntimeBoundary() async throws {
    let path = try Files.temporaryDirectory().appendingPathComponent("reference.json").path
    let runtime = ProgramRuntime(
      binary: URL(fileURLWithPath: "/unused/native"), environment: [:], serve: {},
      inventory: { "[]" },
      ui: { _ in }, execute: { _, _, _ in "" },
      referenceMedia: { actual in
        #expect(actual == path)
        return "reference metadata only"
      })
    #expect(
      try await Program(runtime: runtime).run(Command.parse(["reference-media", path]))
        == "reference metadata only")
    #expect(throws: (any Error).self) { try Command.parse(["reference-media"]) }
    #expect(throws: (any Error).self) { try Command.parse(["reference-media", "relative.json"]) }
    #expect(throws: (any Error).self) { try Command.parse(["reference-media", path, "extra"]) }
  }

  @Test func propagatesTypedBoundaryFailures() async {
    let runtime = ProgramRuntime(
      binary: URL(fileURLWithPath: "/tmp/native"), environment: [:],
      serve: { throw CancellationError() }, inventory: { throw CocoaError(.fileReadNoPermission) },
      ui: { _ in }, execute: { _, _, _ in "" })
    await #expect(throws: CancellationError.self) {
      try await Program(runtime: runtime).run(.serve)
    }
    await #expect(throws: (any Error).self) {
      try await Program(runtime: runtime).run(.capabilities)
    }
  }
}
