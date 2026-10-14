import Foundation
import Testing

@testable import ProAppsCore

struct ConfigurationTests {
  @Test func nativeCommandsHaveNoAmbientOptions() throws {
    #expect(try Command.parse(["serve"]) == .serve)
    #expect(try Command.parse(["capabilities"]) == .capabilities)
  }

  @Test(arguments: [
    [], ["unknown"], ["serve", "--with-ui"], ["capabilities", "x"],
    ["setup"], ["setup", "--repo"], ["setup", "--repo", "relative"],
    ["setup", "--repo", "/tmp/a", "--repo", "/tmp/b"],
    ["setup", "--repo", "/tmp/a", "--approve-registration", "--approve-registration"],
    ["setup", "--repo", "/tmp/a", "--with-ui", "--with-ui"],
    ["serve-ui", "--repo", "/tmp/a", "--approve-registration"],
    ["serve-ui", "--repo", "/tmp/a", "--peekaboo"],
    ["serve-ui", "--repo", "/tmp/a", "--peekaboo", "/a", "--peekaboo", "/b"],
  ])
  func rejectsInvalidArguments(_ arguments: [String]) {
    #expect(throws: (any Error).self) { try Command.parse(arguments) }
  }

  @Test func explicitPathsAndScopeArePreserved() throws {
    let command = try Command.parse([
      "setup", "--repo", "/tmp/pro apps", "--approve-registration", "--with-ui", "--peekaboo",
      "/usr/local/bin/peekaboo",
    ])
    guard case .setup(let config) = command else {
      Issue.record("Expected setup configuration")
      return
    }
    #expect(config.approve && config.withUI)
    #expect(config.peekaboo.path == "/usr/local/bin/peekaboo")
    #expect(config.executor.lastPathComponent == "executor")
    #expect(
      config.environment(from: ["EXECUTOR_SCOPE_DIR": "/wrong", "KEEP": "value"])[
        "EXECUTOR_SCOPE_DIR"] == config.repo.path)
    #expect(config.environment(from: ["KEEP": "value"])["KEEP"] == "value")
    #expect(
      config.uiArguments == [
        "/usr/local/bin/peekaboo", "mcp", "serve", "--transport", "stdio", "--allow-foreground",
        "--input-strategy", "actionFirst",
      ])
    #expect(!Configuration.uiTools.split(separator: ",").contains("shell"))
    guard case .ui(let ui) = try Command.parse(["serve-ui", "--repo", "/tmp/a"]) else {
      Issue.record("Expected UI configuration")
      return
    }
    #expect(ui.peekaboo.path == "/opt/homebrew/bin/peekaboo")
    #expect(!ui.approve && !ui.withUI)
  }

  @Test func validatesExistingExecutablesWithoutLaunchingThem() throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "pro-apps-options-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let configuration = Configuration(
      repo: directory, peekaboo: directory.appendingPathComponent("peekaboo"), approve: false,
      withUI: false)
    #expect(throws: (any Error).self) { try configuration.validate(requireUI: false) }
    try FileManager.default.createDirectory(
      at: directory.appendingPathComponent("scripts"), withIntermediateDirectories: false)
    _ = try Files.writeNew(Data(), to: configuration.executor.path, extensions: [""])
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o700], ofItemAtPath: configuration.executor.path)
    try configuration.validate(requireUI: false)
    #expect(throws: (any Error).self) { try configuration.validate(requireUI: true) }
    _ = try Files.writeNew(Data(), to: configuration.peekaboo.path, extensions: [""])
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o700], ofItemAtPath: configuration.peekaboo.path)
    try configuration.validate(requireUI: true)
  }
}
