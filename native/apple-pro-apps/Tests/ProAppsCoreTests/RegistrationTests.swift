import Foundation
import Testing

@testable import ProAppsCore

private actor MockExecutor {
  var replies: [String]
  var calls: [[String]] = []
  init(_ replies: [String]) { self.replies = replies }
  nonisolated func execute(_ args: [String]) async throws -> String {
    try await record(args)
  }
  private func record(_ args: [String]) throws -> String {
    calls.append(args)
    guard !replies.isEmpty else { throw ProAppsError.invalid("Unexpected test call") }
    return replies.removeFirst()
  }
}

struct RegistrationTests {
  let binary = URL(fileURLWithPath: "/tmp/checkout with spaces/apple-pro-apps")
  let repo = URL(fileURLWithPath: "/tmp/checkout with spaces")
  let catalog = #"{"items":[],"hasMore":false}"#
  let connections = #"{"ok":true,"data":{"connections":[]}}"#
  let ok = #"{"ok":true}"#

  @Test func registersOnlyTwoExplicitOperations() async throws {
    let mock = MockExecutor([catalog, ok, connections, ok])
    try await Registration(approve: false, execute: mock.execute).register(
      binary: binary, repo: repo)
    let calls = await mock.calls
    #expect(calls.count == 4)
    #expect(calls[1].prefix(4) == ["call", "executor", "mcp", "addServer"])
    #expect(calls[3].prefix(5) == ["call", "executor", "coreTools", "connections", "create"])
    struct Payload: Decodable {
      let transport: String
      let command: String
      let args: [String]
      let spawnPerCall: Bool
    }
    let payload = try JSONDecoder().decode(
      Payload.self, from: Data(try #require(calls[1].last).utf8))
    #expect(payload.transport == "stdio")
    #expect(payload.command == binary.path)
    #expect(payload.args == ["serve"])
    #expect(payload.spawnPerCall == false)
    #expect(!calls.joined().contains("policies"))
  }

  @Test func existingConnectionsAreRetained() async throws {
    let mock = MockExecutor([
      #"{"items":[{"id":"apple-pro-apps"}],"hasMore":false}"#,
      #"{"ok":true,"data":{"connections":[{"owner":"user","integration":"apple-pro-apps","name":"default"}]}}"#,
    ])
    try await Registration(approve: true, execute: mock.execute).register(
      binary: binary, repo: repo)
    #expect(await mock.calls.count == 2)
  }

  @Test func approvalMustBeExplicitAndCannotRecurse() async throws {
    let paused = "Execution paused\nexecutionId: setup-123\n"
    let blocked = MockExecutor([catalog, paused])
    await #expect(throws: (any Error).self) {
      try await Registration(approve: false, execute: blocked.execute).register(
        binary: binary, repo: repo)
    }
    #expect(await blocked.calls.count == 2)
    let approved = MockExecutor([catalog, paused, ok, connections, paused, ok])
    try await Registration(approve: true, execute: approved.execute).register(
      binary: binary, repo: repo)
    let calls = await approved.calls
    #expect(calls.filter { $0.first == "resume" }.count == 2)
    #expect(
      calls[2] == [
        "resume", "--execution-id", "setup-123", "--action", "accept", "--content", "{}",
      ])
    let nested = MockExecutor([catalog, paused, paused])
    await #expect(throws: (any Error).self) {
      try await Registration(approve: true, execute: nested.execute).register(
        binary: binary, repo: repo)
    }
    #expect(await nested.calls.count == 3)
  }

  @Test(arguments: [
    [#"{"items":[],"hasMore":true}"#], [#"{"items":[],"hasMore":false}"#, #"{"ok":false}"#],
    ["Execution paused"],
  ])
  func rejectsPartialCatalogAndToolErrors(_ replies: [String]) async {
    let mock = MockExecutor(replies)
    await #expect(throws: (any Error).self) {
      try await Registration(approve: true, execute: mock.execute).register(
        binary: binary, repo: repo)
    }
  }

  @Test func malformedRegistrationPreservesDecodingFailure() async {
    let mock = MockExecutor([catalog, "not JSON"])
    await #expect(throws: DecodingError.self) {
      try await Registration(approve: false, execute: mock.execute).register(
        binary: binary, repo: repo)
    }
  }

  @Test func uiFallbackUsesExistingExplicitBinary() async throws {
    let mock = MockExecutor([catalog, ok, connections, ok])
    try await Registration(approve: false, execute: mock.execute).register(
      binary: binary, repo: repo, ui: true,
      peekaboo: URL(fileURLWithPath: "/opt/homebrew/bin/peekaboo"))
    let payload = try #require(await mock.calls[1].last)
    #expect(payload.contains("apple-pro-apps-ui"))
    #expect(payload.contains("serve-ui"))
    #expect(payload.contains("peekaboo"))
    #expect(Registration.approvalID("normal output") == nil)
  }
}
