import Foundation

public struct Registration: Sendable {
  public typealias Execute = @Sendable ([String]) async throws -> String
  private let execute: Execute
  private let approve: Bool

  public init(approve: Bool, execute: @escaping Execute) {
    self.approve = approve
    self.execute = execute
  }

  struct Catalog: Decodable {
    struct Item: Decodable { let id: String }
    let items: [Item]
    let hasMore: Bool
  }
  struct Envelope: Decodable { let ok: Bool }
  struct Connections: Decodable {
    struct Payload: Decodable {
      struct Connection: Decodable {
        let owner: String
        let integration: String
        let name: String
      }
      let connections: [Connection]
    }
    let ok: Bool
    let data: Payload
  }
  struct ServerPayload: Encodable {
    let transport: String
    let slug: String
    let name: String
    let command: String
    let args: [String]
    let spawnPerCall: Bool
  }

  static func json<T: Encodable>(_ value: T) throws -> String {
    String(decoding: try JSONEncoder().encode(value), as: UTF8.self)
  }

  public static func approvalID(_ output: String) -> String? {
    output.split(separator: "\n").first(where: { $0.hasPrefix("executionId: ") }).map {
      String($0.dropFirst("executionId: ".count)).trimmingCharacters(in: .whitespaces)
    }
  }

  private func setupCall(_ arguments: [String]) async throws {
    var output = try await execute(arguments)
    if let id = Self.approvalID(output) {
      guard approve else {
        throw ProAppsError.unavailable(
          "Executor registration approval required for execution \(id). Use --approve-registration only for this setup."
        )
      }
      guard !id.isEmpty, id.utf8.count <= 200, !id.contains("\0") else {
        throw ProAppsError.invalid("Invalid approval execution ID")
      }
      output = try await execute([
        "resume", "--execution-id", id, "--action", "accept", "--content", "{}",
      ])
    }
    // Exactly one opt-in resume. Nested prompts and tool-level failures fail
    // closed; no policy changes or approval of application actions exists here.
    guard Self.approvalID(output) == nil else {
      throw ProAppsError.unavailable("Registration needs an additional explicit approval")
    }
    let envelope = try JSONDecoder().decode(Envelope.self, from: Data(output.utf8))
    guard envelope.ok else { throw ProAppsError.unavailable("Registration failed") }
  }

  public func register(binary: URL, repo: URL, ui: Bool = false, peekaboo: URL? = nil) async throws
  {
    let slug = ui ? "apple-pro-apps-ui" : "apple-pro-apps"
    let catalogText = try await execute(["tools", "integrations", "--limit", "1000"])
    let catalog = try JSONDecoder().decode(Catalog.self, from: Data(catalogText.utf8))
    guard !catalog.hasMore else {
      throw ProAppsError.unavailable("Catalog is truncated; cannot safely infer absence")
    }
    if !catalog.items.contains(where: { $0.id == slug }) {
      var args = ["serve"]
      if ui {
        guard let peekaboo else {
          throw ProAppsError.invalid("UI registration requires an explicit Peekaboo executable")
        }
        args = ["serve-ui", "--repo", repo.path, "--peekaboo", peekaboo.path]
      }
      let payload = ServerPayload(
        transport: "stdio", slug: slug,
        name: ui ? "Apple Pro Apps UI Fallback (Peekaboo)" : "Apple Pro Apps Native (Swift)",
        command: binary.path, args: args, spawnPerCall: false)
      try await setupCall(["call", "executor", "mcp", "addServer", Self.json(payload)])
    }
    let filter = try Self.json(["integration": slug, "owner": "user"])
    let connectionsText = try await execute([
      "call", "executor", "coreTools", "connections", "list", filter,
    ])
    let connections = try JSONDecoder().decode(Connections.self, from: Data(connectionsText.utf8))
    guard connections.ok else { throw ProAppsError.unavailable("Connection list failed") }
    if !connections.data.connections.contains(where: {
      $0.integration == slug && $0.owner == "user" && $0.name == "default"
    }) {
      let payload = try Self.json([
        "owner": "user", "name": "default", "integration": slug, "template": "none",
      ])
      try await setupCall(["call", "executor", "coreTools", "connections", "create", payload])
    }
  }
}
