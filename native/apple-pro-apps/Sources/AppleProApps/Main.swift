import Darwin
import Foundation
import MCP
import ProAppsCore

/// Thin process/SDK adapters. Command planning and all behavior are testable in
/// Configuration, Program, NativeService and the Core module.
@main
struct AppleProApps {
  static func main() async {
    do {
      guard let binary = Bundle.main.executableURL?.resolvingSymlinksInPath() else {
        throw ProAppsError.unavailable("Cannot resolve the Swift executable")
      }
      let runtime = ProgramRuntime(
        binary: binary, environment: ProcessInfo.processInfo.environment,
        serve: serve,
        inventory: {
          String(decoding: try JSONEncoder().encode(await Applications.inventory()), as: UTF8.self)
        },
        ui: replaceWithPeekaboo,
        execute: { try await Runner.checked($0, $1, environment: $2) })
      let command = try Command.parse(Array(CommandLine.arguments.dropFirst()))
      if let output = try await Program(runtime: runtime).run(command) { print(output) }
    } catch {
      let message = NativeFailureMessage.describe(error)
      FileHandle.standardError.write(Data("\(message)\n".utf8))
      exit(1)
    }
  }

  static func serve() async throws {
    let server = await makeServer(service: NativeService())
    try await server.start(transport: StdioTransport())
    await server.waitUntilCompleted()
  }

  /// Inject the service so native SDK framing/handlers can be verified without
  /// running Apple applications or sending controls to real endpoints.
  static func makeServer(service: NativeService) async -> Server {
    let server = Server(
      name: "apple-pro-apps", version: "1.0.0", capabilities: .init(tools: .init()))
    await server.withMethodHandler(ListTools.self) { _ in .init(tools: ToolSpec.all.map(\.tool)) }
    await server.withMethodHandler(CallTool.self) { params in await service.call(params) }
    return server
  }

  static func replaceWithPeekaboo(_ configuration: Configuration) throws {
    guard setenv("PEEKABOO_ALLOW_TOOLS", Configuration.uiTools, 1) == 0 else {
      throw ProAppsError.unavailable("Cannot configure the UI tool allowlist")
    }
    // execv needs a nil-terminated argv. Each allocation belongs to this call,
    // survives until exec, and is freed if allocation/exec fails. No pointer
    // escapes an await; a successful exec replaces the process and its memory.
    var pointers = configuration.uiArguments.map { strdup($0) }
    defer { for pointer in pointers { free(pointer) } }
    guard pointers.allSatisfy({ $0 != nil }) else {
      throw ProAppsError.unavailable("Cannot allocate native arguments")
    }
    pointers.append(nil)
    execv(configuration.peekaboo.path, &pointers)
    throw ProAppsError.unavailable("Cannot start existing Peekaboo MCP")
  }
}
