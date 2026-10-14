import Foundation

public enum Command: Sendable, Equatable {
  /// Upper bound for one encoded native UI request passed as a child argument.
  public static let maximumUIRequestBytes = 65536

  case serve
  case capabilities
  case inspectMedia(String)
  case editMedia(String)
  case referenceMedia(String)
  case muxPCM(String)
  case measureMedia(String, String)
  case uiNative(String)
  case fcpSavePlan(String)
  case setup(Configuration)
  case ui(Configuration)

  public static func parse(_ arguments: [String]) throws -> Command {
    var remaining = arguments
    guard !remaining.isEmpty else {
      throw ProAppsError.invalid("Use serve, capabilities, setup or serve-ui")
    }
    let name = remaining.removeFirst()
    if name == "fcp-save-plan" {
      guard remaining.count == 1, let path = remaining.first else {
        throw ProAppsError.invalid("fcp-save-plan requires exactly one local JSON path")
      }
      return .fcpSavePlan(try Files.absolute(path).path)
    }
    if name == "ui-native" {
      guard remaining.count == 1, let request = remaining.first,
        request.utf8.count <= Self.maximumUIRequestBytes
      else {
        throw ProAppsError.invalid("ui-native requires exactly one bounded JSON request")
      }
      return .uiNative(request)
    }
    if name == "measure-media" {
      guard remaining.count == 2 else {
        throw ProAppsError.invalid("measure-media requires an operation and a local JSON path")
      }
      return .measureMedia(remaining[0], try Files.absolute(remaining[1]).path)
    }
    if name == "mux-pcm" {
      guard remaining.count == 1, let path = remaining.first else {
        throw ProAppsError.invalid("mux-pcm requires exactly one local JSON path")
      }
      return .muxPCM(try Files.absolute(path).path)
    }
    if name == "reference-media" {
      guard remaining.count == 1, let path = remaining.first else {
        throw ProAppsError.invalid("reference-media requires exactly one local JSON path")
      }
      return .referenceMedia(try Files.absolute(path).path)
    }
    if name == "inspect-media" || name == "edit-media" {
      guard remaining.count == 1, let path = remaining.first else {
        throw ProAppsError.invalid("\(name) requires exactly one local path")
      }
      let absolute = try Files.absolute(path).path
      return name == "inspect-media" ? .inspectMedia(absolute) : .editMedia(absolute)
    }
    if name == "serve" || name == "capabilities" {
      guard remaining.isEmpty else {
        throw ProAppsError.invalid("This command accepts no arguments")
      }
      return name == "serve" ? .serve : .capabilities
    }
    guard name == "setup" || name == "serve-ui" else {
      throw ProAppsError.invalid("Unknown command")
    }
    var repo: String?
    var peekaboo: String?
    var approve = false
    var withUI = false
    while !remaining.isEmpty {
      let flag = remaining.removeFirst()
      switch flag {
      case "--repo":
        guard repo == nil, !remaining.isEmpty else {
          throw ProAppsError.invalid("Supply --repo once with a path")
        }
        repo = remaining.removeFirst()
      case "--peekaboo":
        guard peekaboo == nil, !remaining.isEmpty else {
          throw ProAppsError.invalid("Supply --peekaboo once with a path")
        }
        peekaboo = remaining.removeFirst()
      case "--approve-registration" where name == "setup" && !approve: approve = true
      case "--with-ui" where name == "setup" && !withUI: withUI = true
      default: throw ProAppsError.invalid("Unknown or duplicate option")
      }
    }
    guard let repo else { throw ProAppsError.invalid("--repo is required") }
    let configuration = Configuration(
      repo: try Files.absolute(repo).resolvingSymlinksInPath(),
      peekaboo: try Files.absolute(peekaboo ?? "/opt/homebrew/bin/peekaboo"),
      approve: approve, withUI: withUI)
    return name == "setup" ? .setup(configuration) : .ui(configuration)
  }
}

public struct Configuration: Sendable, Equatable {
  public let repo: URL
  public let peekaboo: URL
  public let approve: Bool
  public let withUI: Bool

  public var executor: URL { repo.appendingPathComponent("scripts/executor") }

  public func validate(requireUI: Bool) throws {
    guard FileManager.default.isExecutableFile(atPath: executor.path) else {
      throw ProAppsError.unavailable("Expected this dotfiles checkout's Executor wrapper")
    }
    if requireUI && !FileManager.default.isExecutableFile(atPath: peekaboo.path) {
      throw ProAppsError.unavailable("Select the existing Peekaboo executable with --peekaboo")
    }
  }

  public func environment(from original: [String: String]) -> [String: String] {
    var environment = original
    environment["EXECUTOR_SCOPE_DIR"] = repo.path
    return environment
  }

  public var uiArguments: [String] {
    [
      peekaboo.path, "mcp", "serve", "--transport", "stdio", "--allow-foreground",
      "--input-strategy", "actionFirst",
    ]
  }

  public static let uiTools =
    "permissions,app,window,menu,dialog,see,inspect_ui,image,click,type,press,scroll,drag,move,set_value,action,verify_state,paste,clipboard"
}
