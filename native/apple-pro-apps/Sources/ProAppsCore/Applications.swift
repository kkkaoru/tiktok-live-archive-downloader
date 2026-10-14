import AppKit
import Foundation

public enum ProApp: String, Codable, CaseIterable, Sendable {
  case motion, compressor, finalCutPro, logicPro, mainStage

  public var bundleIDs: [String] {
    switch self {
    case .motion: return ["com.apple.motionappApp", "com.apple.motionapp"]
    case .compressor: return ["com.apple.CompressorApp", "com.apple.Compressor"]
    case .finalCutPro: return ["com.apple.FinalCutApp", "com.apple.FinalCut"]
    case .logicPro: return ["com.apple.mobilelogic", "com.apple.logic10"]
    case .mainStage: return ["com.apple.MainStageApp", "com.apple.mainstage3"]
    }
  }

  public var documentExtensions: Set<String> {
    switch self {
    case .motion: return ["motn", "moti", "motr", "moef", "mogen"]
    case .compressor: return ["compressor"]
    case .finalCutPro: return ["fcpxml", "fcpxmld", "fcpbundle"]
    case .logicPro:
      return ["logicx", "logic", "mid", "midi", "wav", "aif", "aiff", "caf", "m4a", "mp3"]
    case .mainStage: return ["concert", "patch"]
    }
  }
}

public struct InstalledApp: Codable, Sendable {
  public let app: ProApp
  public let bundleID: String
  public let path: String
  public let version: String
  public let running: Bool

  public init(app: ProApp, bundleID: String, path: String, version: String, running: Bool) {
    self.app = app
    self.bundleID = bundleID
    self.path = path
    self.version = version
    self.running = running
  }
}

/// Injectable AppKit boundary. Tests supply isolated bundles and an open callback;
/// production uses only LaunchServices metadata and an exact Open Document target.
@MainActor
public struct ApplicationAccess {
  public var locate: (String) -> URL? = {
    NSWorkspace.shared.urlForApplication(withBundleIdentifier: $0)
  }
  public var running: (String) -> Bool = {
    !NSRunningApplication.runningApplications(withBundleIdentifier: $0).isEmpty
  }
  public var open: (URL, URL) async throws -> String? = { document, application in
    let configuration = NSWorkspace.OpenConfiguration()
    configuration.activates = false
    configuration.promptsUserIfNeeded = false
    return try await NSWorkspace.shared.open(
      [document], withApplicationAt: application, configuration: configuration
    ).bundleIdentifier
  }
  public init() {}
}

@MainActor
public enum Applications {
  public static func inventory(access: ApplicationAccess? = nil) -> [InstalledApp] {
    let access = access ?? ApplicationAccess()
    return ProApp.allCases.flatMap { app in
      app.bundleIDs.compactMap { id in
        guard let url = access.locate(id),
          let bundle = Bundle(url: url), bundle.bundleIdentifier == id
        else { return nil }
        return InstalledApp(
          app: app, bundleID: id, path: url.path,
          version: bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
            ?? "unknown",
          running: access.running(id))
      }
    }
  }

  public static func resolve(_ app: ProApp, bundleID: String?, access: ApplicationAccess? = nil)
    throws -> InstalledApp
  {
    let id = bundleID ?? app.bundleIDs[0]
    guard app.bundleIDs.contains(id) else {
      throw ProAppsError.invalid("Bundle ID does not match the selected app")
    }
    guard
      let installed = inventory(access: access).first(where: { $0.app == app && $0.bundleID == id })
    else {
      throw ProAppsError.unavailable("Selected app edition is not installed; inspect capabilities")
    }
    return installed
  }

  public static func openDocument(
    app: ProApp, bundleID: String?, path: String, access: ApplicationAccess? = nil
  ) async throws
    -> String
  {
    let access = access ?? ApplicationAccess()
    let installed = try resolve(app, bundleID: bundleID, access: access)
    let url = try Files.absolute(path).resolvingSymlinksInPath()
    guard app.documentExtensions.contains(url.pathExtension.lowercased()),
      FileManager.default.fileExists(atPath: url.path)
    else {
      throw ProAppsError.invalid("Expected an existing document supported by the selected app")
    }
    let recipient = try await access.open(url, URL(fileURLWithPath: installed.path))
    guard recipient == installed.bundleID else {
      throw ProAppsError.unavailable("Open-document recipient did not match the selected edition")
    }
    return
      "Open-document delivery accepted by \(installed.bundleID). Import completion is not verified; a license or import dialog may still require the user."
  }
}
