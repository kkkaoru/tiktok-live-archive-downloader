import Dispatch
import Foundation

public struct FCPXMLValidation: Codable, Sendable {
  public let version: String
  public let dtdName: String
  public let validDTD: Bool
  public let importVerified: Bool
  public let mediaReferencesVerified: Bool
}

/// The caller supplies the selected installed Final Cut bundle's DTD directory,
/// never a caller-selected schema from MCP. Validated bytes are copied privately
/// before invoking the system validator to avoid source-file replacement races.
public actor FCPXMLValidator {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.fcpxml-validation")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }
  private let directory: URL
  private let workspace: URL?

  public init(dtdDirectory: URL, workspace: URL? = nil) {
    directory = dtdDirectory
    self.workspace = workspace
  }

  public func validate(path: String) async throws -> FCPXMLValidation {
    try Task.checkCancellation()
    let data = try Files.read(Files.existing(path, extensions: ["fcpxml"]))
    let summary = try Interchange.inspect(data, kind: .fcpxml)
    guard let version = summary.version, version.count <= 4 else {
      throw ProAppsError.invalid("FCPXML requires a supported explicit version")
    }
    let components = version.split(separator: ".", omittingEmptySubsequences: false)
    guard components.count == 2, components[0] == "1", let minor = Int(components[1]),
      (0...99).contains(minor), version == "1.\(minor)"
    else { throw ProAppsError.invalid("FCPXML version must be canonical 1.N") }
    let name = "FCPXMLv1_\(minor).dtd"
    let schema = try Files.read(
      Files.existing(directory.appendingPathComponent(name).path, extensions: ["dtd"]))
    guard let text = String(data: schema, encoding: .utf8),
      !text.contains("SYSTEM"), !text.contains("PUBLIC")
    else {
      throw ProAppsError.invalid(
        "Installed DTD must be UTF-8 and self-contained; external schema references are refused")
    }
    let workspace = try self.workspace ?? Files.temporaryDirectory()
    let copy = try Files.reserveOutput(directory: workspace.path, name: "input.fcpxml", kind: .edit)
    defer {
      Cleanup.perform { try FileManager.default.removeItem(at: copy.deletingLastPathComponent()) }
    }
    let localDTD = copy.deletingLastPathComponent().appendingPathComponent("schema.dtd")
    _ = try Files.writeNew(data, to: copy.path, extensions: ["fcpxml"])
    _ = try Files.writeNew(schema, to: localDTD.path, extensions: ["dtd"])
    try Task.checkCancellation()
    let result = try await Runner.run(
      URL(fileURLWithPath: "/usr/bin/xmllint"),
      ["--nonet", "--noout", "--dtdvalid", localDTD.path, copy.path],
      environment: ["XML_CATALOG_FILES": "", "SGML_CATALOG_FILES": ""], timeout: .seconds(30))
    guard result.status == 0 else {
      throw ProAppsError.invalid(
        "Document did not validate against the selected installed Apple DTD")
    }
    return FCPXMLValidation(
      version: version, dtdName: name, validDTD: true, importVerified: false,
      mediaReferencesVerified: false)
  }
}
