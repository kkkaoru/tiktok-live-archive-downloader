import Foundation
import Testing

@testable import ProAppsCore

struct FCPXMLValidatorTests {
  private func directory() throws -> URL {
    let url = try Files.temporaryDirectory().appendingPathComponent(
      "fcpxml-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
    return url
  }

  private let schema =
    "<!ELEMENT fcpxml (resources)><!ATTLIST fcpxml version CDATA #FIXED '1.14'><!ELEMENT resources EMPTY>"

  @Test(arguments: [true, false])
  func dtdValidationIsStricterThanWellFormedXMLAndCleansSnapshots(_ valid: Bool) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    try Data(schema.utf8).write(to: root.appendingPathComponent("FCPXMLv1_14.dtd"))
    let text =
      valid
      ? "<!DOCTYPE fcpxml><fcpxml version='1.14'><resources/></fcpxml>"
      : "<fcpxml version='1.14'><wrong/></fcpxml>"
    let data = Data(text.utf8)
    let source = root.appendingPathComponent("input.fcpxml")
    try data.write(to: source)
    #expect(try Interchange.inspect(data, kind: .fcpxml).root == "fcpxml")
    let validator = FCPXMLValidator(dtdDirectory: root, workspace: root)
    if valid {
      let result = try await validator.validate(path: source.path)
      #expect(result.validDTD)
      #expect(!result.importVerified)
      #expect(!result.mediaReferencesVerified)
      #expect(result.dtdName == "FCPXMLv1_14.dtd")
      #expect(
        try JSONDecoder().decode(FCPXMLValidation.self, from: JSONEncoder().encode(result)).version
          == "1.14")
      #expect(try await FCPXMLValidator(dtdDirectory: root).validate(path: source.path).validDTD)
    } else {
      await #expect(throws: (any Error).self) { try await validator.validate(path: source.path) }
    }
    #expect(try Data(contentsOf: source) == data)
    let names = try FileManager.default.contentsOfDirectory(atPath: root.path)
    #expect(!names.contains { $0.hasPrefix("edit-") })
  }

  @Test(arguments: ["", "1", "1.014", "1.-1", "2.14", "1.xx", "1.99"])
  func invalidAndMissingVersionSchemasFailClosed(_ version: String) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = root.appendingPathComponent("input.fcpxml")
    let attribute = version.isEmpty ? "" : " version='\(version)'"
    try Data("<fcpxml\(attribute)/>".utf8).write(to: source)
    await #expect(throws: (any Error).self) {
      try await FCPXMLValidator(dtdDirectory: root).validate(path: source.path)
    }
  }

  @Test(arguments: [
    "<!ENTITY % extra SYSTEM 'https://invalid.example/schema'>%extra;",
    "<!ENTITY % extra PUBLIC 'untrusted' 'https://invalid.example/schema'>%extra;",
  ])
  func externalSchemaResolutionIsRefusedBeforeTheProcess(_ external: String) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    try Data(external.utf8).write(to: root.appendingPathComponent("FCPXMLv1_14.dtd"))
    let source = root.appendingPathComponent("input.fcpxml")
    try Data("<fcpxml version='1.14'><resources/></fcpxml>".utf8).write(to: source)
    await #expect(throws: (any Error).self) {
      try await FCPXMLValidator(dtdDirectory: root).validate(path: source.path)
    }
  }
}
