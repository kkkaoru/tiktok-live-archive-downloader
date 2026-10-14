import Foundation
import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

struct FCPXMLServiceTests {
  @Test func validationResultNeverClaimsAnImport() async throws {
    var interfaces = NativeInterfaces()
    interfaces.validateFCPXML = { path, bundle in
      #expect(path == "/tmp/synthetic.fcpxml")
      #expect(bundle == "com.apple.FinalCutApp")
      return try JSONDecoder().decode(
        FCPXMLValidation.self,
        from: Data(
          "{\"version\":\"1.14\",\"dtdName\":\"FCPXMLv1_14.dtd\",\"validDTD\":true,\"importVerified\":false,\"mediaReferencesVerified\":false}"
            .utf8))
    }
    let result = await NativeService(interfaces: interfaces).call(
      .init(
        name: "fcpxml_validate",
        arguments: [
          "path": .string("/tmp/synthetic.fcpxml"), "bundleID": .string("com.apple.FinalCutApp"),
        ]))
    #expect(result.isError == false)
    #expect(
      result.structuredContent?.objectValue?["validation"]?.objectValue?["validDTD"] == .bool(true))
    #expect(
      result.structuredContent?.objectValue?["validation"]?.objectValue?["importVerified"]
        == .bool(false))
    #expect(result.structuredContent?.objectValue?["sourceModified"] == .bool(false))
  }

  @Test func defaultBoundaryRejectsAnUnrelatedEditionBeforeFileAccess() async {
    await #expect(throws: (any Error).self) {
      try await NativeInterfaces().validateFCPXML("/tmp/not-read.fcpxml", "invalid.bundle")
    }
  }
}
