import Foundation
import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

struct HeadServiceTests {
  @Test func schemaRefusesCropsAndEmptySamples() throws {
    let spec = try #require(ToolSpec.all.first { $0.name == "video_head_detect" })
    #expect(spec.readOnly)
    #expect(throws: (any Error).self) {
      try validate(
        .object(["path": .string("/tmp/source.mp4"), "samples": .array([])]),
        schema: spec.tool.inputSchema)
    }
    #expect(throws: (any Error).self) {
      try validate(
        .object([
          "path": .string("/tmp/source.mp4"),
          "samples": .array([
            .object(["timeSeconds": .int(0), "region": .object([:])])
          ]),
        ]), schema: spec.tool.inputSchema)
    }
  }

  @Test(arguments: [Bool?.none, .some(false), .some(true)])
  func serviceAndMeasurementRoutePreserveEmptyDetectionAndActualTime(compare: Bool?) async throws {
    let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let source = package.appendingPathComponent("Tests/ProAppsNativeTests/Fixtures/black.mp4")
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      let text = try await NativeService().measureLocalFile(name: name, path: path)
      return try JSONDecoder().decode(Value.self, from: Data(text.utf8))
    }
    var arguments: [String: Value] = [
      "path": .string(source.path),
      "samples": .array([.object(["timeSeconds": .int(0)])]),
    ]
    if let compare { arguments["compareFaceDetectors"] = .bool(compare) }
    let response = await NativeService(interfaces: interfaces).call(
      .init(name: "video_head_detect", arguments: arguments))
    #expect(response.isError != true)
    let measurements = try #require(
      response.structuredContent?.objectValue?["measurement"]?.arrayValue)
    let first = try #require(measurements.first?.objectValue)
    #expect(first["actualTimeSeconds"] == .int(0))
    #expect(first["faces"] == .array([]) && first["bodies"] == .array([]))
    #expect((first["faceDetectorComparison"] != nil) == (compare == true))
    #expect(throws: (any Error).self) {
      let spec = try #require(ToolSpec.all.first { $0.name == "video_head_detect" })
      try validate(
        .object([
          "path": .string(source.path),
          "samples": .array([.object(["timeSeconds": .int(0)])]),
          "compareFaceDetectors": .string("true"),
        ]), schema: spec.tool.inputSchema)
    }
    #expect(response.structuredContent?.objectValue?["sourceModified"] == .bool(false))
  }
}
