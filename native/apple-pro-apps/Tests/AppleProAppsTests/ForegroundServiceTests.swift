import Foundation
import MCP
import Testing

@testable import AppleProApps
@testable import ProAppsCore

struct ForegroundServiceTests {
  private func face(value: Int) -> Value {
    .object([
      "presentation": .object(["value": .int(value), "timescale": .int(30)]),
      "x": .int(2), "y": .int(2), "width": .int(2), "height": .int(2),
      "minimumFaceCoverage": .double(0.5),
    ])
  }

  @Test(arguments: [Value.array([]), .array(Array(repeating: .null, count: 9)), .array([.null])])
  func schemaRejectsInvalidFaceBatches(_ faces: Value) throws {
    let spec = try #require(ToolSpec.all.first { $0.name == "video_foreground_measure" })
    #expect(throws: (any Error).self) {
      try validate(
        .object([
          "path": .string("/unused.mp4"),
          "samples": .array([.object(["value": .int(0), "timescale": .int(30)])]),
          "headFaces": faces,
        ]), schema: spec.tool.inputSchema)
    }
  }

  @Test func schemaRejectsNineValidFaceEntries() throws {
    let spec = try #require(ToolSpec.all.first { $0.name == "video_foreground_measure" })
    #expect(throws: (any Error).self) {
      try validate(
        .object([
          "path": .string("/unused.mp4"),
          "samples": .array([.object(["value": .int(0), "timescale": .int(30)])]),
          "headFaces": .array(Array(repeating: face(value: 0), count: 9)),
        ]), schema: spec.tool.inputSchema)
    }
  }

  @Test func boundedBatchedHeadFacesReachTheNativeAPIWithExactOrderedClocks() async throws {
    let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let source = package.appendingPathComponent(
      "Tests/ProAppsNativeTests/Fixtures/quadrants-30.mp4")
    let original = try Data(contentsOf: source)
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      let output = try await NativeService().measureLocalFile(name: name, path: path)
      return try JSONDecoder().decode(Value.self, from: Data(output.utf8))
    }
    let result = await NativeService(interfaces: interfaces).call(
      .init(
        name: "video_foreground_measure",
        arguments: [
          "path": .string(source.path),
          "samples": .array([
            .object(["value": .int(0), "timescale": .int(30)]),
            .object(["value": .int(1), "timescale": .int(30)]),
          ]),
          "headFaces": .array([face(value: 0), face(value: 1)]),
        ]))
    try #require(result.isError == false)
    let rows = try #require(result.structuredContent?.objectValue?["measurement"]?.arrayValue)
    #expect(rows.count == 2)
    #expect(
      rows.first?.objectValue?["requested"] == .object(["value": .int(0), "timescale": .int(30)]))
    #expect(
      rows.last?.objectValue?["requested"] == .object(["value": .int(1), "timescale": .int(30)]))
    #expect(rows.allSatisfy { $0.objectValue?["headTop"]?.objectValue != nil })
    #expect(
      rows.allSatisfy {
        $0.objectValue?["headTop"]?.objectValue?["captionPlacementApproved"] == .bool(false)
      })
    #expect(try Data(contentsOf: source) == original)
  }

  @Test(arguments: [
    Value.array([]),
    .array(Array(repeating: .object(["value": .int(0), "timescale": .int(7)]), count: 9)),
    .array([.object(["value": .bool(false), "timescale": .int(1)])]),
    .array([.object(["value": .double(0.5), "timescale": .int(1)])]),
    .array([.object(["value": .int(0), "timescale": .int(0)])]),
    .array([.object(["value": .int(0), "timescale": .int(1_000_000_001)])]),
    .array([.object(["value": .int(0), "timescale": .int(1), "region": .object([:])])]),
    .array([.object(["timeSeconds": .int(0)])]),
  ])
  func schemaRejectsInvalidSamples(_ samples: Value) throws {
    let spec = try #require(ToolSpec.all.first { $0.name == "video_foreground_measure" })
    #expect(spec.readOnly)
    #expect(throws: (any Error).self) {
      try validate(
        .object(["path": .string("/unused.mp4"), "samples": samples]), schema: spec.tool.inputSchema
      )
    }
  }

  @Test(arguments: [0, 1, 2, 3])
  func serviceDispatchPreservesAllObservationStates(_ state: Int) async throws {
    let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let source = package.appendingPathComponent("Tests/ProAppsNativeTests/Fixtures/black.mp4")
    let original = try Data(contentsOf: source)
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      #expect(name == "video_foreground_measure")
      let text = try await NativeService().measureLocalFile(name: name, path: path) { request in
        #expect(request.path == source.path)
        #expect(request.samples == [.init(value: 0, timescale: 7)])
        var operations = ForegroundOperations()
        operations.observe = { _, image, _ in
          if state == 0 { return nil }
          return try ForegroundObservation.collect(labels: state == 1 ? [] : [1, 3]) {
            var bytes = Data(repeating: 0, count: image.width * image.height)
            if state == 3 { bytes[image.width + 2] = 1 }
            return try ForegroundMaskStatistics.scan(
              bytes, width: image.width, height: image.height, bytesPerRow: image.width)
          }
        }
        return try await FrameProbe().measureForeground(
          path: request.path, samples: request.samples, operations: operations)
      }
      return try JSONDecoder().decode(Value.self, from: Data(text.utf8))
    }
    let result = await NativeService(interfaces: interfaces).call(
      .init(
        name: "video_foreground_measure",
        arguments: [
          "path": .string(source.path),
          "samples": .array([.object(["value": .int(0), "timescale": .int(7)])]),
        ]))
    #expect(result.isError != true)
    let envelope = try #require(result.structuredContent?.objectValue)
    let rows = try #require(envelope["measurement"]?.arrayValue)
    #expect(rows.count == 1)
    let row = try #require(rows.first?.objectValue)
    #expect(row["requested"] == .object(["value": .int(0), "timescale": .int(7)]))
    #expect(row["actual"]?.objectValue?["value"] == .int(0))
    #expect(row["width"] == .int(16) && row["height"] == .int(16))
    #expect(row["observationPresent"] == .bool(state != 0))
    #expect(row["captionPlacementApproved"] == .bool(false))
    #expect(row["maskThreshold8Bit"] == .int(1))
    switch state {
    case 0:
      #expect(row["instanceLabels"] == .null && row["unionMask"] == .null)
    case 1:
      #expect(row["instanceLabels"] == .array([]) && row["unionMask"] == .null)
    case 2:
      #expect(row["instanceLabels"] == .array([.int(1), .int(3)]))
      #expect(
        row["unionMask"]
          == .object([
            "width": .int(16), "height": .int(16), "nonzeroPixels": .int(0), "nonzeroBounds": .null,
          ]))
    default:
      #expect(row["instanceLabels"] == .array([.int(1), .int(3)]))
      #expect(
        row["unionMask"]
          == .object([
            "width": .int(16), "height": .int(16), "nonzeroPixels": .int(1),
            "nonzeroBounds": .object([
              "x": .int(2), "y": .int(1), "width": .int(1), "height": .int(1),
            ]),
          ]))
    }
    #expect(envelope["sourceModified"] == .bool(false))
    #expect(envelope["playbackPerformed"] == .bool(false))
    #expect(NativeInterfaces.measurementTimeout("video_foreground_measure") == .seconds(60))
    #expect(try Data(contentsOf: source) == original)
  }

  @Test func defaultLocalHandlerRejectsDuplicateClocksWithoutInference() async throws {
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      let text = try await NativeService().measureLocalFile(name: name, path: path)
      return try JSONDecoder().decode(Value.self, from: Data(text.utf8))
    }
    let result = await NativeService(interfaces: interfaces).call(
      .init(
        name: "video_foreground_measure",
        arguments: [
          "path": .string("/unused.mp4"),
          "samples": .array([
            .object(["value": .int(0), "timescale": .int(7)]),
            .object(["value": .int(0), "timescale": .int(7)]),
          ]),
        ]))
    #expect(result.isError == true)
    #expect(result.structuredContent?.objectValue?["retrySafe"] == .bool(false))
  }
}
