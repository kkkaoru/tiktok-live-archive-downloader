import Foundation
import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

struct FaceTrackServiceTests {
  private func localService() -> NativeService {
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      let text = try await NativeService().measureLocalFile(name: name, path: path)
      return try JSONDecoder().decode(Value.self, from: Data(text.utf8))
    }
    return NativeService(interfaces: interfaces)
  }
  @Test func requiresAnExplicitBoundedPlanAndRefusesUnknownFields() throws {
    let spec = try #require(ToolSpec.all.first { $0.name == "video_face_track" })
    #expect(spec.readOnly)
    #expect(throws: (any Error).self) {
      try validate(.object(["path": .string("/unused.mp4")]), schema: spec.tool.inputSchema)
    }
    #expect(throws: (any Error).self) {
      try validate(
        .object([
          "path": .string("/unused.mp4"),
          "plan": .object([
            "startSeconds": .int(0), "durationSeconds": .int(1),
            "maximumFrames": .int(7201), "redetectIntervalSeconds": .double(0.5),
          ]),
        ]), schema: spec.tool.inputSchema)
    }
    #expect(throws: (any Error).self) {
      try validate(
        .object([
          "path": .string("/unused.mp4"),
          "plan": .object([
            "startSeconds": .int(0), "durationSeconds": .int(1), "maximumFrames": .int(30),
            "redetectIntervalSeconds": .double(0.5), "defaultHeadY": .int(500),
          ]),
        ]), schema: spec.tool.inputSchema)
    }
  }

  @Test func cliAndServiceUseTheSameReadOnlyTrackingImplementation() async throws {
    let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let source = package.appendingPathComponent("Tests/ProAppsNativeTests/Fixtures/black.mp4")
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      let text = try await NativeService().measureLocalFile(name: name, path: path)
      return try JSONDecoder().decode(Value.self, from: Data(text.utf8))
    }
    let response = await NativeService(interfaces: interfaces).call(
      .init(
        name: "video_face_track",
        arguments: [
          "path": .string(source.path),
          "plan": .object([
            "startSeconds": .int(0), "durationSeconds": .int(1), "maximumFrames": .int(1),
            "redetectIntervalSeconds": .double(0.5), "sourceTimescale": .int(30000),
          ]),
        ]))
    #expect(response.isError != true)
    let frames = try #require(response.structuredContent?.objectValue?["measurement"]?.arrayValue)
    #expect(frames.count == 1)
    let first = try #require(frames.first?.objectValue)
    #expect(first["presentationValue"] == .int(0))
    #expect(first["method"] == .string("missing"))
    #expect(first["faces"] == .array([]))
    #expect(response.structuredContent?.objectValue?["sourceModified"] == .bool(false))
    #expect(response.structuredContent?.objectValue?["playbackPerformed"] == .bool(false))
  }

  @Test func evidenceOptionRequiresABooleanAndRetainsExistingReadOnlyBudget() throws {
    let spec = try #require(ToolSpec.all.first { $0.name == "video_face_track" })
    #expect(spec.readOnly)
    let plan: Value = .object([
      "startSeconds": .int(0), "durationSeconds": .int(1), "maximumFrames": .int(1),
      "redetectIntervalSeconds": .double(0.5),
    ])
    try validate(
      .object(["path": .string("/unused.mp4"), "plan": plan, "objectEvidence": .bool(true)]),
      schema: spec.tool.inputSchema)
    #expect(throws: (any Error).self) {
      try validate(
        .object(["path": .string("/unused.mp4"), "plan": plan, "objectEvidence": .int(1)]),
        schema: spec.tool.inputSchema)
    }
  }

  @Test(arguments: [false, true])
  func explicitOptionUsesTheSameServiceAndKeepsFaceAbsence(_ evidence: Bool) async throws {
    let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let source = package.appendingPathComponent("Tests/ProAppsNativeTests/Fixtures/black.mp4")
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      let text = try await NativeService().measureLocalFile(name: name, path: path)
      return try JSONDecoder().decode(Value.self, from: Data(text.utf8))
    }
    let response = await NativeService(interfaces: interfaces).call(
      .init(
        name: "video_face_track",
        arguments: [
          "path": .string(source.path), "objectEvidence": .bool(evidence),
          "plan": .object([
            "startSeconds": .int(0), "durationSeconds": .int(1), "maximumFrames": .int(1),
            "redetectIntervalSeconds": .double(0.5), "sourceTimescale": .int(30000),
          ]),
        ]))
    #expect(response.isError != true)
    let frames = try #require(response.structuredContent?.objectValue?["measurement"]?.arrayValue)
    let first = try #require(frames.first?.objectValue)
    if evidence {
      let face = try #require(first["faceFrame"]?.objectValue)
      #expect(face["method"] == .string("missing") && face["faces"] == .array([]))
      #expect(first["objects"] == .array([]))
      #expect(first["faceObserved"] == .bool(false))
      #expect(first["objectEvidenceOnly"] == .bool(true))
      #expect(first["semanticHeadVerified"] == .bool(false))
      #expect(first["captionPlacementApproved"] == .bool(false))
    } else {
      #expect(first["method"] == .string("missing") && first["faces"] == .array([]))
      #expect(first["faceFrame"] == nil && first["objects"] == nil)
    }
    #expect(response.structuredContent?.objectValue?["sourceModified"] == .bool(false))
    #expect(response.structuredContent?.objectValue?["playbackPerformed"] == .bool(false))
  }

  @Test func partialObjectOptionRequiresRawEvidenceAndAnExplicitPolicy() async throws {
    let plan: Value = .object([
      "startSeconds": .int(0), "durationSeconds": .int(1), "maximumFrames": .int(1),
      "redetectIntervalSeconds": .double(0.5), "sourceTimescale": .int(30000),
    ])
    let missing = await localService().call(
      .init(
        name: "video_face_track",
        arguments: [
          "path": .string("/unused.mp4"), "plan": plan, "partialObjectEvidence": .bool(true),
        ]))
    #expect(missing.isError == true)
    let policy: Value = .object([
      "maximumSeeds": .int(4), "minimumVisibleFraction": .double(0.25), "boundaries": .array([]),
    ])
    let notEnabled = await localService().call(
      .init(
        name: "video_face_track",
        arguments: [
          "path": .string("/unused.mp4"), "plan": plan, "partialObjectPlan": policy,
        ]))
    #expect(notEnabled.isError == true)
    let missingRaw = await localService().call(
      .init(
        name: "video_face_track",
        arguments: [
          "path": .string("/unused.mp4"), "plan": plan, "partialObjectEvidence": .bool(true),
          "partialObjectPlan": policy, "objectEvidence": .bool(false),
        ]))
    #expect(missingRaw.isError == true)
  }

  @Test func optInPartialServiceRecordsMissingSupportWithoutChangingLegacyFaces() async throws {
    let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let source = package.appendingPathComponent("Tests/ProAppsNativeTests/Fixtures/black.mp4")
    let response = await localService().call(
      .init(
        name: "video_face_track",
        arguments: [
          "path": .string(source.path), "objectEvidence": .bool(true),
          "partialObjectEvidence": .bool(true),
          "partialObjectPlan": .object([
            "maximumSeeds": .int(4), "minimumVisibleFraction": .double(0.25),
            "boundaries": .array([]),
          ]),
          "plan": .object([
            "startSeconds": .int(0), "durationSeconds": .int(1), "maximumFrames": .int(1),
            "redetectIntervalSeconds": .double(0.5), "sourceTimescale": .int(30000),
          ]),
        ]))
    #expect(response.isError != true)
    let first = try #require(
      response.structuredContent?.objectValue?["measurement"]?.arrayValue?.first?.objectValue)
    #expect(first["partialObjects"] == .array([]))
    #expect(first["faceObserved"] == .bool(false))
    #expect(first["partialObjectEvidenceOnly"] == .bool(true))
    #expect(first["captionPlacementApproved"] == .bool(false))
    #expect(first["faceFrame"]?.objectValue?["faces"] == .array([]))
  }
}
