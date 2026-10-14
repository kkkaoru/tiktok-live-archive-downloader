import Foundation
import MCP
import Testing

@testable import AppleProApps
@testable import ProAppsCore

struct SourceTextSessionServiceTests {
  private func arguments(count: Int = 64) -> Value {
    .object([
      "checkoutRoot": .string("/checkout"), "path": .string("/checkout/source.mp4"),
      "checkpointDirectory": .string("/checkout/edits/work/session"),
      "width": .int(1080), "height": .int(1920), "maximumSeconds": .int(60),
      "samples": .array(
        (0..<count).map {
          .object(["value": .int($0 * 3000), "timescale": .int(90000)])
        }),
    ])
  }

  @Test func explicitCLISchemaKeepsOldPublicToolAndMCPAdvertisementUnchanged() throws {
    #expect(!ToolSpec.all.contains { $0.name == "video_text_recognize_session" })
    #expect(!ToolSpec.sourceTextSession.readOnly)
    try validate(arguments(), schema: ToolSpec.sourceTextSession.tool.inputSchema)
    for count in [0, 65] {
      #expect(throws: (any Error).self) {
        try validate(arguments(count: count), schema: ToolSpec.sourceTextSession.tool.inputSchema)
      }
    }
    let old = try #require(ToolSpec.all.first { $0.name == "video_text_recognize" })
    #expect(old.readOnly)
    #expect(throws: (any Error).self) {
      try validate(
        .object([
          "path": .string("/checkout/source.mp4"),
          "samples": .array(Array(repeating: .object(["timeSeconds": .int(0)]), count: 9)),
        ]), schema: old.tool.inputSchema)
    }
    var extra = try #require(arguments().objectValue)
    extra["region"] = .object([:])
    #expect(throws: (any Error).self) {
      try validate(.object(extra), schema: ToolSpec.sourceTextSession.tool.inputSchema)
    }
  }

  @Test func directCLIDispatchDecodesTypedRequestWithoutStartingInference() async throws {
    let request = try Files.reserveOutput(
      directory: Files.temporaryDirectory().path,
      name: "session-request.json", kind: .edit)
    defer {
      do { try FileManager.default.removeItem(at: request.deletingLastPathComponent()) } catch {
        Issue.record(error)
      }
    }
    _ = try Files.writeNew(
      JSONEncoder().encode(arguments()), to: request.path, extensions: ["json"])
    let output = try await NativeService().measureLocalFile(
      name: "video_text_recognize_session", path: request.path,
      textSession: { input in
        #expect(
          input.samples.count == 64 && input.samples.last == .init(value: 189000, timescale: 90000))
        #expect(input.path == "/checkout/source.mp4" && input.width == 1080 && input.height == 1920)
        #expect(input.workers == nil)
        return SourceTextSessionReport(
          sourceIdentity: SourceTextSessionIdentity(
            device: 1, inode: 2, bytes: 3, links: 1,
            modifiedSeconds: 4, modifiedNanoseconds: 5, changedSeconds: 6, changedNanoseconds: 7),
          frames: 64, blocks: [], timing: SourceTextSessionTiming())
      })
    let value = try JSONDecoder().decode(Value.self, from: Data(output.utf8))
    #expect(value.objectValue?["protocolVersion"] == .string("source-caption-ocr-session-v1"))
    #expect(value.objectValue?["sourceHashVerified"] == .bool(false))
    #expect(value.objectValue?["qualityApproved"] == .bool(false))
    await #expect(throws: ProAppsError.unavailable("Synthetic")) {
      try await NativeService().measureLocalFile(
        name: "video_text_recognize_session", path: request.path,
        textSession: { _ in throw ProAppsError.unavailable("Synthetic") })
    }
  }

  @Test(arguments: [nil, 1, 2] as [Int?])
  func compiledCLIProcessesOnlyTwoSyntheticOriginalFrames(workers: Int?) async throws {
    let request = try Files.reserveOutput(
      directory: Files.temporaryDirectory().path,
      name: "request.json", kind: .edit)
    let root = request.deletingLastPathComponent()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let checkout = package.deletingLastPathComponent().deletingLastPathComponent()
    let checkpoints = root.appendingPathComponent("checkpoints")
    try FileManager.default.createDirectory(
      at: checkpoints, withIntermediateDirectories: false,
      attributes: [.posixPermissions: 0o700])
    let input = SourceTextSessionRequest(
      checkoutRoot: checkout.path,
      path: package.appendingPathComponent("Tests/ProAppsNativeTests/Fixtures/quadrants-30.mp4")
        .path,
      checkpointDirectory: checkpoints.path,
      samples: [.init(value: 0, timescale: 30), .init(value: 1, timescale: 30)],
      width: 16, height: 16, maximumSeconds: 60, workers: workers,
      allowedSourceRoot: workers == 2 ? package.path : nil)
    let encoded = try JSONDecoder().decode(Value.self, from: JSONEncoder().encode(input))
    #expect(encoded.objectValue?["workers"] == workers.map { .int($0) })
    #expect(
      encoded.objectValue?["allowedSourceRoot"] == (workers == 2 ? .string(package.path) : nil))
    _ = try Files.writeNew(JSONEncoder().encode(input), to: request.path, extensions: ["json"])
    let products = Bundle(for: TestBundle.self).bundleURL.deletingLastPathComponent()
    var environment = ProcessInfo.processInfo.environment
    environment["LLVM_PROFILE_FILE"] =
      products.appendingPathComponent("codecov/session-%p-%m.profraw").path
    let result = try await Runner.run(
      products.appendingPathComponent("apple-pro-apps"),
      ["measure-media", "video_text_recognize_session", request.path], environment: environment,
      timeout: .seconds(15))
    try #require(result.status == 0)
    let value = try JSONDecoder().decode(Value.self, from: Data(result.stdout.utf8))
    #expect(value.objectValue?["frames"] == .int(2))
    #expect(value.objectValue?["sourceHashVerified"] == .bool(false))
    #expect(
      try FileManager.default.contentsOfDirectory(atPath: checkpoints.path).sorted()
        == ["block-0-clocks.json", "block-0.json", "complete.json"])
  }

  @Test(arguments: [0, 3, -1])
  func workerCountIsRejectedBeforeNativeWork(_ workers: Int) async throws {
    var values = try #require(arguments().objectValue)
    values["workers"] = .int(workers)
    #expect(throws: (any Error).self) {
      try validate(.object(values), schema: ToolSpec.sourceTextSession.tool.inputSchema)
    }
    let input = try JSONDecoder().decode(
      SourceTextSessionRequest.self, from: JSONEncoder().encode(Value.object(values)))
    await #expect(throws: ProAppsError.invalid("Invalid OCR worker count")) {
      try await SourceTextSession.run(input)
    }
  }
}
