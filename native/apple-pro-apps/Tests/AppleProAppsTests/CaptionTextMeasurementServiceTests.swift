import Foundation
import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

struct CaptionTextMeasurementServiceTests {
  @Test(arguments: [
    String(repeating: "a", count: 120),
    String(repeating: "字", count: 120),
    String(repeating: "😀", count: 60),
    String(repeating: "字", count: 512),
    String(repeating: "😀", count: 256),
  ])
  func schemaAcceptsEveryUTF16BoundaryEncoding(text: String) throws {
    let spec = try #require(ToolSpec.all.first { $0.name == "caption_text_measure" })
    try validate(
      .object([
        "fontName": .string("ArialMT"), "assFontSize": .int(66),
        "border": .int(0), "rim": .int(0), "texts": .array([.string(text)]),
      ]), schema: spec.tool.inputSchema)
  }

  @Test func schemaRejectsBytesBeyondTheSafeUTF16UpperBound() throws {
    let spec = try #require(ToolSpec.all.first { $0.name == "caption_text_measure" })
    let text = String(repeating: "字", count: 512) + "a"
    #expect(text.utf8.count == 1537)
    #expect(throws: ProAppsError.self) {
      try validate(
        .object([
          "fontName": .string("ArialMT"), "assFontSize": .int(66),
          "border": .int(0), "rim": .int(0), "texts": .array([.string(text)]),
        ]), schema: spec.tool.inputSchema)
    }
  }

  @Test func schemaIsReadOnlyAndNativeDispatchUsesTheRequest() async throws {
    let spec = try #require(ToolSpec.all.first { $0.name == "caption_text_measure" })
    #expect(spec.readOnly)
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, path in
      #expect(name == "caption_text_measure")
      let request = try JSONDecoder().decode(
        CaptionTextMeasurementRequest.self, from: Files.read(URL(fileURLWithPath: path)))
      #expect(request.texts == ["字幕"])
      return .object(["motionRenderVerified": .bool(false)])
    }
    let result = await NativeService(interfaces: interfaces).call(
      .init(
        name: "caption_text_measure",
        arguments: [
          "fontName": .string("ArialMT"), "assFontSize": .int(66),
          "border": .int(3), "rim": .int(1), "texts": .array([.string("字幕")]),
        ]))
    #expect(result.isError == false)
    #expect(
      result.structuredContent?.objectValue?["measurement"]?.objectValue?["motionRenderVerified"]
        == .bool(false))
  }

  @Test func localMeasurementPreservesItsRequestAndReportsNativeBounds() async throws {
    let request = try Files.reserveOutput(
      directory: Files.temporaryDirectory().path, name: "request.json", kind: .edit)
    defer {
      do { try FileManager.default.removeItem(at: request.deletingLastPathComponent()) } catch {
        Issue.record(error)
      }
    }
    let data = try JSONEncoder().encode(
      CaptionTextMeasurementRequest(
        fontName: "ArialMT", assFontSize: 66, border: 3, rim: 1, texts: ["test"]))
    _ = try Files.writeNew(data, to: request.path, extensions: ["json"])
    let output = try await NativeService().measureLocalFile(
      name: "caption_text_measure", path: request.path)
    let value = try JSONDecoder().decode(Value.self, from: Data(output.utf8))
    #expect(value.objectValue?["motionRenderVerified"] == .bool(false))
    #expect(value.objectValue?["bounds"]?.arrayValue?.count == 1)
    #expect(value.objectValue?["font"]?.objectValue?["postScriptName"] == .string("ArialMT"))
    #expect(try Files.read(request) == data)
  }
}
