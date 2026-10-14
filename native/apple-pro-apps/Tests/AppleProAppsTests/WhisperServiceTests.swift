import Foundation
import MCP
import Testing

@testable import AppleProApps

struct WhisperServiceTests {
  @Test func boundedLocalWhisperDispatchUsesTheReadOnlyMeasurementBoundary() async throws {
    let spec = try #require(ToolSpec.all.first { $0.name == "audio_transcribe_whisper" })
    #expect(spec.readOnly)
    #expect(NativeInterfaces.measurementTimeout("audio_transcribe_whisper") == .seconds(180))
    #expect(NativeInterfaces.measurementTimeout("audio_transcribe") == .seconds(60))
    var interfaces = NativeInterfaces()
    interfaces.measureMedia = { name, requestPath in
      #expect(name == "audio_transcribe_whisper")
      let request = try JSONDecoder().decode(
        Value.self, from: Data(contentsOf: URL(fileURLWithPath: requestPath)))
      #expect(request.objectValue?["path"] == .string("/tmp/synthetic.wav"))
      return .object(["onDevice": .bool(true), "segments": .array([])])
    }
    let response = await NativeService(interfaces: interfaces).call(
      .init(
        name: "audio_transcribe_whisper",
        arguments: [
          "path": .string("/tmp/synthetic.wav"), "modelDirectory": .string("/tmp/model"),
          "tokenizerDirectory": .string("/tmp/tokenizer"),
        ]))
    #expect(response.isError == false)
    #expect(response.structuredContent?.objectValue?["sourceModified"] == .bool(false))
    #expect(throws: (any Error).self) {
      try validate(
        .object(["path": .string("/tmp/synthetic.wav"), "download": .bool(true)]),
        schema: spec.tool.inputSchema)
    }
  }
}
