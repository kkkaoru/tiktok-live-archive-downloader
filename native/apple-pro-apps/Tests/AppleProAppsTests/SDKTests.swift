import Foundation
import Logging
import MCP
import Testing

@testable import AppleProApps

private actor MemoryTransport: Transport {
  nonisolated let logger = Logger(label: "apple-pro-apps.synthetic-transport")
  let incoming: AsyncThrowingStream<Data, any Error>
  let outgoing: AsyncThrowingStream<Data, any Error>.Continuation

  init(
    incoming: AsyncThrowingStream<Data, any Error>,
    outgoing: AsyncThrowingStream<Data, any Error>.Continuation
  ) {
    self.incoming = incoming
    self.outgoing = outgoing
  }
  func connect() async throws {}
  func disconnect() async { outgoing.finish() }
  func receive() -> AsyncThrowingStream<Data, any Error> { incoming }
  func send(_ data: Data) async throws {
    guard case .enqueued = outgoing.yield(data) else { throw CocoaError(.fileWriteUnknown) }
  }
}

struct SDKTests {
  @Test(.timeLimit(.minutes(1)))
  func nativeSDKListsAndCallsThroughTheActualHandlers() async throws {
    let toServer = AsyncThrowingStream<Data, any Error>.makeStream(
      bufferingPolicy: .bufferingOldest(8))
    let toClient = AsyncThrowingStream<Data, any Error>.makeStream(
      bufferingPolicy: .bufferingOldest(8))
    defer {
      toServer.continuation.finish()
      toClient.continuation.finish()
    }
    let serverTransport = MemoryTransport(
      incoming: toServer.stream, outgoing: toClient.continuation)
    let clientTransport = MemoryTransport(
      incoming: toClient.stream, outgoing: toServer.continuation)
    var interfaces = NativeInterfaces()
    interfaces.inventory = { [] }
    let server = await AppleProApps.makeServer(service: NativeService(interfaces: interfaces))
    let client = Client(name: "synthetic-client", version: "1")
    do {
      try await server.start(transport: serverTransport)
      _ = try await client.connect(transport: clientTransport)
      let catalog = try await client.listTools()
      // The registered specification groups currently declare 58 tools;
      // local-only reference-media/mux-pcm commands do not enter this catalog.
      let expectedToolCount = 58
      let names = Set(catalog.tools.map(\.name))
      #expect(catalog.tools.count == expectedToolCount)
      #expect(names.count == expectedToolCount)
      #expect(names.contains("video_foreground_measure"))
      #expect(names.contains("caption_text_measure"))
      #expect(names.contains("app_activate"))
      #expect(names.contains("fcp_save_navigate"))
      #expect(catalog.nextCursor == nil)
      let result = try await client.callTool(name: "app_capabilities", arguments: [:])
      #expect(result.isError == false)
      #expect(result.content.count == 1)
      let refused = try await client.callTool(name: "not_a_tool", arguments: [:])
      #expect(refused.isError == true)
      await client.disconnect()
      await server.stop()
      await server.waitUntilCompleted()
    } catch {
      await client.disconnect()
      await server.stop()
      await server.waitUntilCompleted()
      throw error
    }
  }
}
