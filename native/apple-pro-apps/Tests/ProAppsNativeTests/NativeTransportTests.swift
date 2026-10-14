import CoreMIDI
import Darwin
import Foundation
import Testing

@testable import ProAppsCore

struct NativeTransportTests {
  @Test(.timeLimit(.minutes(1)))
  func sendsMIDIOnlyToItsOwnTemporaryDestination() async throws {
    let name = "ProAppsTest-\(UUID().uuidString)"
    var client: MIDIClientRef = 0
    #expect(MIDIClientCreateWithBlock(name as CFString, &client, nil) == noErr)
    defer { MIDIClientDispose(client) }
    let stream = AsyncStream<UInt32>.makeStream(bufferingPolicy: .bufferingNewest(1))
    defer { stream.continuation.finish() }
    var destination: MIDIEndpointRef = 0
    let status = MIDIDestinationCreateWithProtocol(client, name as CFString, ._1_0, &destination) {
      list, _ in
      guard list.pointee.numPackets == 1, list.pointee.packet.wordCount == 1 else { return }
      stream.continuation.yield(list.pointee.packet.words.0)
    }
    try #require(status == noErr)
    defer { MIDIEndpointDispose(destination) }
    var id: MIDIUniqueID = 0
    try #require(MIDIObjectGetIntegerProperty(destination, kMIDIPropertyUniqueID, &id) == noErr)
    let identity = try #require(MIDIControl.destinations().first { $0.uniqueID == id })
    #expect(throws: (any Error).self) {
      try MIDIControl.send(id: id, name: "wrong-name", word: 0x20b0_0764)
    }
    try MIDIControl.send(id: id, name: identity.name, word: 0x20b0_0764)
    let received = try await withThrowingTaskGroup(of: UInt32.self) { group in
      group.addTask {
        for await value in stream.stream { return value }
        throw ProAppsError.unavailable("MIDI test stream ended")
      }
      group.addTask {
        try await Task.sleep(for: .seconds(2))
        throw ProAppsError.timedOut
      }
      defer { group.cancelAll() }
      let first = try await group.next()
      return try #require(first)
    }
    #expect(received == 0x20b0_0764)
  }

  @Test func sendsOSCOnlyToItsOwnLoopbackReceiver() throws {
    let descriptor = socket(AF_INET, SOCK_DGRAM, 0)
    try #require(descriptor >= 0)
    defer { Darwin.close(descriptor) }
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
    let bound = withUnsafePointer(to: &address) { pointer in
      pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
        bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
      }
    }
    try #require(bound == 0)
    var length = socklen_t(MemoryLayout<sockaddr_in>.size)
    let located = withUnsafeMutablePointer(to: &address) { pointer in
      pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
        getsockname(descriptor, $0, &length)
      }
    }
    try #require(located == 0)
    let port = Int(UInt16(bigEndian: address.sin_port))
    var timeout = timeval(tv_sec: 1, tv_usec: 0)
    try #require(
      setsockopt(
        descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size)) == 0)
    try OSC.send(port: port, path: "/gain", value: 1)
    var bytes = [UInt8](repeating: 0, count: 64)
    let count = bytes.withUnsafeMutableBytes { recv(descriptor, $0.baseAddress, $0.count, 0) }
    #expect(count == 16)
    #expect(
      Array(bytes.prefix(16)) == [47, 103, 97, 105, 110, 0, 0, 0, 44, 102, 0, 0, 63, 128, 0, 0])
  }
}
