import AVFoundation
import CryptoKit
import Foundation
import Testing

@testable import audio_clock_probe

private struct PacketMutation: Codable {
  let name: String
  let sourceSHA256: String
  let originalBytes: Int
  let offset: Int
  let removedBytes: Int
  let replacedBytes: Int
}

/// Clone only a caller-supplied tiny synthetic fixture. Never alter the original.
@Test(.timeLimit(.minutes(1))) func compressedPacketMutationFixture() async throws {
  let root = try testRoot()
  let environment = ProcessInfo.processInfo.environment
  let source = URL(fileURLWithPath: try #require(environment["AUDIO_NEGATIVE_FIXTURE_SOURCE"]))
  let directory = URL(
    fileURLWithPath: try #require(environment["AUDIO_NEGATIVE_FIXTURE_DIRECTORY"]))
  try #require(source.path.hasPrefix(root.path + "/edits/work/"))
  try #require(directory.path.hasPrefix(root.path + "/edits/work/"))
  try #require(source.resolvingSymlinksInPath() == source)
  try #require(directory.resolvingSymlinksInPath() == directory)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
  let size = try #require(source.resourceValues(forKeys: [.fileSizeKey]).fileSize)
  try #require(size < 65536)
  let original = try Data(contentsOf: source)
  try #require(original.count < 65536)
  let hash = SHA256.hash(data: original).map { String(format: "%02x", $0) }.joined()
  let request = ClockProbeRequest(
    source: source.path, outputDirectory: directory.path,
    startValue: 0, durationValue: 1, timescale: 1)
  let compressed = try await CompressedClockProbe.read(request)
  try JSONEncoder().encode(compressed).write(
    to: directory.appendingPathComponent("original-packets.json"), options: .withoutOverwriting)
  try #require(compressed.packets.count >= 3)
  let last = try #require(compressed.packets.last)
  let middle = compressed.packets[compressed.packets.count / 2]
  let cutoff = Int(last.offset + last.bytes / 2)
  try #require(cutoff > 0 && cutoff < original.count)
  let lower = Int(middle.offset)
  let upper = Int(middle.offset + middle.bytes)
  try #require(lower > 0 && upper <= original.count)
  var malformed = original
  malformed.replaceSubrange(lower..<upper, with: Data(repeating: 0, count: upper - lower))
  let mutations: [(String, Data, Int, Int, Int)] = [
    ("truncated-container", original.prefix(16), 16, original.count - 16, 0),
    ("truncated-packet", original.prefix(cutoff), cutoff, original.count - cutoff, 0),
    ("malformed-packet", malformed, lower, 0, upper - lower),
  ]
  for (name, bytes, offset, removed, replaced) in mutations {
    let clone = directory.appendingPathComponent(name + ".mov")
    try bytes.write(to: clone, options: .withoutOverwriting)
    // Corrupt decoding belongs in the existing process watchdog, not this test process.
    let observation = PacketMutation(
      name: name, sourceSHA256: hash, originalBytes: original.count, offset: offset,
      removedBytes: removed, replacedBytes: replaced)
    try JSONEncoder().encode(observation).write(
      to: directory.appendingPathComponent(name + ".json"), options: .withoutOverwriting)
  }
  #expect(try Data(contentsOf: source) == original)
}
