import Foundation

public struct MIDINote: Codable, Sendable {
  public let note: Int
  public let velocity: Int
  public let startTick: Int
  public let durationTick: Int
  public let channel: Int

  public init(note: Int, velocity: Int, startTick: Int, durationTick: Int, channel: Int = 1) {
    self.note = note
    self.velocity = velocity
    self.startTick = startTick
    self.durationTick = durationTick
    self.channel = channel
  }
}

public enum MIDIFile {
  public static func variableLength(_ value: Int) throws -> [UInt8] {
    guard (0...0x0fff_ffff).contains(value) else {
      throw ProAppsError.invalid("MIDI delta exceeds four bytes")
    }
    var remaining = value
    var bytes = [UInt8(remaining & 0x7f)]
    remaining >>= 7
    while remaining > 0 {
      bytes.insert(UInt8(remaining & 0x7f) | 0x80, at: 0)
      remaining >>= 7
    }
    return bytes
  }

  public static func create(notes: [MIDINote], bpm: Double, ticksPerQuarter: Int = 480) throws
    -> Data
  {
    guard bpm.isFinite, (20...400).contains(bpm), (1...32767).contains(ticksPerQuarter),
      !notes.isEmpty, notes.count <= 4096
    else { throw ProAppsError.invalid("Invalid MIDI tempo, division or note count") }
    struct Event {
      let tick: Int
      let order: Int
      let bytes: [UInt8]
    }
    var events: [Event] = []
    for (index, note) in notes.enumerated() {
      guard (0...127).contains(note.note), (1...127).contains(note.velocity),
        (1...16).contains(note.channel),
        (0...0x07ff_ffff).contains(note.startTick), (1...0x07ff_ffff).contains(note.durationTick)
      else { throw ProAppsError.invalid("Invalid MIDI note") }
      events.append(
        Event(
          tick: note.startTick, order: 4096 + index,
          bytes: [0x90 | UInt8(note.channel - 1), UInt8(note.note), UInt8(note.velocity)]))
      events.append(
        Event(
          tick: note.startTick + note.durationTick, order: index,
          bytes: [0x80 | UInt8(note.channel - 1), UInt8(note.note), 0]))
    }
    events.sort { $0.tick == $1.tick ? $0.order < $1.order : $0.tick < $1.tick }
    let tempo = UInt32((60_000_000 / bpm).rounded())
    var track = Data([
      0, 0xff, 0x51, 3, UInt8((tempo >> 16) & 0xff), UInt8((tempo >> 8) & 0xff),
      UInt8(tempo & 0xff),
    ])
    var lastTick = 0
    for event in events {
      track.append(contentsOf: try variableLength(event.tick - lastTick))
      track.append(contentsOf: event.bytes)
      lastTick = event.tick
    }
    track.append(contentsOf: [0, 0xff, 0x2f, 0])
    var result = Data("MThd".utf8)
    result.append(contentsOf: [
      0, 0, 0, 6, 0, 0, 0, 1, UInt8(ticksPerQuarter >> 8), UInt8(ticksPerQuarter & 0xff),
    ])
    result.append(Data("MTrk".utf8))
    let size = UInt32(track.count)
    result.append(contentsOf: [24, 16, 8, 0].map { UInt8(truncatingIfNeeded: size >> $0) })
    result.append(track)
    return result
  }
}
