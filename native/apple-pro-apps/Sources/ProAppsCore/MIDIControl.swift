import CoreMIDI
import Foundation

public struct MIDIDestination: Codable, Sendable {
  public let uniqueID: Int32
  public let name: String
}

public enum MIDIControl {
  public enum Kind: String, Codable, Sendable { case controlChange, programChange, pitchBend }

  public static func message(kind: Kind, channel: Int, number: Int, value: Int) throws -> UInt32 {
    guard (1...16).contains(channel), (0...127).contains(number) else {
      throw ProAppsError.invalid("MIDI channel must be 1–16 and number 0–127")
    }
    switch kind {
    case .controlChange:
      guard (0...127).contains(value) else { throw ProAppsError.invalid("CC value must be 0–127") }
      return MIDI1UPControlChange(0, UInt8(channel - 1), UInt8(number), UInt8(value))
    case .programChange:
      guard value == 0 else {
        throw ProAppsError.invalid("Program change uses number; value must be zero")
      }
      return MIDI1UPProgramChange(0, UInt8(channel - 1), UInt8(number))
    case .pitchBend:
      guard number == 0, (0...16383).contains(value) else {
        throw ProAppsError.invalid("Pitch bend uses value 0–16383 and number zero")
      }
      return MIDI1UPPitchBend(0, UInt8(channel - 1), UInt8(value & 127), UInt8(value >> 7))
    }
  }

  private static func describe(_ endpoint: MIDIEndpointRef) -> MIDIDestination? {
    var id: MIDIUniqueID = 0
    var name: Unmanaged<CFString>?
    guard MIDIObjectGetIntegerProperty(endpoint, kMIDIPropertyUniqueID, &id) == noErr,
      MIDIObjectGetStringProperty(endpoint, kMIDIPropertyDisplayName, &name) == noErr,
      let name
    else { return nil }
    return MIDIDestination(uniqueID: id, name: name.takeRetainedValue() as String)
  }

  public static func destinations() -> [MIDIDestination] {
    (0..<MIDIGetNumberOfDestinations()).compactMap { describe(MIDIGetDestination($0)) }
  }

  public static func send(id: Int32, name: String, word: UInt32) throws {
    let matches = (0..<MIDIGetNumberOfDestinations()).map { MIDIGetDestination($0) }.filter {
      guard let identity = describe($0) else { return false }
      return identity.uniqueID == id && identity.name == name
    }
    guard matches.count == 1, let endpoint = matches.first else {
      throw ProAppsError.invalid("Destination ID/name must match exactly one current endpoint")
    }
    var client: MIDIClientRef = 0
    guard MIDIClientCreateWithBlock("Executor Apple Pro Apps" as CFString, &client, nil) == noErr
    else {
      throw ProAppsError.unavailable("Cannot create CoreMIDI client")
    }
    defer { MIDIClientDispose(client) }
    var port: MIDIPortRef = 0
    guard MIDIOutputPortCreate(client, "Explicit destination only" as CFString, &port) == noErr
    else {
      throw ProAppsError.unavailable("Cannot create MIDI output port")
    }
    defer { MIDIPortDispose(port) }
    var list = MIDIEventList()
    var message = word
    let status = withUnsafeMutablePointer(to: &list) { pointer -> OSStatus in
      let packet = MIDIEventListInit(pointer, ._1_0)
      _ = MIDIEventListAdd(pointer, MemoryLayout<MIDIEventList>.size, packet, 0, 1, &message)
      guard pointer.pointee.numPackets == 1 else { return -1 }
      return MIDISendEventList(port, endpoint, pointer)
    }
    guard status == noErr else { throw ProAppsError.unavailable("CoreMIDI send failed") }
  }
}
