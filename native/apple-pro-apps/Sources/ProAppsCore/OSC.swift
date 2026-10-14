import Darwin
import Foundation

public enum OSC {
  public static func packet(path: String, value: Double) throws -> Data {
    guard path.hasPrefix("/"), path.utf8.count <= 512,
      path.unicodeScalars.allSatisfy({
        CharacterSet(
          charactersIn: "/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
        ).contains($0)
      }),
      value.isFinite, Float(value).isFinite
    else {
      throw ProAppsError.invalid(
        "Expected an exact OSC path (no wildcards) and a finite Float32 value")
    }
    func padded(_ string: String) -> Data {
      var bytes = Data(string.utf8)
      bytes.append(0)
      while bytes.count % 4 != 0 { bytes.append(0) }
      return bytes
    }
    var bytes = padded(path)
    bytes.append(padded(",f"))
    let word = Float(value).bitPattern
    bytes.append(contentsOf: [24, 16, 8, 0].map { UInt8(truncatingIfNeeded: word >> $0) })
    return bytes
  }

  /// Only an explicitly supplied loopback UDP port. No discovery, external host,
  /// listener, broadcast, retries or claim of application-level acknowledgment.
  public static func send(port: Int, path: String, value: Double) throws {
    guard (1024...65535).contains(port) else {
      throw ProAppsError.invalid("OSC port must be 1024–65535")
    }
    let bytes = try packet(path: path, value: value)
    let descriptor = socket(AF_INET, SOCK_DGRAM, 0)
    guard descriptor >= 0 else { throw POSIXError(.EIO) }
    defer { Darwin.close(descriptor) }
    var deadline = timeval(tv_sec: 1, tv_usec: 0)
    guard
      setsockopt(
        descriptor, SOL_SOCKET, SO_SNDTIMEO, &deadline, socklen_t(MemoryLayout<timeval>.size)) == 0
    else {
      throw POSIXError(.EIO)
    }
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = UInt16(port).bigEndian
    address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
    let sent = bytes.withUnsafeBytes { data in
      withUnsafePointer(to: &address) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
          sendto(
            descriptor, data.baseAddress, data.count, 0, $0,
            socklen_t(MemoryLayout<sockaddr_in>.size))
        }
      }
    }
    guard sent == bytes.count else { throw ProAppsError.unavailable("OSC datagram was not sent") }
  }
}
