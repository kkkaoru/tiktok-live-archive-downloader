import Darwin
import Foundation

public enum ProAppsError: Error, Equatable, Sendable, CustomStringConvertible {
  case invalid(String)
  case unavailable(String)
  case commandFailed(Int32)
  case timedOut
  case outputLimit

  public var description: String {
    switch self {
    case .invalid(let reason): return "Invalid request: \(reason)"
    case .unavailable(let reason): return "Unavailable: \(reason)"
    case .commandFailed(let code):
      return
        "Native command exited \(code). Its effect may be partial; inspect state before retrying."
    case .timedOut:
      return "Native command timed out. Its effect may be partial; do not automatically retry."
    case .outputLimit:
      return "Native output exceeded its bounded capture limit. Do not automatically retry."
    }
  }
}

public enum Files {
  public static let maximumBytes = 8 * 1024 * 1024

  public enum OutputKind: String, Sendable { case compressor, edit }

  /// Honor an explicit process scratch directory without silently falling back
  /// when it is invalid. Foundation on macOS may ignore TMPDIR. Callers own the
  /// configured root; each operation must still reserve its own private child.
  public static func temporaryDirectory(
    path: String? = ProcessInfo.processInfo.environment["TMPDIR"]
  ) throws -> URL {
    let root = try absolute(path ?? FileManager.default.temporaryDirectory.path)
    let values = try root.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
    guard values.isDirectory == true, values.isSymbolicLink != true else {
      throw ProAppsError.invalid("Temporary directory must be an existing non-symlink directory")
    }
    return root.resolvingSymlinksInPath()
  }

  /// Each encoder/render owns a private directory. Failed dispatch may retain it
  /// for audit; this never selects or replaces an existing media destination.
  public static func reserveOutput(directory: String, name: String, kind: OutputKind) throws -> URL
  {
    guard !name.isEmpty, name != ".", name != "..", !name.contains("/"), !name.contains("\0"),
      name.utf8.count <= 180, !URL(fileURLWithPath: name).pathExtension.isEmpty
    else { throw ProAppsError.invalid("Expected an output basename with extension") }
    let parent = try absolute(directory).resolvingSymlinksInPath()
    guard try parent.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true else {
      throw ProAppsError.invalid("Output directory must exist")
    }
    let reserved = parent.appendingPathComponent("\(kind.rawValue)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: reserved, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    return reserved.appendingPathComponent(name)
  }

  /// Unknown future errno values map to a defined I/O failure, never success.
  static func systemError(_ code: Int32) -> POSIXError {
    POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
  }

  public static func absolute(_ path: String) throws -> URL {
    guard path.hasPrefix("/"), path.utf8.count <= 4096, !path.contains("\0") else {
      throw ProAppsError.invalid("An absolute local path without NUL is required")
    }
    return URL(fileURLWithPath: path).standardizedFileURL
  }

  public static func existing(_ path: String, extensions: Set<String>? = nil) throws -> URL {
    let url = try absolute(path).resolvingSymlinksInPath()
    let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    guard values.isRegularFile == true else {
      throw ProAppsError.invalid("Expected a regular file")
    }
    if let extensions, !extensions.contains(url.pathExtension.lowercased()) {
      throw ProAppsError.invalid("Unsupported file extension")
    }
    return url
  }

  public static func read(_ url: URL) throws -> Data {
    // Validate the opened descriptor, not just pre-open path metadata. NONBLOCK
    // prevents a raced FIFO from hanging; NOFOLLOW refuses final-path symlinks.
    let descriptor = Darwin.open(url.path, O_RDONLY | O_NONBLOCK | O_NOFOLLOW)
    guard descriptor >= 0 else { throw systemError(errno) }
    defer { Darwin.close(descriptor) }
    var metadata = stat()
    guard Darwin.fstat(descriptor, &metadata) == 0,
      metadata.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG)
    else { throw ProAppsError.invalid("Expected an opened regular file") }
    guard metadata.st_size <= maximumBytes else { throw ProAppsError.invalid("File exceeds 8 MiB") }
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
    var data = Data()
    while true {
      let chunk = try handle.read(upToCount: min(65536, maximumBytes + 1 - data.count)) ?? Data()
      if chunk.isEmpty { break }
      data.append(chunk)
      guard data.count <= maximumBytes else { throw ProAppsError.invalid("File exceeds 8 MiB") }
    }
    return data
  }

  /// Publish a private regular file with an atomic, non-overwriting hard link.
  /// No existing destination (including a dangling symlink) is ever removed.
  public static func writeNew(_ data: Data, to path: String, extensions: Set<String>) throws -> URL
  {
    guard data.count <= maximumBytes else { throw ProAppsError.invalid("Content exceeds 8 MiB") }
    let url = try absolute(path)
    guard extensions.contains(url.pathExtension.lowercased()) else {
      throw ProAppsError.invalid("Unsupported output extension")
    }
    let parent = url.deletingLastPathComponent().resolvingSymlinksInPath()
    guard try parent.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true else {
      throw ProAppsError.invalid("Output parent must already exist")
    }
    let destination = parent.appendingPathComponent(url.lastPathComponent)
    let temporary = parent.appendingPathComponent(".apple-pro-apps-\(UUID().uuidString)")
    let descriptor = Darwin.open(temporary.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o600)
    guard descriptor >= 0 else { throw systemError(errno) }
    // Cleanup errors must not replace the primary publication error. The only
    // removed path is our private staging file, never the destination.
    defer {
      Darwin.close(descriptor)
      Cleanup.perform { try FileManager.default.removeItem(at: temporary) }
    }
    // Foundation owns the byte-buffer interoperability; this scope owns the
    // descriptor. No handwritten raw buffer manipulation is needed for writes.
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
    try handle.write(contentsOf: data)
    guard Darwin.fsync(descriptor) == 0 else { throw POSIXError(.EIO) }
    try FileManager.default.linkItem(at: temporary, to: destination)
    return destination
  }
}
