import Foundation

public enum Compressor {
  public enum Control: String, Codable, Sendable { case pause, resume, cancel }

  public struct TimeRange: Codable, Sendable {
    public let startSeconds: Int
    public let durationSeconds: Int

    public init(startSeconds: Int, durationSeconds: Int) {
      self.startSeconds = startSeconds
      self.durationSeconds = durationSeconds
    }

    /// Whole-second source timecodes with a bounded duration; no frame-rate
    /// assumption or source mutation. The encoder determines frame rounding.
    public func arguments() throws -> [String] {
      guard (0...86399).contains(startSeconds), (1...600).contains(durationSeconds) else {
        throw ProAppsError.invalid("Range requires start 0–86399 and duration 1–600 seconds")
      }
      return [
        "-in", Self.timecode(startSeconds), "-out", Self.timecode(startSeconds + durationSeconds),
      ]
    }

    private static func timecode(_ seconds: Int) -> String {
      [seconds / 3600, (seconds / 60) % 60, seconds % 60]
        .map { $0 < 10 ? "0\($0)" : String($0) }.joined(separator: ":") + ";00"
    }
  }

  /// Compressor documents a URL, not a filesystem path, for source analysis.
  public static func inspectionArguments(source: URL) -> [String] {
    ["-checkstream", source.absoluteString]
  }

  public static func monitoringArguments(id: String, job: Bool, control: Control? = nil) throws
    -> [String]
  {
    guard id.range(of: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$", options: .regularExpression) != nil
    else {
      throw ProAppsError.invalid("Invalid Compressor job/batch ID")
    }
    let selector = job ? "-jobid" : "-batchid"
    if let control {
      return [
        control == .cancel ? "-kill" : "-\(control.rawValue)", selector, id, "-outputformat",
        "json",
      ]
    }
    // Creator Studio 5.3 rejects the legacy -format flag even though its help
    // still mentions it. Use the current shared submission/monitoring option.
    return ["-monitor", selector, id, "-once", "-timeout", "10", "-outputformat", "json"]
  }

  public static func submissionArguments(
    source: URL, preset: URL, output: URL, batchName: String, range: TimeRange? = nil
  )
    throws -> [String]
  {
    guard !batchName.isEmpty, batchName.utf8.count <= 200, !batchName.contains("\0") else {
      throw ProAppsError.invalid("Batch name must contain 1–200 bytes without NUL")
    }
    let locations: [URL] = [source, preset, output]
    let allLocal = locations.allSatisfy { location in
      guard location.isFileURL else { return false }
      guard let host = location.host else { return true }
      return host.isEmpty || host == "localhost"
    }
    guard allLocal else {
      throw ProAppsError.invalid("Compressor submission requires local file URLs")
    }
    // Regular-file jobs use filesystem paths. Creator Studio leaves percent
    // escapes encoded when a file URL is supplied here, unlike -checkstream.
    // Arguments are passed directly, never interpreted by a shell.
    var arguments = ["-batchname", batchName, "-jobpath", source.path]
    // Attach the source interval before configuring this job's output target.
    if let range { arguments.append(contentsOf: try range.arguments()) }
    arguments.append(contentsOf: [
      "-settingpath", preset.path, "-locationpath", output.path, "-outputformat", "json",
    ])
    return arguments
  }

  /// Reserve a private, unique output directory so the encoder cannot overwrite
  /// an existing output. A failed/indeterminate submission retains it for audit.
  public static func reserveOutput(directory: String, name: String) throws -> URL {
    try Files.reserveOutput(directory: directory, name: name, kind: .compressor)
  }

  @MainActor public static func executable(bundleID: String?, access: ApplicationAccess? = nil)
    throws -> URL
  {
    let installed = try Applications.resolve(.compressor, bundleID: bundleID, access: access)
    let binary = URL(fileURLWithPath: installed.path).appendingPathComponent(
      "Contents/MacOS/Compressor")
    guard FileManager.default.isExecutableFile(atPath: binary.path) else {
      throw ProAppsError.unavailable("Compressor executable is missing")
    }
    return binary
  }
}
