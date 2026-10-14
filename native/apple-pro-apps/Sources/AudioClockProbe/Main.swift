import Foundation

@main
struct AudioClockProbeCommand {
  static func main() async {
    do {
      try await execute(
        arguments: CommandLine.arguments,
        root: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
    } catch {
      FileHandle.standardError.write(Data("Audio clock observation failed: \(error)\n".utf8))
      exit(1)
    }
  }

  static func execute(arguments: [String], root: URL) async throws {
    guard arguments.count == 2 else { throw ClockProbeError.invalidRequest }
    let requestURL = URL(fileURLWithPath: arguments[1])
    let handle = try FileHandle(forReadingFrom: requestURL)
    defer { handle.closeFile() }
    guard let data = try handle.read(upToCount: 65537), data.count <= 65536 else {
      throw ClockProbeError.invalidRequest
    }
    let request = try JSONDecoder().decode(ClockProbeRequest.self, from: data)
    try await AudioClockProbe.run(request, root: root)
  }
}
