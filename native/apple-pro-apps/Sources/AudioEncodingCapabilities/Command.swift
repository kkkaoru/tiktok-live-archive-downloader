import Foundation

@main
struct AudioEncodingCapabilitiesCommand {
  static func main() {
    exit(
      run(
        arguments: CommandLine.arguments, output: FileHandle.standardOutput.write,
        error: FileHandle.standardError.write))
  }

  static func run(
    arguments: [String],
    execute: ([String]) throws -> Data = { try AudioEncodingCapabilities.execute(arguments: $0) },
    output: (Data) -> Void, error: (Data) -> Void
  ) -> Int32 {
    do {
      output(try execute(arguments) + Data("\n".utf8))
      return 0
    } catch let failure {
      error(Data("Audio capability observation failed: \(failure)\n".utf8))
      return 1
    }
  }
}
