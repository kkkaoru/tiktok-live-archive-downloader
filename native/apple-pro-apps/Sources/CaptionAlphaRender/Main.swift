import Foundation
import ProAppsCore

/// Draws one caption part natively, from a finished Motion caption part (`--moti`) or a request
/// file, as a ProRes 4444 alpha movie and prints a JSON timing report; `--inspect` prints what
/// the part's Motion document holds and `--skeleton` the skeleton that pins its template.
@main
struct CaptionAlphaRenderCommand {
  static func main() async throws {
    let command: MotionCaptionCommand
    do {
      command = try MotionCaptionMovie.command(arguments: CommandLine.arguments)
    } catch {
      FileHandle.standardError.write(Data("\(error)\n".utf8))
      exit(2)
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    switch command {
    case .render(let request):
      let report = try await MotionCaptionMovie.render(request)
      try FileHandle.standardOutput.write(contentsOf: encoder.encode(report) + Data("\n".utf8))
    case .inspect(let part):
      try FileHandle.standardOutput.write(contentsOf: encoder.encode(part) + Data("\n".utf8))
    case .skeleton(let text):
      try FileHandle.standardOutput.write(contentsOf: Data(text.utf8))
    }
  }
}
