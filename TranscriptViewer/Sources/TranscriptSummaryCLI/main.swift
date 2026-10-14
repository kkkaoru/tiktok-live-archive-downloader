import Foundation
import TranscriptSummaries

@main
struct TranscriptSummaryCLI {
    static func main() async {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments == ["--help"] || arguments == ["-h"] {
            print(SummaryCommand.usage)
            return
        }
        do {
            let command = try SummaryCommand(arguments: arguments)
            exit(try await command.run())
        } catch {
            print("ERROR \(SummaryDiagnostics.errorCode(error))")
            if let error = error as? SummaryCommandError { print(error.localizedDescription) }
            exit(2)
        }
    }
}
