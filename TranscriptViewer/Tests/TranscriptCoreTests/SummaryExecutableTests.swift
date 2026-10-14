import Foundation
import Testing

@testable import TranscriptSummaries

private final class SummaryTestBundle: NSObject {}

struct SummaryExecutableTests {
    @Test(.timeLimit(.minutes(1)), arguments: ["--help", "-h"])
    func helpExitsSuccessfully(_ option: String) async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let help = try await run([option], in: fixture)
        #expect(help.status == 0)
        #expect(help.output.hasPrefix("transcript-summary generate --library DIRECTORY"))
        #expect(help.output.contains("Exit codes: 0 = all selected videos ready"))
    }

    @Test(.timeLimit(.minutes(1)))
    func executableReportsSetupErrorsAndMissingResults() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let invalid = try await run([], in: fixture)
        #expect(invalid.status == 2)
        #expect(invalid.output.contains("Invalid arguments. Use transcript-summary --help."))
        let inaccessible = try await run(
            ["inspect", "--library", fixture.root.appendingPathComponent("absent").path],
            in: fixture)
        #expect(inaccessible.status == 2)
        #expect(inaccessible.output.hasPrefix("ERROR "))
        #expect(!inaccessible.output.contains(fixture.root.path))
        let missing = try await run(["inspect", "--library", fixture.root.path], in: fixture)
        #expect(missing.status == 1)
        #expect(missing.output.contains("MISSING 1"))
        #expect(missing.output.contains("FINISHED total=1 ready=0 failed=1"))
    }

    @Test(.timeLimit(.minutes(1)), arguments: ["inspect", "generate"])
    func cachedSummaryIsReusedWithoutChangingItsPointer(_ action: String) async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        _ = try await fixture.generate("旅行の準備についての説明。")
        let pointer = try Data(contentsOf: fixture.current)
        let ready = try await run([action, "--library", fixture.root.path], in: fixture)
        #expect(ready.status == 0)
        #expect(ready.output.contains("READY 1"))
        #expect(ready.output.contains("FINISHED total=1 ready=1 failed=0"))
        #expect(try Data(contentsOf: fixture.current) == pointer)
    }

    @Test(.timeLimit(.minutes(1)))
    func catalogExecutableDistinguishesLoadedInvalidAndInaccessibleLibraries() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let loaded = try await run([fixture.root.path], in: fixture, executable: "transcript-check")
        #expect(loaded.status == 0)
        #expect(loaded.output.contains("LOADED 1 ISSUES 0"))
        #expect(loaded.output.contains("cues=1"))
        try Data("not a subtitle".utf8).write(to: fixture.video.transcriptURL)
        let invalid = try await run(
            [fixture.root.path], in: fixture, executable: "transcript-check")
        #expect(invalid.status == 1)
        #expect(invalid.output.contains("NG "))
        #expect(invalid.output.contains("LOADED 0 ISSUES 1"))
        let absent = try await run(
            [fixture.root.appendingPathComponent("absent").path], in: fixture,
            executable: "transcript-check")
        #expect(absent.status == 2)
        #expect(absent.output.hasPrefix("ERROR "))
    }

    @concurrent
    func run(
        _ arguments: [String], in fixture: SummaryFixture, executable: String = "transcript-summary"
    ) async throws -> (
        status: Int32, output: String
    ) {
        let directory = Bundle(for: SummaryTestBundle.self).bundleURL.deletingLastPathComponent()
        let log = fixture.root.appendingPathComponent("cli-\(UUID().uuidString).log")
        try Data().write(to: log)
        let handle = try FileHandle(forWritingTo: log)
        defer { #expect(throws: Never.self) { try handle.close() } }
        let process = Process()
        process.executableURL = directory.appendingPathComponent(executable)
        process.arguments = arguments
        process.standardOutput = handle
        process.standardError = handle
        let status = try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<Int32, any Error>) in
            process.terminationHandler = { continuation.resume(returning: $0.terminationStatus) }
            do { try process.run() } catch {
                process.terminationHandler = nil
                continuation.resume(throwing: error)
            }
        }
        return (status, try String(contentsOf: log, encoding: .utf8))
    }
}
