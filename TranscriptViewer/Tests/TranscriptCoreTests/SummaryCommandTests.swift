import Foundation
import Testing

@testable import TranscriptCore
@testable import TranscriptSummaries

private actor DiagnosticReplyGate {
    private let arrival = AsyncStream<Void>.makeStream()
    private let releaseSignal = AsyncStream<Void>.makeStream()

    func reply() async -> String {
        arrival.continuation.finish()
        for await _ in releaseSignal.stream {}
        return "説明文を生成している。"
    }

    func wait() async { for await _ in arrival.stream {} }
    func release() { releaseSignal.continuation.finish() }
}

struct SummaryCommandTests {
    @Test(arguments: [
        [], ["unknown"], ["generate"], ["generate", "--library"],
        ["generate", "--library", "x", "--limit", "0"],
        ["generate", "--library", "x", "--limit", "text"],
        ["generate", "--library", "x", "--include-text"],
        ["generate", "--library", "x", "--force", "--force"],
        ["generate", "--library", "x", "--library", "y"],
        ["generate", "--library", "x", "--unknown"],
        ["inspect", "--library", "x", "--force"],
    ])
    func rejectsInvalidArguments(_ arguments: [String]) {
        #expect(throws: SummaryCommandError.self) { try SummaryCommand(arguments: arguments) }
    }

    @Test func generatesAndInspectsTheSamePersistedFormatWithoutAnotherInference() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let generate = try SummaryCommand(arguments: [
            "generate", "--library", fixture.root.path,
            "--transcript", fixture.video.transcriptURL.path, "--limit", "1", "--force",
        ])
        #expect(generate.force)
        #expect(generate.limit == 1)
        let status = try await generate.run(respond: { _ in "旅行の準備について説明している。" }, output: { _ in })
        #expect(status == 0)
        let inspect = try SummaryCommand(arguments: ["inspect", "--library", fixture.root.path])
        #expect(
            try await inspect.run(
                respond: { _ in
                    Issue.record("Inspection must not generate")
                    throw SummaryGenerationError.emptyResponse
                }, output: { _ in }) == 0)
        #expect(try await SummaryStore().load(video: fixture.video)?.overview == "旅行の準備について説明している。")
    }

    @Test func missingSelectionAndMissingOutputAreDistinct() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let inspect = try SummaryCommand(arguments: ["inspect", "--library", fixture.root.path])
        #expect(try await inspect.run() == 1)
        let selected = try SummaryCommand(arguments: [
            "inspect", "--library", fixture.root.path,
            "--transcript", fixture.root.appendingPathComponent("missing.srt").path,
        ])
        await #expect(throws: SummaryCommandError.self) { try await selected.run(output: { _ in }) }
        let missing = try SummaryCommand(arguments: [
            "inspect", "--library", fixture.root.appendingPathComponent("missing").path,
        ])
        await #expect(throws: TranscriptError.self) { try await missing.run(output: { _ in }) }
    }

    @Test(arguments: [false, true])
    func diagnosticsArePrivateAndRawTextIsExplicitlyOptIn(_ includeText: Bool) async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let root = fixture.root.appendingPathComponent("diagnostics")
        var arguments = ["generate", "--library", fixture.root.path, "--diagnostics", root.path]
        if includeText { arguments.append("--include-text") }
        let command = try SummaryCommand(arguments: arguments)
        #expect(try await command.run(respond: { _ in "秘密のテスト用説明文です。" }, output: { _ in }) == 0)
        let runs = try FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil)
        let run = try #require(runs.first)
        let response = run.appendingPathComponent("000001-response.json")
        let event = try JSONDecoder().decode(
            SummaryDiagnostics.Event.self, from: Data(contentsOf: response))
        #expect(event.video == 1)
        #expect(event.kind == .section)
        #expect(event.sectionStartSeconds == 0)
        #expect(event.maximumResponseTokens == 464)
        #expect(event.responseBytes == 39)
        #expect((event.response != nil) == includeText)
        #expect((event.prompt != nil) == includeText)
        #expect((event.instructions != nil) == includeText)
        let attrs = try FileManager.default.attributesOfItem(atPath: response.path)
        #expect((attrs[.posixPermissions] as? NSNumber)?.intValue == 0o600)
        #expect(
            FileManager.default.fileExists(
                atPath: run.appendingPathComponent("video-1-outcome.json").path))
    }

    @Test func failureIsRecordedWithoutLeakingProviderContextAndWithoutPublishingOutput()
        async throws
    {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let root = fixture.root.appendingPathComponent("diagnostics")
        let command = try SummaryCommand(arguments: [
            "generate", "--library", fixture.root.path,
            "--diagnostics", root.path,
        ])
        #expect(
            try await command.run(
                respond: { _ in
                    throw SummaryGenerationError.providerFailure(
                        code: "model_refusal", message: "private-debug-context")
                }, output: { _ in }) == 1)
        let run = try #require(
            FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil).first
        )
        let data = try Data(contentsOf: run.appendingPathComponent("000001-error.json"))
        let event = try JSONDecoder().decode(SummaryDiagnostics.Event.self, from: data)
        #expect(event.errorCode == "model_refusal")
        #expect(!String(decoding: data, as: UTF8.self).contains("private-debug-context"))
        #expect(!FileManager.default.fileExists(atPath: fixture.current.path))
    }

    @Test func cancellationPropagatesAndDoesNotPublish() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let command = try SummaryCommand(arguments: [
            "generate", "--library", fixture.root.path,
            "--diagnostics", fixture.root.appendingPathComponent("diagnostics").path,
        ])
        await #expect(throws: CancellationError.self) {
            try await command.run(respond: { _ in throw CancellationError() }, output: { _ in })
        }
        #expect(!FileManager.default.fileExists(atPath: fixture.current.path))
    }

    @Test(arguments: [
        (SummaryGenerationError.emptyTranscript, "empty_transcript"),
        (.emptyResponse, "empty_response"), (.nonNarrativeResponse, "non_narrative_response"),
        (.refusedResponse, "refused_response"), (.oversizedCharacter, "oversized_character"),
        (.insufficientCompression, "insufficient_compression"),
        (.unavailable("offline"), "unavailable"),
        (.providerFailure(code: "model_refusal", message: "unavailable"), "model_refusal"),
    ])
    func generationFailuresHaveStableCodes(_ pair: (SummaryGenerationError, String)) {
        #expect(SummaryDiagnostics.errorCode(pair.0) == pair.1)
        #expect(pair.0.errorDescription != nil)
    }

    @Test func setupFailuresAndStorageErrorsHaveSafeDescriptions() {
        #expect(SummaryCommand.usage.contains("--include-text"))
        #expect(SummaryCommandError.arguments.errorDescription?.contains("--help") == true)
        #expect(
            SummaryCommandError.emptySelection.errorDescription?.contains("No matching") == true)
        #expect(SummaryDiagnostics.errorCode(SummaryStoreError.unsafePath) == "invalid_storage")
        #expect(
            SummaryDiagnostics.errorCode(CocoaError(.fileReadNoSuchFile))
                == "io_or_unexpected_error")
        #expect(
            SummaryDiagnostics.errorCode(
                DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "private")))
                == "invalid_json")
        #expect(SummaryStoreError.invalidRecord.errorDescription != nil)
        #expect(SummaryStoreError.unsafePath.errorDescription != nil)
        #expect(SummaryStoreError.oversizedRecord.errorDescription != nil)
    }

    @Test(.timeLimit(.minutes(1))) func diagnosticsKeepTheOriginalVideoAcrossSuspension()
        async throws
    {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let gate = DiagnosticReplyGate()
        let trace = try await SummaryDiagnostics.create(
            in: fixture.root.appendingPathComponent("diagnostics")
        ) { _ in
            await gate.reply()
        }
        try await trace.beginVideo(fixture.video, index: 1)
        async let response = trace.respond(SummaryRequest(kind: .section, source: "資料。"))
        await gate.wait()
        let next: Result<Void, Error>
        do {
            try await trace.beginVideo(fixture.video, index: 2)
            next = .success(())
        } catch { next = .failure(error) }
        await gate.release()
        _ = try await response
        try next.get()
        let event = try JSONDecoder().decode(
            SummaryDiagnostics.Event.self,
            from: Data(contentsOf: trace.directory.appendingPathComponent("000001-response.json")))
        #expect(event.video == 1)
    }

    @Test func oneFailureDoesNotPreventTheNextVideoFromBeingSaved() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let transcript = fixture.root.appendingPathComponent("second.srt")
        try "1\n00:00:00,000 --> 00:00:01,000\n別の話題。\n".write(
            to: transcript, atomically: false, encoding: .utf8)
        try Data().write(to: fixture.root.appendingPathComponent("second.mp4"))
        let command = try SummaryCommand(arguments: ["generate", "--library", fixture.root.path])
        #expect(
            try await command.run(
                respond: { request in
                    if request.source.contains("旅行") { throw SummaryGenerationError.emptyResponse }
                    return "別の話題を説明している。"
                }, output: { _ in }) == 1)
        #expect(
            FileManager.default.fileExists(
                atPath: transcript.appendingPathExtension("summaries").appendingPathComponent(
                    "current.json"
                ).path))
        #expect(!FileManager.default.fileExists(atPath: fixture.current.path))
    }
}
