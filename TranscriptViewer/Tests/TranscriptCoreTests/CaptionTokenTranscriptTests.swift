import CryptoKit
import Foundation
import Testing

@testable import TranscriptCore
@testable import TranscriptSummaries

private struct CaptionTokenFixture {
    let fixture: SummaryFixture
    let manifest: URL
    let tokens: URL

    init(
        tokenJSON: String =
            #"{"version":1,"clock":"output","tokens":[{"text":"やばい","start":0.2,"end":0.4,"utterance":"first"},{"text":"やばい","start":0.4,"end":0.6,"utterance":"second"}]}"#,
        indexJSON: String =
            #"{"version":1,"rate":30,"segments":[{"segment":0,"offsetFrame":18000,"frames":300,"jobs":[0]}]}"#,
        domain: String = "edited-output", badPin: Bool = false, missingBinding: Bool = false,
        extraJob: Bool = false, sectionStarts: [Int]? = nil, additionalTokens: [String] = []
    ) throws {
        fixture = try SummaryFixture()
        manifest = fixture.root.appendingPathComponent("tokens-manifest.json")
        tokens = fixture.root.appendingPathComponent("tokens.json")
        let index = fixture.root.appendingPathComponent("index.json")
        let reflow = fixture.root.appendingPathComponent("reflow.json")
        try Data(tokenJSON.utf8).write(to: tokens)
        try Data(indexJSON.utf8).write(to: index)
        let tokenPin = try Self.pin(tokens)
        let indexPin = try Self.pin(index)
        var tokenPins = [tokenPin]
        for (index, json) in additionalTokens.enumerated() {
            let path = fixture.root.appendingPathComponent("tokens-\(index + 1).json")
            try Data(json.utf8).write(to: path)
            tokenPins.append(try Self.pin(path))
        }
        let bindings =
            missingBinding
            ? [:]
            : Dictionary(
                uniqueKeysWithValues:
                    tokenPins.map { ($0.path, $0.sha256) } + [(index.path, indexPin.sha256)])
        try JSONEncoder().encode(Reflow(inputs: bindings)).write(to: reflow)
        let selectedPin = badPin ? Pin(path: tokens.path, sha256: "wrong") : tokenPin
        var jobs = tokenPins.enumerated().map { index, pin in
            Manifest.Job(job: index, tokens: index == 0 ? selectedPin : pin)
        }
        if extraJob { jobs.append(.init(job: tokenPins.count, tokens: selectedPin)) }
        let document = Manifest(
            version: 1, clockDomain: domain, media: fixture.video.movieURL.path,
            displaySRT: try Self.pin(fixture.video.transcriptURL), segmentIndex: indexPin,
            reflow: try Self.pin(reflow),
            jobs: jobs, sectionStarts: sectionStarts)
        try JSONEncoder().encode(document).write(to: manifest)
    }

    private struct Pin: Encodable {
        let path: String
        let sha256: String
    }
    private struct Reflow: Encodable { let inputs: [String: String] }
    private struct Manifest: Encodable {
        struct Job: Encodable {
            let job: Int
            let tokens: Pin
        }
        let version: Int
        let clockDomain: String
        let media: String
        let displaySRT: Pin
        let segmentIndex: Pin
        let reflow: Pin
        let jobs: [Job]
        let sectionStarts: [Int]?
    }
    private static func pin(_ url: URL) throws -> Pin {
        Pin(
            path: url.path,
            sha256: SHA256.hash(data: try Data(contentsOf: url))
                .map { String(format: "%02x", $0) }.joined())
    }
}

struct CaptionTokenTranscriptTests {
    @Test func silentJobsKeepTheirPinsWhileSpeechIsMappedOnce() throws {
        let empty = #"{"version":1,"clock":"output","tokens":[]}"#
        let speech =
            #"{"version":1,"clock":"output","tokens":[{"text":"説明","start":0.2,"end":0.4,"utterance":"speech"}]}"#
        let f = try CaptionTokenFixture(
            tokenJSON: empty,
            indexJSON:
                #"{"version":1,"rate":30,"segments":[{"segment":0,"offsetFrame":18000,"frames":300,"jobs":[0,1,2]}]}"#,
            additionalTokens: [speech, empty])
        defer { #expect(throws: Never.self) { try f.fixture.remove() } }
        let before = try Data(contentsOf: f.tokens)
        let video = try CaptionTokenTranscript(manifest: f.manifest).load()
        #expect(video.cues.map(\.text) == ["説明"])
        #expect(video.cues.map(\.startSeconds) == [600.2])
        #expect(video.cues.map(\.endSeconds) == [600.4])
        #expect(try Data(contentsOf: f.tokens) == before)
        try Data("changed".utf8).write(to: f.tokens)
        #expect(throws: SummaryStoreError.invalidRecord) {
            try CaptionTokenTranscript(manifest: f.manifest).load()
        }
    }

    @Test func oneSilentSegmentCannotHideBehindAnotherSpeechSegment() throws {
        let f = try CaptionTokenFixture(
            indexJSON:
                #"{"version":1,"rate":30,"segments":[{"segment":0,"offsetFrame":0,"frames":300,"jobs":[0]},{"segment":1,"offsetFrame":300,"frames":300,"jobs":[1]}]}"#,
            additionalTokens: [#"{"version":1,"clock":"output","tokens":[]}"#])
        defer { #expect(throws: Never.self) { try f.fixture.remove() } }
        #expect(throws: SummaryStoreError.invalidRecord) {
            try CaptionTokenTranscript(manifest: f.manifest).load()
        }
    }

    @Test(arguments: [
        #"{"version":2,"clock":"output","tokens":[]}"#,
        #"{"version":1,"clock":"source","tokens":[]}"#,
    ])
    func silentJobsStillRequireTheDeclaredVersionAndClock(json: String) throws {
        let f = try CaptionTokenFixture(
            indexJSON:
                #"{"version":1,"rate":30,"segments":[{"segment":0,"offsetFrame":0,"frames":300,"jobs":[0,1]}]}"#,
            additionalTokens: [json])
        defer { #expect(throws: Never.self) { try f.fixture.remove() } }
        #expect(throws: SummaryStoreError.invalidRecord) {
            try CaptionTokenTranscript(manifest: f.manifest).load()
        }
    }

    @Test func anUnusedSilentJobCannotBeRemovedFromMembershipValidation() throws {
        let f = try CaptionTokenFixture(
            additionalTokens: [#"{"version":1,"clock":"output","tokens":[]}"#])
        defer { #expect(throws: Never.self) { try f.fixture.remove() } }
        #expect(throws: SummaryStoreError.invalidRecord) {
            try CaptionTokenTranscript(manifest: f.manifest).load()
        }
    }

    @Test func representativeCliBindsEditedClockAndRejectsEmptyArtifacts() async throws {
        let f = try CaptionTokenFixture()
        defer { #expect(throws: Never.self) { try f.fixture.remove() } }
        let output = f.fixture.root.appendingPathComponent("representative.json")
        let args = [
            "--caption-tokens", f.manifest.path, "--excerpts-output", output.path,
            "--excerpt-selection", "representative",
        ]
        #expect(
            try await SummaryCommand(arguments: ["generate"] + args).run(
                respond: { _ in throw SummaryGenerationError.emptyResponse }, output: { _ in }) == 0
        )
        #expect(
            try await SummaryCommand(arguments: ["inspect"] + args).run(
                respond: { _ in throw SummaryGenerationError.emptyResponse }, output: { _ in }) == 0
        )
        let artifact = try JSONDecoder().decode(
            RepresentativeSourceSummary.self, from: Data(contentsOf: output))
        #expect(artifact.clockDomain == .editedOutput)
        #expect(artifact.clocks.first?.startSeconds == 600.2)
        let empty = f.fixture.root.appendingPathComponent("empty.json")
        try Data().write(to: empty)
        #expect(
            try await SummaryCommand(arguments: [
                "inspect", "--caption-tokens", f.manifest.path,
                "--excerpts-output", empty.path, "--excerpt-selection", "representative",
            ]).run(
                respond: { _ in throw SummaryGenerationError.emptyResponse }, output: { _ in }) == 1
        )
    }

    @Test func mapsSegmentOffsetOnceAndKeepsRealRepeatedWords() async throws {
        let f = try CaptionTokenFixture()
        defer { #expect(throws: Never.self) { try f.fixture.remove() } }
        let reader = CaptionTokenTranscript(manifest: f.manifest)
        #expect(reader.manifest == f.manifest)
        let video = try reader.load()
        #expect(video.cues.map(\.startSeconds) == [600.2, 600.4])
        #expect(video.cues.map(\.text) == ["やばい", "やばい"])
        #expect(video.summary.sections.map(\.text) == ["やばいやばい"])
        #expect(video.summary.sections.map(\.startSeconds) == [600])
        #expect(video.transcriptURL == f.manifest)
        let preview = try SummaryCommand(arguments: [
            "preview", "--caption-tokens", f.manifest.path,
        ])
        #expect(
            try await preview.run(
                respond: { _ in throw SummaryGenerationError.emptyResponse },
                output: { _ in }) == 0)
    }

    @Test(arguments: ["source-presentation", "unknown"])
    func rejectsWrongDomain(domain: String) throws {
        let f = try CaptionTokenFixture(domain: domain)
        defer { #expect(throws: Never.self) { try f.fixture.remove() } }
        #expect(throws: SummaryStoreError.invalidRecord) {
            try CaptionTokenTranscript(manifest: f.manifest).load()
        }
    }

    @Test(arguments: [
        #"{"version":1,"rate":0,"segments":[]}"#,
        #"{"version":1,"rate":30,"segments":[{"segment":0,"offsetFrame":-1,"frames":300,"jobs":[0]}]}"#,
        #"{"version":1,"rate":30,"segments":[{"segment":0,"offsetFrame":0,"frames":300,"jobs":[1]}]}"#,
        #"{"version":1,"rate":30,"segments":[{"segment":0,"offsetFrame":0,"frames":300,"jobs":[0,0]}]}"#,
    ])
    func rejectsInvalidIndex(json: String) throws {
        let f = try CaptionTokenFixture(indexJSON: json)
        defer { #expect(throws: Never.self) { try f.fixture.remove() } }
        #expect(throws: SummaryStoreError.invalidRecord) {
            try CaptionTokenTranscript(manifest: f.manifest).load()
        }
    }

    @Test(arguments: [
        #"{"version":1,"clock":"output","tokens":[]}"#,
        #"{"version":1,"clock":"source","tokens":[]}"#,
        #"{"version":1,"clock":"output","tokens":[{"text":"a","start":-1,"end":0,"utterance":"u"}]}"#,
        #"{"version":1,"clock":"output","tokens":[{"text":"a","start":0,"end":11,"utterance":"u"}]}"#,
        #"{"version":1,"clock":"output","tokens":[{"text":" ","start":0,"end":1,"utterance":"u"}]}"#,
    ])
    func rejectsInvalidToken(json: String) throws {
        let f = try CaptionTokenFixture(tokenJSON: json)
        defer { #expect(throws: Never.self) { try f.fixture.remove() } }
        #expect(throws: SummaryStoreError.invalidRecord) {
            try CaptionTokenTranscript(manifest: f.manifest).load()
        }
    }

    @Test func rejectsMissingBindingsChangedContentAndSymlinks() throws {
        let f = try CaptionTokenFixture(missingBinding: true)
        defer { #expect(throws: Never.self) { try f.fixture.remove() } }
        #expect(throws: SummaryStoreError.invalidRecord) {
            try CaptionTokenTranscript(manifest: f.manifest).load()
        }
        let good = try CaptionTokenFixture()
        defer { #expect(throws: Never.self) { try good.fixture.remove() } }
        try Data("changed".utf8).write(to: good.tokens)
        #expect(throws: SummaryStoreError.invalidRecord) {
            try CaptionTokenTranscript(manifest: good.manifest).load()
        }
        let link = good.fixture.root.appendingPathComponent("link.json")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: good.manifest)
        #expect(throws: SummaryStoreError.unsafePath) {
            try CaptionTokenTranscript(manifest: link).load()
        }
    }

    @Test func rejectsConflictingCLIInput() {
        #expect(throws: SummaryCommandError.arguments) {
            try SummaryCommand(arguments: ["preview", "--caption-tokens", "/a", "--library", "/b"])
        }
    }

    @Test func rejectsUnusedJobsOversizedManifestAndWrongDeclaredPin() throws {
        let unused = try CaptionTokenFixture(extraJob: true)
        defer { #expect(throws: Never.self) { try unused.fixture.remove() } }
        #expect(throws: SummaryStoreError.invalidRecord) {
            try CaptionTokenTranscript(manifest: unused.manifest).load()
        }
        let wrong = try CaptionTokenFixture(badPin: true)
        defer { #expect(throws: Never.self) { try wrong.fixture.remove() } }
        #expect(throws: SummaryStoreError.invalidRecord) {
            try CaptionTokenTranscript(manifest: wrong.manifest).load()
        }
        try Data(repeating: 0, count: (4 << 20) + 1).write(to: unused.manifest)
        #expect(throws: SummaryStoreError.oversizedRecord) {
            try CaptionTokenTranscript(manifest: unused.manifest).load()
        }
        try Data().write(to: unused.manifest)
        #expect(throws: DecodingError.self) {
            try CaptionTokenTranscript(manifest: unused.manifest).load()
        }
    }

    @Test(arguments: [[], [1], [600, 600], [1200]])
    func rejectsMissingOrInvalidSectionSelection(starts: [Int]) throws {
        let f = try CaptionTokenFixture(sectionStarts: starts)
        defer { #expect(throws: Never.self) { try f.fixture.remove() } }
        #expect(throws: SummaryStoreError.invalidRecord) {
            try CaptionTokenTranscript(manifest: f.manifest).load()
        }
    }

    @Test func explicitSingleSectionRetainsItsOriginalClock() throws {
        let f = try CaptionTokenFixture(sectionStarts: [600])
        defer { #expect(throws: Never.self) { try f.fixture.remove() } }
        let video = try CaptionTokenTranscript(manifest: f.manifest).load()
        #expect(video.summary.sections.map(\.startSeconds) == [600])
    }
}
