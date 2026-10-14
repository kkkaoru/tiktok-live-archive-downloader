import Foundation
import Testing

@testable import TranscriptCore
@testable import TranscriptSummaries

struct SourceLabelTranscriptTests {
    @Test func tokenLayoutPreservesRepeatedCharactersAndSectionClock() throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let labels = fixture.root.appendingPathComponent("tokens.json")
        try Data(
            #"{"labels":[{"start":0.738,"end":1,"text":"こ"},{"start":1,"end":1,"text":"こ"},{"start":1,"end":2,"text":"は"},{"start":600,"end":601,"text":"次"}]}"#
                .utf8
        ).write(to: labels)
        let args = [
            "generate", "--source-labels", labels.path, "--source-media",
            fixture.video.movieURL.path, "--clock-domain", "source-presentation",
            "--source-label-layout", "tokens",
        ]
        let command = try SummaryCommand(arguments: args)
        let input = try #require(command.sourceLabels)
        var video = try input.load()
        #expect(video.summary.sections.map(\.text) == ["ここは", "次"])
        #expect(video.summary.sections.map(\.startSeconds) == [0, 600])
        #expect(video.cues.map(\.startSeconds) == [0.738, 1, 1, 600])
        let originalFingerprint = try SummaryStore.fingerprint(video)
        let captions = try SourceLabelTranscript(
            labels: labels, media: fixture.video.movieURL, clockDomain: "source-presentation"
        ).load()
        #expect(try SummaryStore.fingerprint(captions) != originalFingerprint)
        video.cues[2].text = "です"
        #expect(video.summary.sections.map(\.text) == ["ここです", "次"])
    }

    @Test func invalidTokenLayoutArgumentsFail() {
        #expect(throws: SummaryCommandError.self) {
            try SummaryCommand(arguments: [
                "generate", "--library", "x", "--source-label-layout", "tokens",
            ])
        }
        #expect(throws: SummaryCommandError.self) {
            try SummaryCommand(arguments: [
                "generate", "--source-labels", "a", "--source-media", "b", "--clock-domain",
                "source-presentation", "--source-label-layout", "guess",
            ])
        }
    }

    @Test func tokenWordsRepeatsAndUtteranceGapsRemainIntact() {
        let cues = [
            TranscriptCue(
                id: 0, startSeconds: 0, endSeconds: 0.2, startLabel: "", endLabel: "", text: "やばい"),
            TranscriptCue(
                id: 1, startSeconds: 0.2, endSeconds: 0.4, startLabel: "", endLabel: "", text: "やばい"
            ),
            TranscriptCue(
                id: 2, startSeconds: 3, endSeconds: 3.2, startLabel: "", endLabel: "", text: "フォ"),
            TranscriptCue(
                id: 3, startSeconds: 3.2, endSeconds: 3.4, startLabel: "", endLabel: "", text: "ロワー"
            ),
            TranscriptCue(
                id: 4, startSeconds: 3.4, endSeconds: 3.6, startLabel: "", endLabel: "", text: "さん"),
            TranscriptCue(
                id: 5, startSeconds: 600, endSeconds: 600.2, startLabel: "", endLabel: "",
                text: "次の話。"),
        ]
        #expect(
            TranscriptSummary(cues: cues, inputKind: .tokenSequence).sections.map(\.text) == [
                "やばいやばい\nフォロワーさん", "次の話。",
            ])
    }

    @Test func previewDoesNotInferOrWriteSummary() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let command = try SummaryCommand(arguments: ["preview", "--library", fixture.root.path])
        #expect(
            try await command.run(
                respond: { _ in
                    Issue.record("Preview must not infer")
                    throw SummaryGenerationError.emptyResponse
                }, output: { _ in }) == 0)
        #expect(!FileManager.default.fileExists(atPath: fixture.current.path))
    }

    @Test func preservesSourceOriginOverlapsAndPointsAndReusesSummary() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let labels = fixture.root.appendingPathComponent("source.json")
        try Data(
            #"{"labels":[{"start":0.738,"end":2,"text":"旅行について話している。"},{"start":1,"end":1,"text":"準備する。"}]}"#
                .utf8
        ).write(to: labels)
        let input = try SourceLabelTranscript(
            labels: labels, media: fixture.video.movieURL,
            clockDomain: "source-presentation")
        let video = try input.load()
        #expect(video.cues.map(\.startSeconds) == [0.738, 1])
        #expect(video.cues.map(\.endSeconds) == [2, 1])
        #expect(video.transcriptURL == labels)
        let args = [
            "--source-labels", labels.path, "--source-media", fixture.video.movieURL.path,
            "--clock-domain", "source-presentation",
        ]
        let generate = try SummaryCommand(arguments: ["generate"] + args)
        #expect(try await generate.run(respond: { _ in "旅行の準備について説明している。" }, output: { _ in }) == 0)
        let inspect = try SummaryCommand(arguments: ["inspect"] + args)
        #expect(try await inspect.run(output: { _ in }) == 0)
        #expect(try await SummaryStore().load(video: video) != nil)
    }

    @Test(arguments: [
        "",
        #"{"labels":[]}"#,
        #"{"labels":[{"start":-1,"end":2,"text":"a"}]}"#,
        #"{"labels":[{"start":2,"end":1,"text":"a"}]}"#,
        #"{"labels":[{"start":0,"end":1,"text":"  "}]}"#,
        #"{"labels":[{"start":2,"end":3,"text":"a"},{"start":1,"end":4,"text":"b"}]}"#,
        #"{"labels":[{"start":"NaN","end":1,"text":"a"}]}"#,
        #"{"labels":[{"start":1e300,"end":1e300,"text":"a"}]}"#,
        #"{"labels":[{"start":0,"end":1}]}"#,
    ])
    func rejectsInvalidLabels(_ json: String) throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let labels = fixture.root.appendingPathComponent("source.json")
        try Data(json.utf8).write(to: labels)
        let input = try SourceLabelTranscript(
            labels: labels, media: fixture.video.movieURL,
            clockDomain: "source-presentation")
        #expect(throws: (any Error).self) { try input.load() }
    }

    @Test func rejectsSymlinksDirectoriesAndOversizedData() throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let link = fixture.root.appendingPathComponent("link.json")
        try FileManager.default.createSymbolicLink(
            at: link, withDestinationURL: fixture.video.transcriptURL)
        let linked = try SourceLabelTranscript(
            labels: link, media: fixture.video.movieURL,
            clockDomain: "source-presentation")
        #expect(throws: SummaryStoreError.unsafePath) { try linked.load() }
        let directory = try SourceLabelTranscript(
            labels: fixture.root, media: fixture.video.movieURL,
            clockDomain: "source-presentation")
        #expect(throws: SummaryStoreError.unsafePath) { try directory.load() }
        let huge = fixture.root.appendingPathComponent("large.json")
        try Data().write(to: huge)
        let handle = try FileHandle(forWritingTo: huge)
        try handle.truncate(atOffset: UInt64(SourceLabelTranscript.maximumBytes + 1))
        try handle.close()
        let oversized = try SourceLabelTranscript(
            labels: huge, media: fixture.video.movieURL,
            clockDomain: "source-presentation")
        #expect(throws: SummaryStoreError.oversizedRecord) { try oversized.load() }
    }

    @Test(arguments: [
        ["--source-labels", "a"],
        ["--source-labels", "a", "--source-media", "b", "--clock-domain", "continuous-pcm"],
        [
            "--source-labels", "a", "--source-media", "b", "--clock-domain", "source-presentation",
            "--library", "c",
        ],
        [
            "--source-labels", "a", "--source-media", "b", "--clock-domain", "source-presentation",
            "--limit", "1",
        ],
    ])
    func rejectsAmbiguousInputs(_ args: [String]) {
        #expect(throws: SummaryCommandError.self) {
            try SummaryCommand(arguments: ["generate"] + args)
        }
    }
}
