import Foundation
import Testing
import ViewInspector

@testable import TranscriptCore
@testable import TranscriptSummaries
@testable import TranscriptViewer

private struct RepresentativeFixture {
    let base: ExcerptFixture
    let video: VideoTranscript
    let input: URL
    let artifact: URL
    let pointer: URL
    let candidate: URL?
    let edited: Bool

    private struct Tokens: Encodable {
        struct Token: Encodable {
            let text: String
            let start: Double
            let end: Double
            let utterance: String
        }
        let version = 1
        let clock = "output"
        let tokens: [Token]
    }
    private struct Index: Encodable {
        struct Segment: Encodable {
            let segment = 0
            let offsetFrame = 0
            let frames = 24000
            let jobs = [0]
        }
        let version = 1
        let rate = 30
        let segments = [Segment()]
    }
    private struct Reflow: Encodable { let inputs: [String: String] }
    private struct Manifest: Encodable {
        struct Job: Encodable {
            let job = 0
            let tokens: RepresentativeSummaryStore.Pin
        }
        let version = 1
        let clockDomain = "edited-output"
        let media: String
        let displaySRT: RepresentativeSummaryStore.Pin
        let segmentIndex: RepresentativeSummaryStore.Pin
        let reflow: RepresentativeSummaryStore.Pin
        let jobs: [Job]
    }
    private struct Delivery: Encodable {
        let version = 1
        let status = "unverified-candidate"
        let movie: String
        let movieSize = 6
        let transcript: String
        let transcriptSha256: String
    }

    init(edited: Bool = false, differentSource: Bool = false, sourceText: String = "挨拶だけ。") throws {
        self.edited = edited
        base = try ExcerptFixture(text: sourceText)
        artifact = base.root.appending(path: "representative.json")
        let source: VideoTranscript
        if edited {
            let folder = base.root.appending(path: "replay-123")
            let delivery = folder.appending(path: "revisions/new/delivery")
            try FileManager.default.createDirectory(at: delivery, withIntermediateDirectories: true)
            let movie = delivery.appending(path: "candidate.mp4")
            let srt = delivery.appending(path: "candidate.srt")
            try Data("1\n00:10:00,000 --> 00:10:02,000\n\(sourceText)\n".utf8).write(to: srt)
            try FileManager.default.createSymbolicLink(
                at: movie, withDestinationURL: base.source.movieURL)
            video = VideoTranscript(
                title: "candidate", movieURL: movie, transcriptURL: srt, cues: base.source.cues)
            let evidence = folder.appending(path: "evidence")
            try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
            let record = evidence.appending(path: "candidate-selection.json")
            try JSONEncoder().encode(
                Delivery(
                    movie: "revisions/new/delivery/candidate.mp4",
                    transcript: "revisions/new/delivery/candidate.srt",
                    transcriptSha256: ExcerptFixture.hash(srt))
            ).write(to: record)
            candidate = record
            let tokens = base.root.appending(path: "tokens.json")
            let index = base.root.appending(path: "index.json")
            let reflow = base.root.appending(path: "reflow.json")
            try JSONEncoder().encode(
                Tokens(tokens: [.init(text: sourceText, start: 600, end: 602, utterance: "one")])
            ).write(to: tokens)
            try JSONEncoder().encode(Index()).write(to: index)
            try JSONEncoder().encode(
                Reflow(inputs: [
                    tokens.path: ExcerptFixture.hash(tokens),
                    index.path: ExcerptFixture.hash(index),
                ])
            ).write(to: reflow)
            input = base.root.appending(path: "manifest.json")
            try JSONEncoder().encode(
                Manifest(
                    media: base.source.movieURL.path, displaySRT: Self.pin(srt),
                    segmentIndex: Self.pin(index), reflow: Self.pin(reflow),
                    jobs: [.init(tokens: Self.pin(tokens))])
            ).write(to: input)
            source = try CaptionTokenTranscript(manifest: input).load()
        } else {
            input = base.source.transcriptURL
            candidate = nil
            if differentSource {
                let other = base.root.appending(path: "other.mp4")
                try Data("edited".utf8).write(to: other)
                video = VideoTranscript(
                    title: "other", movieURL: other, transcriptURL: input, cues: base.source.cues)
            } else {
                video = base.source
            }
            source = base.source
        }
        let summary = try RepresentativeSourceSummary.generate(
            video: source,
            fingerprint: SummaryStore.fingerprint(source),
            clockDomain: edited ? .editedOutput : .sourcePresentation)
        try JSONEncoder().encode(summary).write(to: artifact)
        let directory = SummaryStore().directory(for: video)
        try SummaryStore.prepareDirectory(directory)
        pointer = directory.appending(path: "representative-summary-selection.json")
        try write()
    }

    static func pin(_ url: URL) throws -> RepresentativeSummaryStore.Pin {
        .init(path: url.path, sha256: try ExcerptFixture.hash(url))
    }

    func writeTopics() throws {
        let source = try edited ? CaptionTokenTranscript(manifest: input).load() : base.source
        let topics = try SourceTopicSummary.generate(
            video: source, fingerprint: SummaryStore.fingerprint(source),
            clockDomain: edited ? .editedOutput : .sourcePresentation)
        try JSONEncoder().encode(topics).write(to: artifact)
        try write()
    }

    /// Explicit synthetic tokenizer measurements for a concrete planned trip.
    func writeProse() async throws -> URL {
        let source = try edited ? CaptionTokenTranscript(manifest: input).load() : base.source
        #expect(source.summary.sections[0].text == "京都へ旅行する予定。")
        var lower = 0
        let tokens: [TranscriptMorphology.Token] = [
            ("京都", "名詞"), ("へ", "助詞"), ("旅行", "名詞"), ("する", "動詞"), ("予定", "名詞"), ("。", "記号"),
        ].map { text, pos in
            let start = lower
            lower += text.count
            return .init(
                text: text, lowerCharacter: start, upperCharacter: lower,
                partOfSpeech: pos, features: [pos, "一般"])
        }
        let morphology = TranscriptMorphology(
            version: 1, method: "vibrato-ipadic-v1",
            transcriptFingerprint: try SummaryStore.fingerprint(source),
            dictionary: .init(
                path: "/synthetic/system.dic", sha256: TranscriptMorphology.dictionarySHA256),
            sections: [
                .init(
                    startSeconds: 600, endSeconds: 1200,
                    textSHA256: TranscriptMorphology.textHash("京都へ旅行する予定。"), tokens: tokens)
            ])
        let path = base.root.appending(path: "morphology.json")
        try JSONEncoder().encode(morphology).write(to: path)
        let prose = try await SourceSupportedProseGenerator(morphology: morphology) { _ in "[0]" }
            .generate(
                video: source, fingerprint: SummaryStore.fingerprint(source),
                clockDomain: edited ? .editedOutput : .sourcePresentation)
        try JSONEncoder().encode(prose).write(to: artifact)
        try write(morphologyPin: Self.pin(path))
        return path
    }
    func write(
        version: Int = 1, kind: String? = nil, inputPin: RepresentativeSummaryStore.Pin? = nil,
        candidatePin: RepresentativeSummaryStore.Pin? = nil, size: Int = 6,
        morphologyPin: RepresentativeSummaryStore.Pin? = nil,
        artifactPin: RepresentativeSummaryStore.Pin? = nil
    ) throws {
        let selection = try RepresentativeSummaryStore.Selection(
            version: version,
            inputKind: kind ?? (edited ? "caption-tokens" : "source-labels"),
            input: inputPin ?? Self.pin(input), artifact: artifactPin ?? Self.pin(artifact),
            selectedMovie: video.movieURL.resolvingSymlinksInPath().path,
            selectedMovieSize: size,
            selectedTranscriptSha256: ExcerptFixture.hash(video.transcriptURL),
            sourceMedia: edited ? nil : base.source.movieURL.path,
            candidateSelection: candidatePin ?? candidate.map(Self.pin), morphology: morphologyPin)
        try JSONEncoder().encode(selection).write(to: pointer)
    }
}

struct RepresentativeSummaryStoreTests {
    @Test(arguments: ["valid", "corrupt", "wrong-method"])
    func independentProseBindingPreservesOldTopicsAndCannotSilentlyFallBack(_ mutation: String)
        async throws
    {
        let f = try RepresentativeFixture(edited: true, sourceText: "京都へ旅行する予定。")
        defer { #expect(throws: Never.self) { try f.base.remove() } }
        try f.writeTopics()
        let oldArtifact = f.base.root.appending(path: "old-topics.json")
        let oldBytes = try Data(contentsOf: f.artifact)
        try oldBytes.write(to: oldArtifact)
        try f.write(artifactPin: RepresentativeFixture.pin(oldArtifact))
        let oldBinding = try Data(contentsOf: f.pointer)
        _ = try await f.writeProse()
        let newPointer = f.pointer.deletingLastPathComponent().appending(
            path: RepresentativeSummaryStore.proseSelectionFilename)
        if mutation == "wrong-method" {
            try f.write(artifactPin: RepresentativeFixture.pin(oldArtifact))
        }
        let proseBinding = try Data(contentsOf: f.pointer)
        try proseBinding.write(to: newPointer)
        try oldBinding.write(to: f.pointer)
        if mutation == "corrupt" { try Data("invalid".utf8).write(to: newPointer) }
        if mutation == "valid" {
            let result = try #require(await RepresentativeSummaryStore().load(video: f.video))
            #expect(result.prose?.label == "発話抜粋による要約・未検証")
            #expect(result.topics == nil)
            #expect(result.canSeekSelectedMovie)
        } else {
            await #expect(throws: (any Error).self) {
                try await RepresentativeSummaryStore().load(video: f.video)
            }
        }
        #expect(try Data(contentsOf: f.pointer) == oldBinding)
        #expect(try Data(contentsOf: oldArtifact) == oldBytes)
    }

    @Test(arguments: [false, true]) func validatesSupportedProseWithItsPinnedMorphology(
        edited: Bool
    )
        async throws
    {
        let f = try RepresentativeFixture(edited: edited, sourceText: "京都へ旅行する予定。")
        defer { #expect(throws: Never.self) { try f.base.remove() } }
        _ = try await f.writeProse()
        let result = try #require(await RepresentativeSummaryStore().load(video: f.video))
        let prose = try #require(result.prose)
        #expect(prose.overview == "配信では「京都へ旅行する予定。」と話している。")
        #expect(result.summary.clocks.first?.startSeconds == 600)
        #expect(result.summary == prose.representativeEvidence)
        #expect(result.topics == nil)
        #expect(result.canSeekSelectedMovie)
        #expect(try await SummaryStore().load(video: f.video) == nil)
    }

    @Test(arguments: ["missing", "hash", "changed", "unexpected-topic", "unexpected-excerpts"])
    func proseMorphologyBindingCannotBeMissingChangedOrUsedForAnotherMethod(_ mutation: String)
        async throws
    {
        let f = try RepresentativeFixture(sourceText: "京都へ旅行する予定。")
        defer { #expect(throws: Never.self) { try f.base.remove() } }
        let path = try await f.writeProse()
        switch mutation {
        case "missing": try f.write()
        case "hash": try f.write(morphologyPin: .init(path: path.path, sha256: "changed"))
        case "changed": try Data("changed".utf8).write(to: path)
        case "unexpected-topic":
            try f.writeTopics()
            try f.write(morphologyPin: RepresentativeFixture.pin(path))
        case "unexpected-excerpts":
            let plain = try RepresentativeSourceSummary.generate(
                video: f.base.source,
                fingerprint: SummaryStore.fingerprint(f.base.source),
                clockDomain: .sourcePresentation)
            try JSONEncoder().encode(plain).write(to: f.artifact)
            try f.write(morphologyPin: RepresentativeFixture.pin(path))
        default: Issue.record("Unknown fixture mutation")
        }
        await #expect(throws: (any Error).self) {
            try await RepresentativeSummaryStore().load(video: f.video)
        }
    }
    @Test(arguments: [false, true]) func validatesSourceBoundTopicsWithoutChangingNarrative(
        edited: Bool
    )
        async throws
    {
        let f = try RepresentativeFixture(edited: edited)
        defer { #expect(throws: Never.self) { try f.base.remove() } }
        try f.writeTopics()
        let result = try #require(await RepresentativeSummaryStore().load(video: f.video))
        let topics = try #require(result.topics)
        #expect(topics.label == "キーワードと発話抜粋による要約・未検証")
        #expect(topics.sections.first?.excerpts.first?.startSeconds == 600)
        #expect(topics.sections.first?.keywords.first?.text == "挨拶")
        #expect(result.summary == topics.representative)
        #expect(result.canSeekSelectedMovie)
        #expect(try await SummaryStore().load(video: f.video) == nil)
    }

    @Test(.timeLimit(.minutes(1))) func executableUsesTheSameReaderWithoutInference() async throws {
        let f = try RepresentativeFixture(edited: true, sourceText: "京都へ旅行する予定。")
        let execution = try SummaryFixture()
        defer {
            #expect(throws: Never.self) { try f.base.remove() }
            #expect(throws: Never.self) { try execution.remove() }
        }
        let arguments = ["--representative", f.video.movieURL.path, f.video.transcriptURL.path]
        let ready = try await SummaryExecutableTests().run(
            arguments, in: execution, executable: "transcript-check")
        #expect(ready.status == 0)
        #expect(ready.output.contains("CLOCK edited-output SECTIONS 1 EXCERPTS 1 SEEK true"))
        _ = try await f.writeProse()
        let prose = try await SummaryExecutableTests().run(
            arguments, in: execution, executable: "transcript-check")
        #expect(prose.status == 0)
        #expect(prose.output.contains("発話抜粋による要約・未検証"))
        #expect(prose.output.contains("CLOCK edited-output SECTIONS 1 EXCERPTS 1 SEEK true"))
        try FileManager.default.removeItem(at: f.pointer)
        let missing = try await SummaryExecutableTests().run(
            arguments, in: execution, executable: "transcript-check")
        #expect(missing.status == 1)
        #expect(missing.output.contains("REPRESENTATIVE_MISSING"))
        let invalid = try await SummaryExecutableTests().run(
            ["--representative"], in: execution, executable: "transcript-check")
        #expect(invalid.status == 2)
    }

    @Test(arguments: [false, true]) func validatesBothClockDomains(edited: Bool) async throws {
        let f = try RepresentativeFixture(edited: edited)
        defer { #expect(throws: Never.self) { try f.base.remove() } }
        let result = try #require(await RepresentativeSummaryStore().load(video: f.video))
        #expect(result.canSeekSelectedMovie)
        #expect(result == result)
        #expect(result.summary.clocks.first?.startSeconds == 600)
        #expect(result.summary.label == "発話抜粋による要約・未検証")
        #expect(try await SummaryStore().load(video: f.video) == nil)
    }

    @Test func sourceClockCannotSeekAnEditedMovie() async throws {
        let f = try RepresentativeFixture(differentSource: true)
        defer { #expect(throws: Never.self) { try f.base.remove() } }
        #expect(
            try await RepresentativeSummaryStore().load(video: f.video)?.canSeekSelectedMovie
                == false)
    }

    @Test(arguments: [
        "version", "kind", "size", "hash", "relative", "empty", "oversized", "symlink", "artifact",
        "candidate", "candidate-binding",
    ])
    func invalidBindingsFailClosed(change: String) async throws {
        let f = try RepresentativeFixture(edited: true)
        defer { #expect(throws: Never.self) { try f.base.remove() } }
        switch change {
        case "version": try f.write(version: 2)
        case "kind": try f.write(kind: "unknown")
        case "size": try f.write(size: 7)
        case "hash": try f.write(inputPin: .init(path: f.input.path, sha256: "bad"))
        case "relative": try f.write(inputPin: .init(path: "manifest.json", sha256: "bad"))
        case "empty": try Data().write(to: f.pointer)
        case "oversized": try Data(repeating: 32, count: 65_537).write(to: f.pointer)
        case "artifact": try Data("changed".utf8).write(to: f.artifact)
        case "candidate": try Data("changed".utf8).write(to: #require(f.candidate))
        case "candidate-binding":
            let record = try #require(f.candidate)
            let text = try String(contentsOf: record, encoding: .utf8).replacingOccurrences(
                of: "unverified-candidate", with: "verified")
            try Data(text.utf8).write(to: record)
            try f.write()
        case "symlink":
            let target = f.base.root.appending(path: "pointer.json")
            try FileManager.default.moveItem(at: f.pointer, to: target)
            try FileManager.default.createSymbolicLink(at: f.pointer, withDestinationURL: target)
        default: Issue.record("Unexpected fixture mutation")
        }
        await #expect(throws: (any Error).self) {
            try await RepresentativeSummaryStore().load(video: f.video)
        }
    }
}

@MainActor struct RepresentativeSummaryPanelTests {
    @Test(arguments: [false, true]) func paragraphPanelShowsSupportAndUsesOnlyMatchingMovieClock(
        edited: Bool
    )
        async throws
    {
        let f = try RepresentativeFixture(
            edited: edited, differentSource: !edited,
            sourceText: "京都へ旅行する予定。")
        defer { #expect(throws: Never.self) { try f.base.remove() } }
        _ = try await f.writeProse()
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { _ in
                Issue.record("Reading pinned paragraphs must not start inference")
                return "unused"
            })
        await model.loadExcerpts(video: f.video)
        var sought: Double?
        let panel = SourceExcerptPanel(state: model.excerptState) { sought = $0 }
        let content = try panel.inspect().find(SourceSupportedProsePanel.self)
        #expect(try content.find(text: "発話抜粋による要約・未検証").string().contains("未検証"))
        #expect(
            try content.findAll(ViewInspector.ViewType.Text.self).filter {
                try $0.string() == "配信では「京都へ旅行する予定。」と話している。"
            }.count == 2)
        #expect(try content.find(text: "根拠の発話：「京都へ旅行する予定。」").string().contains("予定"))
        #expect(model.state == .idle)
        if edited {
            try content.find(button: "600秒から").tap()
            #expect(sought == 600)
            #expect(try content.find(text: "時刻は編集後の発話区間です。").string().contains("編集後"))
        } else {
            #expect(content.findAll(ViewInspector.ViewType.Button.self).isEmpty)
            #expect(try content.find(text: "原動画 600秒から").string().contains("原動画"))
        }
    }
    @Test(arguments: [false, true]) func topicPanelUsesExactWordsAndValidatedSpeechClock(
        edited: Bool
    )
        async throws
    {
        let f = try RepresentativeFixture(edited: edited, differentSource: !edited)
        defer { #expect(throws: Never.self) { try f.base.remove() } }
        try f.writeTopics()
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { _ in
                Issue.record("Source topic display must not start native generation")
                return "unused"
            })
        await model.loadExcerpts(video: f.video)
        var sought: Double?
        let panel = SourceExcerptPanel(state: model.excerptState) { sought = $0 }
        let content = try panel.inspect().find(SourceTopicSummaryPanel.self)
        #expect(try content.find(text: "キーワードと発話抜粋による要約・未検証").string().contains("未検証"))
        #expect(try content.find(text: "「挨拶だけ。」").string() == "「挨拶だけ。」")
        #expect(
            try content.findAll(ViewInspector.ViewType.Text.self).filter {
                try $0.string() == "挨拶"
            }.count == 2)
        if edited {
            try content.find(button: "600秒から").tap()
            #expect(sought == 600)
        } else {
            #expect(content.findAll(ViewInspector.ViewType.Button.self).isEmpty)
        }
        #expect(model.state == .idle)
    }

    @Test func keywordlessSpeechRemainsExplicitlyAnExcerpt() throws {
        let keywordless = SourceTopicSummary(
            version: 1, method: SourceTopicSummary.methodName,
            label: SourceTopicSummary.displayLabel,
            representative: .init(
                version: 1, method: RepresentativeSourceSummary.methodName,
                label: RepresentativeSourceSummary.displayLabel, clockDomain: .editedOutput,
                source: .init(
                    mode: "source-excerpts-v1", fingerprint: "pin", sections: [], overview: []),
                clocks: []), overview: [], sections: [])
        let content = try SourceTopicSummaryPanel(summary: keywordless, canSeek: false) { _ in }
            .inspect()
        #expect(try content.find(text: "発話抜粋").string() == "発話抜粋")
    }

    @Test func showingCandidateLoadsSavedNarrativeWithoutStartingInference() async throws {
        let f = try RepresentativeFixture(edited: true)
        defer { #expect(throws: Never.self) { try f.base.remove() } }
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { _ in
                Issue.record(
                    "Displaying an explicit representative artifact must not start inference")
                return "unused"
            })
        let view = NarrativeSummaryView(video: f.video, model: model) { _ in }
        let request = NarrativeSummaryView.Request(
            video: f.video, attempt: 0, reload: 0, batching: false)
        try await view.inspect().find(ViewInspector.ViewType.ScrollView.self).callTask(id: request)
        #expect(model.state == .failed("この字幕に一致する保存済み要約がありません。"))
        guard case .representative = model.excerptState else {
            Issue.record("The separate unverified excerpt panel must remain visible")
            return
        }
    }

    @Test(arguments: [false, true]) func panelUsesValidatedClockWithoutClaimingApproval(
        edited: Bool
    ) async throws {
        let f = try RepresentativeFixture(edited: edited, differentSource: !edited)
        defer { #expect(throws: Never.self) { try f.base.remove() } }
        let model = NarrativeSummaryModel()
        await model.loadExcerpts(video: f.video)
        guard case .representative(let candidate) = model.excerptState else {
            Issue.record("Expected distinct representative candidate state")
            return
        }
        var sought: Double?
        let panel = SourceExcerptPanel(state: model.excerptState) { sought = $0 }
        let content = try panel.inspect().find(RepresentativeSummaryPanel.self)
        #expect(try content.find(text: "発話抜粋による要約・未検証").string().contains("未検証"))
        #expect(model.state == .idle)
        if edited {
            try content.find(button: "600秒から").tap()
            #expect(sought == 600)
        } else {
            #expect(!candidate.canSeekSelectedMovie)
            #expect(content.findAll(ViewInspector.ViewType.Button.self).isEmpty)
        }
    }
}
