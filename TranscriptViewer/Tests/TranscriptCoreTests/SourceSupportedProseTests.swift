import Foundation
import NaturalLanguage
import Testing

@testable import TranscriptCore
@testable import TranscriptSummaries

struct SourceSupportedProseTests {
    /// Synthetic morphology labels, not a claim of NLP measurements for real recordings.
    private func morphology(_ input: VideoTranscript, fingerprint: String = "pin")
        -> TranscriptMorphology
    {
        TranscriptMorphology(
            version: 1, method: "vibrato-ipadic-v1", transcriptFingerprint: fingerprint,
            dictionary: .init(
                path: "/synthetic/system.dic", sha256: TranscriptMorphology.dictionarySHA256),
            sections: input.summary.sections.map { section in
                let source = section.text
                let tokenizer = NLTokenizer(unit: .word)
                tokenizer.string = source
                tokenizer.setLanguage(.japanese)
                var ends: [String.Index] = []
                tokenizer.enumerateTokens(in: source.startIndex..<source.endIndex) { range, _ in
                    ends.append(range.upperBound)
                    return true
                }
                if ends.last != source.endIndex { ends.append(source.endIndex) }
                var lower = source.startIndex
                let tokens = ends.map { upper in
                    let text = String(source[lower..<upper])
                    let position = source.distance(from: source.startIndex, to: lower)
                    lower = upper
                    let pos: String
                    if text.contains(where: { "。！？!?\n".contains($0) }) {
                        pos = "記号"
                    } else if ["ありがとう", "こんにちは", "いらっしゃい"].contains(text) {
                        pos = "感動詞"
                    } else if ["は", "へ", "の", "に", "が", "を", "と", "か", "で", "も"].contains(text) {
                        pos = "助詞"
                    } else if ["買っ", "する", "いる", "行く", "話し", "考え", "減らす", "変更"].contains(text) {
                        pos = "動詞"
                    } else {
                        pos = "名詞"
                    }
                    return TranscriptMorphology.Token(
                        text: text, lowerCharacter: position, upperCharacter: position + text.count,
                        partOfSpeech: pos, features: [pos, text == "さん" ? "接尾" : "一般"])
                }
                return .init(
                    startSeconds: section.startSeconds, endSeconds: section.endSeconds,
                    textSHA256: TranscriptMorphology.textHash(source), tokens: tokens)
            })
    }
    private func video(_ texts: [String], sections: Bool = false) -> VideoTranscript {
        VideoTranscript(
            title: "Synthetic", movieURL: URL(fileURLWithPath: "/fixture.mov"),
            transcriptURL: URL(fileURLWithPath: "/fixture.json"),
            cues: texts.enumerated().map { index, text in
                let start = Double(index * (sections ? 600 : 20))
                return .init(
                    id: index, startSeconds: start, endSeconds: start + 10,
                    startLabel: "", endLabel: "", text: text)
            }, summaryInputKind: .tokenSequence)
    }

    private func result(_ input: VideoTranscript) async throws -> SourceSupportedProse {
        try await SourceSupportedProseGenerator(morphology: morphology(input)) { request in
            #expect(request.selectsSource && request.supportedSpeechSelection)
            #expect(request.source.utf8.count <= SummaryRequest.inputByteLimit)
            #expect(request.modelInstructions.contains("新しい事実"))
            #expect(request.prompt.contains("既存ID"))
            return "[0]"
        }.generate(video: input, fingerprint: "pin", clockDomain: .editedOutput)
    }

    @Test func constructsParagraphsFromQuestionsPlansAndNegationWithoutNewClaims() async throws {
        let input = video([
            "薬を買ってる?薬を買ったとは言っていない。",
            "京都への旅行を予定しているが、まだ予約はしていない。",
            "もし雨なら電車で行く。晴れるかどうかは分からない。",
        ])
        let summary = try await result(input)
        try summary.validate(
            video: input, fingerprint: "pin", clockDomain: .editedOutput,
            morphology: morphology(input))
        #expect(summary.method == SourceSupportedProse.methodName)
        #expect(summary.label == "発話抜粋による要約・未検証")
        #expect(summary.version == 2)
        #expect(summary.sections.count == 1)
        #expect(summary.sections[0].id == 0)
        #expect(summary.sections[0].text.hasPrefix("配信では「"))
        #expect(summary.sections[0].text.contains("また、"))
        #expect(summary.sections[0].evidence.count == 3)
        #expect(summary.sections[0].text.contains("言っていない"))
        #expect(summary.sections[0].text.contains("予定"))
        #expect(summary.sections[0].text.contains("もし"))
        #expect(summary.overviewEvidence.allSatisfy(summary.sections.flatMap(\.evidence).contains))
        #expect(summary.sections[0].evidence.first?.startSeconds == 0)
        #expect(summary.sections[0].evidence.last?.endSeconds == 50)
        let question = try await result(video(["薬を買ってる?"]))
        #expect(question.overview == "配信では「薬を買ってる?」と話している。")
    }

    @Test func unsupportedParaphrasesAreRejectedEvenWithAnExactRealQuote() async throws {
        let input = video(["関係が重いと話している。交際を断ったら死ぬと言われたという相談だった。"])
        let good = try await result(input)
        for unsupported in [
            "体重を心配している。", "死去した相手の話をしている。",
            "治療に時間がかかるので薬を注文した。", "交際を始めたから問題は解決した。",
            "The speaker decided to die.", "相手は死去した。相手は死去した。",
        ] {
            let bad = SourceSupportedProse(
                version: good.version, method: good.method, label: good.label,
                fingerprint: good.fingerprint, morphologyFingerprint: good.morphologyFingerprint,
                clockDomain: good.clockDomain,
                overview: unsupported, overviewEvidence: good.overviewEvidence,
                sections: good.sections)
            #expect(throws: SourceExcerptError.invalidEvidence) {
                try bad.validate(
                    video: input, fingerprint: "pin", clockDomain: .editedOutput,
                    morphology: morphology(input))
            }
        }
        let badSection = SourceSupportedProse.Section(
            startSeconds: 0, endSeconds: 600, text: "体重を気にしている。",
            evidence: good.sections[0].evidence)
        let bad = SourceSupportedProse(
            version: 1, method: good.method, label: good.label, fingerprint: "pin",
            morphologyFingerprint: good.morphologyFingerprint,
            clockDomain: .editedOutput, overview: good.overview,
            overviewEvidence: good.overviewEvidence, sections: [badSection])
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try bad.validate(
                video: input, fingerprint: "pin", clockDomain: .editedOutput,
                morphology: morphology(input))
        }
    }

    @Test func rawElongationAndAllSourceClocksRemainUnchanged() async throws {
        let long = "予定を変更するか考えている。え" + String(repeating: "ー", count: 700) + "。"
        let input = video([long, "明日の電車は午後二時に出発する予定。"])
        let before = input.summary
        let summary = try await result(input)
        #expect(input.summary == before)
        #expect(input.summary.sections[0].text.contains(long))
        #expect(!summary.sections[0].text.contains(String(repeating: "ー", count: 4)))
        #expect(SourceSupportedProseGenerator.presentation("おーーーい") == "おーーーい")
        #expect(SourceSupportedProseGenerator.presentation("あーーーーい") == "あーーい")
        #expect(SourceSupportedProseGenerator.presentation("へっっっっっ!?") == "へっっ!?")
        #expect(summary.sections[0].evidence.first?.startSeconds == 0)
        #expect(summary.sections[0].evidence.last?.endSeconds == 30)
    }

    @Test func greetingOnlyAndEmptyInputsCannotBecomeReadySummaries() async {
        for input in [video([]), video(["こんにちは。ありがとう。いらっしゃい。"])] {
            await #expect(throws: SummaryGenerationError.emptyTranscript) {
                try await result(input)
            }
        }
        #expect(SourceSupportedProseGenerator.score(tokens: [], text: "こんにちは。ありがとう。") == 0)
    }

    @Test func repeatedGreetingIsNotTheOverviewAndOriginalRepetitionIsRetained() async throws {
        let greeting = "太郎さんありがとう。いらっしゃい。"
        let input = video(
            [
                String(repeating: greeting, count: 20) + "京都への旅行に向けて雨具と靴を準備する。",
                String(repeating: greeting, count: 20) + "予約変更はまだ決めていない。電車の時刻を確認する。",
                String(repeating: greeting, count: 20) + "荷物を減らすため着替えを二日分にする。",
            ], sections: true)
        let original = input.summary
        let summary = try await result(input)
        #expect(summary.sections.count == 3)
        #expect(input.summary == original)
        #expect(summary.overview.contains("旅行") || summary.overview.contains("準備"))
        #expect(summary.overview.contains("予約") || summary.overview.contains("電車"))
        #expect(summary.overview.contains("荷物") || summary.overview.contains("着替え"))
        #expect(!summary.overview.contains("ありがとう"))
    }

    @Test(arguments: ["[99]", "[-1]", "[0,0]", "[]", "[0.5]", "注文した。", "{\"id\":0}"])
    func modelCannotSupplyNewRangesOrRewrittenFacts(_ raw: String) async {
        let input = video((0..<12).map { "\($0)番の電車で京都へ旅行する予定。" })
        await #expect(throws: SourceExcerptError.invalidSelection) {
            try await SourceSupportedProseGenerator(morphology: morphology(input, fingerprint: "p"))
            { _ in raw }.generate(
                video: input, fingerprint: "p", clockDomain: .editedOutput)
        }
    }

    @Test func rangeClockFingerprintAndMethodTamperingFailClosed() async throws {
        let input = video(["電車で京都へ行く予定を話している。"])
        let good = try await result(input)
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try good.validate(
                video: input, fingerprint: "changed", clockDomain: .editedOutput,
                morphology: morphology(input))
        }
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try good.validate(
                video: input, fingerprint: "pin", clockDomain: .sourcePresentation,
                morphology: morphology(input))
        }
        let original = try #require(good.sections[0].evidence.first)
        let shifted = RepresentativeSourceSummary.Clock(
            span: original.span, startSeconds: original.startSeconds + 1,
            endSeconds: original.endSeconds + 1)
        let invented = RepresentativeSourceSummary.Clock(
            span: .init(
                sectionStartSeconds: 0, lowerCharacter: 0, upperCharacter: 5, text: "購入した。"),
            startSeconds: 0, endSeconds: 10)
        for evidence in [[shifted], [invented], [original, original], []] {
            let changed = SourceSupportedProse(
                version: 1, method: good.method, label: good.label, fingerprint: "pin",
                morphologyFingerprint: good.morphologyFingerprint,
                clockDomain: .editedOutput, overview: good.overview,
                overviewEvidence: evidence, sections: good.sections)
            #expect(throws: SourceExcerptError.invalidEvidence) {
                try changed.validate(
                    video: input, fingerprint: "pin", clockDomain: .editedOutput,
                    morphology: morphology(input))
            }
        }
        await #expect(throws: SummaryGenerationError.oversizedCharacter) {
            try await result(video(["a" + String(repeating: "\u{0301}", count: 9000)]))
        }
    }

    @Test func sharedCliWritesImmutableProseAndInspectNeverRunsInference() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let labels = fixture.root.appendingPathComponent("labels.json")
        let output = fixture.root.appendingPathComponent("prose.json")
        let morphologyPath = fixture.root.appendingPathComponent("morphology.json")
        try Data(#"{"labels":[{"start":0.75,"end":1.5,"text":"京都への旅行に向けて雨具を準備する。"}]}"#.utf8)
            .write(to: labels)
        let source = try SourceLabelTranscript(
            labels: labels, media: fixture.video.movieURL,
            clockDomain: "source-presentation", inputKind: .tokenSequence
        ).load()
        try JSONEncoder().encode(morphology(source, fingerprint: SummaryStore.fingerprint(source)))
            .write(to: morphologyPath)
        let args = [
            "--source-labels", labels.path, "--source-media", fixture.video.movieURL.path,
            "--clock-domain", "source-presentation", "--source-label-layout", "tokens",
            "--excerpts-output", output.path, "--excerpt-selection", "supported-prose",
            "--morphology", morphologyPath.path,
        ]
        let preview = [
            "preview", "--source-labels", labels.path, "--source-media",
            fixture.video.movieURL.path,
            "--clock-domain", "source-presentation", "--source-label-layout", "tokens",
            "--excerpt-selection", "supported-prose", "--morphology", morphologyPath.path,
        ]
        let previewStatus = try await SummaryCommand(arguments: preview).run(
            respond: { _ in throw SourceExcerptError.invalidSelection },
            output: { text in
                #expect(text.contains("\"legacy\"") && text.contains("\"contextual\""))
                #expect(text.contains("京都への旅行に向けて雨具を準備する。"))
            })
        #expect(previewStatus == 0)
        #expect(!FileManager.default.fileExists(atPath: output.path))
        #expect(
            try await SummaryCommand(arguments: ["generate"] + args).run(
                respond: { _ in "[0]" }, output: { _ in }) == 0)
        let result = try JSONDecoder().decode(
            SourceSupportedProse.self, from: Data(contentsOf: output))
        #expect(result.clockDomain == .sourcePresentation)
        #expect(result.overviewEvidence.first?.startSeconds == 0.75)
        #expect(result.overview.contains("準備"))
        #expect(
            try await SummaryCommand(arguments: ["inspect"] + args).run(
                respond: { _ in throw SourceExcerptError.invalidSelection }, output: { _ in }) == 0)
        await #expect(throws: SummaryStoreError.invalidRecord) {
            try await SummaryCommand(arguments: ["generate"] + args).run(output: { _ in })
        }
        #expect(!FileManager.default.fileExists(atPath: fixture.current.path))
    }

    @Test func proseMetadataAndSectionEvidenceCannotBeReassigned() async throws {
        let input = video(
            [
                "電車で京都へ行く予定を話している。", "荷物は明日準備する予定。",
            ], sections: true)
        let good = try await result(input)
        let original = good.sections[0]
        let mismatched = SourceSupportedProse.Section(
            startSeconds: original.startSeconds + 1, endSeconds: original.endSeconds,
            text: original.text, evidence: original.evidence)
        let differentEnd = SourceSupportedProse.Section(
            startSeconds: original.startSeconds, endSeconds: original.endSeconds - 1,
            text: original.text, evidence: original.evidence)
        let otherClock = good.sections[1].evidence
        let wrongOwner = SourceSupportedProse.Section(
            startSeconds: original.startSeconds, endSeconds: original.endSeconds,
            text: try SourceSupportedProse.paragraph(otherClock), evidence: otherClock)
        for sections in [
            [mismatched, good.sections[1]], [differentEnd, good.sections[1]],
            [wrongOwner, good.sections[1]], [],
        ] {
            let changed = SourceSupportedProse(
                version: 1, method: good.method, label: good.label, fingerprint: "pin",
                morphologyFingerprint: good.morphologyFingerprint,
                clockDomain: .editedOutput, overview: good.overview,
                overviewEvidence: good.overviewEvidence, sections: sections)
            #expect(throws: SourceExcerptError.invalidEvidence) {
                try changed.validate(
                    video: input, fingerprint: "pin", clockDomain: .editedOutput,
                    morphology: morphology(input))
            }
        }
        for (version, method, label, morphologyPin) in [
            (3, good.method, good.label, good.morphologyFingerprint),
            (1, "unbound-paraphrase", good.label, good.morphologyFingerprint),
            (1, good.method, "検証済み", good.morphologyFingerprint),
            (1, good.method, good.label, "wrong"),
        ] {
            let changed = SourceSupportedProse(
                version: version, method: method, label: label, fingerprint: "pin",
                morphologyFingerprint: morphologyPin,
                clockDomain: .editedOutput, overview: good.overview,
                overviewEvidence: good.overviewEvidence, sections: good.sections)
            #expect(throws: SourceExcerptError.invalidEvidence) {
                try changed.validate(
                    video: input, fingerprint: "pin", clockDomain: .editedOutput,
                    morphology: morphology(input))
            }
        }
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try SourceSupportedProse.paragraph([])
        }
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try SourceSupportedProse.paragraph(Array(repeating: original.evidence[0], count: 4))
        }
    }

    @Test func legacyQuotationArtifactStillValidatesWithoutRelabeling() async throws {
        let input = video(["京都への旅行は雨具を準備する予定。"])
        let current = try await result(input)
        let legacy = SourceSupportedProse(
            version: 1, method: current.method, label: "発話根拠付きの文章要約・未検証",
            fingerprint: current.fingerprint, morphologyFingerprint: current.morphologyFingerprint,
            clockDomain: current.clockDomain, overview: current.overview,
            overviewEvidence: current.overviewEvidence, sections: current.sections)
        try legacy.validate(
            video: input, fingerprint: "pin", clockDomain: .editedOutput,
            morphology: morphology(input))
        #expect(legacy.label == "発話根拠付きの文章要約・未検証")
    }

    @Test func immediateSameTopicReplyIsKeptWithOriginalClockAndWords() throws {
        let first = "仕事は適当でよい。"
        let reply = "いや仕事は真面目に取り組みたい。"
        let input = VideoTranscript(
            title: "Synthetic", movieURL: URL(fileURLWithPath: "/fixture.mov"),
            transcriptURL: URL(fileURLWithPath: "/fixture.json"),
            cues: [
                .init(
                    id: 0, startSeconds: 10, endSeconds: 12, startLabel: "", endLabel: "",
                    text: first),
                .init(
                    id: 1, startSeconds: 12, endSeconds: 15, startLabel: "", endLabel: "",
                    text: reply),
            ], summaryInputKind: .tokenSequence)
        let layout = try #require(RepresentativeSourceSummary.layouts(input)[0])
        let selected = try SourceSupportedProseGenerator.candidates(
            input.summary.sections[0], morphology: morphology(input).sections[0], layout: layout)
        #expect(selected.count == 1)
        #expect(selected.first?.span.text == first + reply)
        #expect(selected.first?.startSeconds == 10)
        #expect(selected.first?.endSeconds == 15)
        let legacy = try SourceSupportedProseGenerator.candidates(
            input.summary.sections[0], morphology: morphology(input).sections[0], layout: layout,
            policy: .legacy)
        #expect(legacy.count == 2)
        #expect(legacy.first?.span.text == first)
    }

    @Test func unrelatedOrSeparatedRepliesDoNotMerge() throws {
        for texts in [
            ["仕事は適当でよい。", "いや旅行は京都に行く予定。"],
            ["仕事は適当でよい。", "いや仕事は真面目に取り組みたい。"],
        ] {
            let input = video(texts)
            let layout = try #require(RepresentativeSourceSummary.layouts(input)[0])
            let candidates = try SourceSupportedProseGenerator.candidates(
                input.summary.sections[0], morphology: morphology(input).sections[0], layout: layout
            )
            #expect(candidates.count == 2)
            #expect(candidates.first?.endSeconds == 10)
            #expect(candidates.last?.startSeconds == 20)
        }
    }

    @Test func mixedGreetingLengthNoLongerImprovesContentRank() {
        let input = video(
            [
                "京都への旅行は雨具を準備する予定。",
                "京都への旅行は雨具を準備する予定太郎さんありがとう花子さんいらっしゃい。",
            ], sections: true)
        let pos = morphology(input)
        let clear = SourceSupportedProseGenerator.score(
            tokens: pos.sections[0].tokens, text: input.summary.sections[0].text)
        let mixed = SourceSupportedProseGenerator.score(
            tokens: pos.sections[1].tokens, text: input.summary.sections[1].text)
        #expect(clear > mixed)
        #expect(input.summary.sections[1].text.contains("ありがとう"))
    }

    @Test func nounPhrasesStayEligibleWithoutBeatingPredicateSupportedContent() {
        let nouns = ["京都", "旅行"].enumerated().map { index, word in
            TranscriptMorphology.Token(
                text: word, lowerCharacter: index * 2,
                upperCharacter: index * 2 + 2, partOfSpeech: "名詞", features: ["名詞", "一般"])
        }
        let predicate = TranscriptMorphology.Token(
            text: "行く", lowerCharacter: 5,
            upperCharacter: 7, partOfSpeech: "動詞", features: ["動詞", "自立"])
        let nominal = SourceSupportedProseGenerator.score(tokens: nouns, text: "京都旅行案内。")
        let clear = SourceSupportedProseGenerator.score(
            tokens: nouns + [predicate], text: "京都旅行へ行く。")
        let social = SourceSupportedProseGenerator.score(
            tokens: nouns + [predicate], text: "初見しんざんよろしく京都旅行へ行く。")
        #expect(nominal > 0)
        #expect(clear > nominal)
        #expect(clear > social)
    }

    @Test func oversizedReplyPairRetainsResponseInsteadOfTruncatingIt() throws {
        let first = String(repeating: "職業", count: 140) + "仕事。"
        let reply = "いや仕事は真面目に取り組みたい。"
        let input = VideoTranscript(
            title: "Synthetic", movieURL: URL(fileURLWithPath: "/fixture.mov"),
            transcriptURL: URL(fileURLWithPath: "/fixture.json"),
            cues: [
                .init(
                    id: 0, startSeconds: 0, endSeconds: 5,
                    startLabel: "", endLabel: "", text: first + reply)
            ],
            summaryInputKind: .tokenSequence)
        let layout = try #require(RepresentativeSourceSummary.layouts(input)[0])
        let candidates = try SourceSupportedProseGenerator.candidates(
            input.summary.sections[0], morphology: morphology(input).sections[0], layout: layout)
        #expect(candidates.count == 1)
        #expect(candidates.first?.span.text == reply)
        #expect(candidates.first?.startSeconds == 0)
        #expect(candidates.first?.endSeconds == 5)
    }

    @Test func overviewMustUseSelectedOriginalSpeechInSourceOrder() async throws {
        let input = video(
            [
                "電車で京都へ行く予定を話している。", "荷物は明日準備する予定。", "予約はまだしていない。",
            ], sections: true)
        let good = try await result(input)
        let reordered = Array(good.overviewEvidence.reversed())
        let bad = SourceSupportedProse(
            version: 1, method: good.method, label: good.label, fingerprint: "pin",
            morphologyFingerprint: good.morphologyFingerprint,
            clockDomain: .editedOutput, overview: try SourceSupportedProse.paragraph(reordered),
            overviewEvidence: reordered, sections: good.sections)
        #expect(throws: SourceExcerptError.invalidEvidence) {
            try bad.validate(
                video: input, fingerprint: "pin", clockDomain: .editedOutput,
                morphology: morphology(input))
        }
    }

    @Test func morphologyCliRequiresCompatibleModeAndRejectsUnsafeOrChangedEvidence() async throws {
        for args in [
            [
                "generate", "--caption-tokens", "/manifest", "--excerpts-output", "/out",
                "--excerpt-selection", "supported-prose",
            ],
            [
                "generate", "--caption-tokens", "/manifest", "--excerpts-output", "/out",
                "--excerpt-selection", "topics", "--morphology", "/pos",
            ],
        ] {
            #expect(throws: SummaryCommandError.arguments) { try SummaryCommand(arguments: args) }
        }
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let labels = fixture.root.appendingPathComponent("labels.json")
        let pos = fixture.root.appendingPathComponent("morphology.json")
        let alias = fixture.root.appendingPathComponent("alias.json")
        let output = fixture.root.appendingPathComponent("prose.json")
        try Data(#"{"labels":[{"start":1,"end":2,"text":"京都へ旅行する予定。"}]}"#.utf8)
            .write(to: labels)
        try Data("{}".utf8).write(to: pos)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: pos)
        func args(_ path: URL) -> [String] {
            [
                "generate", "--source-labels", labels.path,
                "--source-media", fixture.video.movieURL.path,
                "--clock-domain", "source-presentation", "--source-label-layout", "tokens",
                "--excerpts-output", output.path, "--excerpt-selection", "supported-prose",
                "--morphology", path.path,
            ]
        }
        #expect(
            try await SummaryCommand(arguments: args(alias)).run(
                respond: { _ in throw SourceExcerptError.invalidSelection }, output: { _ in }) == 1)
        #expect(!FileManager.default.fileExists(atPath: output.path))
        let source = try SourceLabelTranscript(
            labels: labels, media: fixture.video.movieURL,
            clockDomain: "source-presentation", inputKind: .tokenSequence
        ).load()
        try JSONEncoder().encode(morphology(source, fingerprint: "changed")).write(to: pos)
        #expect(
            try await SummaryCommand(arguments: args(pos)).run(
                respond: { _ in throw SourceExcerptError.invalidSelection }, output: { _ in }) == 1)
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }

    @Test func oversizedClausesAreNotBrokenIntoMisleadingFragments() async throws {
        let text = (0..<30).map { "\($0)番の電車で京都へ旅行する予定を話している" }.joined()
        let input = video([text])
        let layout = try #require(RepresentativeSourceSummary.layouts(input)[0])
        let candidates = try SourceSupportedProseGenerator.candidates(
            input.summary.sections[0], morphology: morphology(input).sections[0], layout: layout)
        #expect(candidates.isEmpty)
        await #expect(throws: SummaryGenerationError.emptyTranscript) { try await result(input) }
        let supported = try await result(video([text, "京都への旅行は雨具を準備する予定。"]))
        #expect(!supported.overview.contains("29番"))
        #expect(supported.overview.contains("雨具"))
    }

    @Test func grammaticalBoundariesKeepQuestionNegationAndModifierChainsTogether() {
        func token(_ text: String, _ pos: String, _ subtype: String, _ form: String = "基本形")
            -> TranscriptMorphology.Token
        {
            .init(
                text: text, lowerCharacter: 0, upperCharacter: text.count, partOfSpeech: pos,
                features: [pos, subtype, "*", "*", "*", form])
        }
        let go = token("行く", "動詞", "自立")
        let particles = token("ね", "助詞", "終助詞")
        let noun = token("予定", "名詞", "一般")
        #expect(!SourceSupportedProseGenerator.completesClause(go, next: noun))
        #expect(!SourceSupportedProseGenerator.completesClause(go, next: token("ない", "助動詞", "*")))
        #expect(!SourceSupportedProseGenerator.completesClause(go, next: token("か", "助詞", "終助詞")))
        #expect(!SourceSupportedProseGenerator.completesClause(go, next: token("いる", "動詞", "非自立")))
        #expect(!SourceSupportedProseGenerator.completesClause(go, next: token("ない", "形容詞", "非自立")))
        #expect(
            !SourceSupportedProseGenerator.completesClause(
                token("行っ", "動詞", "自立", "連用タ接続"), next: noun))
        #expect(!SourceSupportedProseGenerator.completesClause(noun, next: nil))
        #expect(SourceSupportedProseGenerator.completesClause(particles, next: noun))
        #expect(SourceSupportedProseGenerator.completesClause(go, next: token("明らかに", "副詞", "一般")))
        #expect(SourceSupportedProseGenerator.completesClause(go, next: nil))
    }

    @Test func cliSelectionRequestsPreserveOnlyKnownIdsAndRecordPrivateDiagnostics() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let labels = fixture.root.appendingPathComponent("labels.json")
        let pos = fixture.root.appendingPathComponent("morphology.json")
        struct Labels: Encodable {
            struct Label: Encodable {
                let start: Double
                let end: Double
                let text: String
            }
            let labels: [Label]
        }
        try JSONEncoder().encode(
            Labels(
                labels: (0..<12).map { index in
                    .init(
                        start: Double(index * 20), end: Double(index * 20 + 10),
                        text: "\(index)番の電車で京都へ旅行する予定。")
                })
        ).write(to: labels)
        let source = try SourceLabelTranscript(
            labels: labels, media: fixture.video.movieURL,
            clockDomain: "source-presentation", inputKind: .tokenSequence
        ).load()
        try JSONEncoder().encode(morphology(source, fingerprint: SummaryStore.fingerprint(source)))
            .write(to: pos)
        for withDiagnostics in [false, true] {
            let output = fixture.root.appendingPathComponent("prose-\(withDiagnostics).json")
            var args = [
                "generate", "--source-labels", labels.path,
                "--source-media", fixture.video.movieURL.path,
                "--clock-domain", "source-presentation", "--source-label-layout", "tokens",
                "--excerpts-output", output.path, "--excerpt-selection", "supported-prose",
                "--morphology", pos.path,
            ]
            if withDiagnostics {
                args += [
                    "--diagnostics", fixture.root.appendingPathComponent("diagnostics").path,
                    "--include-text",
                ]
            }
            let status = try await SummaryCommand(arguments: args).run(
                respond: { request in
                    #expect(request.selectsSource && request.supportedSpeechSelection)
                    return "[0]"
                }, output: { _ in })
            #expect(status == 0)
            let summary = try JSONDecoder().decode(
                SourceSupportedProse.self, from: Data(contentsOf: output))
            #expect(summary.sections[0].evidence.count == 3)
            #expect(summary.sections[0].evidence.first?.startSeconds == 0)
            #expect(summary.sections[0].evidence.last?.startSeconds == 160)
        }
        #expect(!FileManager.default.fileExists(atPath: fixture.current.path))
    }
}
