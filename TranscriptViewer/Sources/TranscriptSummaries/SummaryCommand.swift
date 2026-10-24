import Foundation
import TranscriptCore

public enum SummaryCommandError: Error, LocalizedError {
    case arguments
    case emptySelection

    public var errorDescription: String? {
        switch self {
        case .arguments: "Invalid arguments. Use transcript-summary --help."
        case .emptySelection:
            "No matching loaded videos. Check the library and transcript selection."
        }
    }
}

/// CLI parsing and execution are testable without launching a GUI or a native model.
public struct SummaryCommand: Sendable {
    public enum Action: String, Sendable {
        case generate, inspect, preview
        case cleanData = "clean-data"
    }
    public enum ExcerptSelection: String, Sendable {
        case model, representative, topics
        case supportedProse = "supported-prose"
    }
    public let action: Action
    public let library: URL?
    public let sourceLabels: SourceLabelTranscript?
    public let captionTokens: CaptionTokenTranscript?
    public let transcript: URL?
    public let limit: Int?
    public let force: Bool
    public let diagnostics: URL?
    public let includeText: Bool
    public let repairSections: Set<Double>
    public let excerptsOutput: URL?
    public let excerptSelection: ExcerptSelection
    public let morphology: URL?
    /// `clean-data`: a published viewer data JSON and where its original is kept first.
    public let cleanData: (data: URL, backup: URL)?

    public static let usage = """
        transcript-summary generate --library DIRECTORY [--transcript FILE] [--limit N]
            [--force] [--diagnostics PRIVATE_DIRECTORY] [--include-text]
        transcript-summary inspect --library DIRECTORY [--transcript FILE] [--limit N]
        transcript-summary clean-data --data VIEWER_DATA_JSON --backup NEW_FILE
        transcript-summary generate|inspect --source-labels JSON --source-media FILE
            --clock-domain source-presentation [--diagnostics DIRECTORY] [--force]

        Uses the same catalog, newest-first order, native model, validation and saved
        summaries as TranscriptViewer. --limit limits whole videos, never transcript input.
        inspect reads saved results without inference. generate reuses matching results.
        --force creates a new revision, retaining previous results on success or failure.
        --repair-sections START,START regenerates only matching section starts and the overview.
        It requires a fingerprint-matching saved result; all other sections are retained.
        Diagnostics contain request phase, interval, sizes, budgets, latency and error codes.
        --include-text additionally stores private prompts/model responses; never publish them.
        Results are saved beside the transcript in FILE.summaries/ (0700; files 0600).
        Exit codes: 0 = all selected videos ready, 1 = missing/failed results, 2 = setup error.
        Source labels require caller-verified source-presentation timestamps, preserved as-is.
        Their summaries are stored beside JSON; this does not publish a Viewer candidate.
        --source-label-layout tokens joins ASR tokens without inserted line breaks or
        cumulative-caption collapse. Default captions preserves the existing subtitle contract.
        Token gaps greater than 1.5 seconds retain a newline; ten-minute start clocks stay intact.
        preview outputs the exact section input JSON without model execution or saved-result mutation.
        --excerpts-output NEW_JSON explicitly selects source excerpts rather than generated prose.
        It requires one source-label input and writes an immutable, fingerprint-bound artifact.
        inspect with this option validates source ranges; it does not mark Viewer summaries ready.
        --caption-tokens MANIFEST uses pinned recognition tokens on their segment/output clock,
        before display-stage expansion. Its summary fingerprint is separate from display SRT.
        --excerpt-selection representative uses deterministic lexical/time diversity with
        canonical tokens, labels the result as unreviewed speech-excerpt summary, and calls no model.
        --excerpt-selection topics adds source-bound keywords and section/overview structure.
        Keywords preserve exact character ranges and recognition clock envelopes; no prose is invented.
        Both deterministic modes retain explicit unverified labels and never replace narrative results.
        --excerpt-selection supported-prose constructs paragraphs from selected original speech.
        The model selects existing IDs only; source ranges and native clocks are independently checked.
        It is labelled unverified source-supported prose, not freely rewritten factual paraphrase.
        This mode requires --morphology JSON from the pinned Japanese morphology producer.
        preview with --excerpt-selection supported-prose --morphology JSON compares legacy
        and context-aware candidate ranges/scores without calling a model or writing a summary.
        Omit --excerpts-output for this candidate-only preview.
        clean-data drops exactly repeated sentences from a published data JSON's summary texts
        (the same removal as saved summaries get), after copying the original to --backup.
        """

    public init(arguments: [String]) throws {
        guard let first = arguments.first, let action = Action(rawValue: first) else {
            throw SummaryCommandError.arguments
        }
        if action == .cleanData {
            guard arguments.count == 5, arguments[1] == "--data", arguments[3] == "--backup",
                !arguments[2].isEmpty, !arguments[4].isEmpty
            else { throw SummaryCommandError.arguments }
            self.action = action
            cleanData = (URL(fileURLWithPath: arguments[2]), URL(fileURLWithPath: arguments[4]))
            (library, sourceLabels, captionTokens, transcript, limit) = (nil, nil, nil, nil, nil)
            (force, diagnostics, includeText, repairSections) = (false, nil, false, [])
            (excerptsOutput, excerptSelection, morphology) = (nil, .model, nil)
            return
        }
        cleanData = nil
        var values: [String: String] = [:]
        var flags: Set<String> = []
        var index = 1
        while index < arguments.count {
            let key = arguments[index]
            if ["--force", "--include-text"].contains(key) {
                guard flags.insert(key).inserted else { throw SummaryCommandError.arguments }
                index += 1
            } else {
                guard
                    [
                        "--library", "--transcript", "--limit", "--diagnostics", "--source-labels",
                        "--source-media", "--clock-domain",
                        "--repair-sections",
                        "--source-label-layout",
                        "--excerpts-output",
                        "--caption-tokens",
                        "--excerpt-selection",
                        "--morphology",
                    ].contains(key),
                    values[key] == nil, index + 1 < arguments.count,
                    !arguments[index + 1].isEmpty, !arguments[index + 1].hasPrefix("--")
                else { throw SummaryCommandError.arguments }
                values[key] = arguments[index + 1]
                index += 2
            }
        }
        if let path = values["--caption-tokens"] {
            guard
                [
                    "--source-labels", "--source-media", "--clock-domain", "--library",
                    "--transcript", "--limit", "--source-label-layout",
                ].allSatisfy({ values[$0] == nil })
            else { throw SummaryCommandError.arguments }
            captionTokens = CaptionTokenTranscript(manifest: URL(fileURLWithPath: path))
            sourceLabels = nil
            library = nil
        } else if let labels = values["--source-labels"], let media = values["--source-media"],
            let domain = values["--clock-domain"], values["--library"] == nil,
            values["--transcript"] == nil, values["--limit"] == nil
        {
            let layout = values["--source-label-layout"] ?? "captions"
            guard ["tokens", "captions"].contains(layout) else {
                throw SummaryCommandError.arguments
            }
            sourceLabels = try SourceLabelTranscript(
                labels: URL(fileURLWithPath: labels), media: URL(fileURLWithPath: media),
                clockDomain: domain,
                inputKind: layout == "tokens" ? .tokenSequence : .cumulativeCaptions)
            library = nil
            captionTokens = nil
        } else {
            guard let root = values["--library"], values["--source-labels"] == nil,
                values["--source-media"] == nil, values["--clock-domain"] == nil,
                values["--source-label-layout"] == nil
            else { throw SummaryCommandError.arguments }
            sourceLabels = nil
            captionTokens = nil
            library = URL(fileURLWithPath: root).standardizedFileURL
        }
        let limit = values["--limit"].flatMap(Int.init)
        if let selection = values["--repair-sections"] {
            let fields = selection.split(separator: ",", omittingEmptySubsequences: false)
            let numbers = fields.compactMap { Double($0) }
            guard !numbers.isEmpty, numbers.count == fields.count, numbers.count <= 100,
                numbers.allSatisfy({ $0.isFinite && $0 >= 0 }),
                Set(numbers).count == numbers.count, action == .generate, !flags.contains("--force")
            else { throw SummaryCommandError.arguments }
            repairSections = Set(numbers)
        } else {
            repairSections = []
        }
        if values["--limit"] != nil, limit.map({ $0 > 0 }) != true {
            throw SummaryCommandError.arguments
        }
        guard !flags.contains("--include-text") || values["--diagnostics"] != nil,
            action == .generate || (flags.isEmpty && values["--diagnostics"] == nil)
        else { throw SummaryCommandError.arguments }
        self.action = action
        transcript = values["--transcript"].map { URL(fileURLWithPath: $0).standardizedFileURL }
        self.limit = limit
        force = flags.contains("--force")
        diagnostics = values["--diagnostics"].map { URL(fileURLWithPath: $0).standardizedFileURL }
        includeText = flags.contains("--include-text")
        excerptsOutput = values["--excerpts-output"].map { URL(fileURLWithPath: $0) }
        let selection = values["--excerpt-selection"] ?? "model"
        guard let excerptSelection = ExcerptSelection(rawValue: selection),
            values["--excerpt-selection"] == nil || excerptsOutput != nil
                || (action == .preview && excerptSelection == .supportedProse)
        else { throw SummaryCommandError.arguments }
        self.excerptSelection = excerptSelection
        morphology = values["--morphology"].map { URL(fileURLWithPath: $0) }
        guard (excerptSelection == .supportedProse) == (morphology != nil) else {
            throw SummaryCommandError.arguments
        }
        if excerptsOutput != nil {
            guard sourceLabels != nil || captionTokens != nil,
                action != .preview, !force, repairSections.isEmpty
            else {
                throw SummaryCommandError.arguments
            }
        }
    }

    public func run(
        respond: @escaping NarrativeSummaryGenerator.Respond = AppleSummaryService.respond,
        output: @escaping @Sendable (String) -> Void = { print($0) }
    ) async throws -> Int32 {
        if let cleanData {
            let removed = try await SummaryStore.io {
                try PublishedSummaryCleaner.clean(data: cleanData.data, backup: cleanData.backup)
            }
            output("CLEANED removed_characters=\(removed)")
            return 0
        }
        let scan = try await SummaryStore.io {
            if let sourceLabels {
                return (videos: [try sourceLabels.load()], issueCount: 0)
            }
            if let captionTokens {
                return (videos: [try captionTokens.load()], issueCount: 0)
            }
            guard let library else { throw SummaryCommandError.arguments }
            let catalog = try TranscriptCatalog.scan(in: library)
            return (videos: catalog.videos, issueCount: catalog.issues.count)
        }
        var videos = AppleSummaryService.newestFirst(scan.videos)
        if let transcript {
            videos = videos.filter { $0.transcriptURL.standardizedFileURL == transcript }
        }
        if let limit { videos = Array(videos.prefix(limit)) }
        guard !videos.isEmpty else { throw SummaryCommandError.emptySelection }
        if let excerptsOutput {
            guard excerptsOutput.path == excerptsOutput.standardizedFileURL.path,
                excerptsOutput.path == excerptsOutput.resolvingSymlinksInPath().path
            else { throw SummaryStoreError.unsafePath }
            try SummaryStore.checkDirectory(excerptsOutput.deletingLastPathComponent())
            if action == .generate,
                FileManager.default.fileExists(atPath: excerptsOutput.path)
            {
                throw SummaryStoreError.invalidRecord
            }
        }
        if action == .preview {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            for video in videos {
                let data: Data
                if excerptSelection == .supportedProse {
                    let fingerprint = try SummaryStore.fingerprint(video)
                    guard let bound = try await boundMorphology(video, fingerprint: fingerprint)
                    else {
                        throw SummaryCommandError.arguments
                    }
                    data = try encoder.encode(
                        SourceSupportedProseGenerator.preview(
                            video: video, fingerprint: fingerprint, morphology: bound))
                } else {
                    data = try encoder.encode(video.summary)
                }
                output(String(decoding: data, as: UTF8.self))
            }
            return 0
        }
        let trace: SummaryDiagnostics?
        if let diagnostics {
            trace = try await SummaryDiagnostics.create(
                in: diagnostics, includeText: includeText, provider: respond)
        } else {
            trace = nil
        }
        let service = AppleSummaryService(store: SummaryStore()) { request in
            if let trace { return try await trace.respond(request) }
            return try await respond(request)
        }
        output(
            "SELECTED \(videos.count) CATALOG_ISSUES \(scan.issueCount) PIPELINE \(SummaryStore.pipelineVersion)"
        )
        if let trace { output("DIAGNOSTICS \(trace.directory.path)") }
        var failures = 0
        let total = videos.count
        for (offset, video) in videos.enumerated() {
            try Task.checkCancellation()
            let index = offset + 1
            try await trace?.beginVideo(video, index: index)
            output("VIDEO \(index)/\(total) DATE \(video.broadcastStartJST ?? "unknown")")
            do {
                if let excerptsOutput {
                    try await sourceExcerpts(
                        video, output: excerptsOutput, trace: trace,
                        respond: respond)
                    let verification =
                        excerptSelection == .supportedProse
                        ? "SOURCE_SUPPORTED_PROSE_VERIFIED" : "SOURCE_EXCERPTS_VERIFIED"
                    output("\(verification) \(index)")
                } else if action == .inspect {
                    if try await SummaryStore().load(video: video) != nil {
                        output("READY \(index)")
                    } else {
                        failures += 1
                        output("MISSING \(index)")
                    }
                } else {
                    _ = try await service.generate(
                        video: video, force: force, repairing: repairSections
                    ) { done, sections in
                        output("PROGRESS \(index)/\(total) SECTIONS \(done)/\(sections)")
                    }
                    output("READY \(index)")
                }
                try await trace?.finishVideo(index: index, errorCode: nil)
            } catch is CancellationError {
                try await trace?.finishVideo(index: index, errorCode: "cancelled")
                throw CancellationError()
            } catch {
                failures += 1
                let code = SummaryDiagnostics.errorCode(error)
                try await trace?.finishVideo(index: index, errorCode: code)
                output("FAILED \(index) CODE \(code)")
            }
        }
        output("FINISHED total=\(total) ready=\(total - failures) failed=\(failures)")
        return failures == 0 ? 0 : 1
    }

    private func boundMorphology(_ video: VideoTranscript, fingerprint: String) async throws
        -> TranscriptMorphology?
    {
        guard let morphology else { return nil }
        return try await SummaryStore.io {
            guard morphology.path == morphology.standardizedFileURL.path,
                morphology.path == morphology.resolvingSymlinksInPath().path,
                !morphology.pathComponents.contains(where: {
                    ["private", "downloads"].contains($0)
                }),
                try morphology.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile
                    == true
            else { throw SummaryStoreError.unsafePath }
            let handle = try FileHandle(forReadingFrom: morphology)
            let read = Result { try handle.read(upToCount: 64 * 1024 * 1024 + 1) ?? Data() }
            try handle.close()
            let bytes = try read.get()
            guard bytes.count <= 64 * 1024 * 1024 else {
                throw SummaryStoreError.oversizedRecord
            }
            let parsed = try JSONDecoder().decode(TranscriptMorphology.self, from: bytes)
            try parsed.validate(source: video.summary, fingerprint: fingerprint)
            return parsed
        }
    }

    private func sourceExcerpts(
        _ video: VideoTranscript, output: URL, trace: SummaryDiagnostics?,
        respond: @escaping NarrativeSummaryGenerator.Respond
    ) async throws {
        let fingerprint = try SummaryStore.fingerprint(video)
        let clockDomain: RepresentativeSourceSummary.ClockDomain =
            captionTokens == nil ? .sourcePresentation : .editedOutput
        let boundMorphology = try await boundMorphology(video, fingerprint: fingerprint)
        if action == .inspect {
            let data = try await SummaryStore.io {
                guard try output.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true
                else { throw SummaryStoreError.unsafePath }
                let handle = try FileHandle(forReadingFrom: output)
                let read = Result { try handle.read(upToCount: 8 * 1024 * 1024 + 1) ?? Data() }
                try handle.close()
                let data = try read.get()
                guard data.count <= 8 * 1024 * 1024 else {
                    throw SummaryStoreError.oversizedRecord
                }
                return data
            }
            switch excerptSelection {
            case .supportedProse:
                guard let boundMorphology else { throw SummaryCommandError.arguments }
                try JSONDecoder().decode(SourceSupportedProse.self, from: data)
                    .validate(
                        video: video, fingerprint: fingerprint, clockDomain: clockDomain,
                        morphology: boundMorphology)
            case .topics:
                try JSONDecoder().decode(SourceTopicSummary.self, from: data)
                    .validate(video: video, fingerprint: fingerprint, clockDomain: clockDomain)
            case .representative:
                try JSONDecoder().decode(RepresentativeSourceSummary.self, from: data)
                    .validate(video: video, fingerprint: fingerprint, clockDomain: clockDomain)
            case .model:
                try JSONDecoder().decode(SourceExcerptSummary.self, from: data)
                    .validate(source: video.summary, fingerprint: fingerprint)
            }
        } else {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
            let data: Data
            switch excerptSelection {
            case .supportedProse:
                guard let boundMorphology else { throw SummaryCommandError.arguments }
                let result = try await SourceSupportedProseGenerator(morphology: boundMorphology) {
                    request in
                    if let trace { return try await trace.respond(request) }
                    return try await respond(request)
                }.generate(video: video, fingerprint: fingerprint, clockDomain: clockDomain)
                data = try encoder.encode(result)
            case .topics:
                data = try encoder.encode(
                    SourceTopicSummary.generate(
                        video: video, fingerprint: fingerprint, clockDomain: clockDomain))
            case .representative:
                data = try encoder.encode(
                    RepresentativeSourceSummary.generate(
                        video: video, fingerprint: fingerprint, clockDomain: clockDomain))
            case .model:
                let result = try await SourceExcerptGenerator { request in
                    if let trace { return try await trace.respond(request) }
                    return try await respond(request)
                }.generate(video.summary, fingerprint: fingerprint)
                data = try encoder.encode(result)
            }
            try await SummaryStore.io {
                try SummaryStore.writePrivate(data, to: output, atomic: false)
            }
        }
    }
}
