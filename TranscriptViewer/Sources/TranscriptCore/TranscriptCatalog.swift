import Foundation

/// Result of scanning a library folder. Unloadable entries are kept here so a
/// single incomplete delivery never hides the readable ones.
public struct CatalogScan: Equatable, Sendable {
    public init() {}

    public var videos: [VideoTranscript] = []
    /// A transcript whose movie has not landed yet (for example while
    /// Compressor is still encoding).
    public var pendingMovieLabels: [String] = []
    /// Movies that have no matching transcript CSV.
    public var moviesWithoutTranscript: [URL] = []
    /// Transcript CSVs that could not be read, with the reason.
    public var unreadableTranscriptLabels: [String] = []
    /// A source probe exists but lacks a validated broadcast timestamp.
    public var unknownBroadcastDateLabels: [String] = []
    /// Multiple deliveries without a unique canonical, dated selection.
    public var ambiguousDeliveryLabels: [String] = []
    /// Supersession records that could not be read; their deliveries stay eligible.
    public var invalidSupersededRecordLabels: [String] = []
    /// Invalid explicit revision selections fail closed for that source.
    public var invalidCandidateSelectionLabels: [String] = []
    /// Dated folders holding neither a transcript CSV nor a movie, so the
    /// pipeline never delivered anything for them.
    public var emptyFolders: [String] = []

    /// Human-readable list of unprocessed entries; empty means nothing is missing.
    public var issues: [String] {
        pendingMovieLabels.map { "動画待ち: \($0)" }
            + moviesWithoutTranscript.map { "字幕なし: \($0.lastPathComponent)" }
            + unreadableTranscriptLabels.map { "読み込み不可: \($0)" }
            + unknownBroadcastDateLabels.map { "配信日時不明: \($0)" }
            + ambiguousDeliveryLabels.map { "成果物が重複: \($0)" }
            + invalidSupersededRecordLabels.map { "置き換え記録が不正: \($0)" }
            + invalidCandidateSelectionLabels.map { "候補の選択が不正: \($0)" }
            + emptyFolders.map { "未処理: \($0)" }
    }

    /// Add another library; a valid current candidate replaces older deliveries of
    /// the same dated recording. Unrelated entries remain, with newest broadcasts first.
    public mutating func merge(_ other: CatalogScan) {
        for candidate in other.videos where candidate.isUnverifiedCandidate {
            guard let recordID = candidate.recordingID else { continue }
            let previous = videos.filter { $0.recordingID == recordID }
            guard previous.allSatisfy({ $0.broadcastStartJST == candidate.broadcastStartJST })
            else {
                ambiguousDeliveryLabels.append(recordID)
                continue
            }
            videos.removeAll { $0.recordingID == recordID }
        }
        let known = Set(videos.map(\.id))
        videos += other.videos.filter {
            !known.contains($0.id)
                && !($0.isUnverifiedCandidate
                    && ambiguousDeliveryLabels.contains($0.recordingID ?? ""))
        }
        videos.sort { left, right in
            let leftDate = left.broadcastStartJST ?? ""
            let rightDate = right.broadcastStartJST ?? ""
            return leftDate == rightDate
                ? left.movieURL.path < right.movieURL.path : leftDate > rightDate
        }
        pendingMovieLabels += other.pendingMovieLabels
        // A movie another library pairs with its transcript (an offloaded store) is not missing one.
        let paired = Set(videos.map { $0.movieURL.standardizedFileURL.path })
        moviesWithoutTranscript = (moviesWithoutTranscript + other.moviesWithoutTranscript)
            .filter { !paired.contains($0.standardizedFileURL.path) }
        unreadableTranscriptLabels += other.unreadableTranscriptLabels
        unknownBroadcastDateLabels += other.unknownBroadcastDateLabels
        ambiguousDeliveryLabels += other.ambiguousDeliveryLabels
        invalidSupersededRecordLabels += other.invalidSupersededRecordLabels
        invalidCandidateSelectionLabels += other.invalidCandidateSelectionLabels
        emptyFolders += other.emptyFolders
    }
}

public enum TranscriptCatalog {
    private struct SourceProbe: Decodable {
        struct Format: Decodable { let filename: String }
        let format: Format
    }

    /// The fields of a record mirror's `edit-config.json` that locate its delivery.
    private struct EditConfig: Decodable {
        let source: String?
        let storageRoot: String?

        enum CodingKeys: String, CodingKey {
            case source
            case storageRoot = "storage_root"
        }
    }

    /// A delivery that `replay.edit_completion supersede` recorded as missing a
    /// required edit stage. The file is kept as evidence, never shown as current.
    private struct SupersededDelivery: Decodable {
        let delivery: String
        let size: Int
    }

    /// Use the checkout's edit library only for an app built in its dist folder.
    public static func checkoutLibrary(near bundle: URL, fileManager: FileManager = .default)
        -> URL?
    {
        guard bundle.pathExtension == "app",
            bundle.deletingLastPathComponent().lastPathComponent == "dist"
        else { return nil }
        let package = bundle.deletingLastPathComponent().deletingLastPathComponent()
        guard package.lastPathComponent == "TranscriptViewer" else { return nil }
        let edits = package.deletingLastPathComponent()
            .appending(path: "edits", directoryHint: .isDirectory)
        var isDirectory: ObjCBool = false
        guard
            fileManager.fileExists(
                atPath: edits.path(percentEncoded: false), isDirectory: &isDirectory),
            isDirectory.boolValue
        else { return nil }
        return edits
    }

    public static func scan(in root: URL, fileManager: FileManager = .default) throws -> CatalogScan
    {
        guard
            !root.resolvingSymlinksInPath().pathComponents.contains(where: {
                ["private", "downloads"].contains($0)
            })
        else {
            throw TranscriptError.unreadableDirectory(root.path(percentEncoded: false))
        }
        let contents: [URL]
        do {
            contents = try fileManager.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
        } catch {
            throw TranscriptError.unreadableDirectory(root.path(percentEncoded: false))
        }
        var scan = CatalogScan()
        let rootName = root.lastPathComponent
        // Match the repository's record-ID contract. A source root is one record,
        // so its revisions must not also enter as independent generic folders.
        let sourceRoot = RecordingIdentity.isRecordingID(rootName)
        let narrowRecording = recordingFolder(for: root)
        let folders: [URL]
        if let narrowRecording {
            folders = [narrowRecording]
        } else {
            folders =
                sourceRoot
                ? [root]
                : contents.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) + [root]
        }
        for folder in folders {
            guard
                !folder.resolvingSymlinksInPath().pathComponents.contains(where: {
                    ["private", "downloads"].contains($0)
                })
            else { continue }
            var isDirectory: ObjCBool = false
            guard
                folder == root
                    || (fileManager.fileExists(
                        atPath: folder.path(percentEncoded: false), isDirectory: &isDirectory)
                        && isDirectory.boolValue)
            else { continue }
            let title = folder.lastPathComponent
            // A folder of recording folders is another library's media store, not a delivery.
            if folder != root, !RecordingIdentity.isRecordingID(title),
                holdsRecordingFolders(folder, fileManager: fileManager)
            {
                continue
            }
            if root.lastPathComponent == "edits",
                folder == root || (!title.hasPrefix("replay-") && !title.hasPrefix("youtube-"))
            {
                continue
            }
            var broadcastStartJST: String?
            let probeURL = folder.appending(path: "evidence/source-probe.json")
            if fileManager.fileExists(atPath: probeURL.path(percentEncoded: false)) {
                do {
                    broadcastStartJST = try validatedBroadcastStartJST(
                        in: folder, probeURL: probeURL, fileManager: fileManager)
                } catch {
                    // Report invalid evidence only for folders that actually contain deliveries.
                    broadcastStartJST = nil
                }
            }
            let config = editConfig(in: folder)
            if broadcastStartJST == nil, title.hasPrefix("replay-"), let source = config?.source {
                broadcastStartJST = sourceStartJST(source, folder: folder)
            }
            let store = config?.storageRoot.flatMap {
                offloadedStore($0, for: folder, fileManager: fileManager)
            }
            let candidate: CandidateSelection.Selection?
            do {
                candidate = try CandidateSelection.load(
                    in: folder, library: narrowRecording ?? root, broadcast: broadcastStartJST,
                    fileManager: fileManager)
                if narrowRecording != nil {
                    let delivery =
                        root.deletingLastPathComponent().lastPathComponent == "revisions"
                        ? root.appending(path: "delivery") : root
                    guard
                        candidate?.movie.deletingLastPathComponent().standardizedFileURL.path
                            == delivery.standardizedFileURL.path
                    else {
                        throw CandidateSelection.Invalid.provenance
                    }
                }
            } catch {
                scan.invalidCandidateSelectionLabels.append(title)
                continue
            }
            let transcripts = try transcriptURLs(in: folder, fileManager: fileManager)
            var storedTranscripts: [URL: URL] = [:]
            if let store, candidate == nil {
                for transcript in try movieCandidates(
                    in: folder, extensions: ["csv", "srt"], fileManager: fileManager)
                {
                    let movie = store.appending(
                        path: transcript.pathComponents.dropFirst(folder.pathComponents.count)
                            .joined(separator: "/")
                    ).deletingPathExtension().appendingPathExtension("mp4")
                    var isDirectory: ObjCBool = false
                    if fileManager.fileExists(
                        atPath: movie.path(percentEncoded: false), isDirectory: &isDirectory),
                        !isDirectory.boolValue,
                        !movie.resolvingSymlinksInPath().pathComponents.contains(where: {
                            ["private", "downloads"].contains($0)
                        })
                    {
                        // Sorted paths put .srt after .csv, so the SRT wins as in transcriptURL.
                        storedTranscripts[movie] = transcript
                    }
                }
            }
            let base = { (movie: URL) in storedTranscripts[movie] == nil ? folder : store ?? folder
            }
            let discoveredMovies: [URL]
            if let candidate {
                discoveredMovies = [candidate.movie]
            } else if folder == root && !sourceRoot {
                discoveredMovies = contents.filter { item in
                    var isDirectory: ObjCBool = false
                    return ["mov", "mp4"].contains(item.pathExtension.lowercased())
                        && !item.deletingPathExtension().lastPathComponent.hasSuffix(".incomplete")
                        && fileManager.fileExists(
                            atPath: item.path(percentEncoded: false), isDirectory: &isDirectory)
                        && !isDirectory.boolValue
                }
            } else {
                discoveredMovies =
                    try movieCandidates(in: folder, fileManager: fileManager)
                    + storedTranscripts.keys.sorted { $0.path < $1.path }
            }
            if title.hasPrefix("replay-"), broadcastStartJST == nil, !discoveredMovies.isEmpty {
                scan.unknownBroadcastDateLabels.append(title)
            }
            // Only a dated, provenance-bound replay can enter the processed library.
            // Keep revisions on disk as evidence; expose one stable delivery per source.
            let movies: [URL]
            if let candidate {
                movies = [candidate.movie]
            } else if title.hasPrefix("replay-") {
                let eligible = broadcastStartJST == nil ? [] : discoveredMovies
                let stamp = broadcastStartJST?
                    .replacingOccurrences(of: " JST", with: "")
                    .replacingOccurrences(of: " ", with: "_")
                    .replacingOccurrences(of: ":", with: "-")
                let superseded = supersededDeliveries(in: folder, fileManager: fileManager)
                if !superseded.invalidRecords.isEmpty {
                    scan.invalidSupersededRecordLabels.append(
                        contentsOf: superseded.invalidRecords.map { "\(title)/\($0)" })
                }
                let canonical = eligible.filter {
                    $0.lastPathComponent == "\(stamp ?? "")_JST_\(title)-review.mp4"
                        && !superseded.matches($0, in: base($0), fileManager: fileManager)
                }
                let selected = canonical.isEmpty ? eligible : canonical
                if selected.count > 1 {
                    scan.ambiguousDeliveryLabels.append(title)
                    movies = []
                } else {
                    movies = selected
                }
            } else if title.hasPrefix("youtube-") {
                // A YouTube edit is named after its validated publication time, like replays.
                let canonical = discoveredMovies.filter {
                    youtubeStartJST(movie: $0, title: title) != nil
                }
                let reviews = canonical.filter {
                    $0.deletingPathExtension().lastPathComponent.hasSuffix("_JST_\(title)-review")
                }
                movies = reviews.isEmpty ? canonical : reviews
                if movies.isEmpty, !discoveredMovies.isEmpty {
                    scan.unknownBroadcastDateLabels.append(title)
                }
            } else {
                movies = discoveredMovies
            }
            let movieStems = Set(movies.map { $0.deletingPathExtension().lastPathComponent })
            for transcript in transcripts
            where !movieStems.contains(transcript.deletingPathExtension().lastPathComponent) {
                scan.pendingMovieLabels.append(
                    "\(title)/\(transcript.deletingPathExtension().lastPathComponent)")
            }
            for movieURL in movies {
                guard
                    let transcript = candidate?.transcript ?? storedTranscripts[movieURL]
                        ?? transcriptURL(
                            for: movieURL, in: folder, fileManager: fileManager)
                else {
                    scan.moviesWithoutTranscript.append(movieURL)
                    continue
                }
                do {
                    let cues =
                        try transcript.pathExtension.lowercased() == "srt"
                        ? TranscriptSRT.cues(contentsOf: transcript)
                        : TranscriptCSV.cues(contentsOf: transcript)
                    let relative = Array(
                        movieURL.pathComponents.dropFirst(base(movieURL).pathComponents.count))
                    let revision = zip(relative, relative.dropFirst())
                        .first { $0.0 == "revisions" }?.1
                    let label = revision.map { "\(title)/\($0)" } ?? title
                    scan.videos.append(
                        VideoTranscript(
                            title: candidate == nil ? label : "\(label) [未検証候補]",
                            movieURL: movieURL, transcriptURL: transcript, cues: cues,
                            broadcastStartJST: candidate?.identity.broadcastStartJST
                                ?? broadcastStartJST
                                ?? youtubeStartJST(movie: movieURL, title: title),
                            isUnverifiedCandidate: candidate != nil)
                    )
                } catch let error as TranscriptError {
                    scan.unreadableTranscriptLabels.append(
                        "\(title)/\(transcript.lastPathComponent) (\(error.message))"
                    )
                } catch {
                    scan.unreadableTranscriptLabels.append(
                        "\(title)/\(transcript.lastPathComponent) (\(error.localizedDescription))"
                    )
                }
            }
            if folder != root, candidate == nil, transcripts.isEmpty, discoveredMovies.isEmpty,
                !(root.lastPathComponent == "edits" && title.hasPrefix("replay-")),
                // A record mirror's deliveries are reported by the library that stores them.
                config == nil
            {
                scan.emptyFolders.append(title)
            }
        }
        scan.videos.sort { left, right in
            let leftDate = left.broadcastStartJST ?? ""
            let rightDate = right.broadcastStartJST ?? ""
            return leftDate == rightDate
                ? left.movieURL.path < right.movieURL.path : leftDate > rightDate
        }
        return scan
    }

    /// A narrow delivery/revision can read only its exact recording's selection.
    /// The selected lexical delivery must still match the opened folder; its
    /// pinned media target may be a sibling compressor output in that recording.
    private static func recordingFolder(for root: URL) -> URL? {
        let parent = root.deletingLastPathComponent()
        let revision =
            root.lastPathComponent == "delivery"
                && parent.deletingLastPathComponent().lastPathComponent == "revisions"
            ? parent : root
        let revisions = revision.deletingLastPathComponent()
        let recording = revisions.deletingLastPathComponent()
        guard revisions.lastPathComponent == "revisions",
            RecordingIdentity.isRecordingID(recording.lastPathComponent),
            recording.resolvingSymlinksInPath() == recording.standardizedFileURL
        else { return nil }
        return recording
    }

    /// Scan off the UI thread and deliver the result on the main actor.
    public static func scan(
        in root: URL,
        completion: @escaping @MainActor @Sendable (Result<CatalogScan, Error>) -> Void
    ) {
        DispatchQueue.global(qos: .userInitiated).async {
            let result = Result { try scan(in: root) }
            DispatchQueue.main.async { completion(result) }
        }
    }

    private static func validatedBroadcastStartJST(
        in folder: URL, probeURL: URL, fileManager: FileManager
    ) throws -> String? {
        let expectedEvidence = folder.resolvingSymlinksInPath()
            .appending(path: "evidence", directoryHint: .isDirectory)
        guard folder.lastPathComponent.hasPrefix("replay-"),
            probeURL.resolvingSymlinksInPath().deletingLastPathComponent()
                == expectedEvidence.standardizedFileURL,
            let size = try fileManager.attributesOfItem(
                atPath: probeURL.path(percentEncoded: false))[
                    .size] as? NSNumber,
            size.intValue > 0, size.intValue <= 1_048_576
        else { return nil }
        let source = try JSONDecoder().decode(
            SourceProbe.self, from: Data(contentsOf: probeURL)
        ).format.filename
        return sourceStartJST(source, folder: folder)
    }

    /// The broadcast time named by the original's canonical download filename.
    private static func sourceStartJST(_ source: String, folder: URL) -> String? {
        let name = URL(fileURLWithPath: source).lastPathComponent
        guard source.hasPrefix("/") || source == "downloads/\(name)" else { return nil }
        let suffix = "_JST_\(folder.lastPathComponent).mp4"
        guard name.hasSuffix(suffix) else { return nil }
        return jstLabel(stamp: String(name.dropLast(suffix.count)))
    }

    /// "YYYY-MM-DD_HH-mm-ss" in JST as "YYYY-MM-DD HH:mm:ss JST"; nil unless canonical.
    static func jstLabel(stamp: String) -> String? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        guard let jst = TimeZone(secondsFromGMT: 9 * 3_600) else { return nil }
        formatter.timeZone = jst
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        formatter.isLenient = false
        guard stamp.count == 19, let date = formatter.date(from: stamp),
            formatter.string(from: date) == stamp
        else { return nil }
        let parts = stamp.split(separator: "_")
        guard parts.count == 2 else { return nil }
        return "\(parts[0]) \(parts[1].replacingOccurrences(of: "-", with: ":")) JST"
    }

    /// Date from "<stamp>_JST_youtube-<id>-…" movie names inside youtube-<id>.
    static func youtubeStartJST(movie: URL, title: String) -> String? {
        guard title.hasPrefix("youtube-") else { return nil }
        let name = movie.deletingPathExtension().lastPathComponent
        let marker = "_JST_\(title)"
        guard let range = name.range(of: marker),
            name[range.upperBound...].isEmpty || name[range.upperBound...].hasPrefix("-")
        else { return nil }
        return jstLabel(stamp: String(name[..<range.lowerBound]))
    }

    private static func editConfig(in folder: URL) -> EditConfig? {
        let url = folder.appending(path: "edit-config.json")
        guard url.resolvingSymlinksInPath() == url.standardizedFileURL,
            let data = try? Data(contentsOf: url), data.count <= 1_048_576
        else { return nil }
        return try? JSONDecoder().decode(EditConfig.self, from: data)
    }

    /// A record mirror's storage root that holds only media: its own evidence would make
    /// its library scan authoritative, so a mirror never re-selects one of its revisions.
    private static func offloadedStore(_ path: String, for folder: URL, fileManager: FileManager)
        -> URL?
    {
        let store = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard path.hasPrefix("/"), store.lastPathComponent == folder.lastPathComponent,
            store.resolvingSymlinksInPath() != folder.resolvingSymlinksInPath(),
            !store.resolvingSymlinksInPath().pathComponents.contains(where: {
                ["private", "downloads"].contains($0)
            }),
            fileManager.fileExists(
                atPath: store.path(percentEncoded: false), isDirectory: &isDirectory),
            isDirectory.boolValue,
            !fileManager.fileExists(
                atPath: store.appending(path: "evidence").path(percentEncoded: false))
        else { return nil }
        return store
    }

    private static func holdsRecordingFolders(_ folder: URL, fileManager: FileManager) -> Bool {
        let names =
            (try? fileManager.contentsOfDirectory(atPath: folder.path(percentEncoded: false))) ?? []
        return names.contains(where: RecordingIdentity.isRecordingID)
    }

    private static func transcriptURLs(in folder: URL, fileManager: FileManager) throws -> [URL] {
        try fileManager.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        .filter { ["csv", "srt"].contains($0.pathExtension.lowercased()) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private static func transcriptURL(for movie: URL, in root: URL, fileManager: FileManager)
        -> URL?
    {
        let stem = movie.deletingPathExtension().lastPathComponent
        var directory = movie.deletingLastPathComponent()
        let rootPath = root.standardizedFileURL.path
        while directory.standardizedFileURL.path == rootPath
            || directory.standardizedFileURL.path.hasPrefix(
                rootPath.hasSuffix("/") ? rootPath : "\(rootPath)/")
        {
            for ext in ["srt", "csv"] {
                let candidate = directory.appending(path: "\(stem).\(ext)")
                if fileManager.fileExists(atPath: candidate.path(percentEncoded: false)) {
                    return candidate
                }
            }
            if directory.standardizedFileURL == root.standardizedFileURL
                || directory.deletingLastPathComponent().lastPathComponent == "revisions"
            {
                break
            }
            directory = directory.deletingLastPathComponent()
        }
        return nil
    }

    /// Superseded deliveries of one replay folder, keyed by folder-relative path.
    struct SupersededSet: Equatable, Sendable {
        var sizesByRelativePath: [String: Int] = [:]
        var invalidRecords: [String] = []

        /// A movie is superseded only while its recorded path and size still match.
        /// A delivery alias is sized by the movie it names, as it is when found eligible.
        func matches(_ movie: URL, in folder: URL, fileManager: FileManager) -> Bool {
            let relative = movie.pathComponents.dropFirst(folder.pathComponents.count)
                .joined(separator: "/")
            guard let recordedSize = sizesByRelativePath[relative] else { return false }
            do {
                let attributes = try fileManager.attributesOfItem(
                    atPath: movie.resolvingSymlinksInPath().path(percentEncoded: false))
                // An unreadable or changed file is not the recorded one, so it stays eligible.
                return (attributes[.size] as? Int) == recordedSize
            } catch {
                return false
            }
        }
    }

    static func supersededDeliveries(in folder: URL, fileManager: FileManager) -> SupersededSet {
        let directory = folder.appending(path: "evidence/superseded")
        var result = SupersededSet()
        var isDirectory: ObjCBool = false
        guard
            fileManager.fileExists(
                atPath: directory.path(percentEncoded: false), isDirectory: &isDirectory)
        else { return result }
        let names: [String]
        do {
            guard isDirectory.boolValue else {
                throw TranscriptError.unreadableDirectory(directory.path)
            }
            names = try fileManager.contentsOfDirectory(
                atPath: directory.path(percentEncoded: false))
        } catch {
            result.invalidRecords.append("evidence/superseded")
            return result
        }
        for name in names.sorted() where name.hasSuffix(".json") {
            let record = directory.appending(path: name)
            do {
                let decoded = try JSONDecoder().decode(
                    SupersededDelivery.self, from: Data(contentsOf: record))
                result.sizesByRelativePath[decoded.delivery] = decoded.size
            } catch {
                result.invalidRecords.append(name)
            }
        }
        return result
    }

    private static func movieCandidates(
        in folder: URL, extensions: Set<String> = ["mov", "mp4"], fileManager: FileManager
    ) throws -> [URL] {
        let enumerator = fileManager.enumerator(
            at: folder,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        var urls: [URL] = []
        while let item = enumerator?.nextObject() as? URL {
            if ["private", "downloads", "work", "audio", "evidence", "transcripts"].contains(
                item.lastPathComponent
            ) {
                // Enumerators do not descend symlinks. Skipping one instead skips the
                // following real directory, which can be the completed delivery.
                if try item.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true {
                    enumerator?.skipDescendants()
                }
                continue
            }
            var isDirectory: ObjCBool = false
            guard extensions.contains(item.pathExtension.lowercased()),
                !item.deletingPathExtension().lastPathComponent.hasSuffix(".incomplete"),
                !item.resolvingSymlinksInPath().pathComponents.contains(where: {
                    ["private", "downloads"].contains($0)
                }),
                fileManager.fileExists(
                    atPath: item.path(percentEncoded: false), isDirectory: &isDirectory),
                !isDirectory.boolValue
            else { continue }
            let relative = item.pathComponents.dropFirst(folder.pathComponents.count)
            if folder.lastPathComponent.hasPrefix("replay-"), !relative.contains("delivery") {
                continue
            }
            if relative.contains("render") && !relative.contains("delivery") {
                continue
            }
            urls.append(item)
        }
        return urls.sorted { $0.path < $1.path }
    }
}
