import Foundation

/// One saved BGM track. It is unrelated to replay videos and their transcripts.
public struct BGMTrack: Identifiable, Equatable, Sendable {
    public let id: String
    public let title: String
    public let channel: String
    public let channelURL: URL?
    public let pageURL: URL
    public let audioURL: URL
    public let thumbnailURL: URL
    public let duration: Double?
    /// A rebuilt BGM loop, played again from the start when it ends.
    public let loops: Bool
}

/// Reads `ROOT/<video id>/track.json` written by `python -m replay.bgm_library add`.
public enum BGMLibrary {
    public static let folderName = "bgm-library"
    /// `edits/bgm-library.json`, `{"version":1,"root":"/absolute/library"}`: where the tracks
    /// live (for example an external SSD) instead of the checkout's `edits/bgm-library`.
    public static let configName = "bgm-library.json"
    private static let recordLimit = 65_536

    private struct Config: Decodable {
        let version: Int
        let root: String
    }

    private struct Record: Decodable {
        let version: Int
        let id: String
        let title: String
        let channel: String
        let channelURL: String?
        let url: String
        let duration: Double?
        let audio: String
        let thumbnail: String
        let loop: Bool?
    }

    /// The configured library, else the checkout's beside its edit library; nil outside a dist
    /// app or when the configuration is invalid.
    public static func root(near bundle: URL, fileManager: FileManager = .default) -> URL? {
        guard let edits = TranscriptCatalog.checkoutLibrary(near: bundle, fileManager: fileManager)
        else { return nil }
        let config = edits.appending(path: configName)
        // Not following links, so a linked configuration is refused rather than ignored.
        guard (try? fileManager.attributesOfItem(atPath: config.path(percentEncoded: false))) != nil
        else {
            return edits.appending(path: folderName, directoryHint: .isDirectory)
        }
        return configuredRoot(config)
    }

    /// An absolute, normalized, unlinked directory path from a small plain configuration file.
    static func configuredRoot(_ config: URL) -> URL? {
        guard
            let values = try? config.resourceValues(forKeys: [
                .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
            ]),
            values.isRegularFile == true, values.isSymbolicLink != true,
            let size = values.fileSize, size <= recordLimit,
            let data = try? Data(contentsOf: config),
            let value = try? JSONDecoder().decode(Config.self, from: data), value.version == 1,
            value.root.hasPrefix("/"), !value.root.contains("\0")
        else { return nil }
        let url = URL(fileURLWithPath: value.root, isDirectory: true)
        guard url.standardizedFileURL.path == value.root,
            url.resolvingSymlinksInPath() == url.standardizedFileURL,
            !url.pathComponents.contains(where: { ["private", "downloads"].contains($0) })
        else { return nil }
        return url
    }

    /// Valid tracks sorted by title, plus one issue per unreadable track folder.
    public static func load(in root: URL, fileManager: FileManager = .default)
        -> (tracks: [BGMTrack], issues: [String])
    {
        // An unmounted external library is reported, not shown as an empty one.
        guard
            let folders = try? fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        else { return ([], [root.path(percentEncoded: false)]) }
        var tracks: [BGMTrack] = []
        var issues: [String] = []
        for folder in folders.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            do {
                tracks.append(try track(in: folder))
            } catch {
                issues.append(folder.lastPathComponent)
            }
        }
        return (tracks.sorted { $0.title < $1.title }, issues)
    }

    /// Every query word must appear in the title or channel, ignoring case and width.
    public static func search(_ tracks: [BGMTrack], query: String) -> [BGMTrack] {
        let words = fold(query).split(whereSeparator: \.isWhitespace)
        return tracks.filter { track in
            let text = fold("\(track.title) \(track.channel)")
            return words.allSatisfy { text.contains($0) }
        }
    }

    private static func fold(_ text: String) -> String {
        text.folding(
            options: [.caseInsensitive, .widthInsensitive, .diacriticInsensitive], locale: nil)
    }

    private static func track(in folder: URL) throws -> BGMTrack {
        let recordURL = folder.appending(path: "track.json")
        let data = try Data(contentsOf: try regularFile(recordURL, in: folder))
        guard data.count <= recordLimit else {
            throw TranscriptError.unreadableDirectory(folder.path)
        }
        let record = try JSONDecoder().decode(Record.self, from: data)
        guard record.version == 1, record.id == folder.lastPathComponent,
            record.url == "https://www.youtube.com/watch?v=\(record.id)",
            let page = URL(string: record.url)
        else { throw TranscriptError.unreadableDirectory(folder.path) }
        return BGMTrack(
            id: record.id, title: record.title, channel: record.channel,
            channelURL: record.channelURL.flatMap {
                $0.hasPrefix("https://www.youtube.com/") ? URL(string: $0) : nil
            },
            pageURL: page,
            audioURL: try regularFile(folder.appending(path: record.audio), in: folder),
            thumbnailURL: try regularFile(folder.appending(path: record.thumbnail), in: folder),
            duration: record.duration, loops: record.loop ?? false)
    }

    /// A plain file directly inside the track folder, never a link or a nested path.
    private static func regularFile(_ url: URL, in folder: URL) throws -> URL {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true,
            url.deletingLastPathComponent().standardizedFileURL.path
                == folder.standardizedFileURL.path
        else { throw TranscriptError.unreadableDirectory(url.path) }
        return url
    }
}
