import Foundation

/// YouTube channel ID → streamer label, read from the web viewer's Worker
/// (`/api/channels.json`), the same record the web viewer groups its streamers by. A video's
/// channel comes from `channel.json` beside its original (`<originals>/<stamp>_JST_youtube-<id>/`).
public struct YouTubeChannels: Equatable, Sendable {
    public var labels: [String: String]

    public init(labels: [String: String] = [:]) {
        self.labels = labels
    }

    private struct Wire: Decodable {
        struct Entry: Decodable { let label: String }
        let channels: [String: Entry]
    }

    private struct ChannelFile: Decodable {
        let channelID: String
        enum CodingKeys: String, CodingKey { case channelID = "channel_id" }
    }

    public static func decode(_ data: Data) throws -> YouTubeChannels {
        YouTubeChannels(
            labels: try JSONDecoder().decode(Wire.self, from: data).channels.mapValues(\.label))
    }

    /// The channel of a `youtube-<id>` title's original, or nil for any other title.
    public static func channelID(
        title: String, originals: URL, fileManager: FileManager = .default
    ) -> String? {
        guard title.hasPrefix("youtube-"),
            let names = try? fileManager.contentsOfDirectory(
                atPath: originals.path(percentEncoded: false))
        else { return nil }
        let folders = names.filter { $0.hasSuffix("_JST_\(title)") }
        guard folders.count == 1, let folder = folders.first,
            let data = try? Data(
                contentsOf: originals.appending(path: folder).appending(path: "channel.json")),
            data.count <= 65_536
        else { return nil }
        return try? JSONDecoder().decode(ChannelFile.self, from: data).channelID
    }
}
