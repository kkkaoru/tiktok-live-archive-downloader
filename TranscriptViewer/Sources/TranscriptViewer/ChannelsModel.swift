import Foundation
import Observation
import TranscriptCore

/// The streamer label of each local YouTube video: its channel from `channel.json` beside the
/// original, labelled by the web viewer's Worker so both viewers show the same names.
@MainActor @Observable
final class ChannelsModel {
    private(set) var channels = YouTubeChannels()
    /// Video title → channel ID.
    private(set) var channelIDs: [String: String] = [:]
    let api: ClipsAPI?
    let originals: URL?
    @ObservationIgnored var fetch: ClipsModel.Fetch = ClipsModel.download

    /// The checkout's YouTube originals, beside which `replay.youtube_delivery` writes channel.json.
    static var configuredOriginals: URL? {
        BGMLibraryModel.checkout?.appending(path: "youtube-originals", directoryHint: .isDirectory)
    }

    init(api: ClipsAPI?, originals: URL?) {
        self.api = api
        self.originals = originals
    }

    func label(for video: VideoTranscript) -> String? {
        channelIDs[video.title].flatMap { channels.labels[$0] }
    }

    /// Reads the channels off the main actor, then the labels; an unreachable Worker keeps the
    /// labels read before.
    func load(titles: [String]) async {
        guard let api, let originals else { return }
        let youtube = titles.filter { $0.hasPrefix("youtube-") }
        channelIDs = await Task.detached {
            youtube.reduce(into: [String: String]()) { ids, title in
                ids[title] = YouTubeChannels.channelID(title: title, originals: originals)
            }
        }.value
        guard !channelIDs.isEmpty, let (data, status) = try? await fetch(api.channels()),
            status == 200, let decoded = try? YouTubeChannels.decode(data)
        else { return }
        channels = decoded
    }
}
