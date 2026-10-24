import AVFoundation
import Foundation
import Observation
import TranscriptCore

/// A BGM played under the selected video: chosen and leveled per video, started and paused
/// with the video, and looped so a short track covers a long stream.
@MainActor @Observable
final class VideoBGMModel {
    /// Reads a movie's precomputed per-second levels; nil when none are stored.
    typealias LoadLevels = @Sendable (URL) async -> [Double]?
    /// Where `python -m replay.excitement` caches them, in this dist app's checkout.
    nonisolated static let levelCache = BGMLibraryModel.checkout?.appending(
        path: "edits/excitement-cache", directoryHint: .isDirectory)
    static let settingsKey = "TranscriptViewer.videoBGM"

    let library: BGMLibraryModel
    let player = AVPlayer()
    var query = ""
    /// The open video's setting; every change is saved for that video.
    var setting = VideoBGMSetting() {
        didSet {
            guard let key, setting != oldValue else { return }
            settings[key] = setting
            store(settings)
            refresh()
        }
    }
    /// Silences the BGM for every video without changing any video's setting, until resumed.
    var stopped = false {
        didSet { refresh() }
    }
    private(set) var key: String?
    private(set) var playingID: BGMTrack.ID?
    private(set) var videoPlaying = false
    private(set) var loopRestarts = 0
    private var settings: [String: VideoBGMSetting]
    private var movie: URL?
    /// Smoothed levels per second of each video read so far; empty when none are stored.
    private var videoMeasured: [URL: [Double]] = [:]
    private var loading: Set<URL> = []
    private var seconds = 0.0
    @ObservationIgnored private var loopObserver: NSObjectProtocol?
    @ObservationIgnored private var statusObservation: NSKeyValueObservation?
    @ObservationIgnored var store: ([String: VideoBGMSetting]) -> Void = {
        UserDefaults.standard.set(try? JSONEncoder().encode($0), forKey: VideoBGMModel.settingsKey)
    }
    @ObservationIgnored var loadLevels: LoadLevels = {
        VideoBGMModel.storedLevels($0, cache: VideoBGMModel.levelCache)
    }
    @ObservationIgnored private weak var video: AVPlayer?

    init(library: BGMLibraryModel, settings: [String: VideoBGMSetting] = VideoBGMModel.saved()) {
        self.library = library
        self.settings = settings
        library.willPlay = { [weak self] in self?.player.pause() }
        library.defaultChanged = { [weak self] in self?.refresh() }
    }

    nonisolated static func storedLevels(_ movie: URL, cache: URL?) -> [Double]? {
        guard let cache else { return nil }
        return BGMLevel.cachedLevels(for: movie, in: cache)
    }

    static func saved(_ defaults: UserDefaults = .standard) -> [String: VideoBGMSetting] {
        defaults.data(forKey: settingsKey).flatMap {
            try? JSONDecoder().decode([String: VideoBGMSetting].self, from: $0)
        } ?? [:]
    }

    var trackID: BGMTrack.ID? { setting.trackID(default: library.defaultID) }
    var track: BGMTrack? { library.tracks.first { $0.id == trackID } }
    var defaultTrack: BGMTrack? { library.tracks.first { $0.id == library.defaultID } }
    var results: [BGMTrack] { BGMLibrary.search(library.tracks, query: query) }
    var videoLevels: [Double]? { movie.flatMap { videoMeasured[$0] } }
    /// Reading the video's stored levels; typical levels apply meanwhile.
    var isLoading: Bool { movie.map(loading.contains) ?? false }
    /// No stored levels for this video: typical levels apply until they are computed.
    var lacksLevels: Bool { videoLevels?.isEmpty == true }

    /// Shows the video's saved BGM; it starts when the video plays.
    func open(_ video: VideoTranscript?) {
        if library.tracks.isEmpty { library.reload() }
        let key = video.map(VideoBGMSetting.key(for:))
        self.key = nil
        setting = key.flatMap { settings[$0] } ?? VideoBGMSetting()
        self.key = key
        movie = video?.movieURL
        query = ""
        refresh()
    }

    func select(_ selection: VideoBGMSetting.Selection) { setting.selection = selection }

    /// The BGM plays exactly while the video plays (or buffers to play).
    func follow(_ video: AVPlayer) {
        self.video = video
        statusObservation = video.observe(\.timeControlStatus, options: [.initial, .new]) {
            [weak self] observed, _ in
            let playing = observed.timeControlStatus != .paused
            Task { @MainActor in self?.videoPlaybackChanged(playing) }
        }
    }

    func videoPlaybackChanged(_ playing: Bool) {
        videoPlaying = playing
        refresh()
    }

    /// Follows the video's level at its playhead in automatic mode.
    func update(at seconds: Double) {
        self.seconds = seconds
        player.volume = volume
    }

    /// The BGM player's volume now, scaled with the video's own volume. Automatic mode uses
    /// stored levels and assumes typical ones where none are stored, never the manual volume.
    var volume: Float {
        let videoVolume = video.map { $0.isMuted ? 0 : $0.volume } ?? 1
        guard setting.mode == .automatic else { return setting.volume * videoVolume }
        var videoDB = BGMLevel.typicalVideoDB
        if let videoLevels, !videoLevels.isEmpty {
            videoDB = videoLevels[min(max(Int(seconds), 0), videoLevels.count - 1)]
        }
        let bgmDB = track?.loudness ?? BGMLevel.typicalBGMDB
        return BGMLevel.gain(video: videoDB, bgm: bgmDB, offsetDB: setting.offsetDB) * videoVolume
    }

    private func refresh() {
        guard let track else {
            stop()
            return
        }
        if setting.mode == .automatic, let movie { levels(movie) }
        player.volume = volume
        // Paused, not unloaded: resuming continues the track where it stopped.
        guard videoPlaying, !stopped else {
            player.pause()
            return
        }
        if playingID != track.id { load(track) }
        library.pause()
        player.play()
    }

    private func stop() {
        player.pause()
        player.replaceCurrentItem(with: nil)
        playingID = nil
    }

    private func load(_ track: BGMTrack) {
        let item = AVPlayerItem(url: track.audioURL)
        player.replaceCurrentItem(with: item)
        if let loopObserver { NotificationCenter.default.removeObserver(loopObserver) }
        loopObserver = NotificationCenter.default.addObserver(
            forName: AVPlayerItem.didPlayToEndTimeNotification, object: item, queue: .main
        ) { [weak self] _ in MainActor.assumeIsolated { self?.restart() } }
        playingID = track.id
    }

    private func restart() {
        loopRestarts += 1
        player.seek(to: .zero)
        player.play()
    }

    /// Reads a movie's stored levels once per launch, off the main thread.
    private func levels(_ movie: URL) {
        guard videoMeasured[movie] == nil, loading.insert(movie).inserted else { return }
        let loadLevels = loadLevels
        Task {
            let levels = await loadLevels(movie)
            loading.remove(movie)
            videoMeasured[movie] = BGMLevel.smoothed(levels ?? [])
            player.volume = volume
        }
    }
}
