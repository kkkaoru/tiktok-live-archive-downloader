import AVFoundation
import AppKit
import Foundation
import Observation
import TranscriptCore

/// The BGM window's state: a library separate from replay videos, audio-only playback.
@MainActor @Observable
final class BGMLibraryModel {
    /// Runs `python -m replay.bgm_library add URL` and returns its exit status and output.
    typealias Downloader = @Sendable (String, URL) async -> (status: Int32, output: String)
    static let volumeKey = "TranscriptViewer.bgmVolume"
    static let defaultKey = "TranscriptViewer.defaultBGM"

    let root: URL?
    let player = AVPlayer()
    var tracks: [BGMTrack] = []
    var issues: [String] = []
    var query = ""
    /// Choosing a track in the list starts it from the beginning.
    var selectedID: BGMTrack.ID? {
        didSet {
            if selectedID != oldValue, let selected { play(selected) }
        }
    }
    private(set) var playingID: BGMTrack.ID?
    private(set) var loopRestarts = 0
    @ObservationIgnored private var loopObserver: NSObjectProtocol?
    /// One app-level BGM volume for every track, independent of the system output volume.
    var volume: Float {
        get { level }
        set {
            level = min(max(newValue, 0), 1)
            player.volume = level
            storeVolume(level)
        }
    }
    private var level: Float
    @ObservationIgnored private var volumeObservation: NSKeyValueObservation?
    /// The BGM a video plays when it has none of its own.
    var defaultID: BGMTrack.ID? {
        didSet {
            guard defaultID != oldValue else { return }
            storeDefault(defaultID)
            defaultChanged()
        }
    }
    var newURL = ""
    private(set) var isAdding = false
    private(set) var addMessage = ""
    @ObservationIgnored var storeVolume: (Float) -> Void = {
        UserDefaults.standard.set($0, forKey: BGMLibraryModel.volumeKey)
    }
    @ObservationIgnored var storeDefault: (BGMTrack.ID?) -> Void = {
        UserDefaults.standard.set($0, forKey: BGMLibraryModel.defaultKey)
    }
    /// Set by the video BGM, so only one BGM plays at a time.
    @ObservationIgnored var willPlay: () -> Void = {}
    @ObservationIgnored var defaultChanged: () -> Void = {}
    @ObservationIgnored var download: Downloader = { url, root in
        await BGMLibraryModel.runCheckoutDownloader(url, root, checkout: BGMLibraryModel.checkout)
    }
    @ObservationIgnored var reveal: (URL) -> Void = {
        NSWorkspace.shared.activateFileViewerSelecting([$0])
    }
    @ObservationIgnored var copy: (String) -> Void = { text in
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    init(
        root: URL? = BGMLibrary.root(near: Bundle.main.bundleURL),
        volume: Float = UserDefaults.standard.object(forKey: BGMLibraryModel.volumeKey) as? Float
            ?? 1,
        defaultID: BGMTrack.ID? = UserDefaults.standard.string(forKey: BGMLibraryModel.defaultKey)
    ) {
        self.root = root
        self.defaultID = defaultID
        level = min(max(volume, 0), 1)
        player.volume = level
        // The seek bar's own speaker control changes the same player; keep one shared value.
        // Read the player when the task runs, so echoes of earlier changes never undo later ones.
        volumeObservation = player.observe(\.volume) { [weak self] _, _ in
            Task { @MainActor in
                if let self, self.level != self.player.volume { self.volume = self.player.volume }
            }
        }
    }

    var results: [BGMTrack] { BGMLibrary.search(tracks, query: query) }
    var selected: BGMTrack? { tracks.first { $0.id == selectedID } }
    var playing: BGMTrack? { tracks.first { $0.id == playingID } }

    func reload() {
        guard let root else { return }
        (tracks, issues) = BGMLibrary.load(in: root)
    }

    /// Starts the track, or pauses it when it is already playing.
    func togglePlayback(_ track: BGMTrack) {
        if playingID == track.id {
            pause()
            return
        }
        play(track)
    }

    func pause() {
        player.pause()
        playingID = nil
    }

    func toggleDefault(_ track: BGMTrack) {
        defaultID = defaultID == track.id ? nil : track.id
    }

    private func play(_ track: BGMTrack) {
        let item = AVPlayerItem(url: track.audioURL)
        player.replaceCurrentItem(with: item)
        if let loopObserver { NotificationCenter.default.removeObserver(loopObserver) }
        loopObserver =
            track.loops
            ? NotificationCenter.default.addObserver(
                forName: AVPlayerItem.didPlayToEndTimeNotification, object: item, queue: .main
            ) { [weak self] _ in MainActor.assumeIsolated { self?.restart() } }
            : nil
        willPlay()
        player.play()
        playingID = track.id
    }

    private func restart() {
        loopRestarts += 1
        player.seek(to: .zero)
        player.play()
    }

    func revealInFinder(_ track: BGMTrack) { reveal(track.audioURL) }

    func copyPath(_ track: BGMTrack) { copy(track.audioURL.path(percentEncoded: false)) }

    /// Saves one canonical YouTube URL through the repository's downloader, then reloads.
    func add() async {
        let url = newURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let root, !url.isEmpty, !isAdding else { return }
        isAdding = true
        addMessage = "保存中: \(url)"
        let result = await download(url, root)
        isAdding = false
        reload()
        if result.status == 0 {
            newURL = ""
            addMessage = "保存しました"
        } else {
            addMessage = "保存できませんでした: \(result.output)"
        }
    }

    /// The checkout of this dist app; the library itself may live elsewhere.
    nonisolated static let checkout = TranscriptCatalog.checkoutLibrary(
        near: Bundle.main.bundleURL)?
        .deletingLastPathComponent()

    /// The checkout's locked Python runs the module; no shell, PATH only for node/ffmpeg.
    nonisolated static func runCheckoutDownloader(_ url: String, _ root: URL, checkout: URL?)
        async -> (status: Int32, output: String)
    {
        guard let checkout else { return (-1, "checkout not found") }
        let process = Process()
        process.executableURL = checkout.appending(path: ".venv/bin/python")
        process.arguments = ["-B", "-m", "replay.bgm_library", "add", url, "--root", root.path]
        process.currentDirectoryURL = checkout
        process.environment = ["PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do {
            try process.run()
        } catch {
            return (-1, error.localizedDescription)
        }
        let data = await Task.detached { pipe.fileHandleForReading.readDataToEndOfFile() }.value
        process.waitUntilExit()
        return (process.terminationStatus, String(decoding: data, as: UTF8.self))
    }
}
