import AVFoundation
import AppKit
import Foundation
import Observation
import SwiftUI
import TranscriptCore

@MainActor
@Observable
final class PlaybackSession {
    typealias CatalogLoader =
        @MainActor (
            URL, @escaping @MainActor @Sendable (Result<CatalogScan, Error>) -> Void
        ) -> Void
    typealias ManifestLoader =
        @MainActor (
            URL, @escaping @MainActor @Sendable (Result<[URL], Error>) -> Void
        ) -> Void
    static let catalogRefreshInterval: Duration = .seconds(30)

    var videos: [VideoTranscript] = []
    var selectedVideoID: URL?
    private(set) var utterances: [TranscriptCue] = []
    var selectedUtteranceID: Int?
    var browseAnchorID: Int?
    var timelineFollowsPlayback = true
    var currentSeconds = 0.0
    var searchQuery = ""
    var searchScope = SearchScope.currentVideo
    var searchHits: [SearchHit] = []
    var isSearching = false
    private(set) var catalogRevision = 0
    private(set) var usesDefaultLibraries = false
    var showsSearch = false
    var columnVisibility: NavigationSplitViewVisibility = .all
    var showsSummary = true
    var summaryBatchRun: UUID?
    var errorText = ""
    var showsError = false
    var scanIssues: [String] = []
    let player = AVPlayer()
    let narrativeSummary: NarrativeSummaryModel
    var playbackSpeed = PlaybackSpeed.normal {
        didSet {
            player.defaultRate = playbackSpeed.rawValue
            if player.rate != 0 {
                player.rate = playbackSpeed.rawValue
            }
        }
    }

    private var libraryRoot: URL?
    private var scanningRoot: URL?
    private var scanGeneration = 0
    private var openedVideoID: URL?

    var libraryName: String { libraryRoot?.lastPathComponent ?? "未選択" }
    var isScanning: Bool { scanningRoot != nil }

    private var timeObserver: Any?
    /// Captions are burned into the video; an embedded subtitle track would show twice.
    private(set) var legibleSelection: Task<Void, Never>?
    private var utterancePanel: NSWindow?
    private let readLibraryRoot: () -> URL?
    private let saveLibraryRoot: (URL) -> Void
    private let defaultLibrary: URL
    private let defaultAdditionalLibraries: [URL]
    private let defaultManifest: URL
    private let readManifest: ManifestLoader
    /// Explicit additions and configured ordinary-startup libraries are kept separate.
    private var additionalRoots: [URL] = []
    private let presentWindow: (NSWindow) -> Void
    private let scanCatalog: CatalogLoader
    private let launchArguments: [String]

    init(
        readLibraryRoot: @escaping () -> URL? = {
            UserDefaults.standard.url(forKey: "TranscriptViewer.libraryRoot")
        },
        saveLibraryRoot: @escaping (URL) -> Void = {
            UserDefaults.standard.set($0, forKey: "TranscriptViewer.libraryRoot")
        },
        defaultLibrary: URL? = nil,
        defaultAdditionalLibraries: [URL]? = nil,
        defaultManifest: URL? = nil,
        readManifest: @escaping ManifestLoader = DefaultLibraryManifest.load(at:completion:),
        presentWindow: @escaping (NSWindow) -> Void = { $0.makeKeyAndOrderFront(nil) },
        scanCatalog: @escaping CatalogLoader = TranscriptCatalog.scan(in:completion:),
        narrativeSummary: NarrativeSummaryModel = NarrativeSummaryModel(),
        launchArguments: [String] = []
    ) {
        self.readLibraryRoot = readLibraryRoot
        self.saveLibraryRoot = saveLibraryRoot
        self.defaultLibrary = defaultLibrary ?? Self.defaultRoot
        self.defaultAdditionalLibraries = defaultAdditionalLibraries ?? []
        self.defaultManifest =
            defaultManifest
            ?? (defaultLibrary ?? Self.defaultRoot).appending(path: DefaultLibraryManifest.filename)
        self.readManifest = readManifest
        self.presentWindow = presentWindow
        self.scanCatalog = scanCatalog
        self.narrativeSummary = narrativeSummary
        self.launchArguments = launchArguments
    }

    var selectedVideo: VideoTranscript? {
        videos.first { $0.id == selectedVideoID }
    }

    var duration: Double {
        if let item = player.currentItem {
            let seconds = item.duration.seconds
            if seconds.isFinite, seconds > 0 {
                return seconds
            }
        }
        return CueTimeline.duration(of: utterances)
    }

    var sliderUpperBound: Double {
        max(duration, 0.001)
    }

    var currentUtterance: TranscriptCue? {
        utterances.first { $0.id == selectedUtteranceID }
    }

    var nearbyUtterances: [TranscriptCue] {
        let anchor = TimelineFollow.anchorID(
            followsPlayback: timelineFollowsPlayback,
            playbackID: selectedUtteranceID,
            browseID: browseAnchorID
        )
        let index = utterances.firstIndex { $0.id == anchor }
        return CueTimeline.window(around: index, in: utterances, radius: 24)
    }

    func loadDefaultLibrary() {
        do {
            let launchRoot = try LaunchLibrary.resolve(arguments: launchArguments)
            // A directory Open Document event can arrive before this view's task.
            guard libraryRoot == nil else { return }
            if let launchRoot {
                load(root: launchRoot)
            } else {
                load(
                    root: readLibraryRoot() ?? defaultLibrary,
                    additional: defaultAdditionalLibraries, includeConfiguredLibraries: true)
            }
        } catch {
            errorText = error.localizedDescription
            showsError = true
        }
    }

    /// Open an explicitly supplied library without changing the saved preference.
    func openLibrary(url: URL) {
        do {
            load(root: try LaunchLibrary.resolve(url: url))
        } catch {
            errorText = error.localizedDescription
            showsError = true
        }
    }

    static var defaultRoot: URL {
        if let edits = TranscriptCatalog.checkoutLibrary(near: Bundle.main.bundleURL) {
            return edits
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Movies", directoryHint: .isDirectory)
            .appending(path: "Apple-Pro-Apps-Verification", directoryHint: .isDirectory)
    }

    func chooseFolder(panel: NSOpenPanel = NSOpenPanel()) {
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "動画とCSV/SRT字幕があるフォルダを選んでください"
        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }
        load(root: url)
        saveLibraryRoot(url)
    }

    func load(root: URL, additional: [URL] = []) {
        load(root: root, additional: additional, includeConfiguredLibraries: false)
    }

    private func load(root: URL, additional: [URL], includeConfiguredLibraries: Bool) {
        libraryRoot = root
        usesDefaultLibraries = includeConfiguredLibraries
        additionalRoots = additional.filter { $0.standardizedFileURL != root.standardizedFileURL }
        rescan(root: root, keepSelection: false)
    }

    /// Adds deliveries that appeared after the last scan. Keeps playing the
    /// current video when it is still present.
    func refreshLibrary() {
        guard let libraryRoot, scanningRoot != libraryRoot else {
            return
        }
        rescan(root: libraryRoot, keepSelection: true)
    }

    /// Each window's lifecycle cancels its observer. The shared session coalesces
    /// concurrent ticks while a scan is running, so closing one window cannot stop others.
    func observeDefaultLibraryUpdates(
        waitForUpdate: () async throws -> Void = {
            try await Task.sleep(for: catalogRefreshInterval)
        }
    ) async {
        guard usesDefaultLibraries else { return }
        do {
            while usesDefaultLibraries {
                try await waitForUpdate()
                try Task.checkCancellation()
                guard usesDefaultLibraries else { return }
                refreshLibrary()
            }
        } catch is CancellationError {
            return
        } catch {
            errorText = error.localizedDescription
            showsError = true
        }
    }

    private func rescan(root: URL, keepSelection: Bool) {
        scanGeneration += 1
        let generation = scanGeneration
        scanningRoot = root
        if usesDefaultLibraries {
            readManifest(defaultManifest) { [weak self] result in
                guard let self, self.scanGeneration == generation, self.libraryRoot == root else {
                    return
                }
                var initial = CatalogScan()
                let configured: [URL]
                switch result {
                case .success(let libraries): configured = libraries
                case .failure(let error):
                    configured = []
                    initial.unreadableTranscriptLabels.append(
                        "\(DefaultLibraryManifest.filename) (\(error.localizedDescription))")
                }
                self.scanLibraries(
                    root: root, additions: self.additionalRoots + configured, initial: initial,
                    generation: generation, keepSelection: keepSelection)
            }
        } else {
            scanLibraries(
                root: root, additions: additionalRoots, initial: CatalogScan(),
                generation: generation, keepSelection: keepSelection)
        }
    }

    private func scanLibraries(
        root: URL, additions: [URL], initial: CatalogScan, generation: Int, keepSelection: Bool
    ) {
        var roots = [root]
        for addition in additions
        where !roots.contains(where: {
            $0.standardizedFileURL.path == addition.standardizedFileURL.path
        }) {
            roots.append(addition)
        }
        scanAll(roots: roots, merged: initial) { [weak self] result in
            guard let self, self.scanGeneration == generation, self.libraryRoot == root else {
                return
            }
            self.scanningRoot = nil
            switch result {
            case .success(let scan):
                self.apply(scan: scan, keepSelection: keepSelection)
            case .failure(let error):
                self.errorText =
                    (error as? LocalizedError)?.errorDescription
                    ?? "読み込みに失敗しました"
                self.showsError = true
            }
        }
    }

    /// Scans the libraries one after another and folds them into one catalog.
    /// An explicit primary library must scan. Ordinary startup can retain valid
    /// additions when its saved primary is missing; failed roots are reported as issues.
    private func scanAll(
        roots: [URL], merged: CatalogScan,
        completion: @escaping @MainActor @Sendable (Result<CatalogScan, Error>) -> Void
    ) {
        guard let next = roots.first else {
            completion(.success(merged))
            return
        }
        let isPrimary = next == libraryRoot
        scanCatalog(next) { [weak self] result in
            guard let self else { return }
            var combined = merged
            switch result {
            case .success(let scan):
                combined.merge(scan)
            case .failure(let error):
                if isPrimary && (!usesDefaultLibraries || roots.count == 1) {
                    completion(.failure(error))
                    return
                }
                combined.unreadableTranscriptLabels.append(
                    "\(next.lastPathComponent) (\(error.localizedDescription))")
            }
            self.scanAll(roots: Array(roots.dropFirst()), merged: combined, completion: completion)
        }
    }

    private func apply(scan: CatalogScan, keepSelection: Bool) {
        scanIssues = scan.issues
        let previousID = selectedVideoID
        let previousVideo = selectedVideo
        videos = scan.videos
        catalogRevision += 1
        if keepSelection, let previousID,
            let selected = scan.videos.first(where: { $0.id == previousID })
        {
            utterances = CueTimeline.utterances(from: selected.cues)
            updateCue(at: currentSeconds)
            return
        }
        if keepSelection, let previousVideo, let recordID = previousVideo.recordingID,
            let current = scan.videos.first(where: {
                $0.recordingID == recordID
                    && $0.broadcastStartJST == previousVideo.broadcastStartJST
            })
        {
            // A new revision can have a different edit clock: select the same recording,
            // but open it paused at zero instead of inventing a cross-revision time mapping.
            selectedVideoID = current.id
            openSelection()
            return
        }
        selectedVideoID = scan.videos.first?.id
        openSelection()
    }

    func handleVideoChange() {
        // All windows share this session. Repeated view notifications must not
        // replace an already opened item or reset a search/summary seek.
        guard openedVideoID != selectedVideo?.id else { return }
        openSelection()
    }

    func openSelection(startingAt seconds: Double? = nil) {
        stopObserving()
        openedVideoID = selectedVideo?.id
        utterances = CueTimeline.utterances(from: selectedVideo?.cues ?? [])
        selectedUtteranceID = nil
        browseAnchorID = nil
        timelineFollowsPlayback = true
        currentSeconds = seconds ?? 0
        guard let video = selectedVideo else {
            player.replaceCurrentItem(with: nil)
            return
        }
        let asset = AVURLAsset(url: video.movieURL)
        let item = AVPlayerItem(asset: asset)
        item.preferredForwardBufferDuration = 2
        item.audioTimePitchAlgorithm = .timeDomain
        player.appliesMediaSelectionCriteriaAutomatically = false
        legibleSelection?.cancel()
        legibleSelection = Task { [weak self] in
            // An unreadable movie (for example one still being written) must be
            // reported as a movie problem; its transcript stays listed regardless.
            let playable: Bool
            do {
                try Task.checkCancellation()
                playable = try await asset.load(.isPlayable)
                try Task.checkCancellation()
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled else { return }
                self?.errorText = "動画を開けません（書き出し中または破損）: \(error.localizedDescription)"
                self?.showsError = true
                return
            }
            guard playable else {
                self?.errorText = "この動画は再生できない形式です: \(video.movieURL.lastPathComponent)"
                self?.showsError = true
                return
            }
            do {
                if let group = try await asset.loadMediaSelectionGroup(for: .legible) {
                    item.select(nil, in: group)
                }
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled else { return }
                self?.errorText = "字幕トラックを無効化できません: \(error.localizedDescription)"
                self?.showsError = true
            }
        }
        player.automaticallyWaitsToMinimizeStalling = true
        player.replaceCurrentItem(with: item)
        let interval = CMTime(value: 1, timescale: 10)
        timeObserver = player.addPeriodicTimeObserver(forInterval: interval, queue: .main) {
            [weak self] time in
            Task { @MainActor in
                self?.updateCue(at: time.seconds)
            }
        }
        if let seconds {
            seek(to: seconds)
        } else {
            player.pause()
            player.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero)
            updateCue(at: 0)
        }
    }

    func seek(toUtterance identifier: Int) {
        guard let utterance = utterances.first(where: { $0.id == identifier }) else {
            return
        }
        resumeTimelineFollow()
        seek(to: utterance.startSeconds)
    }

    func beginTimelineBrowse() {
        timelineFollowsPlayback = false
        browseAnchorID = TimelineFollow.anchorID(
            followsPlayback: false,
            playbackID: selectedUtteranceID,
            browseID: browseAnchorID
        )
    }

    func resumeTimelineFollow() {
        timelineFollowsPlayback = true
        browseAnchorID = nil
    }

    func seek(to seconds: Double) {
        let clamped = min(max(0, seconds), sliderUpperBound)
        let time = CMTime(seconds: clamped, preferredTimescale: 600)
        player.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero)
        player.play()
        updateCue(at: clamped)
    }

    func play(_ hit: SearchHit) {
        showsSearch = false
        resumeTimelineFollow()
        if selectedVideoID != hit.videoID {
            selectedVideoID = hit.videoID
            openSelection(startingAt: hit.cue.startSeconds)
            return
        }
        seek(to: hit.cue.startSeconds)
    }

    func updateCue(at seconds: Double) {
        currentSeconds = seconds
        if let index = CueTimeline.index(at: seconds, in: utterances) {
            selectedUtteranceID = utterances[index].id
        } else {
            selectedUtteranceID = nil
        }
        let visible = Set(nearbyUtterances.map(\.id))
        if TimelineFollow.shouldFollowPlayback(
            browsing: !timelineFollowsPlayback,
            currentID: selectedUtteranceID,
            visibleIDs: visible
        ) {
            timelineFollowsPlayback = true
            browseAnchorID = nil
        }
    }

    func showUtteranceWindow() {
        if let utterancePanel, utterancePanel.isVisible {
            presentWindow(utterancePanel)
            return
        }
        let hosting = NSHostingController(rootView: UtteranceWindow(session: self))
        let window = NSWindow(contentViewController: hosting)
        window.title = "Transcripts"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 560, height: 720))
        window.minSize = NSSize(width: 400, height: 300)
        window.isReleasedWhenClosed = false
        presentWindow(window)
        utterancePanel = window
    }

    func stopObserving() {
        legibleSelection?.cancel()
        if let timeObserver {
            player.removeTimeObserver(timeObserver)
            self.timeObserver = nil
        }
    }
}
