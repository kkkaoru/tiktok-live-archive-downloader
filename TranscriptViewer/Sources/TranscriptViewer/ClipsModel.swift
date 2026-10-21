import Foundation
import Observation
import TranscriptCore

/// The selected video's clip points, read from the web viewer's Worker (where they are computed
/// and stored per movie content), and the clip being played range by range.
@MainActor @Observable
final class ClipsModel {
    enum State: Equatable {
        case idle
        case loading
        case ready([ClipPoint])
        case pending
        case mismatch
        case missing
        case noExcitement
        /// No API configuration in this checkout.
        case unavailable
        case failed(String)
    }

    typealias Fetch = @Sendable (URLRequest) async throws -> (Data, Int)
    typealias MovieKey = @Sendable (URL) throws -> String

    static let pollInterval: Duration = .seconds(15)

    private(set) var state = State.idle
    private(set) var cursor: ClipCursor?
    let api: ClipsAPI?
    @ObservationIgnored var fetch: Fetch = ClipsModel.download
    @ObservationIgnored var movieKey: MovieKey = BGMLevel.movieKey
    @ObservationIgnored var pollInterval = ClipsModel.pollInterval
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var movie: URL?
    @ObservationIgnored private var key: String?

    /// The API of this dist app's checkout, when it has one.
    static var configured: ClipsAPI? { BGMLibraryModel.checkout.flatMap(ClipsAPI.load(checkout:)) }

    init(api: ClipsAPI?) {
        self.api = api
    }

    /// The HTTP status, or 0 for a response that is not HTTP.
    nonisolated static func download(_ request: URLRequest) async throws -> (Data, Int) {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { return (data, 0) }
        return (data, http.statusCode)
    }

    var clips: [ClipPoint] {
        if case .ready(let clips) = state { return clips }
        return []
    }

    var canReload: Bool { api != nil && movie != nil && state != .pending && state != .loading }

    /// Loads the clips of `movie`, polling while they are being computed.
    func open(_ movie: URL?) {
        task?.cancel()
        cursor = nil
        self.movie = movie
        key = nil
        guard let movie else {
            state = .idle
            return
        }
        guard api != nil else {
            state = .unavailable
            return
        }
        state = .loading
        task = Task { [weak self] in await self?.poll(movie) }
    }

    /// Asks the Worker to compute the clips again, then polls for the new ones.
    func reload() {
        guard let api, let movie else { return }
        task?.cancel()
        cursor = nil
        state = .pending
        task = Task { [weak self] in
            guard let self else { return }
            do {
                let (_, status) = try await fetch(api.reload(videoID: ClipsAPI.videoID(movie)))
                guard status == 200 else {
                    state = .failed("再計算を開始できませんでした（HTTP \(status)）")
                    return
                }
                try await Task.sleep(for: pollInterval)
                await poll(movie)
            } catch is CancellationError {
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }

    private func poll(_ movie: URL) async {
        guard let api else { return }
        do {
            let key = try await keyOf(movie)
            while !Task.isCancelled, self.movie == movie {
                let (data, status) = try await fetch(
                    api.lookup(videoID: ClipsAPI.videoID(movie), movieKey: key))
                guard status == 200 || status == 404 else {
                    state = .failed("切り抜きを読み込めませんでした（HTTP \(status)）")
                    return
                }
                switch try ClipStatus.decode(data) {
                case .ready(let clips):
                    state = .ready(clips)
                    return
                case .mismatch:
                    state = .mismatch
                    return
                case .missing:
                    state = .missing
                    return
                case .noExcitement:
                    state = .noExcitement
                    return
                case .pending: state = .pending
                }
                try await Task.sleep(for: pollInterval)
            }
        } catch is CancellationError {
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    /// The content key reads 2 MiB of the movie, so it runs off the main actor once per video.
    private func keyOf(_ movie: URL) async throws -> String {
        if let key { return key }
        let read = movieKey
        let key = try await Task.detached { try read(movie) }.value
        self.key = key
        return key
    }

    /// Starts `clip` at range `index`; returns where to seek.
    func start(_ clip: ClipPoint, range index: Int = 0, loop: Bool = false) -> Double? {
        let cursor = ClipCursor(clip: clip, index: index, loop: loop)
        self.cursor = cursor
        return cursor.range?.start
    }

    func stop() { cursor = nil }

    /// Advances the playing clip; the session seeks or pauses as told.
    func step(at seconds: Double) -> ClipCursor.Step {
        guard var cursor else { return .keep }
        let step = cursor.step(at: seconds)
        self.cursor = step == .finish || step == .leave ? nil : cursor
        return step
    }
}
