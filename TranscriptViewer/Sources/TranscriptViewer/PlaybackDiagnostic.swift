import AVKit
import Foundation

/// Read the native player, independently of the displayed transcript selection.
struct PlaybackDiagnostic: Equatable {
    enum Field: String, CaseIterable {
        case selectedMovie, loadedMovie, boundPlayerMatches, selectionMatches, itemState,
            durationSeconds
        case currentSeconds, playbackRate

        var label: String {
            switch self {
            case .selectedMovie: "選択動画"
            case .loadedMovie: "再生動画"
            case .boundPlayerMatches: "プレーヤー一致"
            case .selectionMatches: "動画一致"
            case .itemState: "読込状態"
            case .durationSeconds: "動画時間"
            case .currentSeconds: "動画の再生位置"
            case .playbackRate: "動画の再生速度"
            }
        }
    }

    enum ItemState: String {
        case empty
        case loading
        case ready
        case failed
    }

    let selectedMovie: URL?
    let loadedMovie: URL?
    let boundPlayerMatches: Bool
    let itemState: ItemState
    let durationSeconds: Double?
    let currentSeconds: Double?
    let playbackRate: Float?

    var selectionMatches: Bool? {
        guard let selectedMovie, let loadedMovie else { return nil }
        return selectedMovie.standardizedFileURL == loadedMovie.standardizedFileURL
    }

    func value(for field: Field) -> String {
        let value: String
        switch field {
        case .selectedMovie: value = selectedMovie?.path ?? "none"
        case .loadedMovie: value = loadedMovie?.path ?? "none"
        case .boundPlayerMatches: value = String(boundPlayerMatches)
        case .selectionMatches: value = selectionMatches.map { String($0) } ?? "unknown"
        case .itemState: value = itemState.rawValue
        case .durationSeconds: value = durationSeconds.map { String($0) } ?? "unknown"
        case .currentSeconds: value = currentSeconds.map { String($0) } ?? "unknown"
        case .playbackRate: value = playbackRate.map { String($0) } ?? "unknown"
        }
        return "\(field.rawValue)=\(value)"
    }

    @MainActor
    static func capture(player: AVPlayer?, expectedPlayer: AVPlayer?, selectedMovie: URL?)
        -> Self
    {
        let item = player?.currentItem
        let state: ItemState
        switch item?.status {
        case nil: state = .empty
        case .unknown: state = .loading
        case .readyToPlay: state = .ready
        case .failed: state = .failed
        @unknown default: state = .loading
        }
        let seconds = item?.duration.seconds
        let current = player?.currentTime().seconds
        return Self(
            selectedMovie: selectedMovie,
            loadedMovie: (item?.asset as? AVURLAsset)?.url,
            boundPlayerMatches: player != nil && player === expectedPlayer,
            itemState: state,
            durationSeconds: seconds.flatMap { $0.isFinite && $0 > 0 ? $0 : nil },
            currentSeconds: current.flatMap { $0.isFinite && $0 >= 0 ? $0 : nil },
            playbackRate: player?.rate
        )
    }
}

/// A dedicated read-only element leaves the native AVKit control roles intact.
/// Its live value also works while paused, without a playback callback/redraw.
@MainActor
final class PlaybackDiagnosticLabel: NSTextField {
    weak var nativePlayerView: AVPlayerView?
    weak var expectedPlayer: AVPlayer?
    var selectedMovie: URL?
    var field: PlaybackDiagnostic.Field = .itemState

    override func accessibilityValue() -> String? {
        PlaybackDiagnostic.capture(
            player: nativePlayerView?.player, expectedPlayer: expectedPlayer,
            selectedMovie: selectedMovie
        ).value(for: field)
    }
}
