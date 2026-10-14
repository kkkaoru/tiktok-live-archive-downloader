import AVKit
import AppKit
import Foundation
import SwiftUI
import Testing

@testable import TranscriptViewer

@MainActor
struct PlaybackDiagnosticTests {
    @Test func noLoadedItemIsNotAnAcceptedSelection() {
        let missing = PlaybackDiagnostic.capture(
            player: nil, expectedPlayer: nil, selectedMovie: nil)
        #expect(missing.itemState == .empty)
        #expect(!missing.boundPlayerMatches)
        #expect(missing.selectionMatches == nil)
        #expect(missing.durationSeconds == nil)
        #expect(missing.value(for: .loadedMovie) == "loadedMovie=none")
        #expect(missing.value(for: .selectionMatches) == "selectionMatches=unknown")
        #expect(missing.value(for: .durationSeconds) == "durationSeconds=unknown")
        #expect(missing.value(for: .currentSeconds) == "currentSeconds=unknown")
        #expect(missing.value(for: .playbackRate) == "playbackRate=unknown")

        let player = AVPlayer()
        let bound = PlaybackDiagnostic.capture(
            player: player, expectedPlayer: player,
            selectedMovie: URL(fileURLWithPath: "/synthetic/selected.mp4"))
        #expect(bound.boundPlayerMatches)
        #expect(bound.itemState == .empty)
        #expect(bound.loadedMovie == nil)
        #expect(bound.value(for: .itemState) == "itemState=empty")
    }

    @Test func differentPlayersAndUnknownAssetRemainExplicit() {
        let player = AVPlayer(playerItem: AVPlayerItem(asset: AVMutableComposition()))
        defer { player.replaceCurrentItem(with: nil) }
        let diagnostic = PlaybackDiagnostic.capture(
            player: player, expectedPlayer: AVPlayer(),
            selectedMovie: URL(fileURLWithPath: "/synthetic/selected.mp4"))
        #expect(!diagnostic.boundPlayerMatches)
        #expect(diagnostic.loadedMovie == nil)
        #expect(diagnostic.selectionMatches == nil)
        #expect(diagnostic.itemState == .loading)
        #expect(diagnostic.durationSeconds == nil)
    }

    @Test func nativeAccessibilityReadsTheCurrentItemWithoutAViewRedraw() throws {
        let keyWindow = NSApplication.shared.keyWindow
        let first = URL(fileURLWithPath: "/synthetic/first.mp4")
        let second = URL(fileURLWithPath: "/synthetic/second.mp4")
        let player = AVPlayer(playerItem: AVPlayerItem(url: first))
        defer { player.replaceCurrentItem(with: nil) }
        let host = NSHostingView(rootView: PlayerSurface(player: player, selectedMovie: first))
        host.setFrameSize(NSSize(width: 320, height: 180))
        host.layoutSubtreeIfNeeded()
        let views = findFields(in: host)
        #expect(views.map(\.field) == PlaybackDiagnostic.Field.allCases)
        #expect(
            views.allSatisfy {
                $0.isAccessibilityElement() && $0.accessibilityRole() == .staticText
            })
        #expect(
            views.allSatisfy {
                $0.accessibilityIdentifier() == "playback-media-state-\($0.field.rawValue)"
            })
        #expect(views.allSatisfy { $0.selectedMovie == first && $0.expectedPlayer === player })
        #expect(views.allSatisfy { $0.stringValue == " " })
        #expect(views.allSatisfy { $0.accessibilityValue()?.contains("\n") == false })
        let selected = try #require(views.first { $0.field == .selectedMovie })
        let loaded = try #require(views.first { $0.field == .loadedMovie })
        let matches = try #require(views.first { $0.field == .selectionMatches })
        #expect(matches.accessibilityValue() == "selectionMatches=true")

        player.replaceCurrentItem(with: AVPlayerItem(url: second))
        #expect(loaded.accessibilityValue() == "loadedMovie=\(second.path)")
        #expect(selected.accessibilityValue() == "selectedMovie=\(first.path)")
        #expect(matches.accessibilityValue() == "selectionMatches=false")
        host.rootView = PlayerSurface(player: player, selectedMovie: second)
        host.layoutSubtreeIfNeeded()
        #expect(selected.selectedMovie == second)
        #expect(matches.accessibilityValue() == "selectionMatches=true")
        #expect(NSApplication.shared.keyWindow === keyWindow)
    }

    @Test(.timeLimit(.minutes(1)))
    func failedMediaIsReportedRatherThanUsingTheTranscriptDuration() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let item = AVPlayerItem(url: fixture.video.movieURL)
        let player = AVPlayer(playerItem: item)
        defer { player.replaceCurrentItem(with: nil) }
        while item.status == .unknown {
            try Task.checkCancellation()
            await Task.yield()
        }
        let failed = PlaybackDiagnostic.capture(
            player: player, expectedPlayer: player, selectedMovie: fixture.video.movieURL)
        #expect(failed.itemState == .failed)
        #expect(failed.selectionMatches == true)
        #expect(failed.durationSeconds == nil)
        #expect(failed.value(for: .itemState) == "itemState=failed")
        #expect(failed.value(for: .durationSeconds) == "durationSeconds=unknown")
    }

    private func findFields(in root: NSView) -> [PlaybackDiagnosticLabel] {
        if let field = root as? PlaybackDiagnosticLabel { return [field] }
        return root.subviews.flatMap { findFields(in: $0) }
    }
}
