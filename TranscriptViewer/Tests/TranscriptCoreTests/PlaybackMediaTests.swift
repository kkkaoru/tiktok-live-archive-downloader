import AVKit
import AppKit
import Foundation
import SwiftUI
import Testing
import TranscriptCore

@testable import TranscriptViewer

private enum SyntheticMovieError: Error {
    case encoderFailed(Int32)
}

/// Generate only a silent colour fixture; never use a user's recording.
@concurrent
private func writeSyntheticMovie(_ fixture: SummaryFixture, duration: Double = 2) async throws {
    let log = fixture.root.appendingPathComponent("ffmpeg.log")
    try Data().write(to: log)
    let handle = try FileHandle(forWritingTo: log)
    defer { #expect(throws: Never.self) { try handle.close() } }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=\(duration)",
        "-i", fixture.video.transcriptURL.path,
        "-t", String(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:s", "mov_text",
        fixture.video.movieURL.path,
    ]
    process.standardOutput = handle
    process.standardError = handle
    try await withCheckedThrowingContinuation {
        (continuation: CheckedContinuation<Void, any Error>) in
        process.terminationHandler = { finished in
            if finished.terminationStatus == 0 {
                continuation.resume()
            } else {
                continuation.resume(
                    throwing: SyntheticMovieError.encoderFailed(finished.terminationStatus))
            }
        }
        do {
            try process.run()
        } catch {
            process.terminationHandler = nil
            continuation.resume(throwing: error)
        }
    }
}

@MainActor
struct PlaybackMediaTests {
    @Test(.timeLimit(.minutes(1)))
    func libraryReplacementKeepsTheNewAssetAndSeekAcrossWindowNotifications() async throws {
        let first = try SummaryFixture()
        let second = try SummaryFixture()
        defer {
            #expect(throws: Never.self) { try first.remove() }
            #expect(throws: Never.self) { try second.remove() }
        }
        try await writeSyntheticMovie(first)
        try await writeSyntheticMovie(second, duration: 4)
        let session = PlaybackSession(scanCatalog: { root, completion in
            completion(Result { try TranscriptCatalog.scan(in: root) })
        })
        defer {
            session.player.pause()
            session.stopObserving()
            session.player.replaceCurrentItem(with: nil)
        }
        session.load(root: first.root)
        await session.legibleSelection?.value
        let firstItem = try #require(session.player.currentItem)
        try await waitUntilReady(firstItem)
        #expect(abs(session.duration - 2) < 0.01)
        let hosts = (0..<2).map { _ in
            let host = NSHostingView(
                rootView: PlayerSurface(player: session.player, selectedMovie: first.video.movieURL)
            )
            host.setFrameSize(NSSize(width: 320, height: 180))
            host.layoutSubtreeIfNeeded()
            return host
        }
        let views = try hosts.map { try #require(findField(in: $0, field: .durationSeconds)) }
        #expect(views.count == 2)
        for view in views {
            #expect(view.nativePlayerView?.player?.currentItem === firstItem)
            #expect(view.accessibilityValue()?.contains("durationSeconds=2.0") == true)
        }

        session.openLibrary(url: second.root)
        await session.legibleSelection?.value
        let secondItem = try #require(session.player.currentItem)
        try await waitUntilReady(secondItem)
        session.stopObserving()
        #expect(secondItem !== firstItem)
        #expect(try #require(secondItem.asset as? AVURLAsset).url == second.video.movieURL)
        #expect(abs(session.duration - 4) < 0.01)
        session.seek(to: 1.25)
        session.handleVideoChange()
        session.handleVideoChange()
        #expect(session.player.currentItem === secondItem)
        #expect(session.currentSeconds == 1.25)
        #expect(session.selectedVideoID == second.video.id)
        #expect(abs(session.duration - 4) < 0.01)
        for (host, view) in zip(hosts, views) {
            host.rootView = PlayerSurface(
                player: session.player, selectedMovie: second.video.movieURL)
            host.layoutSubtreeIfNeeded()
            #expect(view.nativePlayerView?.player?.currentItem === secondItem)
            #expect(view.accessibilityValue() == "durationSeconds=4.0")
            #expect(
                findField(in: host, field: .loadedMovie)?.accessibilityValue()
                    == "loadedMovie=\(second.video.movieURL.path)")
            #expect(
                findField(in: host, field: .itemState)?.accessibilityValue() == "itemState=ready")
            #expect(
                findField(in: host, field: .boundPlayerMatches)?.accessibilityValue()
                    == "boundPlayerMatches=true")
            #expect(
                findField(in: host, field: .selectionMatches)?.accessibilityValue()
                    == "selectionMatches=true")
            #expect(
                findField(in: host, field: .playbackRate)?.accessibilityValue()
                    == "playbackRate=1.0")
        }
        session.player.pause()
        let sought = await session.player.seek(
            to: CMTime(seconds: 1.25, preferredTimescale: 600),
            toleranceBefore: .zero, toleranceAfter: .zero)
        #expect(sought)
        #expect(
            findField(in: try #require(hosts.first), field: .currentSeconds)?.accessibilityValue()
                == "currentSeconds=1.25")
        #expect(
            findField(in: try #require(hosts.last), field: .playbackRate)?.accessibilityValue()
                == "playbackRate=0.0")
    }

    private func findField(in root: NSView, field: PlaybackDiagnostic.Field)
        -> PlaybackDiagnosticLabel?
    {
        if let label = root as? PlaybackDiagnosticLabel, label.field == field { return label }
        for child in root.subviews {
            if let label = findField(in: child, field: field) { return label }
        }
        return nil
    }

    private func waitUntilReady(_ item: AVPlayerItem) async throws {
        while item.status == .unknown {
            try Task.checkCancellation()
            await Task.yield()
        }
        #expect(item.status == .readyToPlay)
    }

    @Test(.timeLimit(.minutes(1)))
    func nativePlaybackUsesMediaDurationAndDisablesEmbeddedSubtitles() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        try await writeSyntheticMovie(fixture)
        let session = PlaybackSession()
        defer {
            session.player.pause()
            session.stopObserving()
            session.player.replaceCurrentItem(with: nil)
        }
        session.videos = [fixture.video]
        session.selectedVideoID = fixture.video.id
        session.openSelection()
        await session.legibleSelection?.value
        #expect(!session.showsError)
        let item = try #require(session.player.currentItem)
        while item.status == .unknown {
            try Task.checkCancellation()
            await Task.yield()
        }
        #expect(item.status == .readyToPlay)
        #expect(abs(session.duration - 2) < 0.01)
        let group = try #require(try await item.asset.loadMediaSelectionGroup(for: .legible))
        #expect(!group.options.isEmpty)
        #expect(item.currentMediaSelection.selectedMediaOption(in: group) == nil)
        session.playbackSpeed = .oneAndHalf
        session.player.play()
        while session.currentSeconds <= 0.1 {
            try Task.checkCancellation()
            await Task.yield()
        }
        #expect(session.currentSeconds < 2.1)
        #expect(session.player.defaultRate == 1.5)
        #expect(session.currentUtterance?.text == "旅行の準備を説明する。")
    }
}
