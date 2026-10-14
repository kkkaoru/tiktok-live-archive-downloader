import AVKit
import AppKit
import SwiftUI
import Testing
import TranscriptCore

@testable import TranscriptViewer

@MainActor
struct NativeViewTests {
    @Test func nativePlayerKeepsItsControlsAndUpdatesTheBoundPlayer() throws {
        let keyWindow = NSApplication.shared.keyWindow
        let first = AVPlayer()
        let second = AVPlayer()
        let host = NSHostingView(rootView: PlayerSurface(player: first))
        host.setFrameSize(NSSize(width: 320, height: 180))
        host.layoutSubtreeIfNeeded()
        let native = try #require(findPlayer(in: host))
        #expect(native.player === first)
        #expect(native.controlsStyle == .inline)
        #expect(native.showsFullScreenToggleButton)
        #expect(native.videoGravity == .resizeAspect)
        host.rootView = PlayerSurface(player: second)
        host.layoutSubtreeIfNeeded()
        #expect(native.player === second)
        #expect(NSApplication.shared.keyWindow === keyWindow)
    }

    @Test func renderedTimelineDistinguishesSelectedSpeechOtherSpeechAndPlayhead() throws {
        let cues = [
            TranscriptCue(
                id: 1, startSeconds: 0.2, endSeconds: 0.4, startLabel: "", endLabel: "", text: "選択"),
            TranscriptCue(
                id: 2, startSeconds: 0.6, endSeconds: 0.8, startLabel: "", endLabel: "",
                text: "別の発話"),
        ]
        let view = TimelineBar(utterances: cues, duration: 1, currentSeconds: 0.5, selectedID: 1)
            .frame(width: 200, height: 20).environment(\.colorScheme, .light)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 1
        let bitmap = NSBitmapImageRep(cgImage: try #require(renderer.cgImage))
        #expect(bitmap.pixelsWide == 200)
        #expect(bitmap.pixelsHigh == 20)
        let selected = try #require(bitmap.colorAt(x: 60, y: 10))
        let other = try #require(bitmap.colorAt(x: 140, y: 10))
        let background = try #require(bitmap.colorAt(x: 20, y: 10))
        let head = try #require(bitmap.colorAt(x: 100, y: 10))
        #expect(selected.alphaComponent > other.alphaComponent)
        #expect(other.alphaComponent > background.alphaComponent)
        #expect(head.alphaComponent > other.alphaComponent)
    }

    private func findPlayer(in root: NSView) -> AVPlayerView? {
        if let player = root as? AVPlayerView { return player }
        for child in root.subviews {
            if let player = findPlayer(in: child) { return player }
        }
        return nil
    }
}
