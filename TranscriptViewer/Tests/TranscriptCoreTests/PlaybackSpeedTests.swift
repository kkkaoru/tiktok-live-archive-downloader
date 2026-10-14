import AVFoundation
import Testing
import TranscriptCore

@testable import TranscriptViewer

@MainActor
struct PlaybackSpeedTests {
    @Test func changingSpeedWhilePausedDoesNotStartPlayback() {
        let session = PlaybackSession()
        #expect(session.playbackSpeed == .normal)
        session.playbackSpeed = .oneAndHalf
        #expect(session.player.defaultRate == 1.5)
        #expect(session.player.rate == 0)
        session.playbackSpeed = .double
        #expect(session.player.defaultRate == 2)
        #expect(session.player.rate == 0)
        session.playbackSpeed = .normal
        #expect(session.player.defaultRate == 1)
    }

    @Test func playingAndJumpingUseSelectedRate() {
        let session = PlaybackSession()
        defer { session.player.pause() }
        session.player.rate = 1
        session.playbackSpeed = .oneAndHalf
        #expect(session.player.rate == 1.5)
        session.player.pause()
        session.player.play()
        #expect(session.player.rate == 1.5)
        session.playbackSpeed = .double
        session.seek(to: 0)
        #expect(session.player.rate == 2)
        #expect(session.player.defaultRate == 2)
    }
}
