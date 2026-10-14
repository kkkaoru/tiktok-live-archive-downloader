import Foundation
import Testing

@testable import TranscriptCore

struct RecordingIdentityTests {
    @Test(arguments: [
        ("2026-10-04_14-30-09_JST_replay-123-review.mp4", "replay-123"),
        ("2024-02-29_23-59-59_JST_youtube-AbC012_-xyz-review.mov", "youtube-AbC012_-xyz"),
        ("2026-10-04_14-30-09_JST_replay-123-review.m4v", "replay-123"),
    ])
    func readsExactCanonicalDateAndID(name: String, recordingID: String) throws {
        let identity = try #require(
            RecordingIdentity(movie: URL(fileURLWithPath: "/media/\(name)")))
        #expect(identity.recordingID == recordingID)
        #expect(
            identity.broadcastStartJST
                == (recordingID.hasPrefix("youtube-")
                    ? "2024-02-29 23:59:59 JST" : "2026-10-04 14:30:09 JST"))
    }

    @Test(arguments: [
        "2026-10-04_14-30-09_JST_replay-123-review.srt",
        "2026-10-04_14-30-09_JST_replay-123-other.mp4",
        "2026-10-04_14-30-09_JST_replay-123-review-extra.mp4",
        "2026-10-04_14-30-09_JST_replay-123-review\n.mp4",
        "2026-10-04_14-30-09_JST_replay--review.mp4",
        "2026-10-04_14-30-09_JST_youtube-short-review.mp4",
        "2026-02-30_14-30-09_JST_replay-123-review.mp4",
        "2026-10-04_24-30-09_JST_replay-123-review.mp4",
        "2026-10-04_14-30-09_UTC_replay-123-review.mp4",
        "short-review.mp4",
        String(repeating: "x", count: 161) + "-review.mp4",
    ])
    func rejectsNonCanonicalOrImpossibleIdentity(name: String) {
        #expect(RecordingIdentity(movie: URL(fileURLWithPath: "/media/\(name)")) == nil)
    }
}
