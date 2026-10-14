import Testing

@testable import TranscriptCore

struct TimelineFollowTests {
    @Test func playbackAnchorWhenFollowing() {
        #expect(TimelineFollow.anchorID(followsPlayback: true, playbackID: 4, browseID: 9) == 4)
    }

    @Test func browseAnchorWhenNotFollowing() {
        #expect(TimelineFollow.anchorID(followsPlayback: false, playbackID: 4, browseID: 9) == 9)
        #expect(TimelineFollow.anchorID(followsPlayback: false, playbackID: 4, browseID: nil) == 4)
    }

    @Test func followResumesWhenCurrentIsVisible() {
        #expect(
            TimelineFollow.shouldFollowPlayback(
                browsing: true, currentID: 3, visibleIDs: [2, 3, 4])
        )
    }

    @Test func stayPutWhenCurrentLeavesWindow() {
        #expect(
            TimelineFollow.shouldFollowPlayback(
                browsing: true, currentID: 9, visibleIDs: [2, 3, 4])
                == false
        )
    }

    @Test func browsingWithoutCurrentCueDoesNotResumeFollow() {
        #expect(
            !TimelineFollow.shouldFollowPlayback(
                browsing: true, currentID: nil, visibleIDs: [1, 2]))
    }

    @Test func followWhileNotBrowsing() {
        #expect(TimelineFollow.shouldFollowPlayback(browsing: false, currentID: 1, visibleIDs: []))
    }
}
