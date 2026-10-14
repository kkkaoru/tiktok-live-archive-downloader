public enum TimelineFollow {
    public static func anchorID(followsPlayback: Bool, playbackID: Int?, browseID: Int?) -> Int? {
        if followsPlayback {
            return playbackID
        }
        return browseID ?? playbackID
    }

    public static func shouldFollowPlayback(browsing: Bool, currentID: Int?, visibleIDs: Set<Int>)
        -> Bool
    {
        if !browsing {
            return true
        }
        guard let currentID else {
            return false
        }
        return visibleIDs.contains(currentID)
    }
}
