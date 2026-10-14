public enum CueTimeline {
    public static let mergeGapSeconds = 1.5

    public static func duration(of cues: [TranscriptCue]) -> Double {
        cues.map(\.endSeconds).max() ?? 0
    }

    public static func utterances(from cues: [TranscriptCue]) -> [TranscriptCue] {
        var result: [TranscriptCue] = []
        for cue in cues {
            guard let last = result.last, isContinuation(last, cue) else {
                result.append(cue)
                continue
            }
            result[result.count - 1] = TranscriptCue(
                id: last.id,
                startSeconds: last.startSeconds,
                endSeconds: max(last.endSeconds, cue.endSeconds),
                startLabel: last.startLabel,
                endLabel: cue.endLabel,
                text: cue.text
            )
        }
        return result
    }

    static func isContinuation(_ previous: TranscriptCue, _ next: TranscriptCue) -> Bool {
        next.startSeconds <= previous.endSeconds + mergeGapSeconds
            && !previous.text.isEmpty
            && next.text.hasPrefix(previous.text)
    }

    public static func window(around index: Int?, in cues: [TranscriptCue], radius: Int)
        -> [TranscriptCue]
    {
        guard !cues.isEmpty, radius >= 0 else {
            return []
        }
        let center = min(max(index ?? 0, 0), cues.count - 1)
        let start = max(center - radius, 0)
        let end = min(center + radius, cues.count - 1)
        return Array(cues[start...end])
    }

    public static func index(at seconds: Double, in cues: [TranscriptCue]) -> Int? {
        guard !cues.isEmpty, seconds.isFinite else {
            return nil
        }
        if seconds < cues[0].startSeconds {
            return nil
        }
        var low = 0
        var high = cues.count
        while low < high {
            let middle = low + (high - low) / 2
            if cues[middle].startSeconds <= seconds {
                low = middle + 1
            } else {
                high = middle
            }
        }
        let candidate = low - 1
        let cue = cues[candidate]
        if seconds < cue.endSeconds || (candidate == cues.count - 1 && seconds == cue.endSeconds) {
            return candidate
        }
        return nil
    }
}
