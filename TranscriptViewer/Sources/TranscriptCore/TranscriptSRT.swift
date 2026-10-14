import Foundation

public enum TranscriptSRT {
    public static func cues(from text: String) throws -> [TranscriptCue] {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(
            of: "\r", with: "\n")
        let lines = normalized.components(separatedBy: "\n")
        var blocks: [[String]] = []
        var current: [String] = []
        for line in lines {
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                if !current.isEmpty {
                    blocks.append(current)
                    current = []
                }
            } else {
                current.append(line)
            }
        }
        if !current.isEmpty {
            blocks.append(current)
        }

        return try blocks.enumerated().map { index, block in
            guard block.count >= 3, Int(block[0]) != nil else {
                throw TranscriptError.malformedSRT(index + 1)
            }
            let range = block[1].components(separatedBy: "-->")
            guard range.count == 2 else {
                throw TranscriptError.malformedSRT(index + 1)
            }
            let startLabel = range[0].trimmingCharacters(in: .whitespaces)
            guard
                let endLabel = range[1].split(whereSeparator: \.isWhitespace).first.map(String.init)
            else {
                throw TranscriptError.malformedSRT(index + 1)
            }
            let start = try seconds(from: startLabel)
            let end = try seconds(from: endLabel)
            guard end >= start else {
                throw TranscriptError.invalidClock(startLabel)
            }
            let cueText = block.dropFirst(2).joined(separator: "\n").trimmingCharacters(
                in: .whitespacesAndNewlines)
            guard !cueText.isEmpty else {
                throw TranscriptError.malformedSRT(index + 1)
            }
            return TranscriptCue(
                id: index,
                startSeconds: start,
                endSeconds: end,
                startLabel: startLabel,
                endLabel: endLabel,
                text: cueText
            )
        }
    }

    public static func cues(contentsOf url: URL) throws -> [TranscriptCue] {
        try cues(from: String(contentsOf: url, encoding: .utf8))
    }

    private static func seconds(from value: String) throws -> Double {
        let milliseconds = value.components(separatedBy: ",")
        guard milliseconds.count == 2,
            milliseconds[1].count == 3,
            let millisecondValue = Int(milliseconds[1]),
            (0..<1_000).contains(millisecondValue)
        else {
            throw TranscriptError.invalidClock(value)
        }
        let clock = milliseconds[0].split(separator: ":", omittingEmptySubsequences: false)
        guard clock.count == 3,
            let hours = Int(clock[0]), hours >= 0,
            let minutes = Int(clock[1]), (0..<60).contains(minutes),
            let seconds = Int(clock[2]), (0..<60).contains(seconds)
        else {
            throw TranscriptError.invalidClock(value)
        }
        return Double(hours * 3_600 + minutes * 60 + seconds) + Double(millisecondValue) / 1_000
    }
}
