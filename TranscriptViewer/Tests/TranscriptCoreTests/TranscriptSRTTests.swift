import Foundation
import Testing

@testable import TranscriptCore

struct TranscriptSRTTests {
    @Test func parsesMultilineCuesAndWindowsNewlines() throws {
        let text =
            "1\r\n00:00:00,000 --> 00:00:01,250\r\nこんにちは\r\n字幕です\r\n\r\n2\r\n00:01:02,300 --> 00:01:03,400 align:start\r\n次の発話\r\n"

        let cues = try TranscriptSRT.cues(from: text)

        #expect(cues.map(\.id) == [0, 1])
        #expect(cues.map(\.startSeconds) == [0, 62.3])
        #expect(cues.map(\.endSeconds) == [1.25, 63.4])
        #expect(cues.map(\.text) == ["こんにちは\n字幕です", "次の発話"])
    }

    @Test func emptySubtitleHasNoCues() throws {
        #expect(try TranscriptSRT.cues(from: "\n  \n").isEmpty)
    }

    @Test func readsUTF8SubtitleFile() throws {
        let directory = try transcriptTestWorkRoot().appending(
            path: "srt-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appending(path: "cues.srt")
        try Data("1\n00:00:00,000 --> 00:00:00,200\n声\n".utf8).write(to: url)

        let cues = try TranscriptSRT.cues(contentsOf: url)

        #expect(cues.map(\.text) == ["声"])
        try FileManager.default.removeItem(at: directory)
    }

    @Test(arguments: [
        "missing timing\ntext",
        "1\nmissing timing\ntext",
        "1\n00:00:00,000 --> 00:00:01,000\n",
    ])
    func rejectsMalformedBlocks(_ text: String) {
        #expect(throws: TranscriptError.malformedSRT(1)) {
            _ = try TranscriptSRT.cues(from: text)
        }
    }

    @Test(arguments: [
        "1\n00:60:00,000 --> 00:00:01,000\ntext",
        "1\n00:00:00,00 --> 00:00:01,000\ntext",
        "1\n00:00:02,000 --> 00:00:01,000\ntext",
    ])
    func rejectsInvalidTiming(_ text: String) {
        #expect(throws: TranscriptError.self) {
            _ = try TranscriptSRT.cues(from: text)
        }
    }
}
