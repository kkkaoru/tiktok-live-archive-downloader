import Foundation
import Testing

@testable import TranscriptCore

struct TranscriptCSVTests {
    @Test func parsesQuotedCommaAndQuote() throws {
        let text =
            "start_seconds,end_seconds,start,end,text\n"
            + "1.000,2.500,00:00:01.000,00:00:02.500,\"hello, \"\"world\"\"\"\n"
        let rows = try TranscriptCSV.cues(from: text)
        #expect(
            rows == [
                TranscriptCue(
                    id: 0,
                    startSeconds: 1.0,
                    endSeconds: 2.5,
                    startLabel: "00:00:01.000",
                    endLabel: "00:00:02.500",
                    text: "hello, \"world\""
                )
            ]
        )
    }

    @Test func rejectsMissingHeader() {
        #expect(throws: TranscriptError.missingHeader) {
            _ = try TranscriptCSV.cues(from: "1.0,2.0,00:00:01.000,00:00:02.000,hi\n")
        }
    }

    @Test func rejectsMalformedRow() {
        let text = "start_seconds,end_seconds,start,end,text\n1.0,2.0,00:00:01.000\n"
        #expect(throws: TranscriptError.malformedRow(2)) {
            _ = try TranscriptCSV.cues(from: text)
        }
    }

    @Test func rejectsInvalidClock() {
        let text =
            "start_seconds,end_seconds,start,end,text\nnope,2.0,00:00:01.000,00:00:02.000,hi\n"
        #expect(throws: TranscriptError.invalidClock("nope")) {
            _ = try TranscriptCSV.cues(from: text)
        }
    }

    @Test func rejectsInvertedRange() {
        let text =
            "start_seconds,end_seconds,start,end,text\n5.0,1.0,00:00:05.000,00:00:01.000,hi\n"
        #expect(throws: TranscriptError.invalidClock("5.0")) {
            _ = try TranscriptCSV.cues(from: text)
        }
    }

    @Test func parsesCarriageReturnNewlines() throws {
        let text =
            "start_seconds,end_seconds,start,end,text\r\n0.000,1.000,00:00:00.000,00:00:01.000,hi\r\n"
        let rows = try TranscriptCSV.cues(from: text)
        #expect(rows[0].text == "hi")
    }

    @Test func allowsUnclosedQuoteAndTrailingNewline() throws {
        let text =
            "start_seconds,end_seconds,start,end,text\n"
            + "0.000,1.000,00:00:00.000,00:00:01.000,\"hi\n\n"
        let rows = try TranscriptCSV.cues(from: text)
        #expect(rows[0].text == "hi\n\n")
    }

    @Test func headerOnlyWithExtraNewline() throws {
        let rows = try TranscriptCSV.cues(from: "start_seconds,end_seconds,start,end,text\n\n")
        #expect(rows.isEmpty)
    }

    @Test func readsFile() throws {
        let directory = try transcriptTestWorkRoot().appending(
            path: "csv-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appending(path: "cues.csv")
        try Data(
            "start_seconds,end_seconds,start,end,text\n0.000,0.200,00:00:00.000,00:00:00.200,ん、\n"
                .utf8
        )
        .write(to: url)
        let rows = try TranscriptCSV.cues(contentsOf: url)
        #expect(rows.count == 1)
        #expect(rows[0].text == "ん、")
        try FileManager.default.removeItem(at: directory)
    }
}
