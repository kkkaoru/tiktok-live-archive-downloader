import Foundation
import Synchronization
import Testing

@testable import TranscriptSummaries

struct PublishedSummaryCleanerTests {
    static let looped =
        "地域の催しでは参加者が料理の工夫を紹介している。料理の話をした。\n\n地域の催しでは参加者が料理の工夫を紹介している。料理の話をした。"

    @Test func cleanDataDropsRepeatsFromPublishedSummaryTextsAfterABackup() async throws {
        let root = try transcriptTestWorkRoot().appendingPathComponent("clean-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { #expect(throws: Never.self) { try FileManager.default.removeItem(at: root) } }
        let data = root.appendingPathComponent("v.json")
        let backup = root.appendingPathComponent("v.before.json")
        let original = try JSONSerialization.data(withJSONObject: [
            "id": "v", "cues": [["text": "a"]],
            "summary": [
                "overview": Self.looped,
                "sections": [["startSeconds": 0, "endSeconds": 600, "text": Self.looped]],
            ],
        ])
        try original.write(to: data)
        let lines = Mutex<[String]>([])
        let command = try SummaryCommand(arguments: [
            "clean-data", "--data", data.path, "--backup", backup.path,
        ])
        #expect(try await command.run(output: { line in lines.withLock { $0.append(line) } }) == 0)
        #expect(lines.withLock { $0 } == ["CLEANED removed_characters=48"])
        #expect(try Data(contentsOf: backup) == original)
        let cleaned = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: data)) as? [String: Any])
        let summary = try #require(cleaned["summary"] as? [String: Any])
        let unlooped = "地域の催しでは参加者が料理の工夫を紹介している。料理の話をした。\n\n料理の話をした。"
        #expect(summary["overview"] as? String == unlooped)
        #expect((summary["sections"] as? [[String: Any]])?.first?["text"] as? String == unlooped)
        #expect((summary["sections"] as? [[String: Any]])?.first?["endSeconds"] as? Int == 600)
        #expect(cleaned["id"] as? String == "v")
        // The backup is never overwritten, and a data file without a summary is refused.
        await #expect(throws: (any Error).self) { try await command.run(output: { _ in }) }
        for (index, body) in [
            #"{"id":"v"}"#, #"{"summary":1}"#, #"{"summary":{"overview":1,"sections":[]}}"#,
            #"{"summary":{"overview":"a","sections":1}}"#,
        ].enumerated() {
            try Data(body.utf8).write(to: data)
            #expect(throws: SummaryStoreError.invalidRecord) {
                try PublishedSummaryCleaner.clean(
                    data: data, backup: root.appendingPathComponent("other-\(index).json"))
            }
        }
    }

    @Test func cleanDataAlsoCleansASavedSummaryRecord() throws {
        let root = try transcriptTestWorkRoot().appendingPathComponent("clean-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { #expect(throws: Never.self) { try FileManager.default.removeItem(at: root) } }
        let record = root.appendingPathComponent("current.json")
        try JSONSerialization.data(withJSONObject: [
            "version": 1, "fingerprint": "f", "overview": Self.looped,
            "sections": [["startSeconds": 0, "endSeconds": 600, "text": "短い。"]],
        ]).write(to: record)
        #expect(
            try PublishedSummaryCleaner.clean(
                data: record, backup: root.appendingPathComponent("before.json")) == 24)
        let cleaned = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: record)) as? [String: Any])
        #expect(cleaned["fingerprint"] as? String == "f")
        #expect(cleaned["summary"] == nil)
        #expect(
            cleaned["overview"] as? String
                == "地域の催しでは参加者が料理の工夫を紹介している。料理の話をした。\n\n料理の話をした。")
    }

    @Test(arguments: [
        ["clean-data"], ["clean-data", "--data", "x"], ["clean-data", "--data", "x", "--out", "y"],
        ["clean-data", "--data", "", "--backup", "y"],
    ])
    func cleanDataNeedsExactlyItsTwoPaths(_ arguments: [String]) {
        #expect(throws: SummaryCommandError.self) { try SummaryCommand(arguments: arguments) }
    }
}
