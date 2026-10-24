import Foundation
import TranscriptCore

/// Drops exactly repeated sentences from the summary texts of a published viewer data JSON
/// (its `summary`) or of a saved summary record (top-level `overview` and `sections`), the
/// removal saved summaries get on load. Everything else in the file is kept as parsed.
enum PublishedSummaryCleaner {
    /// Copies the original to `backup` (which must not exist), then rewrites `data`.
    /// Returns how many characters were removed.
    static func clean(data: URL, backup: URL) throws -> Int {
        let original = try Data(contentsOf: data)
        guard var document = try JSONSerialization.jsonObject(with: original) as? [String: Any]
        else { throw SummaryStoreError.invalidRecord }
        let nested = document["summary"] != nil
        guard var summary = nested ? document["summary"] as? [String: Any] : document,
            summary["overview"] != nil
        else { throw SummaryStoreError.invalidRecord }
        var removed = 0
        func cleaned(_ value: Any?) throws -> String {
            guard let text = value as? String else { throw SummaryStoreError.invalidRecord }
            let result = SummaryProseQuality.removingRepeatedSentences(text)
            removed += text.count - result.count
            return result
        }
        summary["overview"] = try cleaned(summary["overview"])
        guard let sections = summary["sections"] as? [[String: Any]] else {
            throw SummaryStoreError.invalidRecord
        }
        summary["sections"] = try sections.map { section in
            var section = section
            section["text"] = try cleaned(section["text"])
            return section
        }
        document = nested ? document.merging(["summary": summary]) { $1 } : summary
        let rewritten = try JSONSerialization.data(
            withJSONObject: document, options: [.sortedKeys, .withoutEscapingSlashes])
        try SummaryStore.writePrivate(original, to: backup, atomic: false)
        try SummaryStore.writePrivate(rewritten, to: data, atomic: true)
        return removed
    }
}
