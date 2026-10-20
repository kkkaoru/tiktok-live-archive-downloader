import Foundation

/// Conservative surface checks, not a semantic entailment or factuality proof.
public enum SummaryProseQuality {
    /// Drops a sentence (with its terminator) whose exact text already appeared. Nothing is
    /// reworded or added; the repetition check below still applies to what remains.
    public static func removingRepeatedSentences(_ text: String) -> String {
        var seen: Set<String> = []
        var result = ""
        var sentence = ""
        func flush() {
            let core = sentence.trimmingCharacters(
                in: CharacterSet(charactersIn: "。！？\n").union(.whitespacesAndNewlines))
            if core.count < 12 || seen.insert(core).inserted { result += sentence }
            sentence = ""
        }
        for character in text {
            sentence.append(character)
            if "。！？\n".contains(character) { flush() }
        }
        flush()
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func validate(_ text: String) throws {
        let sentences = text.components(separatedBy: CharacterSet(charactersIn: "。！？\n"))
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.count >= 12 }
        guard Set(sentences).count == sentences.count else {
            throw SummaryGenerationError.repeatedResponse
        }
        // Explicit changes of view and quoted alternatives require semantic review;
        // do not reject them using a surface polarity comparison.
        let contextual = ["当初", "後で", "訂正", "以前", "現在", "一方", "「", "」"]
        guard !contextual.contains(where: text.contains) else { return }
        let clauses = text.components(separatedBy: CharacterSet(charactersIn: "。！？、\n"))
            .map { clause in
                let trimmed = clause.trimmingCharacters(in: .whitespacesAndNewlines)
                return trimmed.hasSuffix("が") ? String(trimmed.dropLast()) : trimmed
            }
        let pairs = [("高い", "低い"), ("多い", "少ない"), ("可能です", "不可能です")]
        for (positive, negative) in pairs {
            let positiveClauses = Set(clauses.filter { $0.contains(positive) && $0.count >= 12 })
            let opposites = clauses.filter { $0.contains(negative) }.map {
                $0.replacingOccurrences(of: negative, with: positive)
            }
            if opposites.contains(where: positiveClauses.contains) {
                throw SummaryGenerationError.contradictoryResponse
            }
        }
    }
}
