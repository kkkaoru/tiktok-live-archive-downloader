import Foundation

public enum TranscriptCSV {
    public static let header = ["start_seconds", "end_seconds", "start", "end", "text"]

    public static func cues(from text: String) throws -> [TranscriptCue] {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(
            of: "\r", with: "\n")
        var parser = CSVParser(text: normalized)
        guard let first = parser.nextRow(), first == header else {
            throw TranscriptError.missingHeader
        }
        var rows: [TranscriptCue] = []
        var index = 1
        while let fields = parser.nextRow() {
            index += 1
            guard fields.count == 5 else {
                throw TranscriptError.malformedRow(index)
            }
            guard let startSeconds = Double(fields[0]), let endSeconds = Double(fields[1]) else {
                throw TranscriptError.invalidClock(fields[0])
            }
            guard startSeconds.isFinite, endSeconds.isFinite, endSeconds >= startSeconds else {
                throw TranscriptError.invalidClock(fields[0])
            }
            rows.append(
                TranscriptCue(
                    id: rows.count,
                    startSeconds: startSeconds,
                    endSeconds: endSeconds,
                    startLabel: fields[2],
                    endLabel: fields[3],
                    text: fields[4]
                )
            )
        }
        return rows
    }

    public static func cues(contentsOf url: URL) throws -> [TranscriptCue] {
        try cues(from: String(contentsOf: url, encoding: .utf8))
    }
}

struct CSVParser {
    private let characters: [Character]
    private var offset = 0

    init(text: String) {
        characters = Array(text)
    }

    mutating func nextRow() -> [String]? {
        if offset >= characters.count {
            return nil
        }
        if offset == characters.count - 1, characters[offset] == "\n" {
            offset += 1
            return nil
        }
        var fields: [String] = []
        while offset < characters.count {
            fields.append(nextField())
            if offset >= characters.count {
                break
            }
            let current = characters[offset]
            if current == "," {
                offset += 1
                continue
            }
            if current == "\n" {
                offset += 1
                break
            }
            fields[fields.count - 1].append(current)
            offset += 1
        }
        return fields
    }

    private mutating func nextField() -> String {
        if offset < characters.count, characters[offset] == "\"" {
            offset += 1
            var field = ""
            while offset < characters.count {
                let current = characters[offset]
                if current == "\"" {
                    offset += 1
                    if offset < characters.count, characters[offset] == "\"" {
                        field.append("\"")
                        offset += 1
                        continue
                    }
                    return field
                }
                field.append(current)
                offset += 1
            }
            return field
        }
        var field = ""
        while offset < characters.count {
            let current = characters[offset]
            if current == "," || current == "\n" {
                break
            }
            field.append(current)
            offset += 1
        }
        return field
    }
}
