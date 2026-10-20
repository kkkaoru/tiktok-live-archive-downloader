import CryptoKit
import Foundation

/// Excitement (盛り上がり) per time bin, written by `python -m replay.excitement` beside a
/// transcript as `<transcript>.excitement.json`. The viewer never computes scores: they are
/// relative to the same video (its peak is 1) and the raw values are supporting detail.
/// A file whose recorded transcript hash no longer matches is ignored, so an estimate for an
/// older transcript version is never drawn and never recomputed here.
public struct ExcitementTimeline: Equatable, Sendable {
    public struct Series: Codable, Equatable, Sendable {
        public let score: [Double]
        public let peaks: [Double]
    }

    public struct Speech: Codable, Equatable, Sendable {
        public let score: [Double]
        public let peaks: [Double]
        public let characters: [Double]
        public let loudnessDb: [Double?]?
    }

    public struct Comments: Codable, Equatable, Sendable {
        public let score: [Double]
        public let peaks: [Double]
        public let count: [Int]
        public let characters: [Double]
    }

    public enum Mode: String, CaseIterable, Sendable {
        case combined
        case split
    }

    public struct Row: Equatable, Sendable {
        public let label: String
        public let series: Series
    }

    private struct Inputs: Codable {
        let transcriptSha256: String
    }

    private struct Record: Codable {
        let version: Int
        let binSeconds: Double
        let duration: Double
        let speech: Speech
        let comments: Comments?
        let combined: Series?
        let inputs: Inputs
    }

    public static let version = 1
    static let maximumBytes = 8 << 20

    public let binSeconds: Double
    public let duration: Double
    public let speech: Speech
    public let comments: Comments?
    public let combined: Series?

    public static func url(for transcriptURL: URL) -> URL {
        transcriptURL.deletingLastPathComponent().appending(
            path: transcriptURL.lastPathComponent + ".excitement.json")
    }

    /// The saved estimate for this transcript, or nil when absent, malformed or stale.
    public static func load(transcriptURL: URL) -> ExcitementTimeline? {
        let url = url(for: transcriptURL)
        guard
            let values = try? url.resourceValues(forKeys: [.isSymbolicLinkKey, .fileSizeKey]),
            values.isSymbolicLink == false, let size = values.fileSize, size <= maximumBytes,
            let data = try? Data(contentsOf: url),
            let record = try? JSONDecoder().decode(Record.self, from: data),
            record.version == version,
            let transcript = try? Data(contentsOf: transcriptURL),
            SHA256.hash(data: transcript).map({ String(format: "%02x", $0) }).joined()
                == record.inputs.transcriptSha256
        else { return nil }
        let timeline = ExcitementTimeline(
            binSeconds: record.binSeconds, duration: record.duration, speech: record.speech,
            comments: record.comments, combined: record.combined)
        return timeline.isConsistent ? timeline : nil
    }

    private var isConsistent: Bool {
        let bins = speech.score.count
        let lengths =
            [speech.characters.count, comments?.score.count, comments?.count.count]
            + [comments?.characters.count, combined?.score.count]
        return bins > 0 && binSeconds > 0 && duration > 0
            && (comments == nil) == (combined == nil)
            && lengths.allSatisfy { $0 == nil || $0 == bins }
    }

    public var binCount: Int { speech.score.count }

    /// Rows to draw: speech alone without comments; otherwise the combined row or both.
    public func rows(_ mode: Mode) -> [Row] {
        let speechRow = Row(label: "発話", series: Series(score: speech.score, peaks: speech.peaks))
        guard let comments, let combined else { return [speechRow] }
        let commentRow = Row(
            label: "コメント", series: Series(score: comments.score, peaks: comments.peaks))
        return mode == .split ? [speechRow, commentRow] : [Row(label: "合算", series: combined)]
    }

    public func bin(atFraction fraction: Double) -> Int {
        min(binCount - 1, max(0, Int(fraction * Double(binCount))))
    }

    public func seconds(ofBin bin: Int) -> Double { Double(bin) * binSeconds }

    /// Relative scores with the absolute values behind them, for one bin.
    public func detail(ofBin bin: Int) -> String {
        let percent = { (value: Double) in "\(Int((value * 100).rounded()))%" }
        var parts = ["\(Self.clock(seconds(ofBin: bin)))〜"]
        if let combined { parts.append("合算 \(percent(combined.score[bin]))") }
        var spoken = "\(Int(speech.characters[bin].rounded()))文字"
        if let levels = speech.loudnessDb, let level = levels[bin] { spoken += "・\(level)dB" }
        parts.append("発話 \(percent(speech.score[bin]))（\(spoken)）")
        if let comments {
            let characters = Int(comments.characters[bin])
            parts.append(
                "コメント \(percent(comments.score[bin]))（\(comments.count[bin])件・\(characters)文字）")
        }
        return parts.joined(separator: " / ")
    }

    public static func clock(_ seconds: Double) -> String {
        let total = Int(seconds)
        return String(format: "%d:%02d:%02d", total / 3600, total % 3600 / 60, total % 60)
    }
}
