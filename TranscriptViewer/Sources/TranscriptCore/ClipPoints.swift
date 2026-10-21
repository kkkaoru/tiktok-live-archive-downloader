import Foundation

/// One range of a clip, in seconds of the movie the transcript follows.
public struct ClipRange: Codable, Equatable, Sendable {
    public let start: Double
    public let end: Double
    public var duration: Double { end - start }

    public init(start: Double, end: Double) {
        self.start = start
        self.end = end
    }
}

/// A clip point computed by the web viewer's Worker: one or more ranges of the stream that
/// play in order, 10-30 s ("short") or 30-90 s ("long") in total, with a generated title and
/// summary.
public struct ClipPoint: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let kind: String
    /// "climax-first" (the peak, then its lead-up) or "hook-to-climax" (a hook rising to the peak).
    public let pattern: String?
    public let duration: Double
    public let ranges: [ClipRange]
    public let title: String
    public let summary: String

    public init(
        id: String, kind: String, pattern: String? = nil, duration: Double, ranges: [ClipRange],
        title: String, summary: String
    ) {
        self.id = id
        self.kind = kind
        self.pattern = pattern
        self.duration = duration
        self.ranges = ranges
        self.title = title
        self.summary = summary
    }

    public var kindLabel: String {
        kind == "long" ? "長尺 30〜90秒" : "短尺 10〜30秒"
    }

    public var patternLabel: String? {
        switch pattern {
        case "climax-first": "山場→経緯"
        case "hook-to-climax": "引き→山場"
        default: nil
        }
    }

    /// Seconds already played when range `index` is at `seconds`.
    public func elapsed(range index: Int, at seconds: Double) -> Double {
        let before = ranges.prefix(index).reduce(0) { $0 + $1.duration }
        guard ranges.indices.contains(index) else { return before }
        let range = ranges[index]
        return before + max(0, min(seconds, range.end) - range.start)
    }

    public static func seconds(_ value: Double) -> String {
        String(format: "%.1f秒", value)
    }
}

/// What the Worker answers for one video.
public enum ClipStatus: Equatable, Sendable {
    case ready([ClipPoint])
    /// Being computed; ask again later.
    case pending
    /// The published movie differs from this one, so its clips do not apply.
    case mismatch
    case missing
    /// Computed before the video had excitement scores, so no clip could be chosen.
    case noExcitement

    private struct Wire: Decodable {
        struct Record: Decodable {
            let clips: [ClipPoint]
            let excitement: Bool?
        }
        let status: String
        let record: Record?
    }

    public static func decode(_ data: Data) throws -> ClipStatus {
        let wire = try JSONDecoder().decode(Wire.self, from: data)
        switch wire.status {
        case "ready":
            guard let record = wire.record else { return .missing }
            return record.excitement == false ? .noExcitement : .ready(record.clips)
        case "pending": return .pending
        case "mismatch": return .mismatch
        default: return .missing
        }
    }
}

/// The web viewer's clip API: its first custom domain (`web-viewer/deploy.env` VIEWER_DOMAIN)
/// and the bearer token the Worker accepts for clips (`web-viewer/secrets.json`
/// CLIPS_API_TOKEN). Both files are untracked; the token is only ever sent to that domain.
public struct ClipsAPI: Equatable, Sendable {
    public let base: URL
    public let token: String

    public init(base: URL, token: String) {
        self.base = base
        self.token = token
    }

    public static func load(checkout: URL) -> ClipsAPI? {
        let folder = checkout.appending(path: "web-viewer", directoryHint: .isDirectory)
        guard
            let env = try? String(
                contentsOf: folder.appending(path: "deploy.env"), encoding: .utf8),
            let secrets = try? Data(contentsOf: folder.appending(path: "secrets.json")),
            let values = try? JSONDecoder().decode([String: String].self, from: secrets),
            let token = values["CLIPS_API_TOKEN"], token.count >= 32,
            let host = domain(in: env),
            let base = URL(string: "https://\(host)")
        else { return nil }
        return ClipsAPI(base: base, token: token)
    }

    public static func domain(in env: String) -> String? {
        env.split(whereSeparator: \.isNewline).lazy
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { $0.hasPrefix("VIEWER_DOMAIN=") }
            .flatMap {
                $0.dropFirst("VIEWER_DOMAIN=".count).split(separator: ",").first.map {
                    $0.trimmingCharacters(in: CharacterSet(charactersIn: " \"'"))
                }
            }
            .flatMap { $0.isEmpty ? nil : $0 }
    }

    /// The web catalog's ID: the movie's file name without its extension.
    public static func videoID(_ movie: URL) -> String {
        movie.deletingPathExtension().lastPathComponent
    }

    public func lookup(videoID: String, movieKey: String) -> URLRequest {
        let url = base.appending(path: "api/clips/\(videoID).json")
            .appending(queryItems: [URLQueryItem(name: "key", value: movieKey)])
        return authorized(URLRequest(url: url), method: "GET")
    }

    public func reload(videoID: String) -> URLRequest {
        authorized(
            URLRequest(url: base.appending(path: "api/clips/\(videoID)/reload")), method: "POST")
    }

    private func authorized(_ request: URLRequest, method: String) -> URLRequest {
        var request = request
        request.httpMethod = method
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        return request
    }
}

/// Plays a clip's ranges one after another as playback time advances.
public struct ClipCursor: Equatable, Sendable {
    public enum Step: Equatable, Sendable {
        case keep
        case jump(to: Double)
        /// The last range ended.
        case finish
        /// Playback moved away from the clip (a seek elsewhere).
        case leave
    }

    /// How far outside the playing range counts as a seek away rather than jitter.
    public static let slack = 1.5
    public let clip: ClipPoint
    public private(set) var index: Int
    /// False until playback reports a time inside the range just sought to: until then the
    /// player may still report the time from before the seek.
    public private(set) var settled = false
    /// Starts over from the first range after the last one instead of finishing.
    public let loop: Bool

    public init(clip: ClipPoint, index: Int = 0, loop: Bool = false) {
        self.clip = clip
        self.loop = loop
        self.index = min(max(index, 0), max(clip.ranges.count - 1, 0))
    }

    public var range: ClipRange? { clip.ranges.indices.contains(index) ? clip.ranges[index] : nil }

    public mutating func step(at seconds: Double) -> Step {
        guard let range else { return .finish }
        if !settled {
            guard seconds >= range.start - Self.slack, seconds <= range.end else { return .keep }
            settled = true
        }
        if seconds < range.start - Self.slack || seconds > range.end + Self.slack { return .leave }
        guard seconds >= range.end else { return .keep }
        guard index + 1 < clip.ranges.count || loop else { return .finish }
        index = (index + 1) % clip.ranges.count
        settled = false
        return .jump(to: clip.ranges[index].start)
    }
}
