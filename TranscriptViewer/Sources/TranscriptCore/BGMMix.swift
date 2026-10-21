import CryptoKit
import Foundation

/// The BGM saved for one video and how loud it plays under that video.
public struct VideoBGMSetting: Codable, Equatable, Sendable {
    public enum Selection: Codable, Equatable, Sendable {
        /// The BGM tab's default track, whichever it is when the video plays.
        case libraryDefault
        case off
        case track(String)
    }

    public enum Mode: String, Codable, CaseIterable, Sendable {
        case automatic
        case manual
    }

    public var selection = Selection.libraryDefault
    public var mode = Mode.automatic
    /// Manual mode: the BGM player's volume, 0...1.
    public var volume: Float = 0.3
    /// Automatic mode: the BGM's level relative to the video's audio around the playhead.
    public var offsetDB = -22.0

    public init() {}

    public func trackID(default defaultID: String?) -> String? {
        switch selection {
        case .libraryDefault: defaultID
        case .off: nil
        case .track(let id): id
        }
    }

    /// One saved setting per recording, so an offloaded or revised movie keeps it.
    public static func key(for video: VideoTranscript) -> String {
        video.recordingID ?? video.movieURL.lastPathComponent
    }
}

/// Levels (dBFS) for following a video's loudness with the BGM. The viewer never decodes audio:
/// a video's per-second levels come from `python -m replay.excitement` (`levels` or `catalog`)
/// and a BGM's mean level from its `track.json`.
public enum BGMLevel {
    /// Quieter seconds are pauses; they never pull the BGM down (`excitement.AUDIBLE_DB`).
    static let silenceDB = -50.0
    static let keySampleBytes: UInt64 = 1 << 20
    static let levelsLimit = 16 << 20
    /// Assumed while a level is missing: talk with peaks near full scale, and a loud mastered
    /// BGM, so an unmeasured video errs on the quiet side.
    public static let typicalVideoDB = -20.0
    public static let typicalBGMDB = -10.0

    /// Each second's level averaged in power over ±`radius` seconds, held through pauses;
    /// empty when nothing is audible.
    public static func smoothed(_ levels: [Double], radius: Int = 5) -> [Double] {
        let power = levels.map { $0 > silenceDB ? pow(10, $0 / 10) : nil }
        let averaged = power.indices.map { index in
            mean(power[max(0, index - radius)...min(power.count - 1, index + radius)])
        }
        guard var held = averaged.compactMap({ $0 }).first else { return [] }
        return averaged.map { value in
            held = value ?? held
            return held
        }
    }

    private static func mean(_ power: ArraySlice<Double?>) -> Double? {
        let audible = power.compactMap { $0 }
        return audible.isEmpty ? nil : 10 * log10(audible.reduce(0, +) / Double(audible.count))
    }

    /// The player volume that puts a BGM of level `bgm` at `offsetDB` from `video`.
    public static func gain(video: Double, bgm: Double, offsetDB: Double) -> Float {
        Float(min(max(pow(10, (video + offsetDB - bgm) / 20), 0), 1))
    }

    /// `replay.excitement.movie_key`: the size, then the first and last MiB, so a moved or
    /// offloaded copy keeps its levels.
    public static func movieKey(_ movie: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: movie)
        defer { try? handle.close() }
        let size = try handle.seekToEnd()
        var hasher = SHA256()
        hasher.update(data: Data(String(size).utf8))
        for offset in [0, size > keySampleBytes ? size - keySampleBytes : 0] {
            try handle.seek(toOffset: offset)
            hasher.update(data: try handle.read(upToCount: Int(keySampleBytes)) ?? Data())
        }
        return String(hasher.finalize().map { String(format: "%02x", $0) }.joined().prefix(32))
    }

    /// The movie's precomputed per-second levels in `cache`; nil when none are stored.
    public static func cachedLevels(for movie: URL, in cache: URL) -> [Double]? {
        guard let key = try? movieKey(movie) else { return nil }
        let url = cache.appending(path: "levels-\(key).json")
        guard
            let values = try? url.resourceValues(forKeys: [
                .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
            ]),
            values.isRegularFile == true, values.isSymbolicLink != true,
            let size = values.fileSize, size <= levelsLimit,
            let data = try? Data(contentsOf: url),
            let levels = try? JSONDecoder().decode([Double].self, from: data)
        else { return nil }
        return levels
    }
}
