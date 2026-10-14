import Foundation
import TranscriptCore
import TranscriptSummaries

let defaultRoot = FileManager.default.homeDirectoryForCurrentUser
    .appending(path: "Movies", directoryHint: .isDirectory)
    .appending(path: "Apple-Pro-Apps-Verification", directoryHint: .isDirectory)
let root =
    CommandLine.arguments.count > 1
    ? URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    : defaultRoot

do {
    if CommandLine.arguments.dropFirst().first == "--representative" {
        guard CommandLine.arguments.count == 4 else { throw SummaryStoreError.invalidRecord }
        let movie = URL(fileURLWithPath: CommandLine.arguments[2])
        let transcript = URL(fileURLWithPath: CommandLine.arguments[3])
        let video = VideoTranscript(
            title: "inspection", movieURL: movie, transcriptURL: transcript,
            cues: try TranscriptSRT.cues(contentsOf: transcript))
        guard let candidate = try await RepresentativeSummaryStore().load(video: video) else {
            print("REPRESENTATIVE_MISSING")
            exit(1)
        }
        print(
            "REPRESENTATIVE_CANDIDATE \(candidate.prose?.label ?? candidate.topics?.label ?? candidate.summary.label) CLOCK \(candidate.summary.clockDomain.rawValue) SECTIONS \(candidate.summary.source.sections.count) EXCERPTS \(candidate.summary.clocks.count) SEEK \(candidate.canSeekSelectedMovie)"
        )
        exit(0)
    }
    let scan = try TranscriptCatalog.scan(in: root)
    for video in scan.videos {
        print("OK \(video.title)/\(video.movieURL.lastPathComponent) cues=\(video.cues.count)")
    }
    for issue in scan.issues {
        print("NG \(issue)")
    }
    print(
        "LOADED \(scan.videos.count) ISSUES \(scan.issues.count) ROOT \(root.path(percentEncoded: false))"
    )
    exit(scan.issues.isEmpty ? 0 : 1)
} catch {
    let reason = error.localizedDescription
    print("ERROR \(reason)")
    exit(2)
}
