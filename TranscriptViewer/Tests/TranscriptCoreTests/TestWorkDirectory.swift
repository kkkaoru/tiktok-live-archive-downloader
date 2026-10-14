import Foundation

private enum TestWorkDirectoryError: Error {
    case outsideCheckout
}

/// Foundation's temporaryDirectory can ignore TMPDIR on macOS; use this checkout explicitly.
func transcriptTestWorkRoot() throws -> URL {
    let checkout = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .resolvingSymlinksInPath()
    let root = checkout.appending(path: "edits/work/transcript-tests", directoryHint: .isDirectory)
        .resolvingSymlinksInPath()
    guard root.path.hasPrefix(checkout.path + "/") else {
        throw TestWorkDirectoryError.outsideCheckout
    }
    return root
}
