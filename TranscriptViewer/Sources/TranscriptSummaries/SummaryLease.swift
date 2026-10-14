import Darwin
import Foundation
import TranscriptCore

/// A stable, kernel-owned per-transcript lock. Closing the handle (including process
/// exit) releases ownership; the file remains so concurrent clients share one inode.
struct SummaryLease: Sendable {
    private let handle: FileHandle

    static func acquire(video: VideoTranscript, store: SummaryStore) async throws -> SummaryLease {
        try await SummaryStore.io {
            let directory = store.directory(for: video)
            try SummaryStore.prepareDirectory(directory)
            let url = directory.appendingPathComponent(".generation.lock")
            if FileManager.default.fileExists(atPath: url.path) {
                let values = try url.resourceValues(forKeys: [
                    .isRegularFileKey, .isSymbolicLinkKey,
                ])
                guard values.isRegularFile == true, values.isSymbolicLink != true else {
                    throw SummaryStoreError.unsafePath
                }
            }
            let descriptor = open(url.path, O_CREAT | O_RDWR | O_NOFOLLOW | O_NONBLOCK, 0o600)
            guard descriptor >= 0 else { throw SummaryStoreError.unsafePath }
            let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
            guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
                let code = errno
                try handle.close()
                if code == EWOULDBLOCK { throw SummaryStoreError.busy }
                throw SummaryStoreError.unsafePath
            }
            return SummaryLease(handle: handle)
        }
    }

    func release() async throws {
        try await SummaryStore.io { try handle.close() }
    }
}
