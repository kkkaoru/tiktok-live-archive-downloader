import Foundation

/// Additional libraries for ordinary startup. Recording selection, dates and
/// transcript integrity remain the catalog's responsibility, not this configuration's.
public struct DefaultLibraryManifest: Codable, Sendable {
    public static let filename = "viewer-default-libraries.json"
    private static let byteLimit = 65_536
    private static let libraryLimit = 16

    public let version: Int
    public let libraries: [String]

    public enum Invalid: Error, LocalizedError {
        case manifest, path

        public var errorDescription: String? {
            switch self {
            case .manifest: "追加ライブラリの設定が不正です。"
            case .path: "追加ライブラリには実体のある絶対パスを指定してください。"
            }
        }
    }

    /// Missing configuration is intentional absence; malformed or linked configuration fails.
    /// A missing library itself stays configured so a later volume mount can be refreshed.
    public static func load(at manifest: URL, fileManager: FileManager = .default) throws -> [URL] {
        guard manifest.isFileURL, manifest.query == nil, manifest.fragment == nil,
            manifest.host == nil || manifest.host == "" || manifest.host == "localhost",
            manifest.resolvingSymlinksInPath() == manifest.standardizedFileURL,
            !manifest.pathComponents.contains(where: { ["private", "downloads"].contains($0) })
        else { throw Invalid.path }
        guard try entryExists(at: manifest, fileManager: fileManager) else { return [] }
        guard
            try fileManager.attributesOfItem(atPath: manifest.path)[.type] as? FileAttributeType
                == .typeRegular
        else { throw Invalid.manifest }
        let properties = try manifest.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard properties.isRegularFile == true, let size = properties.fileSize,
            size > 0, size <= byteLimit
        else { throw Invalid.manifest }
        let value = try JSONDecoder().decode(Self.self, from: Data(contentsOf: manifest))
        guard value.version == 1, value.libraries.count <= libraryLimit else {
            throw Invalid.manifest
        }
        var result: [URL] = []
        for path in value.libraries {
            let url = URL(fileURLWithPath: path, isDirectory: true)
            guard path.hasPrefix("/"), !path.contains("\0"), !path.contains("\\"),
                url.standardizedFileURL.path == path,
                url.resolvingSymlinksInPath() == url.standardizedFileURL,
                !url.pathComponents.contains(where: { ["private", "downloads"].contains($0) }),
                !result.contains(url)
            else { throw Invalid.path }
            if try entryExists(at: url, fileManager: fileManager) {
                guard
                    try fileManager.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType
                        == .typeDirectory
                else { throw Invalid.path }
            }
            result.append(url)
        }
        return result
    }

    private static func entryExists(at url: URL, fileManager: FileManager) throws -> Bool {
        let parent = url.deletingLastPathComponent()
        guard fileManager.fileExists(atPath: parent.path) else { return false }
        return try fileManager.contentsOfDirectory(atPath: parent.path).contains(
            url.lastPathComponent)
    }

    /// Read small configuration off UI isolation, with exactly one callback.
    public static func load(
        at manifest: URL,
        completion: @escaping @MainActor @Sendable (Result<[URL], Error>) -> Void
    ) {
        DispatchQueue.global(qos: .utility).async {
            let result = Result { try load(at: manifest) }
            DispatchQueue.main.async { completion(result) }
        }
    }
}
