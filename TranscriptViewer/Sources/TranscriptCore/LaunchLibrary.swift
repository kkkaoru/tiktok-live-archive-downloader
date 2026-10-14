import Foundation

public enum LaunchLibraryError: Error, LocalizedError {
    case arguments
    case unsupportedURL
    case unreadableDirectory(String)

    public var errorDescription: String? {
        switch self {
        case .arguments:
            "起動引数は --library と絶対パスのフォルダを指定してください。"
        case .unsupportedURL:
            "ライブラリにはこのMacのフォルダURLを指定してください。"
        case .unreadableDirectory(let path):
            "ライブラリを読み込めません: \(path)"
        }
    }
}

/// An explicit library selection; never writes the user's saved folder choice.
public enum LaunchLibrary {
    public static func resolve(arguments: [String], fileManager: FileManager = .default)
        throws(LaunchLibraryError)
        -> URL?
    {
        let options = arguments.filter { !$0.hasPrefix("-psn_") }
        guard !options.isEmpty else { return nil }
        guard options.count == 2, options[0] == "--library", options[1].hasPrefix("/") else {
            throw LaunchLibraryError.arguments
        }
        return try resolve(
            url: URL(fileURLWithPath: options[1], isDirectory: true),
            fileManager: fileManager)
    }

    /// Accept only local directory URLs from an explicit Open Document request.
    public static func resolve(url: URL, fileManager: FileManager = .default)
        throws(LaunchLibraryError) -> URL
    {
        guard url.isFileURL, url.path.hasPrefix("/"),
            url.host == nil || url.host == "" || url.host == "localhost",
            url.query == nil, url.fragment == nil
        else { throw LaunchLibraryError.unsupportedURL }
        let root = url.standardizedFileURL
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: root.path, isDirectory: &isDirectory),
            isDirectory.boolValue, fileManager.isReadableFile(atPath: root.path)
        else {
            throw LaunchLibraryError.unreadableDirectory(root.path)
        }
        return root
    }
}
