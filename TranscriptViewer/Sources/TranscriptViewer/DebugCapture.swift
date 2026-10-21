import AppKit
import Foundation

/// Development aid: on request, writes each visible window's accessibility elements (role,
/// name, value, frame) and flags small or unnamed targets, without any click or focus change.
/// `scripts/viewer-debug-capture.sh` posts the request, adds a real screenshot of each listed
/// window (`screencapture -l`: in-app drawing misses translucent materials) and prints the folder.
@MainActor
enum DebugCapture {
    static let request = Notification.Name("works.kkkaoru.TranscriptViewer.debugCapture")
    /// Older captures are removed so the folder never grows without bound.
    static let keep = 10
    static let elementLimit = 5_000
    /// Apple HIG's default macOS control size; the app's own buttons use `Chrome.hit`.
    static let minimumTarget = 28.0
    /// The title bar's own buttons are the system's, not the app's targets.
    static let windowChrome: Set<NSAccessibility.Subrole> = [
        .closeButton, .minimizeButton, .zoomButton, .fullScreenButton,
    ]
    static let interactiveRoles: Set<NSAccessibility.Role> = [
        .button, .checkBox, .radioButton, .slider, .popUpButton, .menuButton, .textField,
        .comboBox, .link, .disclosureTriangle, .incrementor,
    ]

    struct Element: Codable, Equatable {
        let depth: Int
        let role: String
        let label: String
        let value: String
        let x: Double
        let y: Double
        let width: Double
        let height: Double
        let interactive: Bool
    }

    struct Report: Codable, Equatable {
        let title: String
        /// The CGWindowID that `screencapture -l` takes.
        let windowNumber: Int
        let width: Double
        let height: Double
        let elements: [Element]
        /// Interactive elements below `minimumTarget` in either direction, or with no name.
        let smallTargets: [Element]
        let unlabeled: [Element]
    }

    /// `edits/work/viewer-debug` of this dist app's checkout; nil outside one.
    static var root: URL? {
        BGMLibraryModel.checkout?.appending(
            path: "edits/work/viewer-debug", directoryHint: .isDirectory)
    }

    private static var observer: NSObjectProtocol?

    /// Answers capture requests for the app's lifetime; nothing outside a checkout's dist app.
    static func listen(root: URL? = root, name: Notification.Name = request) {
        guard let root, observer == nil else { return }
        observer = DistributedNotificationCenter.default().addObserver(
            forName: name, object: nil, queue: .main
        ) { _ in
            MainActor.assumeIsolated { handle(root: root) }
        }
    }

    static func handle(root: URL, windows: [NSWindow] = NSApplication.shared.windows) {
        _ = try? capture(windows.filter(\.isVisible), into: root)
    }

    /// Writes `<stamp>/<n>-<title>.json` per window, `windows.tsv` (window number, name) and
    /// last `done.json` listing them, so a waiting script never reads a half-written capture.
    @discardableResult
    static func capture(_ windows: [NSWindow], into root: URL, now: Date = Date()) throws -> URL {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss-SSS"
        let folder = root.appending(path: formatter.string(from: now), directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var files: [String] = []
        var table = ""
        for (index, window) in windows.enumerated() {
            let name = "\(index)-\(safe(window.title))"
            table += "\(window.windowNumber)\t\(name)\n"
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(report(window)).write(to: folder.appending(path: "\(name).json"))
            files.append("\(name).json")
        }
        try Data(table.utf8).write(to: folder.appending(path: "windows.tsv"))
        try JSONEncoder().encode(files).write(to: folder.appending(path: "done.json"))
        try prune(root)
        return folder
    }

    static func safe(_ title: String) -> String {
        let cleaned = title.unicodeScalars.map {
            CharacterSet.alphanumerics.contains($0) ? String($0) : "_"
        }.joined()
        return cleaned.isEmpty ? "window" : String(cleaned.prefix(60))
    }

    static func report(_ window: NSWindow) -> Report {
        var elements: [Element] = []
        var seen: Set<ObjectIdentifier> = []
        walk(window, depth: 0, window: window, seen: &seen, into: &elements)
        let targets = elements.filter(\.interactive)
        return Report(
            title: window.title, windowNumber: window.windowNumber, width: window.frame.width,
            height: window.frame.height,
            elements: elements,
            smallTargets: targets.filter {
                $0.width < minimumTarget || $0.height < minimumTarget
            },
            unlabeled: targets.filter { $0.label.isEmpty })
    }

    /// Accessibility children plus subviews: a hosting view lists its SwiftUI elements as the
    /// former, and an AppKit view inside it only as the latter.
    private static func walk(
        _ node: NSAccessibilityProtocol, depth: Int, window: NSWindow,
        seen: inout Set<ObjectIdentifier>, into elements: inout [Element]
    ) {
        let subviews: [Any] =
            (node as? NSView)?.subviews ?? (node as? NSWindow)?.contentView
            .map { [$0] } ?? []
        for child in (node.accessibilityChildren() ?? []) + subviews {
            guard elements.count < elementLimit, let child = child as? NSAccessibilityProtocol,
                seen.insert(ObjectIdentifier(child)).inserted
            else { continue }
            // A scroll bar's parts are the system's; a zero-size element is not on screen.
            guard child.accessibilityRole() != .scrollBar else { continue }
            let size = child.accessibilityFrame().size
            if child.isAccessibilityElement(), size.width > 0, size.height > 0,
                !windowChrome.contains(child.accessibilitySubrole() ?? .unknown)
            {
                elements.append(element(child, depth: depth, window: window))
            }
            walk(child, depth: depth + 1, window: window, seen: &seen, into: &elements)
        }
    }

    /// Frames are in window points from the top left, like the PNG.
    private static func element(_ node: NSAccessibilityProtocol, depth: Int, window: NSWindow)
        -> Element
    {
        let frame = window.convertFromScreen(node.accessibilityFrame())
        let label =
            [node.accessibilityLabel(), node.accessibilityTitle()]
            .compactMap { $0 }.first { !$0.isEmpty } ?? ""
        let role = node.accessibilityRole()
        return Element(
            depth: depth, role: role?.rawValue ?? "", label: label,
            value: describe(node.accessibilityValue()),
            x: frame.minX, y: window.frame.height - frame.maxY, width: frame.width,
            height: frame.height,
            interactive: role.map(interactiveRoles.contains) == true)
    }

    private static func describe(_ value: Any?) -> String {
        guard let value else { return "" }
        return "\(value)"
    }

    private static func prune(_ root: URL) throws {
        let folders = try FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        for old in folders.sorted(by: { $0.lastPathComponent > $1.lastPathComponent }).dropFirst(
            keep)
        {
            try FileManager.default.removeItem(at: old)
        }
    }
}
