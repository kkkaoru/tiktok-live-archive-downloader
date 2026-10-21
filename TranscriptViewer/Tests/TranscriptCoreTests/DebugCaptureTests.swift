import AppKit
import Foundation
import Testing

@testable import TranscriptViewer

@MainActor
struct DebugCaptureTests {
    static func window() -> NSWindow {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 300, height: 200), styleMask: [.titled],
            backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.title = "Root / 1"
        let content = NSView(frame: NSRect(x: 0, y: 0, width: 300, height: 200))
        let small = NSButton(title: "OK", target: nil, action: nil)
        small.frame = NSRect(x: 10, y: 150, width: 40, height: 20)
        let unnamed = NSButton(frame: NSRect(x: 10, y: 10, width: 60, height: 60))
        unnamed.title = ""
        unnamed.image = NSImage(size: NSSize(width: 8, height: 8))
        let text = NSTextField(labelWithString: "Label")
        text.frame = NSRect(x: 100, y: 100, width: 80, height: 20)
        content.addSubview(small)
        content.addSubview(unnamed)
        content.addSubview(text)
        let scroll = NSScrollView(frame: NSRect(x: 200, y: 10, width: 80, height: 80))
        scroll.hasVerticalScroller = true
        scroll.documentView = NSView(frame: NSRect(x: 0, y: 0, width: 80, height: 400))
        content.addSubview(scroll)
        window.contentView = content
        return window
    }

    @Test func aCaptureSavesEachWindowWithItsTargetsAndKeepsTheLatestTen() throws {
        let root = try transcriptTestWorkRoot().appending(
            path: "debug-capture-\(UUID().uuidString)", directoryHint: .isDirectory)
        defer { #expect(throws: Never.self) { try FileManager.default.removeItem(at: root) } }
        for index in 0..<11 {
            try FileManager.default.createDirectory(
                at: root.appending(path: "20000101-000000-\(100 + index)"),
                withIntermediateDirectories: true)
        }
        let window = Self.window()
        let folder = try DebugCapture.capture(
            [window], into: root, now: Date(timeIntervalSince1970: 1_800_000_000))

        #expect(folder.lastPathComponent.hasPrefix("2027"))
        let listed = try JSONDecoder().decode(
            [String].self, from: Data(contentsOf: folder.appending(path: "done.json")))
        #expect(listed == ["0-Root___1.json"])
        #expect(
            try String(contentsOf: folder.appending(path: "windows.tsv"), encoding: .utf8)
                == "\(window.windowNumber)\t0-Root___1\n")
        let report = try JSONDecoder().decode(
            DebugCapture.Report.self,
            from: Data(contentsOf: folder.appending(path: "0-Root___1.json")))
        #expect(report.title == "Root / 1")
        #expect(report.windowNumber == window.windowNumber)
        #expect(!report.elements.contains { $0.role == "AXScrollBar" })
        #expect(report.smallTargets.map(\.label) == ["OK"])
        #expect(report.unlabeled.count == 1)
        #expect(report.elements.contains { $0.value == "Label" && !$0.interactive })
        let ok = try #require(report.elements.first { $0.label == "OK" })
        // AppKit reports the button with its bezel inset, in window points from the top left.
        #expect((8...10).contains(ok.x))
        #expect((55...65).contains(ok.y))  // below the 28pt title bar
        #expect((40...44).contains(ok.width))
        let kept = try FileManager.default.contentsOfDirectory(atPath: root.path)
        #expect(kept.count == DebugCapture.keep)
        #expect(!kept.contains("20000101-000000-100"))
        #expect(DebugCapture.safe("") == "window")
    }

    @Test func requestsCaptureOnlyVisibleWindowsAndNeedACheckout() throws {
        let root = try transcriptTestWorkRoot().appending(
            path: "debug-capture-\(UUID().uuidString)", directoryHint: .isDirectory)
        defer { #expect(throws: Never.self) { try FileManager.default.removeItem(at: root) } }
        DebugCapture.handle(root: root, windows: [Self.window()])
        let folder = try #require(
            try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
                .first)
        let listed = try JSONDecoder().decode(
            [String].self, from: Data(contentsOf: folder.appending(path: "done.json")))
        #expect(listed.isEmpty)
        DebugCapture.listen(root: nil)
        #expect(DebugCapture.root == nil)
    }

    /// A private request name, so a running Viewer never answers this test.
    @Test func aDistributedRequestTriggersACapture() async throws {
        let root = try transcriptTestWorkRoot().appending(
            path: "debug-capture-\(UUID().uuidString)", directoryHint: .isDirectory)
        defer { #expect(throws: Never.self) { try FileManager.default.removeItem(at: root) } }
        let name = Notification.Name("works.kkkaoru.TranscriptViewer.test-\(UUID().uuidString)")
        DebugCapture.listen(root: root, name: name)
        DistributedNotificationCenter.default().postNotificationName(
            name, object: nil, userInfo: nil, deliverImmediately: true)
        for _ in 0..<500 where !FileManager.default.fileExists(atPath: root.path) {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(FileManager.default.fileExists(atPath: root.path))
    }
}
