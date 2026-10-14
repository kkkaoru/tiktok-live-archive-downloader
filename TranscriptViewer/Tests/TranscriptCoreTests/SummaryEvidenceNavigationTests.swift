import AppKit
import SwiftUI
import Testing
import ViewInspector

@testable import TranscriptCore
@testable import TranscriptSummaries
@testable import TranscriptViewer

private struct EvidenceNavigationFixture {
    let clock = RepresentativeSourceSummary.Clock(
        span: .init(
            sectionStartSeconds: 600, lowerCharacter: 7, upperCharacter: 18,
            text: "明日は京都に行く予定。"),
        startSeconds: 614.25, endSeconds: 619.5)

    func summary(
        fingerprint: String = "synthetic-first", morphology: String = "synthetic-pos",
        version: Int = 2, domain: RepresentativeSourceSummary.ClockDomain = .editedOutput,
        includeSection: Bool = true, includeEvidence: Bool = true
    ) -> SourceSupportedProse {
        .init(
            version: version, method: SourceSupportedProse.methodName,
            label: SourceSupportedProse.displayLabel, fingerprint: fingerprint,
            morphologyFingerprint: morphology, clockDomain: domain,
            overview: "旅行の予定を話している。", overviewEvidence: [clock],
            sections: includeSection
                ? [
                    .init(
                        startSeconds: 600, endSeconds: 1200,
                        text: "京都へ行く予定について話している。",
                        evidence: includeEvidence ? [clock] : [])
                ] : [])
    }

    func candidate(_ summary: SourceSupportedProse?) -> RepresentativeSummaryCandidate {
        .init(
            summary: self.summary().representativeEvidence, canSeekSelectedMovie: true,
            prose: summary)
    }
}

/// This marker measures a hosted scroll destination without desktop AX or screenshots.
private struct EvidenceViewportMarker: NSViewRepresentable {
    final class NativeView: NSView {}
    func makeNSView(context _: Context) -> NativeView { NativeView() }
    func updateNSView(_: NativeView, context _: Context) {}
}

@MainActor
struct SummaryEvidenceNavigationTests {
    @Test func targetRetainsTheExactSourceRangeAndChangesWithItsBoundSource() throws {
        let fixture = EvidenceNavigationFixture()
        let summary = fixture.summary()
        let target = try #require(
            SummaryEvidenceTarget.first(in: .representative(fixture.candidate(summary))))
        #expect(target.transcriptFingerprint == "synthetic-first")
        #expect(target.morphologyFingerprint == "synthetic-pos")
        #expect(target.version == 2)
        #expect(target.clockDomain == "edited-output")
        #expect(target.sectionStartSeconds == 600)
        #expect(target.lowerCharacter == 7)
        #expect(target.upperCharacter == 18)
        #expect(target == SummaryEvidenceTarget(summary: summary, clock: fixture.clock))
        #expect(
            target
                != SummaryEvidenceTarget(
                    summary: fixture.summary(fingerprint: "synthetic-second"), clock: fixture.clock)
        )
        #expect(
            target
                != SummaryEvidenceTarget(
                    summary: fixture.summary(morphology: "different-pos"), clock: fixture.clock))
        #expect(
            target
                != SummaryEvidenceTarget(
                    summary: fixture.summary(version: 1), clock: fixture.clock))
        #expect(
            target
                != SummaryEvidenceTarget(
                    summary: fixture.summary(domain: .sourcePresentation), clock: fixture.clock))
        #expect(fixture.clock.startSeconds == 614.25)
        #expect(fixture.clock.endSeconds == 619.5)
    }

    @Test func missingAndNonProseStatesDoNotInventADestination() {
        let fixture = EvidenceNavigationFixture()
        #expect(SummaryEvidenceTarget.first(in: .absent) == nil)
        #expect(SummaryEvidenceTarget.first(in: .failed) == nil)
        #expect(
            SummaryEvidenceTarget.first(
                in: .ready(
                    .init(
                        summary: fixture.summary().representativeEvidence.source,
                        canSeekSelectedMovie: false))) == nil)
        #expect(
            SummaryEvidenceTarget.first(in: .representative(fixture.candidate(nil))) == nil)
        #expect(
            SummaryEvidenceTarget.first(
                in: .representative(fixture.candidate(fixture.summary(includeSection: false))))
                == nil)
        #expect(
            SummaryEvidenceTarget.first(
                in: .representative(fixture.candidate(fixture.summary(includeEvidence: false))))
                == nil)
    }

    @Test(arguments: [false, true])
    func evidenceAnchorKeepsTheExistingSeekEligibilityAndClock(_ canSeek: Bool) throws {
        let fixture = EvidenceNavigationFixture()
        let summary = fixture.summary()
        var sought: Double?
        let inspected = try SourceSupportedProsePanel(
            summary: summary, canSeek: canSeek, seek: { sought = $0 }
        ).inspect()
        let row = try inspected.find(ViewType.VStack.self) {
            try $0.id()
                == AnyHashable(
                    SummaryEvidenceTarget(summary: summary, clock: fixture.clock))
        }
        #expect(try row.find(text: "根拠の発話：「明日は京都に行く予定。」").string().contains("予定"))
        if canSeek {
            try row.find(button: "614.25秒から").tap()
            #expect(sought == 614.25)
        } else {
            #expect(row.findAll(ViewType.Button.self).isEmpty)
            #expect(try row.find(text: "原動画 614.25秒から").string().contains("原動画"))
            #expect(sought == nil)
        }
        #expect(summary.sections[0].evidence[0] == fixture.clock)
    }

    @Test func missingEvidenceDoesNotShowANavigationButton() throws {
        let host = NSHostingView(
            rootView: SummaryEvidenceNavigation(target: nil) { ScrollView { Text("準備中") } })
        host.setFrameSize(NSSize(width: 620, height: 180))
        host.layoutSubtreeIfNeeded()
        #expect(find(NSButton.self, in: host) == nil)
    }

    @Test func nativeNavigationButtonUsesTheUpdatedActionAndKeepsItsAccessibilityRole() throws {
        let keyWindow = NSApplication.shared.keyWindow
        var firstCalls = 0
        var secondCalls = 0
        let host = NSHostingView(
            rootView: SummaryEvidenceNavigationButton { firstCalls += 1 })
        let window = hostWithoutOrdering(host, size: NSSize(width: 200, height: 44))
        defer { window.close() }
        host.layoutSubtreeIfNeeded()
        let button = try #require(find(NSButton.self, in: host))
        #expect(button.title == "発話根拠へ")
        #expect(button.isAccessibilityElement())
        #expect(button.accessibilityRole() == .button)
        #expect(button.accessibilityIdentifier() == "summary-evidence-navigation")
        #expect(button.accessibilityHelp() == "最初の発話根拠を表示します。再生位置は変えません。")
        button.performClick(nil)
        #expect(firstCalls == 1)
        host.rootView = SummaryEvidenceNavigationButton { secondCalls += 1 }
        host.layoutSubtreeIfNeeded()
        button.performClick(nil)
        #expect(firstCalls == 1)
        #expect(secondCalls == 1)
        #expect(!window.isVisible && !window.isKeyWindow && !window.isMainWindow)
        #expect(NSApplication.shared.keyWindow === keyWindow)
    }

    @Test(arguments: [180.0, 300.0])
    func visibleNativeControlRevealsAnOffscreenEvidenceTarget(_ height: Double) async throws {
        let keyWindow = NSApplication.shared.keyWindow
        let fixture = EvidenceNavigationFixture()
        let target = SummaryEvidenceTarget(summary: fixture.summary(), clock: fixture.clock)
        let overview = String(repeating: "全体の話題を説明する長い文章です。", count: 80)
        let section = String(repeating: "区間の発話を説明する別の長い文章です。", count: 80)
        let host = NSHostingView(
            rootView: SummaryEvidenceNavigation(target: target) {
                ScrollView {
                    VStack(alignment: .leading) {
                        Text(overview).fixedSize(horizontal: false, vertical: true)
                        Text(section).fixedSize(horizontal: false, vertical: true)
                        EvidenceViewportMarker().frame(height: 44).id(target)
                        Text("引用の後に続く本文")
                    }
                }
            })
        let window = hostWithoutOrdering(host, size: NSSize(width: 620, height: height))
        defer { window.close() }
        host.layoutSubtreeIfNeeded()
        let button = try #require(find(NSButton.self, in: host))
        let marker = try #require(find(EvidenceViewportMarker.NativeView.self, in: host))
        let scroll = try #require(find(NSScrollView.self, in: host))
        #expect(host.bounds.contains(host.convert(button.bounds, from: button)))
        let viewport = host.convert(scroll.contentView.bounds, from: scroll.contentView)
        #expect(!viewport.intersects(host.convert(marker.bounds, from: marker)))
        let originalContentBounds = scroll.contentView.bounds
        let originalMarkerBounds = host.convert(marker.bounds, from: marker)
        button.performClick(nil)
        let revealed = try await waitForMarker(marker, in: scroll, hostedBy: host)
        #expect(
            revealed,
            "content \(originalContentBounds) -> \(scroll.contentView.bounds); marker \(originalMarkerBounds) -> \(host.convert(marker.bounds, from: marker))"
        )
        #expect(scroll.contentView.bounds.origin != originalContentBounds.origin)
        #expect(host.bounds.contains(host.convert(button.bounds, from: button)))
        #expect(!window.isVisible && !window.isKeyWindow && !window.isMainWindow)
        #expect(NSApplication.shared.keyWindow === keyWindow)
    }

    private func hostWithoutOrdering<Content: View>(
        _ host: NSHostingView<Content>, size: NSSize
    ) -> NSWindow {
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: size), styleMask: .borderless,
            backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        host.setFrameSize(size)
        return window
    }

    private func find<Native: NSView>(_ type: Native.Type, in root: NSView) -> Native? {
        if let view = root as? Native { return view }
        for child in root.subviews {
            if let view = find(type, in: child) { return view }
        }
        return nil
    }

    private func waitForMarker(
        _ marker: NSView, in scroll: NSScrollView, hostedBy host: NSView
    ) async throws -> Bool {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(2))
        repeat {
            host.layoutSubtreeIfNeeded()
            let viewport = host.convert(scroll.contentView.bounds, from: scroll.contentView)
            if viewport.contains(host.convert(marker.bounds, from: marker)) { return true }
            try Task.checkCancellation()
            await Task.yield()
        } while clock.now < deadline
        return false
    }
}
