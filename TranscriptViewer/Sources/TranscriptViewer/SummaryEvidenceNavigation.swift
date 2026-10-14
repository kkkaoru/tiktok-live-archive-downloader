import AppKit
import SwiftUI
import TranscriptCore

/// An existing source range identifies a scroll destination, never a new playback clock.
struct SummaryEvidenceTarget: Hashable, Sendable {
    let transcriptFingerprint: String
    let morphologyFingerprint: String
    let version: Int
    let clockDomain: String
    let sectionStartSeconds: Double
    let lowerCharacter: Int
    let upperCharacter: Int

    init(summary: SourceSupportedProse, clock: RepresentativeSourceSummary.Clock) {
        transcriptFingerprint = summary.fingerprint
        morphologyFingerprint = summary.morphologyFingerprint
        version = summary.version
        clockDomain = summary.clockDomain.rawValue
        sectionStartSeconds = clock.span.sectionStartSeconds
        lowerCharacter = clock.span.lowerCharacter
        upperCharacter = clock.span.upperCharacter
    }

    @MainActor static func first(in state: NarrativeSummaryModel.ExcerptState) -> Self? {
        switch state {
        case .representative(let candidate):
            guard let summary = candidate.prose,
                let clock = summary.sections.first?.evidence.first
            else { return nil }
            return Self(summary: summary, clock: clock)
        case .absent, .ready, .failed:
            return nil
        }
    }
}

/// Keep evidence navigation outside the long summary's scrolling content.
struct SummaryEvidenceNavigation<Content: View>: View {
    let target: SummaryEvidenceTarget?
    let content: Content

    init(target: SummaryEvidenceTarget?, @ViewBuilder content: () -> Content) {
        self.target = target
        self.content = content()
    }

    var body: some View {
        ScrollViewReader { proxy in
            VStack(alignment: .leading, spacing: 8) {
                if let target {
                    HStack {
                        SummaryEvidenceNavigationButton {
                            proxy.scrollTo(target, anchor: .top)
                        }
                        .fixedSize()
                        .frame(minHeight: Chrome.hit)
                        Spacer()
                    }
                    .fixedSize(horizontal: false, vertical: true)
                }
                content
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
}

/// A normal native button exposes a press action without synthetic desktop input.
struct SummaryEvidenceNavigationButton: NSViewRepresentable {
    static let identifier = "summary-evidence-navigation"
    let action: @MainActor () -> Void

    @MainActor final class Coordinator: NSObject {
        var action: @MainActor () -> Void

        init(action: @escaping @MainActor () -> Void) { self.action = action }

        @objc func activate() { action() }
    }

    func makeCoordinator() -> Coordinator { Coordinator(action: action) }

    func makeNSView(context: Context) -> NSButton {
        let button = NSButton(
            title: "発話根拠へ", target: context.coordinator,
            action: #selector(Coordinator.activate))
        button.bezelStyle = .rounded
        button.toolTip = "最初の発話根拠を表示します。再生位置は変えません。"
        button.setAccessibilityElement(true)
        button.setAccessibilityRole(.button)
        button.setAccessibilityIdentifier(Self.identifier)
        button.setAccessibilityHelp(button.toolTip)
        button.setContentHuggingPriority(.required, for: .horizontal)
        return button
    }

    func updateNSView(_: NSButton, context: Context) {
        context.coordinator.action = action
    }
}
