import SwiftUI

@main
struct TranscriptViewerApp: App {
    @State private var session: PlaybackSession

    init() {
        self.init(arguments: Array(CommandLine.arguments.dropFirst()))
    }

    init(arguments: [String]) {
        _session = State(initialValue: PlaybackSession(launchArguments: arguments))
    }

    var body: some Scene {
        WindowGroup(makeContent: makeWindowContent)
            .windowResizability(.contentMinSize)
            .defaultSize(width: 1200, height: 800)
    }

    func makeRootView() -> TranscriptRootView {
        TranscriptRootView(session: session)
    }

    func makeWindowContent() -> some View {
        makeRootView().onOpenURL(perform: session.openLibrary)
    }
}
