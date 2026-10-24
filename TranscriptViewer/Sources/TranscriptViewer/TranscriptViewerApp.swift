import SwiftUI

@main
struct TranscriptViewerApp: App {
    @State private var session: PlaybackSession
    @State private var bgm: BGMLibraryModel

    init() {
        self.init(arguments: Array(CommandLine.arguments.dropFirst()))
    }

    init(arguments: [String]) {
        let bgm = BGMLibraryModel()
        _bgm = State(initialValue: bgm)
        DebugCapture.listen()
        _session = State(
            initialValue: PlaybackSession(
                videoBGM: VideoBGMModel(library: bgm),
                clips: ClipsModel(api: ClipsModel.configured),
                channels: ChannelsModel(
                    api: ClipsModel.configured, originals: ChannelsModel.configuredOriginals),
                launchArguments: arguments))
    }

    var body: some Scene {
        WindowGroup(makeContent: makeWindowContent)
            .windowResizability(.contentMinSize)
            .defaultSize(width: 1200, height: 800)
        // BGM is a separate library and window, not part of the replay catalog.
        Window("BGM", id: BGMLibraryView.windowID, content: makeBGMView)
            .defaultSize(width: 1000, height: 700)
    }

    func makeBGMView() -> BGMLibraryView {
        BGMLibraryView(model: bgm)
    }

    func makeRootView() -> TranscriptRootView {
        TranscriptRootView(session: session)
    }

    func makeWindowContent() -> some View {
        makeRootView().onOpenURL(perform: session.openLibrary)
    }
}
