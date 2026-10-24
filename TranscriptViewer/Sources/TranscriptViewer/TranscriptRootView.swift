import AVKit
import AppKit
import SwiftUI
import TranscriptCore

struct PlayerSurface: NSViewRepresentable {
    let player: AVPlayer
    var selectedMovie: URL?

    @MainActor final class Coordinator {
        let playerView = AVPlayerView()
        let diagnostics = PlaybackDiagnostic.Field.allCases.map { field in
            // AX diagnostics have a small native frame, without visible developer text.
            let label = PlaybackDiagnosticLabel(labelWithString: " ")
            label.field = field
            return label
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> NSStackView {
        let native = context.coordinator.playerView
        native.controlsStyle = .inline
        native.showsFullScreenToggleButton = true
        // Picture in Picture keeps the video in its own floating window above every app.
        native.allowsPictureInPicturePlayback = true
        native.setAccessibilityLabel("動画プレーヤー")
        for diagnostic in context.coordinator.diagnostics {
            diagnostic.nativePlayerView = native
            diagnostic.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
            diagnostic.textColor = .secondaryLabelColor
            diagnostic.setAccessibilityElement(true)
            diagnostic.setAccessibilityRole(.staticText)
            diagnostic.setAccessibilityLabel(diagnostic.field.label)
            diagnostic.setAccessibilityIdentifier(
                "playback-media-state-\(diagnostic.field.rawValue)")
            diagnostic.setContentHuggingPriority(.required, for: .vertical)
        }
        let diagnostics = NSStackView(views: context.coordinator.diagnostics)
        diagnostics.spacing = 8
        let stack = NSStackView(views: [native, diagnostics])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 4
        native.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        update(context.coordinator)
        return stack
    }

    func updateNSView(_: NSStackView, context: Context) { update(context.coordinator) }

    private func update(_ coordinator: Coordinator) {
        for diagnostic in coordinator.diagnostics {
            diagnostic.expectedPlayer = player
            diagnostic.selectedMovie = selectedMovie
        }
        if coordinator.playerView.player !== player {
            coordinator.playerView.player = player
        }
    }
}

struct TranscriptRootView: View {
    @Bindable var session: PlaybackSession
    @Environment(\.openWindow) private var openWindow
    var directoryPanel: () -> NSOpenPanel = { NSOpenPanel() }
    var revealInFinder: (URL) -> Void = { NSWorkspace.shared.activateFileViewerSelecting([$0]) }

    var body: some View {
        NavigationSplitView(columnVisibility: $session.columnVisibility) {
            VStack(spacing: 0) {
                List(session.videos, selection: $session.selectedVideoID) { video in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(video.broadcastStartJST ?? "配信日時不明")
                            .font(.callout.monospacedDigit())
                        if let streamer = session.channels.label(for: video) {
                            Text(streamer)
                                .font(.caption)
                                .foregroundStyle(.tint)
                        }
                        Text(video.title)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(maxWidth: .infinity, minHeight: Chrome.hit, alignment: .leading)
                    .contentShape(Rectangle())
                    .tag(video.id)
                    .contextMenu { fileMenu(for: video) }
                    .accessibilityLabel(
                        (["動画", video.broadcastStartJST ?? "配信日時不明"]
                            + [session.channels.label(for: video), video.title].compactMap { $0 })
                            .joined(separator: " "))
                }
                .accessibilityLabel("動画一覧")
                .task(id: session.videos.map(\.title)) {
                    await session.channels.load(titles: session.videos.map(\.title))
                }
                .navigationTitle("動画")
                Text("ライブラリ: \(session.libraryName)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
                HStack {
                    if session.narrativeSummary.isBatchRunning {
                        Button("一括生成を停止") { session.summaryBatchRun = nil }
                            .disabled(session.summaryBatchRun == nil)
                    } else {
                        Button("新しい順にすべて要約") { session.summaryBatchRun = UUID() }
                            .disabled(session.videos.isEmpty || session.isScanning)
                            .accessibilityIdentifier("generate-library-summaries")
                            .help("読み込み済み動画だけを新しい順に処理します。完了済みは再利用します。")
                    }
                    Spacer()
                }
                .padding(12)
                if let progress = session.narrativeSummary.libraryProgress {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(
                            "要約：処理済み \(progress.completed)/\(progress.total)・成功 \(progress.succeeded)・失敗 \(progress.failures.count)"
                        )
                        .accessibilityIdentifier("summary-batch-progress")
                        if let date = progress.currentDate {
                            Text("生成中：\(date)")
                        } else if progress.wasCancelled {
                            Text("一括生成を停止しました。完了した要約は再利用できます。")
                        } else if !progress.isRunning {
                            Text(
                                progress.failures.isEmpty
                                    ? "読み込み済み動画の要約が完了しました。" : "失敗した動画があります。再実行で未完了分を試します。")
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .help(progress.failures.joined(separator: "\n"))
                }
                if !session.scanIssues.isEmpty {
                    Text("未読込 \(session.scanIssues.count) 件")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .help(session.scanIssues.joined(separator: "\n"))
                        .accessibilityLabel("未読込 \(session.scanIssues.count) 件")
                        .accessibilityHint("カーソルを合わせると理由を表示します")
                }
            }
            .toolbar {
                Button("フォルダを開く") {
                    session.chooseFolder(panel: directoryPanel())
                }
                .frame(minHeight: Chrome.hit)
                .accessibilityHint("動画とCSV/SRT字幕が入ったフォルダを選びます")
                Button("再読み込み") {
                    session.refreshLibrary()
                }
                .frame(minHeight: Chrome.hit)
                .disabled(session.isScanning)
                .keyboardShortcut("r", modifiers: [.command])
                .accessibilityLabel("ライブラリを再読み込み")
                .accessibilityHint("追加された動画とCSV/SRT字幕を読み直します")
                Button("検索") {
                    session.showsSearch = true
                }
                .frame(minHeight: Chrome.hit)
                .keyboardShortcut("f", modifiers: [.command])
                .accessibilityLabel("検索画面を開く")
                .accessibilityHint("字幕を検索して発話の先頭から再生します")
                Button("BGM") {
                    openWindow(id: BGMLibraryView.windowID)
                }
                .frame(minHeight: Chrome.hit)
                .keyboardShortcut("b", modifiers: [.command])
                .accessibilityLabel("BGMライブラリを開く")
                .accessibilityHint("保存したBGMを検索して再生します")
            }
            .navigationSplitViewColumnWidth(min: 420, ideal: 580, max: 750)
        } detail: {
            if let video = session.selectedVideo {
                VStack(alignment: .leading, spacing: 12) {
                    // Pinned above everything that scrolls, so the video stays in sight while a
                    // clip, the summary or the transcript is browsed.
                    PlayerSurface(player: session.player, selectedMovie: video.movieURL)
                        .frame(maxWidth: .infinity)
                        .frame(height: Chrome.playerHeight)
                        .clipped()
                        .accessibilityLabel("動画プレーヤー \(video.title)")
                    HStack {
                        Picker("再生速度", selection: $session.playbackSpeed) {
                            ForEach(PlaybackSpeed.allCases, id: \.self) { speed in
                                Text(speed.label).tag(speed)
                            }
                        }
                        .pickerStyle(.segmented)
                        .controlSize(.large)
                        .frame(width: 300)
                        .accessibilityLabel("再生速度")
                        Button {
                            session.playbackSpeed = session.playbackSpeed.next
                        } label: {
                            iconLabel("速度を切り替え", systemImage: "speedometer")
                        }
                        .buttonStyle(.borderless)
                        .keyboardShortcut("s", modifiers: [])
                        .help("S: 再生速度を 0.5x→1.0x→1.5x→2.0x の順に切り替え")
                        Button {
                            session.skip(by: -10)
                        } label: {
                            iconLabel("10秒戻す", systemImage: "gobackward.10")
                        }
                        .buttonStyle(.borderless)
                        .keyboardShortcut(.leftArrow, modifiers: [])
                        .help("←: 10秒戻す")
                        Button {
                            session.skip(by: 10)
                        } label: {
                            iconLabel("10秒送る", systemImage: "goforward.10")
                        }
                        .buttonStyle(.borderless)
                        .keyboardShortcut(.rightArrow, modifiers: [])
                        .help("→: 10秒送る")
                        Spacer()
                    }
                    ScrollView {
                        timeline
                    }
                    // Outside the scroll area, so the BGM stays visible however short it gets.
                    VideoBGMControls(model: session.videoBGM)
                    Picker("表示内容", selection: $session.detailTab) {
                        Text("要約").tag(DetailTab.summary)
                        Text("認識結果").tag(DetailTab.utterances)
                        Text("切り抜き").tag(DetailTab.clips)
                    }
                    .pickerStyle(.segmented)
                    .controlSize(.large)
                    switch session.detailTab {
                    case .summary:
                        NarrativeSummaryView(video: video, model: session.narrativeSummary) {
                            seconds in
                            session.resumeTimelineFollow()
                            session.seek(to: seconds)
                        }
                        .id(video.id)
                    case .clips:
                        ScrollView {
                            ClipsPanel(model: session.clips) { clip, index, loop in
                                session.play(clip, range: index, loop: loop)
                            }
                        }
                    case .utterances:
                        HStack(spacing: 8) {
                            Text("認識結果")
                                .font(.title2)
                                .accessibilityAddTraits(.isHeader)
                            Button {
                                session.showUtteranceWindow()
                            } label: {
                                Image(systemName: "macwindow")
                                    .frame(width: Chrome.hit, height: Chrome.hit)
                                    .contentShape(Rectangle())
                            }
                            .buttonStyle(.borderless)
                            .help("認識結果を別ウィンドウで開く")
                            .accessibilityLabel("認識結果を別ウィンドウで開く")
                            Spacer()
                        }
                        UtteranceListView(session: session)
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .navigationTitle(
                    video.broadcastStartJST.map { "\($0) \(video.title)" } ?? video.title)
            } else {
                ContentUnavailableView(
                    "動画を選択",
                    systemImage: "film",
                    description: Text("左の一覧から動画を選ぶか、フォルダを開いてください")
                )
            }
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 1200, minHeight: 520)
        .onChange(of: session.selectedVideoID) { _, _ in
            session.handleVideoChange()
        }
        .task {
            session.loadDefaultLibrary()
        }
        .task(id: session.usesDefaultLibraries) {
            await session.observeDefaultLibraryUpdates()
        }
        .task(id: session.summaryBatchRun) {
            guard session.summaryBatchRun != nil else { return }
            await session.narrativeSummary.generateLibrary(videos: session.videos)
        }
        .onReceive(
            NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification),
            perform: refreshOnActivation
        )
        .sheet(isPresented: $session.showsSearch) {
            SearchScreen(session: session)
        }
        .alert("読み込みエラー", isPresented: $session.showsError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(session.errorText)
        }
    }

    private var timeline: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("タイムライン")
                .font(.title2)
                .accessibilityAddTraits(.isHeader)
            Group {
                if let utterance = session.currentUtterance {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(utterance.text)
                            .font(.title)
                            .lineLimit(3)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Text("\(utterance.startLabel) – \(utterance.endLabel)")
                            .font(.title3.monospacedDigit())
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("現在の発話 \(utterance.startLabel) \(utterance.text)")
                } else {
                    Text("発話なし")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .accessibilityLabel("この位置に発話はありません")
                }
            }
            .frame(
                maxWidth: .infinity,
                minHeight: Chrome.utteranceBlockHeight,
                maxHeight: Chrome.utteranceBlockHeight,
                alignment: .topLeading
            )
            .clipped()
            Slider(
                value: Binding(
                    get: { min(session.currentSeconds, session.sliderUpperBound) },
                    set: { seconds in
                        session.resumeTimelineFollow()
                        session.seek(to: seconds)
                    }
                ),
                in: 0...session.sliderUpperBound
            )
            .controlSize(.large)
            .frame(minHeight: Chrome.hit)
            .accessibilityLabel("再生位置")
            .accessibilityValue(Text(timeLabel(session.currentSeconds)))
            .accessibilityHint("スライダーで再生位置を移動します")
            TimelineBar(
                utterances: session.utterances,
                duration: session.sliderUpperBound,
                currentSeconds: session.currentSeconds,
                selectedID: session.selectedUtteranceID
            )
            .frame(height: 28)
            .accessibilityHidden(true)
            if let video = session.selectedVideo {
                ExcitementSection(
                    transcriptURL: video.transcriptURL, currentSeconds: session.currentSeconds
                ) { seconds in
                    session.resumeTimelineFollow()
                    session.seek(to: seconds)
                }
            }
            ScrollView(.horizontal) {
                HStack(spacing: 12) {
                    ForEach(session.nearbyUtterances) { cue in
                        Button {
                            session.seek(toUtterance: cue.id)
                        } label: {
                            VStack(alignment: .leading, spacing: 6) {
                                Text(cue.startLabel)
                                    .font(.title3.monospacedDigit())
                                Text(cue.text)
                                    .font(.title3)
                                    .lineLimit(8)
                                    .multilineTextAlignment(.leading)
                            }
                            .frame(
                                minWidth: 320, idealWidth: 400, minHeight: 88,
                                alignment: .topLeading
                            )
                            .padding(12)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.bordered)
                        .accessibilityLabel("\(cue.startLabel) \(cue.text)")
                        .accessibilityHint("発話の先頭から再生します")
                        .accessibilityAddTraits(
                            session.selectedUtteranceID == cue.id ? [.isSelected] : [])
                    }
                }
            }
            .frame(height: Chrome.nearbyStripHeight)
            .onScrollPhaseChange(handleScrollPhase)
            .accessibilityLabel("付近の発話")
            .accessibilityHint("スクロールすると再生への自動追従を止めます")
        }
    }

    /// An icon button's whole 44pt square is clickable; VoiceOver reads the title.
    func iconLabel(_ title: String, systemImage: String) -> some View {
        Label(title, systemImage: systemImage)
            .labelStyle(.iconOnly)
            .font(.title2)
            .frame(width: Chrome.hit, height: Chrome.hit)
            .contentShape(Rectangle())
    }

    func fileMenu(for video: VideoTranscript) -> some View {
        Button("Reveal Finder", systemImage: "folder") { revealInFinder(video.movieURL) }
            .help("動画ファイルをFinderで表示します")
    }

    func refreshOnActivation(_: Notification) {
        session.refreshLibrary()
    }

    func handleScrollPhase(_: ScrollPhase, _ phase: ScrollPhase) {
        if phase == .interacting { session.beginTimelineBrowse() }
    }

    private func timeLabel(_ seconds: Double) -> String {
        let total = max(Int(seconds.rounded(.down)), 0)
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let rest = total % 60
        return "\(hours)時間\(minutes)分\(rest)秒"
    }
}

struct TimelineBar: View {
    let utterances: [TranscriptCue]
    let duration: Double
    let currentSeconds: Double
    let selectedID: Int?

    var body: some View {
        Canvas { context, size in
            let span = max(duration, 0.001)
            let track = Path(roundedRect: CGRect(origin: .zero, size: size), cornerRadius: 6)
            context.fill(track, with: .color(.secondary.opacity(0.25)))
            for cue in utterances {
                let x = cue.startSeconds / span * size.width
                let width = max((cue.endSeconds - cue.startSeconds) / span * size.width, 2)
                let rect = CGRect(x: x, y: 4, width: width, height: max(size.height - 8, 4))
                let selected = cue.id == selectedID
                context.fill(
                    Path(roundedRect: rect, cornerRadius: 3),
                    with: .color(selected ? Color.primary : Color.primary.opacity(0.4))
                )
            }
        }
        .drawingGroup()
        .overlay {
            GeometryReader { geometry in
                Rectangle()
                    .fill(.primary)
                    .frame(width: 2, height: geometry.size.height)
                    .offset(x: currentSeconds / max(duration, 0.001) * geometry.size.width)
            }
        }
    }
}
