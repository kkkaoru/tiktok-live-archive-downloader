import AVKit
import AppKit
import SwiftUI
import TranscriptCore

/// Inline transport (seek, time) for audio; the thumbnail is shown beside it, not video.
struct AudioControls: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context _: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .inline
        view.player = player
        view.setAccessibilityLabel("BGM再生コントロール")
        return view
    }

    func updateNSView(_ view: AVPlayerView, context _: Context) { view.player = player }
}

struct BGMLibraryView: View {
    static let windowID = "bgm"
    @Bindable var model: BGMLibraryModel

    var body: some View {
        NavigationSplitView {
            VStack(spacing: 0) {
                searchField
                List(model.results, selection: $model.selectedID) { track in
                    HStack(spacing: 8) {
                        thumbnail(track).frame(width: 96, height: 54)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(track.title).lineLimit(2)
                            Text(track.channel).font(.caption).foregroundStyle(.secondary)
                        }
                        if model.defaultID == track.id {
                            Spacer()
                            Image(systemName: "star.fill")
                                .foregroundStyle(.yellow)
                                .help("デフォルトBGM")
                                .accessibilityLabel("デフォルトBGM")
                        }
                    }
                    .frame(minHeight: Chrome.hit)
                    .tag(track.id)
                    .contextMenu { fileButtons(track) }
                    .accessibilityLabel("BGM \(track.title) \(track.channel)")
                }
                .accessibilityLabel("BGM一覧")
                if let playing = model.playing { nowPlaying(playing) }
                volumeControl
                addForm
            }
            .navigationTitle("BGM")
            .navigationSplitViewColumnWidth(min: 360, ideal: 460)
        } detail: {
            if let track = model.selected ?? model.playing {
                detail(track)
            } else {
                Text(model.root == nil ? "BGMライブラリが見つかりません" : "BGMを選択してください")
                    .foregroundStyle(.secondary)
            }
        }
        .onAppear(perform: model.reload)
    }

    /// Applies to every track and keeps its value across launches; the system volume is untouched.
    var volumeControl: some View {
        HStack(spacing: 8) {
            Image(systemName: "speaker.fill").foregroundStyle(.secondary)
            Slider(value: $model.volume, in: 0...1)
                .accessibilityLabel("BGM音量")
                .accessibilityValue("\(Int((model.volume * 100).rounded()))%")
            Image(systemName: "speaker.wave.3.fill").foregroundStyle(.secondary)
            Text("\(Int((model.volume * 100).rounded()))%")
                .font(.caption.monospacedDigit())
                .frame(minWidth: 36, alignment: .trailing)
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .help("BGM全体の音量（macOSの音量とは別）")
    }

    /// Always visible above the list, unlike a toolbar search field.
    var searchField: some View {
        TextField("タイトル・チャンネル名で検索", text: $model.query)
            .textFieldStyle(.roundedBorder)
            .accessibilityLabel("BGMを検索")
            .padding(12)
    }

    var addForm: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                TextField("YouTube URL", text: $model.newURL)
                    .onSubmit(startAdding)
                Button("保存", action: startAdding)
                    .disabled(model.isAdding || model.root == nil)
            }
            Text(model.addMessage).font(.caption).foregroundStyle(.secondary)
            if !model.issues.isEmpty {
                Text("読み込めないBGM \(model.issues.count) 件")
                    .font(.caption)
                    .help(model.issues.joined(separator: "\n"))
            }
        }
        .padding(12)
    }

    /// The playing track stays visible, with its thumbnail, while another one is browsed.
    func nowPlaying(_ track: BGMTrack) -> some View {
        HStack(spacing: 8) {
            thumbnail(track).frame(width: 64, height: 36)
            Text(track.title).font(.caption).lineLimit(1)
            Spacer()
            Button("一時停止") { model.togglePlayback(track) }
        }
        .padding(.horizontal, 12)
        .accessibilityLabel("再生中 \(track.title)")
    }

    func startAdding() { Task { await model.add() } }

    func detail(_ track: BGMTrack) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                // Clicking the artwork plays or pauses, as clicking a video does.
                Button {
                    model.togglePlayback(track)
                } label: {
                    thumbnail(track)
                        .frame(maxWidth: .infinity, maxHeight: Chrome.playerHeight)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(model.playingID == track.id ? "クリックで一時停止" : "クリックで再生")
                .accessibilityLabel(
                    model.playingID == track.id ? "再生中のサムネイル" : "サムネイル")
                Text(track.title).font(.title2).textSelection(.enabled)
                if let channel = track.channelURL {
                    Link(track.channel, destination: channel)
                } else {
                    Text(track.channel)
                }
                Link("YouTubeで開く", destination: track.pageURL)
                if track.loops {
                    Label("ループ再生", systemImage: "repeat")
                        .help("最後まで再生すると最初から繰り返します")
                }
                HStack {
                    Button(model.playingID == track.id ? "一時停止" : "再生") {
                        model.togglePlayback(track)
                    }
                    .keyboardShortcut(.space, modifiers: [])
                    fileButtons(track)
                }
                Button(
                    model.defaultID == track.id ? "デフォルトを解除" : "デフォルトに設定",
                    systemImage: model.defaultID == track.id ? "star.slash" : "star"
                ) {
                    model.toggleDefault(track)
                }
                .help("BGMを登録していない動画で、このBGMを流します")
                if model.playingID == track.id {
                    AudioControls(player: model.player).frame(height: Chrome.hit)
                }
                Label(track.audioURL.lastPathComponent, systemImage: "music.note")
                    .font(.caption)
                    .onDrag { NSItemProvider(object: track.audioURL as NSURL) }
                    .help("ドラッグしてFinal Cut ProやLogic Proに配置できます")
            }
            .padding(16)
        }
    }

    func fileButtons(_ track: BGMTrack) -> some View {
        Group {
            Button("Reveal Finder", systemImage: "folder") { model.revealInFinder(track) }
                .help("音源ファイルをFinderで表示します")
            Button("パスをコピー", systemImage: "doc.on.doc") { model.copyPath(track) }
        }
    }

    func thumbnail(_ track: BGMTrack) -> some View { BGMThumbnail(track: track) }
}
