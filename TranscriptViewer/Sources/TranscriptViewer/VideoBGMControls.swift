import AppKit
import SwiftUI
import TranscriptCore

struct BGMThumbnail: View {
    let track: BGMTrack

    var body: some View {
        if let image = NSImage(contentsOf: track.thumbnailURL) {
            Image(nsImage: image).resizable().aspectRatio(contentMode: .fit)
        } else {
            Image(systemName: "music.note")
        }
    }
}

/// The video's BGM: a searchable chooser with artwork, and its volume.
struct VideoBGMControls: View {
    @Bindable var model: VideoBGMModel
    @State private var choosing = false

    /// Two rows that wrap instead of squeezing, with 44pt targets and spoken values.
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("BGM")
                    .font(.headline)
                    .accessibilityAddTraits(.isHeader)
                Button {
                    choosing = true
                } label: {
                    HStack(spacing: 8) {
                        Group {
                            if let track = model.track {
                                BGMThumbnail(track: track)
                            } else {
                                Image(systemName: "music.note.slash")
                            }
                        }
                        .frame(width: 64, height: 36)
                        .accessibilityHidden(true)
                        Text(chosenLabel)
                            .lineLimit(2)
                            .multilineTextAlignment(.leading)
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer(minLength: 0)
                        Image(systemName: "chevron.down")
                            .foregroundStyle(.secondary)
                            .accessibilityHidden(true)
                    }
                    .frame(maxWidth: 480, minHeight: Chrome.hit, alignment: .leading)
                    .contentShape(Rectangle())
                }
                .help("この動画で流すBGMを検索して選びます")
                .accessibilityLabel("BGMを選択")
                .accessibilityValue(chosenLabel)
                .accessibilityHint("検索とサムネイル付きの候補を開きます")
                // A popover is its own window, so typing there never triggers S, ← or →.
                .popover(isPresented: $choosing, content: picker)
            }
            HStack(spacing: 12) {
                Picker("BGM音量の調整方法", selection: $model.setting.mode) {
                    Text("自動").tag(VideoBGMSetting.Mode.automatic)
                    Text("手動").tag(VideoBGMSetting.Mode.manual)
                }
                .pickerStyle(.segmented)
                .controlSize(.large)
                .labelsHidden()
                .fixedSize()
                .frame(minHeight: Chrome.hit)
                .accessibilityLabel("BGM音量の調整方法")
                .help("自動: 再生位置の動画の音量に合わせます / 手動: 指定した音量で流します")
                volumeSlider
            }
        }
        .disabled(model.library.tracks.isEmpty)
    }

    func picker() -> VideoBGMPicker { VideoBGMPicker(model: model, done: close) }

    func close() { choosing = false }

    var chosenLabel: String {
        switch model.setting.selection {
        case .off: "BGMなし"
        case .libraryDefault: model.track.map { "デフォルト: \($0.title)" } ?? "デフォルト（未設定）"
        case .track: model.track?.title ?? "見つからないBGM"
        }
    }

    @ViewBuilder var volumeSlider: some View {
        if model.setting.mode == .automatic {
            Slider(value: $model.setting.offsetDB, in: -36...0, step: 1)
                .controlSize(.large)
                .frame(minWidth: 120, maxWidth: 240, minHeight: Chrome.hit)
                .accessibilityLabel("動画の音声に対するBGMの大きさ")
                .accessibilityValue(automaticLabel)
            Text(automaticLabel)
                .font(.callout.monospacedDigit())
                .fixedSize()
                .accessibilityHidden(true)
                .help("動画の音声より何dB小さく流すか。事前計算した音量データを使います。データがない動画は標準的な音量を仮定します")
        } else {
            Slider(value: $model.setting.volume, in: 0...1)
                .controlSize(.large)
                .frame(minWidth: 120, maxWidth: 240, minHeight: Chrome.hit)
                .accessibilityLabel("BGM音量")
                .accessibilityValue(manualLabel)
            Text(manualLabel)
                .font(.callout.monospacedDigit())
                .fixedSize()
                .accessibilityHidden(true)
        }
    }

    var manualLabel: String { "\(Int((model.setting.volume * 100).rounded()))%" }

    var automaticLabel: String {
        let offset = "声との差 \(Int(model.setting.offsetDB))dB"
        if model.isLoading { return "\(offset)・読込中" }
        return model.lacksLevels ? "\(offset)・音量データなし" : offset
    }

}

/// The chooser's content: the default, none, and every track, filtered as you type.
struct VideoBGMPicker: View {
    @Bindable var model: VideoBGMModel
    let done: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            TextField("タイトル・チャンネル名で検索", text: $model.query)
                .textFieldStyle(.roundedBorder)
                .accessibilityLabel("BGMを検索")
                .padding(8)
            List {
                row(.libraryDefault, title: "デフォルト", detail: model.defaultTrack?.title ?? "未設定")
                row(.off, title: "BGMなし", detail: nil)
                ForEach(model.results) { track in
                    row(.track(track.id), title: track.title, detail: track.channel, track: track)
                }
            }
            .accessibilityLabel("BGM候補")
        }
        .frame(width: 420, height: 480)
    }

    func row(
        _ selection: VideoBGMSetting.Selection, title: String, detail: String?,
        track: BGMTrack? = nil
    ) -> some View {
        Button {
            model.select(selection)
            done()
        } label: {
            HStack(spacing: 8) {
                Group {
                    if let track { BGMThumbnail(track: track) } else { Color.clear }
                }
                .frame(width: 80, height: 45)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).lineLimit(2)
                    if let detail { Text(detail).font(.caption).foregroundStyle(.secondary) }
                }
                Spacer()
                if model.setting.selection == selection { Image(systemName: "checkmark") }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .frame(minHeight: Chrome.hit)
        .accessibilityLabel("BGM候補 \(title)")
    }
}
