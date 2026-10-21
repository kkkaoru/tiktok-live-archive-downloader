import SwiftUI
import TranscriptCore

/// The selected video's clip points: each plays its ranges in order and shows its length,
/// band, ranges and generated summary. Reload asks the Worker to compute them again.
struct ClipsPanel: View {
    let model: ClipsModel
    /// Clip, first range, loop.
    let play: (ClipPoint, Int, Bool) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Text("切り抜き").font(.headline).accessibilityAddTraits(.isHeader)
                Text(note).foregroundStyle(.secondary)
                Spacer(minLength: 0)
                Button("切り抜きを再計算", action: model.reload)
                    .disabled(!model.canReload)
                    .frame(minHeight: Chrome.hit)
                    .help("Cloudflareで切り抜き箇所と要約を計算し直します（数分かかります）")
            }
            ForEach(model.clips) { clip in
                ClipRow(clip: clip, cursor: model.cursor, play: play)
            }
        }
    }

    var note: String {
        if let cursor = model.cursor, let range = cursor.range {
            let parts =
                cursor.clip.ranges.count > 1
                ? "・区間\(cursor.index + 1)/\(cursor.clip.ranges.count)" : ""
            return
                "\(cursor.loop ? "ループ再生中" : "再生中"): \(cursor.clip.title)（\(ClipPoint.seconds(cursor.clip.duration))\(parts)、\(ClipPoint.seconds(range.duration))の区間）"
        }
        return Self.note(for: model.state)
    }

    static func note(for state: ClipsModel.State) -> String {
        switch state {
        case .idle: ""
        case .loading: "読み込み中…"
        case .ready(let clips): clips.isEmpty ? "切り抜きに向く区間が見つかりませんでした" : "\(clips.count)件"
        case .pending: "計算中です（数分かかります）…"
        case .mismatch: "Web版の動画と内容が異なるため表示できません"
        case .missing: "この動画の切り抜きはまだありません"
        case .noExcitement: "盛り上がりデータがまだないため作成できません（計算して同期すると自動で作成されます）"
        case .unavailable: "Web版の接続設定がありません"
        case .failed(let message): "読み込めませんでした: \(message)"
        }
    }
}

struct ClipRow: View {
    let clip: ClipPoint
    let cursor: ClipCursor?
    let play: (ClipPoint, Int, Bool) -> Void

    var playing: Bool { cursor?.clip.id == clip.id }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Button {
                    play(clip, 0, false)
                } label: {
                    Label("再生", systemImage: "play.fill")
                        .frame(minWidth: Chrome.hit, minHeight: Chrome.hit)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("\(clip.title)を再生（\(ClipPoint.seconds(clip.duration))）")
                Button {
                    play(clip, 0, true)
                } label: {
                    Label("ループ再生", systemImage: "repeat")
                        .frame(minWidth: Chrome.hit, minHeight: Chrome.hit)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("\(clip.title)をループ再生")
                Text(clip.title).font(.headline)
                Spacer(minLength: 0)
                Text(lengthLabel).font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
            Text(clip.summary).fixedSize(horizontal: false, vertical: true)
            ScrollView(.horizontal) {
                HStack(spacing: 6) {
                    ForEach(Array(clip.ranges.enumerated()), id: \.offset) { index, range in
                        Button(rangeLabel(index, range)) { play(clip, index, false) }
                            .frame(minHeight: Chrome.hit)
                    }
                }
            }
            .accessibilityLabel("\(clip.title)の区間")
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(playing ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.08))
        )
    }

    var lengthLabel: String {
        let parts = clip.ranges.count > 1 ? "・\(clip.ranges.count)区間" : ""
        var pattern = ""
        if let label = clip.patternLabel { pattern = "・\(label)" }
        return "\(ClipPoint.seconds(clip.duration))・\(clip.kindLabel)\(pattern)\(parts)"
    }

    func rangeLabel(_ index: Int, _ range: ClipRange) -> String {
        "\(index + 1). \(ExcitementTimeline.clock(range.start))〜（\(ClipPoint.seconds(range.duration))）"
    }
}
