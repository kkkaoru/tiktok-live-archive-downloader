import SwiftUI
import TranscriptCore

/// Source clocks are actionable only when the selected movie is the pinned source.
struct SourceExcerptPanel: View {
    let state: NarrativeSummaryModel.ExcerptState
    let seek: (Double) -> Void

    var body: some View {
        switch state {
        case .absent:
            EmptyView()
        case .failed:
            Text("原動画の抜粋候補を読み込めません。指定ファイルと照合情報を確認してください。")
                .foregroundStyle(.secondary)
        case .representative(let candidate):
            if let prose = candidate.prose {
                SourceSupportedProsePanel(
                    summary: prose, canSeek: candidate.canSeekSelectedMovie, seek: seek)
            } else if let topics = candidate.topics {
                SourceTopicSummaryPanel(
                    summary: topics, canSeek: candidate.canSeekSelectedMovie, seek: seek)
            } else {
                RepresentativeSummaryPanel(candidate: candidate, seek: seek)
            }
        case .ready(let candidate):
            VStack(alignment: .leading, spacing: 12) {
                Text("原動画の発話抜粋候補")
                    .font(.headline)
                Text("原文の一部を表示しています。話題の代表性・認識精度は未確認で、完成した要約ではありません。")
                    .font(.caption)
                Text(
                    candidate.canSeekSelectedMovie
                        ? "時刻は原動画の区間開始です。押すと原動画を再生します。抜粋の発話開始時刻ではありません。"
                        : "時刻は原動画の区間開始です。編集後の時刻には変換していません。"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                ForEach(candidate.summary.sections, id: \.startSeconds) { section in
                    if candidate.canSeekSelectedMovie {
                        Button("原動画 \(section.startSeconds.formatted())秒から") {
                            seek(section.startSeconds)
                        }
                    } else {
                        Text("原動画 \(section.startSeconds.formatted())秒から")
                            .font(.subheadline.monospacedDigit())
                    }
                    ForEach(section.excerpts, id: \.lowerCharacter) { span in
                        Text(span.text).textSelection(.enabled)
                    }
                }
                Divider()
            }
        }
    }
}
