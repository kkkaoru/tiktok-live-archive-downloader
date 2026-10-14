import SwiftUI
import TranscriptSummaries

struct RepresentativeSummaryPanel: View {
    let candidate: RepresentativeSummaryCandidate
    let seek: (Double) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(candidate.summary.label).font(.headline)
            Text("認識された発話を選んだ抜粋です。話題の代表性・認識精度は未確認です。自然な説明文の要約や映像品質の合格を意味しません。")
                .font(.caption)
            Text(
                candidate.summary.clockDomain == .editedOutput
                    ? "時刻は編集後の引用を含む発話区間です。"
                    : "時刻は原動画の発話区間です。編集後の時刻には変換していません。"
            )
            .font(.caption).foregroundStyle(.secondary)
            ForEach(candidate.summary.clocks.indices, id: \.self) { index in
                let clock = candidate.summary.clocks[index]
                if candidate.canSeekSelectedMovie {
                    Button("\(clock.startSeconds.formatted())秒から") { seek(clock.startSeconds) }
                } else {
                    Text("原動画 \(clock.startSeconds.formatted())秒から")
                }
                Text(clock.span.text).textSelection(.enabled)
            }
            Divider()
        }
    }
}
