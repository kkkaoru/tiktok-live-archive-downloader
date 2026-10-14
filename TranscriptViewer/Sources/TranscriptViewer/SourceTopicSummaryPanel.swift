import SwiftUI
import TranscriptCore

/// Keywords name literal source words; quotations carry every displayed content claim.
struct SourceTopicSummaryPanel: View {
    let summary: SourceTopicSummary
    let canSeek: Bool
    let seek: (Double) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(summary.label).font(.headline)
            Text("認識本文のキーワードと実際の発話をまとめています。話題の代表性・認識精度は未確認です。")
                .font(.caption)
            Text("動画全体のキーワード").font(.subheadline)
            Text(keywordText(summary.overview)).textSelection(.enabled)
            Text(
                summary.representative.clockDomain == .editedOutput
                    ? "時刻は編集後の発話区間です。" : "時刻は原動画の発話区間です。編集後の時刻には変換していません。"
            ).font(.caption).foregroundStyle(.secondary)
            ForEach(summary.sections) { section in
                Divider()
                Text(
                    "\(Duration.seconds(section.startSeconds).formatted(.time(pattern: .hourMinuteSecond))) – \(Duration.seconds(section.endSeconds).formatted(.time(pattern: .hourMinuteSecond)))"
                ).font(.headline.monospacedDigit())
                Text(keywordText(section.keywords)).font(.subheadline).textSelection(.enabled)
                ForEach(section.excerpts.indices, id: \.self) { index in
                    let clock = section.excerpts[index]
                    if canSeek {
                        Button("\(clock.startSeconds.formatted())秒から") { seek(clock.startSeconds) }
                    } else {
                        Text("原動画 \(clock.startSeconds.formatted())秒から")
                    }
                    Text("「\(clock.span.text)」").textSelection(.enabled)
                }
            }
            Divider()
        }
    }

    private func keywordText(_ keywords: [SourceTopicSummary.Keyword]) -> String {
        keywords.isEmpty ? "発話抜粋" : keywords.map(\.text).joined(separator: "・")
    }
}
