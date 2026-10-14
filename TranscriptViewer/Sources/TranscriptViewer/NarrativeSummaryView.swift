import SwiftUI
import TranscriptCore

struct NarrativeSummaryView: View {
    struct Request: Equatable {
        let video: VideoTranscript
        let attempt: Int
        let reload: Int
        let batching: Bool
    }

    let video: VideoTranscript
    @Bindable var model: NarrativeSummaryModel
    let seek: (Double) -> Void

    var body: some View {
        SummaryEvidenceNavigation(
            target: SummaryEvidenceTarget.first(
                in: model.excerptVideo == video ? model.excerptState : .absent)
        ) {
            summaryScroll
        }
        .frame(maxWidth: .infinity, minHeight: 180, maxHeight: .infinity)
    }

    private var summaryScroll: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                SourceExcerptPanel(
                    state: model.excerptVideo == video ? model.excerptState : .absent, seek: seek)
                HStack {
                    Label("Apple Intelligenceによる要約", systemImage: "sparkles")
                        .font(.headline)
                    Spacer()
                    Button("保存済みを読み込む") { model.requestReload += 1 }
                        .disabled(model.isBatchRunning)
                    Button("再生成") { model.requestAttempt += 1 }
                        .disabled(isGenerating)
                }
                Text("このMacで生成します。文字起こしの誤認識やAIの誤りを含む場合があります。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if model.video != video {
                    ProgressView("要約を準備しています…")
                } else {
                    switch model.state {
                    case .loadingSaved:
                        ProgressView("保存済み要約を読み込んでいます…")
                    case .waitingForLibrary:
                        Text("新しい順に一括生成しています。左カラムで進捗を確認できます。動画を切り替えても生成は続きます。")
                            .foregroundStyle(.secondary)
                    case .idle:
                        Text("要約生成が中断されました。「再生成」で再試行できます。")
                    case .generating(let completed, let total):
                        ProgressView(value: Double(completed), total: Double(max(total + 1, 1)))
                        Text(
                            completed < total
                                ? "10分区間の要約を生成中（\(completed)/\(total)）。長い動画は時間がかかります。"
                                : "各区間の内容を統合して、動画全体の要約を生成しています…"
                        )
                        .foregroundStyle(.secondary)
                    case .failed(let message):
                        Label("要約を生成できません", systemImage: "exclamationmark.triangle")
                        Text(message).textSelection(.enabled)
                        Text("発話の抜粋への置き換えは行いません。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    case .ready(let summary):
                        Text("動画全体の要約")
                            .font(.title2)
                            .accessibilityAddTraits(.isHeader)
                        narrative(summary.overview)
                        Divider()
                        Text("10分ごとの要約")
                            .font(.title2)
                            .accessibilityAddTraits(.isHeader)
                        Text("文字起こしがある区間を表示します。時刻を押すと区間の先頭から再生します。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        ForEach(summary.sections) { section in
                            Button {
                                seek(section.source.startSeconds)
                            } label: {
                                Label(
                                    "\(section.source.startLabel) – \(section.source.endLabel)",
                                    systemImage: "play.circle"
                                )
                                .font(.headline.monospacedDigit())
                                .frame(minHeight: Chrome.hit)
                            }
                            .accessibilityHint("この10分区間の先頭から再生します")
                            narrative(section.text)
                            Divider()
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.trailing, 8)
        }
        .task(
            id: Request(
                video: video, attempt: model.requestAttempt, reload: model.requestReload,
                batching: model.isBatchRunning)
        ) {
            await model.loadExcerpts(video: video)
            if model.requestReload != model.handledReload {
                model.handledReload = model.requestReload
                await model.loadSaved(video: video)
            } else {
                let force = model.requestAttempt != model.handledAttempt
                model.handledAttempt = model.requestAttempt
                if case .representative = model.excerptState, !force {
                    await model.loadSaved(video: video)
                } else {
                    await model.load(video: video, force: force)
                }
            }
        }
    }

    private var isGenerating: Bool {
        if model.isBatchRunning { return true }
        guard model.video == video else { return false }
        if case .generating = model.state { return true }
        return false
    }

    private func narrative(_ text: String) -> some View {
        Text(text)
            .font(.body)
            .lineSpacing(5)
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}
