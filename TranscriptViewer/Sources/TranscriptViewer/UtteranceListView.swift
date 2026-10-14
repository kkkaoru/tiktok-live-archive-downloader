import SwiftUI
import TranscriptCore

struct UtteranceListView: View {
    @Bindable var session: PlaybackSession

    var body: some View {
        List(session.utterances) { cue in
            Button {
                session.seek(toUtterance: cue.id)
            } label: {
                HStack(alignment: .top, spacing: 16) {
                    Text(cue.startLabel)
                        .font(.title2.monospacedDigit())
                        .frame(minWidth: 168, minHeight: Chrome.hit, alignment: .leading)
                    Text(cue.text)
                        .font(.title2)
                        .multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, minHeight: Chrome.hit, alignment: .leading)
                }
                .contentShape(Rectangle())
                .padding(.vertical, 6)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("\(cue.startLabel) \(cue.text)")
            .accessibilityHint("発話の先頭から再生します")
            .accessibilityAddTraits(session.selectedUtteranceID == cue.id ? [.isSelected] : [])
        }
        .accessibilityLabel("認識結果一覧")
    }
}

struct UtteranceWindow: View {
    @Bindable var session: PlaybackSession

    var body: some View {
        NavigationStack {
            Group {
                if session.selectedVideo == nil {
                    ContentUnavailableView(
                        "動画が選ばれていません",
                        systemImage: "text.alignleft",
                        description: Text("メインウィンドウで動画を選ぶと認識結果が表示されます")
                    )
                } else {
                    UtteranceListView(session: session)
                }
            }
            .navigationTitle(session.selectedVideo?.title ?? "認識結果")
        }
        .frame(minWidth: 480, minHeight: 360)
    }
}
