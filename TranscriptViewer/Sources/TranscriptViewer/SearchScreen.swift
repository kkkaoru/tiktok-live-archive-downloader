import SwiftUI
import TranscriptCore

struct SearchScreen: View {
    @Bindable var session: PlaybackSession

    private var request: SearchRequest {
        SearchRequest(
            query: session.searchQuery, scope: session.searchScope,
            currentVideoID: session.selectedVideoID, catalogRevision: session.catalogRevision)
    }

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                TextField("字幕の文言", text: $session.searchQuery)
                    .textFieldStyle(.roundedBorder)
                    .frame(minHeight: Chrome.hit)
                    .accessibilityLabel("字幕の文言を検索")
                    .accessibilityHint("一致した発話の先頭から再生できます")
                Picker("検索範囲", selection: $session.searchScope) {
                    Text("この動画").tag(SearchScope.currentVideo)
                    Text("すべての動画").tag(SearchScope.allVideos)
                }
                .pickerStyle(.radioGroup)
                .accessibilityLabel("検索範囲")
                results
            }
            .padding(16)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .navigationTitle("検索")
            .toolbar {
                Button("閉じる") {
                    session.showsSearch = false
                }
                .frame(minHeight: Chrome.hit)
                .keyboardShortcut(.cancelAction)
                .accessibilityLabel("検索を閉じる")
            }
        }
        .frame(minWidth: 640, minHeight: 480)
        .task(id: request) {
            let request = request
            let videos = session.videos
            let query = request.query.trimmingCharacters(in: .whitespacesAndNewlines)
            session.searchHits = []
            guard !query.isEmpty else {
                session.isSearching = false
                return
            }
            session.isSearching = true
            do {
                try await Task.sleep(for: .milliseconds(200))
                let results = await TranscriptSearch.hitsAsync(
                    query: query, videos: videos, scope: request.scope,
                    currentVideoID: request.currentVideoID)
                try Task.checkCancellation()
                session.searchHits = results
                session.isSearching = false
            } catch is CancellationError {
                return
            } catch {
                session.isSearching = false
                session.errorText = "検索に失敗しました"
                session.showsError = true
            }
        }
    }

    @ViewBuilder
    private var results: some View {
        let query = session.searchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        if query.isEmpty {
            ContentUnavailableView(
                "文言を入力",
                systemImage: "magnifyingglass",
                description: Text("この動画、またはすべての動画の最終発話から検索します")
            )
            .frame(maxWidth: .infinity, minHeight: Chrome.hit)
        } else if session.isSearching {
            ProgressView("検索中")
                .frame(maxWidth: .infinity, minHeight: Chrome.hit)
        } else if session.searchHits.isEmpty {
            ContentUnavailableView.search(text: query)
                .accessibilityLabel("検索結果はありません")
        } else {
            List(session.searchHits) { hit in
                Button {
                    session.play(hit)
                } label: {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(hit.videoTitle)
                            .font(.headline)
                            .foregroundStyle(.secondary)
                        Text(hit.cue.startLabel)
                            .font(.title2.monospacedDigit())
                        Text(hit.cue.text)
                            .font(.title2)
                            .multilineTextAlignment(.leading)
                    }
                    .frame(maxWidth: .infinity, minHeight: 56, alignment: .leading)
                    .contentShape(Rectangle())
                    .padding(.vertical, 8)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("\(hit.videoTitle) \(hit.cue.startLabel) \(hit.cue.text)")
                .accessibilityHint("発話の先頭から再生します")
            }
            .accessibilityLabel("検索結果 \(session.searchHits.count)件")
        }
    }
}

struct SearchRequest: Equatable {
    let query: String
    let scope: SearchScope
    let currentVideoID: URL?
    let catalogRevision: Int
}
