import SwiftUI
import TranscriptCore

/// Loads the saved excitement estimate of the playing transcript off the main thread and
/// shows it under the timeline; nothing is shown when no current estimate exists.
struct ExcitementSection: View {
    let transcriptURL: URL
    let currentSeconds: Double
    let seek: (Double) -> Void
    @State private var timeline: ExcitementTimeline?

    var body: some View {
        Group {
            if let timeline {
                ExcitementPanel(timeline: timeline, currentSeconds: currentSeconds, seek: seek)
            }
        }
        .task(id: transcriptURL) {
            let url = transcriptURL
            timeline = await Task.detached { ExcitementTimeline.load(transcriptURL: url) }.value
        }
    }
}

struct ExcitementPanel: View {
    let timeline: ExcitementTimeline
    let currentSeconds: Double
    let seek: (Double) -> Void
    @AppStorage("excitementMode") var modeName = ExcitementTimeline.Mode.combined.rawValue
    @State var detail = ""

    private var mode: ExcitementTimeline.Mode {
        ExcitementTimeline.Mode(rawValue: modeName) ?? .combined
    }

    var body: some View {
        let rows = timeline.rows(mode)
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("盛り上がり（この動画内の相対値）").font(.headline)
                if timeline.comments != nil {
                    Picker("盛り上がりの表示", selection: $modeName) {
                        Text("合算").tag(ExcitementTimeline.Mode.combined.rawValue)
                        Text("別々").tag(ExcitementTimeline.Mode.split.rawValue)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                } else {
                    Text(timeline.speech.loudnessDb == nil ? "発話（文字認識量）から推定" : "発話（文字認識量・音量）から推定")
                        .foregroundStyle(.secondary)
                }
            }
            GeometryReader { geometry in
                ExcitementBars(
                    rows: rows, duration: timeline.duration, currentSeconds: currentSeconds
                )
                .contentShape(Rectangle())
                .onContinuousHover(perform: hoverHandler(width: geometry.size.width))
                .onTapGesture(perform: tapHandler(width: geometry.size.width))
            }
            .frame(height: CGFloat(40 * rows.count))
            .accessibilityLabel("盛り上がりのタイムライン")
            .accessibilityHint("クリックした位置へ移動します")
            Text(detail).font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                .frame(minHeight: 18, alignment: .leading)
            ScrollView(.horizontal) {
                HStack {
                    Text("盛り上がり箇所:").foregroundStyle(.secondary)
                    ForEach(rows, id: \.label) { row in
                        ForEach(row.series.peaks, id: \.self) { start in
                            Button(peakTitle(row: row, start: start, split: rows.count > 1)) {
                                seek(start)
                            }
                        }
                    }
                }
            }
        }
    }

    func peakTitle(row: ExcitementTimeline.Row, start: Double, split: Bool) -> String {
        (split ? "\(row.label) " : "") + ExcitementTimeline.clock(start)
    }

    func hoverHandler(width: CGFloat) -> (HoverPhase) -> Void {
        { phase in
            if case .active(let location) = phase { show(fraction: location.x / max(width, 1)) }
        }
    }

    func tapHandler(width: CGFloat) -> (CGPoint) -> Void {
        { location in jump(fraction: location.x / max(width, 1)) }
    }

    func show(fraction: Double) {
        detail = timeline.detail(ofBin: timeline.bin(atFraction: fraction))
    }

    /// Touch-free trackpads still need the numbers, so a click shows them as it jumps.
    func jump(fraction: Double) {
        let bin = timeline.bin(atFraction: fraction)
        detail = timeline.detail(ofBin: bin)
        seek(timeline.seconds(ofBin: bin))
    }
}

struct ExcitementBars: View {
    let rows: [ExcitementTimeline.Row]
    let duration: Double
    let currentSeconds: Double

    var body: some View {
        Canvas { context, size in
            let track = Path(roundedRect: CGRect(origin: .zero, size: size), cornerRadius: 6)
            context.fill(track, with: .color(.secondary.opacity(0.25)))
            let rowHeight = size.height / CGFloat(max(rows.count, 1))
            for (index, row) in rows.enumerated() {
                let width = size.width / CGFloat(max(row.series.score.count, 1))
                let bottom = rowHeight * CGFloat(index + 1)
                for (bin, score) in row.series.score.enumerated() {
                    let height = CGFloat(score) * (rowHeight - 6)
                    let rect = CGRect(
                        x: CGFloat(bin) * width, y: bottom - height,
                        width: max(width - 1, 1), height: height)
                    context.fill(Path(rect), with: .color(Self.color(row.label)))
                }
                context.draw(
                    Text(row.label).font(.caption).foregroundStyle(.secondary),
                    at: CGPoint(x: 6, y: rowHeight * CGFloat(index) + 4), anchor: .topLeading)
            }
            let x = currentSeconds / max(duration, 0.001) * size.width
            context.fill(
                Path(CGRect(x: x, y: 0, width: 2, height: size.height)), with: .color(.primary))
        }
    }

    static func color(_ label: String) -> Color {
        switch label {
        case "コメント": .green
        case "合算": .orange
        default: .accentColor
        }
    }
}
