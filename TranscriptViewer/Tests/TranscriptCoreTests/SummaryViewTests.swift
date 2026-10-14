import SwiftUI
import Testing
import TranscriptCore
import ViewInspector

@testable import TranscriptSummaries
@testable import TranscriptViewer

actor ViewResponseGate {
    private let arrived = AsyncStream<Void>.makeStream()
    private let released = AsyncStream<Void>.makeStream()

    func respond() async -> String {
        arrived.continuation.finish()
        for await _ in released.stream {}
        return "旅行の準備について説明している。"
    }

    func wait() async { for await _ in arrived.stream {} }
    func release() { released.continuation.finish() }
}

@MainActor
struct SummaryViewTests {
    @Test func reloadAndRegenerateActionsAreConsumedByTheViewTask() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let saved = try await fixture.generate("保存された説明文です。")
        let model = NarrativeSummaryModel(service: AppleSummaryService { _ in "新しく生成された説明文です。" })
        let view = NarrativeSummaryView(video: fixture.video, model: model) { _ in
            Issue.record("Reload and regenerate must not seek")
        }
        let request = NarrativeSummaryView.Request(
            video: fixture.video, attempt: 0, reload: 0, batching: false)
        try view.inspect().find(button: "保存済みを読み込む").tap()
        #expect(model.requestReload == 1)
        try await view.inspect().find(ViewType.ScrollView.self).callTask(id: request)
        #expect(model.handledReload == 1)
        #expect(model.state == .ready(saved))
        try view.inspect().find(button: "再生成").tap()
        #expect(model.requestAttempt == 1)
        try await view.inspect().find(ViewType.ScrollView.self).callTask(id: request)
        #expect(model.handledAttempt == 1)
        guard case .ready(let regenerated) = model.state else {
            Issue.record("Regeneration must display the provider result")
            return
        }
        #expect(regenerated.overview == "新しく生成された説明文です。")
        try await view.inspect().find(ViewType.ScrollView.self).callTask(id: request)
        #expect(model.state == .ready(regenerated))
        #expect(model.handledAttempt == 1)
        #expect(try await SummaryStore().load(video: fixture.video) == saved)
    }

    @Test(arguments: [SummaryRequest.Kind.section, .overview])
    func generationDisplaysPhaseAndPreventsDuplicateRequests(_ phase: SummaryRequest.Kind)
        async throws
    {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let gate = ViewResponseGate()
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { request in
                if request.kind == phase { return await gate.respond() }
                return "旅行の準備について説明している。"
            })
        async let loading: Void = model.load(video: fixture.video)
        await gate.wait()
        let result = Result {
            let inspected = try NarrativeSummaryView(video: fixture.video, model: model) { _ in
                Issue.record("A pending summary cannot seek")
            }.inspect()
            let expected =
                phase == .section
                ? "10分区間の要約を生成中（0/1）。長い動画は時間がかかります。"
                : "各区間の内容を統合して、動画全体の要約を生成しています…"
            #expect(try inspected.find(text: expected).string() == expected)
            #expect(try inspected.find(button: "再生成").isDisabled())
            #expect(try inspected.find(button: "保存済みを読み込む").isDisabled() == false)
            #expect(inspected.findAll(ViewType.ProgressView.self).count == 1)
        }
        await gate.release()
        await loading
        try result.get()
    }

    @Test func unselectedVideoShowsPreparationWithoutStaleSummary() throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { _ in
                Issue.record("Inspecting a view must not start inference")
                return "説明している。"
            })
        let view = NarrativeSummaryView(video: fixture.video, model: model) { _ in
            Issue.record("Preparation must not seek")
        }
        let inspected = try view.inspect()
        #expect(
            try inspected.find(ViewType.ProgressView.self).labelView().text().string()
                == "要約を準備しています…")
        #expect(try inspected.find(button: "再生成").isDisabled() == false)
        #expect(inspected.findAll(ViewType.Button.self).count == 2)
    }

    @Test func providerFailureIsVisibleWithoutAnExtractiveFallback() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { _ in
                throw SummaryGenerationError.unavailable("モデルを利用できません。")
            })
        await model.load(video: fixture.video)
        let view = NarrativeSummaryView(video: fixture.video, model: model) { _ in
            Issue.record("Failed summaries must not seek")
        }
        let inspected = try view.inspect()
        #expect(try inspected.find(text: "モデルを利用できません。").string() == "モデルを利用できません。")
        #expect(try inspected.find(text: "発話の抜粋への置き換えは行いません。").string() == "発話の抜粋への置き換えは行いません。")
        #expect(try inspected.find(button: "再生成").isDisabled() == false)
        #expect(inspected.findAll(ViewType.Button.self).count == 2)
    }

    @Test func cancelledGenerationOffersRetryWithoutReportingFailure() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let model = NarrativeSummaryModel(
            service: AppleSummaryService { _ in
                throw CancellationError()
            })
        await model.load(video: fixture.video)
        let inspected = try NarrativeSummaryView(video: fixture.video, model: model) { _ in
            Issue.record("Cancelled summaries must not seek")
        }.inspect()
        #expect(
            try inspected.find(text: "要約生成が中断されました。「再生成」で再試行できます。").string()
                == "要約生成が中断されました。「再生成」で再試行できます。")
        #expect(try inspected.find(button: "再生成").isDisabled() == false)
    }

    @Test func savedSummaryExposesItsOverviewAndSectionAction() async throws {
        let fixture = try SummaryFixture()
        defer { #expect(throws: Never.self) { try fixture.remove() } }
        let model = NarrativeSummaryModel(service: AppleSummaryService { _ in "旅行の準備について説明している。" })
        await model.load(video: fixture.video)
        var destination: Double?
        let view = NarrativeSummaryView(video: fixture.video, model: model) {
            destination = $0
        }
        let inspected = try view.inspect()
        #expect(try inspected.find(text: "動画全体の要約").string() == "動画全体の要約")
        #expect(
            try inspected.findAll(ViewType.Text.self).filter {
                try $0.string() == "旅行の準備について説明している。"
            }.count == 2)
        let section = try inspected.find(ViewType.Button.self) {
            try $0.labelView().find(ViewType.Label.self).title().text().string().contains("00:00")
        }
        try section.tap()
        #expect(destination == 0)
    }
}
