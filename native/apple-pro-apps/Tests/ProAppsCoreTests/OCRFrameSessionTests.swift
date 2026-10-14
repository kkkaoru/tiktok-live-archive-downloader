import Foundation
import Testing

@testable import ProAppsCore

@Suite(.timeLimit(.minutes(1)))
struct OCRFrameSessionTests {
  private actor OrderGate {
    private var waiting: CheckedContinuation<Void, Never>?
    private var pairWaiting: CheckedContinuation<Void, Never>?
    private var secondFinished = false
    private var active = 0
    private(set) var peak = 0
    private(set) var finished = 0
    private(set) var cancelled = 0

    func enter() {
      active += 1
      peak = max(peak, active)
    }

    func waitForSecond() async {
      if secondFinished { return }
      await withCheckedContinuation { waiting = $0 }
    }

    func waitForPair() async {
      if active == 2 {
        pairWaiting?.resume()
        pairWaiting = nil
        return
      }
      await withCheckedContinuation { pairWaiting = $0 }
    }

    func leave(second: Bool, wasCancelled: Bool = false) {
      active -= 1
      finished += 1
      if wasCancelled { cancelled += 1 }
      if second {
        secondFinished = true
        waiting?.resume()
        waiting = nil
      }
    }
  }

  private enum Mode: Sendable {
    case success, reversed, failure, peerFailure, cancelled, incomplete, duplicate, wrongClock,
      wrongCanvas
    case invalidConfidence, overflowingText, outsideIndex
  }

  private actor Worker: OCRFrameSessionWorker {
    let mode: Mode
    let gate: OrderGate

    init(_ mode: Mode, gate: OrderGate) {
      self.mode = mode
      self.gate = gate
    }

    func process(
      _ request: OCRFrameSessionRequest, indices: Range<Int>, sink: OCRFrameSessionSink
    ) async throws {
      await gate.enter()
      if mode == .reversed || mode == .peerFailure { await gate.waitForPair() }
      if mode == .reversed && indices.lowerBound == 0 { await gate.waitForSecond() }
      do {
        try await deliver(request, indices: indices, sink: sink)
      } catch {
        await gate.leave(second: indices.lowerBound != 0, wasCancelled: error is CancellationError)
        throw error
      }
      await gate.leave(second: indices.lowerBound != 0)
    }

    func deliver(
      _ request: OCRFrameSessionRequest, indices: Range<Int>, sink: OCRFrameSessionSink
    ) async throws {
      if mode == .failure { throw ProAppsError.unavailable("Synthetic worker failure") }
      if mode == .peerFailure {
        if indices.lowerBound == 0 { throw ProAppsError.unavailable("Synthetic peer failure") }
        while !Task.isCancelled { await Task.yield() }
        try Task.checkCancellation()
      }
      if mode == .cancelled { throw CancellationError() }
      if mode == .incomplete { return }
      for index in indices {
        try Task.checkCancellation()
        let sample = request.samples[index]
        let time = try sample.time().seconds
        let frame = SourceTextSessionFrame(
          measurement: FrameTextMeasurement(
            requestedTimeSeconds: time, actualTimeSeconds: time,
            width: mode == .wrongCanvas ? 7 : request.width, height: request.height,
            lines: [
              FrameTextLine(
                text: mode == .overflowingText ? String(repeating: "字", count: 6000) : "字\(index)",
                confidence: mode == .invalidConfidence ? .nan : 0.75, boundsClipped: true,
                region: .init(x: 0, y: 0, width: 1, height: 1))
            ]),
          actual: mode == .wrongClock ? .init(value: 100, timescale: 30) : sample,
          imageGenerationSeconds: 0, ocrSeconds: 0)
        try await sink.receive(frame, index: mode == .outsideIndex ? -1 : index)
        if mode == .duplicate { try await sink.receive(frame, index: index) }
      }
    }
  }

  private func request(
    samples: [ForegroundTime] = [.init(value: 0, timescale: 30), .init(value: 1, timescale: 30)],
    width: Int = 16, height: Int = 16, maximumSeconds: Double = 60
  ) -> OCRFrameSessionRequest {
    OCRFrameSessionRequest(
      path: "synthetic-input-not-opened", samples: samples,
      width: width, height: height, maximumSeconds: maximumSeconds)
  }

  @Test func completionOrderDoesNotChangeInputOrderOrRawFields() async throws {
    let gate = OrderGate()
    let result = try await OCRFrameSession.run(
      request(), factory: { Worker(.reversed, gate: gate) })
    #expect(result.measurements.map { $0.lines.first?.text } == ["字0", "字1"])
    #expect(result.actual.map(\.value) == [0, 1])
    #expect(result.actual.map(\.timescale) == [30, 30])
    #expect(result.measurements.first?.lines.first?.boundsClipped == true)
    #expect(result.measurements.first?.lines.first?.confidence == 0.75)
    #expect(await gate.peak == 2)
    #expect(await gate.finished == 2)
  }

  @Test func oneFrameUsesOnlyOneWorker() async throws {
    let gate = OrderGate()
    let result = try await OCRFrameSession.run(
      request(samples: [.init(value: 1, timescale: 25)]),
      factory: { Worker(.success, gate: gate) })
    #expect(result.actual.map(\.timescale) == [25])
    #expect(await gate.finished == 1)
    #expect(await gate.peak == 1)
  }

  @Test(arguments: [
    Mode.failure, .incomplete, .duplicate, .wrongClock, .wrongCanvas,
    .invalidConfidence, .overflowingText, .outsideIndex,
  ])
  private func invalidOrFailedWorkersNeverReturnPartialSuccess(_ mode: Mode) async {
    let gate = OrderGate()
    await #expect(throws: (any Error).self) {
      try await OCRFrameSession.run(request(), factory: { Worker(mode, gate: gate) })
    }
  }

  @Test func workerCancellationPropagates() async {
    let gate = OrderGate()
    await #expect(throws: CancellationError.self) {
      try await OCRFrameSession.run(request(), factory: { Worker(.cancelled, gate: gate) })
    }
  }

  @Test func failedWorkerCancelsAndJoinsItsActivePeer() async {
    let gate = OrderGate()
    await #expect(throws: ProAppsError.self) {
      try await OCRFrameSession.run(request(), factory: { Worker(.peerFailure, gate: gate) })
    }
    #expect(await gate.finished == 2)
    #expect(await gate.cancelled == 1)
    #expect(await gate.peak == 2)
  }

  @Test func maximumSampleBatchStillUsesOnlyTwoWorkers() async throws {
    let gate = OrderGate()
    let result = try await OCRFrameSession.run(
      request(samples: (0..<64).map { .init(value: Int64($0), timescale: 25) }),
      factory: { Worker(.reversed, gate: gate) })
    #expect(result.measurements.count == 64)
    #expect(result.actual.first?.value == 0)
    #expect(result.actual.last?.value == 63)
    #expect(await gate.finished == 2)
    #expect(await gate.peak == 2)
  }

  @Test func cancelledParentStartsNoNativeWork() async {
    let gate = OrderGate()
    let task = Task {
      await gate.waitForSecond()
      return try await OCRFrameSession.run(request(), factory: { Worker(.success, gate: gate) })
    }
    task.cancel()
    await gate.leave(second: true)
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await gate.peak == 0)
  }

  @Test(arguments: [0, -1, 16385])
  func invalidCanvas(_ width: Int) async {
    await #expect(throws: ProAppsError.self) {
      try await OCRFrameSession.run(request(width: width))
    }
  }

  @Test(arguments: [0.0, -1.0, 61.0, Double.nan, Double.infinity])
  func invalidDeadline(_ maximum: Double) async {
    await #expect(throws: ProAppsError.self) {
      try await OCRFrameSession.run(request(maximumSeconds: maximum))
    }
  }

  @Test func invalidSamplesAndPixelBudget() async {
    await #expect(throws: ProAppsError.self) { try await OCRFrameSession.run(request(samples: [])) }
    await #expect(throws: ProAppsError.self) {
      try await OCRFrameSession.run(
        request(samples: Array(repeating: .init(value: 0, timescale: 30), count: 65)))
    }
    await #expect(throws: ProAppsError.self) {
      try await OCRFrameSession.run(
        request(samples: [.init(value: 2, timescale: 30), .init(value: 1, timescale: 30)]))
    }
    await #expect(throws: ProAppsError.self) {
      try await OCRFrameSession.run(request(width: 16384, height: 16384))
    }
  }

  @Test func expiredSessionRefusesCompletion() async {
    let sink = OCRFrameSessionSink(
      request(), start: ContinuousClock().now.advanced(by: .seconds(-61)))
    await #expect(throws: ProAppsError.self) { try await sink.finish() }
  }

  @Test(arguments: [-1.0, Double.nan, Double.infinity])
  func invalidSetupTimingIsRejected(_ value: Double) async {
    let sink = OCRFrameSessionSink(request())
    var asset = SourceTextSessionTiming()
    asset.assetSetupSeconds = value
    await #expect(throws: ProAppsError.self) { try await sink.prepared(asset) }
    var recognition = SourceTextSessionTiming()
    recognition.ocrSetupSeconds = value
    await #expect(throws: ProAppsError.self) { try await sink.prepared(recognition) }
  }

  @Test func sourceBoundaryFailurePropagatesBeforeNativeWork() async {
    let sink = OCRFrameSessionSink(
      request(),
      beforeFrame: {
        throw ProAppsError.invalid("Source identity changed")
      })
    await #expect(throws: ProAppsError.invalid("Source identity changed")) {
      try await sink.check()
    }
  }

  @Test func nativeWorkersUseFullFrameExactClocks() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "ordered-ocr-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let movie = root.appendingPathComponent("synthetic.mov")
    let rendered = try await Runner.run(
      URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg"),
      [
        "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i", "color=black:s=32x32:r=30",
        "-t", "0.2", "-c:v", "libx264", "-threads", "1", "-video_track_timescale", "60000",
        movie.path,
      ])
    try #require(rendered.status == 0)
    let result = try await OCRFrameSession.run(
      OCRFrameSessionRequest(
        path: movie.path,
        samples: [.init(value: 0, timescale: 30), .init(value: 1, timescale: 30)],
        width: 32, height: 32, maximumSeconds: 60))
    #expect(result.measurements.map(\.width) == [32, 32])
    #expect(result.measurements.map(\.height) == [32, 32])
    #expect(result.measurements.allSatisfy { $0.lines.isEmpty })
    #expect(result.setup.assetSetupSeconds > 0)
    #expect(result.setup.ocrSetupSeconds > 0)
    #expect(result.frames.count == 2)
    #expect(try result.actual.map { try $0.time().seconds } == [0, 1.0 / 30.0])
  }
}
