import CoreGraphics
import CoreText
import Foundation
import Testing

@testable import ProAppsCore

struct SourceTextSessionTests {
  enum Mode: Sendable {
    case normal, shiftedClock, canvas, confidence, region, timing, setupTiming, tooManyLines
    case overflow, mutate, collision, fail, delay, missing, extra, repeatedSetup, directory
  }

  private actor Worker: SourceTextSessionWorker {
    let mode: Mode
    var request: SourceTextSessionRequest?
    var preparations = 0
    var calls = 0
    init(_ mode: Mode = .normal) { self.mode = mode }

    func process(_ request: SourceTextSessionRequest, sink: SourceTextSessionSink) async throws {
      try await sink.begin(prepare(request))
      if mode == .repeatedSetup { try await sink.begin(prepare(request)) }
      for sample in request.samples {
        if mode == .missing && calls == 8 { break }
        try await sink.beforeFrame()
        try await sink.receive(frame(sample))
      }
      if mode == .extra { try await sink.receive(frame(.init(value: 16, timescale: 30))) }
    }

    func prepare(_ request: SourceTextSessionRequest) throws -> SourceTextSessionTiming {
      self.request = request
      preparations += 1
      var timing = SourceTextSessionTiming()
      timing.assetSetupSeconds = mode == .setupTiming ? .nan : 0.1
      timing.ocrSetupSeconds = 0.2
      return timing
    }

    func frame(_ sample: ForegroundTime) async throws -> SourceTextSessionFrame {
      let request = try #require(request)
      calls += 1
      if mode == .delay { try await Task.sleep(for: .milliseconds(5)) }
      if mode == .fail && calls == 9 { throw ProAppsError.unavailable("Synthetic failure") }
      if mode == .mutate && calls == 9 {
        try Data([2, 3]).write(to: URL(fileURLWithPath: request.path))
      }
      if mode == .directory && calls == 9 {
        try FileManager.default.moveItem(
          atPath: request.checkpointDirectory,
          toPath: request.checkpointDirectory + "-moved")
        try FileManager.default.createDirectory(
          atPath: request.checkpointDirectory,
          withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
      }
      if mode == .collision && calls == 8 {
        _ = try Files.writeNew(
          Data([1]), to: request.checkpointDirectory + "/block-0.json", extensions: ["json"])
      }
      let actual =
        mode == .shiftedClock
        ? ForegroundTime(value: sample.value + 1, timescale: sample.timescale) : sample
      let line = FrameTextLine(
        text: String(repeating: "a", count: mode == .overflow ? 4097 : 4096),
        confidence: mode == .confidence ? .nan : 0.75, boundsClipped: false,
        region: .init(x: mode == .region ? -1 : 0, y: 0, width: 1, height: 1))
      return SourceTextSessionFrame(
        measurement: FrameTextMeasurement(
          requestedTimeSeconds: try sample.time().seconds,
          actualTimeSeconds: try actual.time().seconds,
          width: mode == .canvas ? 17 : request.width, height: request.height,
          lines: Array(repeating: line, count: mode == .tooManyLines ? 65 : 1)),
        actual: actual, imageGenerationSeconds: mode == .timing ? -.infinity : 0.3, ocrSeconds: 0.4)
    }
  }

  private func fixture(
    count: Int = 64, maximumSeconds: Double = 60, width: Int = 16,
    samples: [ForegroundTime]? = nil
  ) throws -> (SourceTextSessionRequest, URL) {
    let source = try Files.reserveOutput(
      directory: Files.temporaryDirectory().path, name: "source.bin", kind: .edit)
    let root = source.deletingLastPathComponent()
    _ = try Files.writeNew(Data([1]), to: source.path, extensions: ["bin"])
    let directory = root.appendingPathComponent("checkpoints")
    try FileManager.default.createDirectory(
      at: directory, withIntermediateDirectories: false,
      attributes: [.posixPermissions: 0o700])
    let request = SourceTextSessionRequest(
      checkoutRoot: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().path, path: source.path,
      checkpointDirectory: directory.path,
      samples: samples ?? (0..<count).map { .init(value: Int64($0), timescale: 30) },
      width: width, height: 16, maximumSeconds: maximumSeconds)
    return (request, root)
  }

  private func remove(_ root: URL) {
    do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) }
  }

  @Test(arguments: [
    "valid", "escape", "root-link", "source-link", "hardlink", "missing", "directory", "replaced",
  ])
  func explicitSourceRootRetainsReadonlyIdentityGuards(_ mode: String) throws {
    let (original, root) = try fixture(count: 1)
    defer { remove(root) }
    let allowed = root.appendingPathComponent("allowed")
    try FileManager.default.createDirectory(at: allowed, withIntermediateDirectories: false)
    let source = allowed.appendingPathComponent("source.bin")
    try Data([1]).write(to: source)
    var declared = allowed.path
    var selected = source.path
    switch mode {
    case "escape": selected = original.path
    case "root-link":
      let alias = root.appendingPathComponent("alias")
      try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: allowed)
      declared = alias.path
    case "source-link":
      let alias = allowed.appendingPathComponent("alias.bin")
      try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: source)
      selected = alias.path
    case "hardlink":
      try FileManager.default.linkItem(at: source, to: allowed.appendingPathComponent("copy.bin"))
    case "missing": declared = root.appendingPathComponent("missing").path
    case "directory": selected = allowed.path
    default: break
    }
    let request = SourceTextSessionRequest(
      checkoutRoot: original.checkoutRoot, path: selected,
      checkpointDirectory: original.checkpointDirectory, samples: original.samples,
      width: 16, height: 16, maximumSeconds: 60, allowedSourceRoot: declared)
    if mode != "valid" && mode != "replaced" {
      #expect(throws: (any Error).self) { try SessionPaths(request) }
      return
    }
    let paths = try SessionPaths(request)
    defer { paths.close() }
    try paths.check()
    if mode == "replaced" {
      try FileManager.default.moveItem(at: allowed, to: root.appendingPathComponent("retained"))
      try FileManager.default.createDirectory(at: allowed, withIntermediateDirectories: false)
      try Data([1]).write(to: source)
      #expect(throws: (any Error).self) { try paths.check() }
    }
    #expect(try Data(contentsOf: source) == Data([1]))
    #expect(
      try FileManager.default.contentsOfDirectory(atPath: original.checkpointDirectory).isEmpty)
  }

  @Test(arguments: [1, 8, 9, 64])
  func checkpointsKeepOriginalEightFrameFormatAndReuseOneWorker(count: Int) async throws {
    let (request, root) = try fixture(count: count)
    defer { remove(root) }
    let worker = Worker()
    let report = try await SourceTextSession.run(request, worker: worker)
    #expect(await worker.preparations == 1)
    #expect(await worker.calls == count)
    #expect(report.frames == count && report.blocks.count == (count + 7) / 8)
    #expect(report.timing.assetSetupSeconds == 0.1 && report.timing.ocrSetupSeconds == 0.2)
    #expect(abs(report.timing.imageGenerationSeconds - Double(count) * 0.3) < 1e-10)
    #expect(abs(report.timing.ocrSeconds - Double(count) * 0.4) < 1e-10)
    #expect(report.timing.serializationSeconds >= 0 && report.timing.checkpointWriteSeconds >= 0)
    #expect(!report.sourceHashVerified && !report.qualityApproved && !report.sourceModified)
    for block in report.blocks {
      let rows = try JSONDecoder().decode(
        [FrameTextMeasurement].self,
        from: Files.read(URL(fileURLWithPath: block.measurementsPath)))
      #expect(rows.count == block.requested.count && rows.count <= 8)
      #expect(block.requested == block.actual)
      #expect(rows.allSatisfy { $0.lines.first?.text.utf8.count == 4096 })
      let receipt = try JSONDecoder().decode(
        SourceTextSessionBlock.self,
        from: Files.read(
          URL(
            fileURLWithPath: request.checkpointDirectory
              + "/block-\(block.firstSample / 8)-clocks.json")))
      #expect(receipt.requested == block.requested && receipt.actual == block.actual)
    }
    await #expect(throws: (any Error).self) {
      try await SourceTextSession.run(request, worker: worker)
    }
    #expect(await worker.preparations == 1)
  }

  @Test(arguments: [
    Mode.shiftedClock, .canvas, .confidence, .region, .timing, .setupTiming,
    .tooManyLines, .overflow, .mutate, .collision, .fail, .missing, .extra, .repeatedSetup,
    .directory,
  ])
  func failuresRetainOnlyCompletedBlocksAndNeverPublishCompletion(mode: Mode) async throws {
    let (request, root) = try fixture(count: 16)
    defer { remove(root) }
    await #expect(throws: (any Error).self) {
      try await SourceTextSession.run(request, worker: Worker(mode))
    }
    let names = try FileManager.default.contentsOfDirectory(atPath: request.checkpointDirectory)
      .sorted()
    if mode == .fail || mode == .mutate || mode == .missing {
      #expect(names == ["block-0-clocks.json", "block-0.json"])
    } else if mode == .extra {
      #expect(
        names == ["block-0-clocks.json", "block-0.json", "block-1-clocks.json", "block-1.json"])
    } else if mode == .collision {
      #expect(names == ["block-0.json"])
    } else {
      #expect(names.isEmpty)
    }
  }

  @Test func checkpointDescriptorRejectsInvalidNamesAndKeepsExistingBytes() throws {
    let (request, root) = try fixture(count: 1)
    defer { remove(root) }
    let paths = try SessionPaths(request)
    defer { paths.close() }
    #expect(throws: ProAppsError.self) { try paths.write(Data([1]), name: "../outside.json") }
    try paths.write(Data([2]), name: "block-0.json")
    #expect(throws: (any Error).self) { try paths.write(Data([3]), name: "block-0.json") }
    #expect(
      try Files.read(URL(fileURLWithPath: request.checkpointDirectory + "/block-0.json"))
        == Data([2]))
  }

  @Test(arguments: [0, 65])
  func invalidCountsFailBeforePreparing(count: Int) async throws {
    let (request, root) = try fixture(count: count)
    defer { remove(root) }
    let worker = Worker()
    await #expect(throws: (any Error).self) {
      try await SourceTextSession.run(request, worker: worker)
    }
    #expect(await worker.preparations == 0)
  }

  @Test(arguments: [0.0, -1, 60.001, Double.nan, Double.infinity])
  func invalidDeadlinesFailBeforePreparing(seconds: Double) async throws {
    let (request, root) = try fixture(maximumSeconds: seconds)
    defer { remove(root) }
    await #expect(throws: (any Error).self) {
      try await SourceTextSession.run(request, worker: Worker())
    }
  }

  @Test(arguments: [
    [ForegroundTime(value: -1, timescale: 30)], [.init(value: 1, timescale: 0)],
    [.init(value: 0, timescale: 30), .init(value: 0, timescale: 60)],
    [.init(value: 1, timescale: 30), .init(value: 0, timescale: 30)],
  ])
  func invalidOriginalClocksFail(samples: [ForegroundTime]) async throws {
    let (request, root) = try fixture(samples: samples)
    defer { remove(root) }
    await #expect(throws: (any Error).self) {
      try await SourceTextSession.run(request, worker: Worker())
    }
  }

  @Test func deadlineAndCancellationPreserveFailure() async throws {
    let (request, root) = try fixture(maximumSeconds: 0.001)
    defer { remove(root) }
    await #expect(throws: ProAppsError.timedOut) {
      try await SourceTextSession.run(request, worker: Worker(.delay))
    }
    await withTaskGroup(of: Void.self) { group in
      group.cancelAll()
      group.addTask {
        await #expect(throws: CancellationError.self) {
          try await SourceTextSession.run(request, worker: Worker())
        }
      }
    }
  }

  @Test(arguments: [
    "source-link", "directory-link", "public-directory", "empty-source", "missing-source",
    "outside", "wrong-root", "normalized",
  ])
  func unsafePathsFailBeforePreparing(mode: String) async throws {
    let (original, root) = try fixture(count: 1)
    defer { remove(root) }
    var source = original.path
    var directory = original.checkpointDirectory
    var checkout = original.checkoutRoot
    switch mode {
    case "source-link":
      source = root.appendingPathComponent("link.bin").path
      try FileManager.default.createSymbolicLink(atPath: source, withDestinationPath: original.path)
    case "directory-link":
      directory = root.appendingPathComponent("link").path
      try FileManager.default.createSymbolicLink(
        atPath: directory, withDestinationPath: original.checkpointDirectory)
    case "public-directory":
      try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: directory)
    case "empty-source": try Data().write(to: URL(fileURLWithPath: source))
    case "missing-source": source = root.appendingPathComponent("absent.bin").path
    case "outside": source = "/outside.bin"
    case "wrong-root": checkout = root.path
    default: source = root.path + "/./source.bin"
    }
    let request = SourceTextSessionRequest(
      checkoutRoot: checkout, path: source,
      checkpointDirectory: directory, samples: original.samples, width: 16, height: 16,
      maximumSeconds: 60)
    let worker = Worker()
    await #expect(throws: (any Error).self) {
      try await SourceTextSession.run(request, worker: worker)
    }
    #expect(await worker.preparations == 0)
  }

  @Test func nativeWorkerUsesExactSyntheticFramesAndOldEightLimitStaysIntact() async throws {
    let (original, root) = try fixture(count: 2)
    defer { remove(root) }
    let source = try #require(
      Bundle.module.url(forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures")
    )
    let request = SourceTextSessionRequest(
      checkoutRoot: original.checkoutRoot, path: source.path,
      checkpointDirectory: original.checkpointDirectory, samples: original.samples,
      width: 16, height: 16, maximumSeconds: 60)
    let report = try await SourceTextSession.run(request)
    #expect(report.frames == 2)
    let actual = try #require(report.blocks.first?.actual)
    #expect(try zip(actual, original.samples).allSatisfy { try $0.0.time() == $0.1.time() })
    #expect(report.timing.assetSetupSeconds > 0 && report.timing.ocrSetupSeconds > 0)
    await #expect(throws: (any Error).self) {
      try await FrameProbe().recognizeText(
        path: source.path, samples: Array(repeating: .init(timeSeconds: 0), count: 9))
    }
    await #expect(throws: (any Error).self) {
      try await FrameProbe().sessionSource(path: source.path, samples: [])
    }
    let context = try #require(
      CGContext(
        data: nil, width: 512, height: 256,
        bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    context.setFillColor(CGColor(gray: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 512, height: 256))
    context.textPosition = CGPoint(x: 40, y: 160)
    let text = NSAttributedString(
      string: "TEST",
      attributes: [
        NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName(
          "Helvetica-Bold" as CFString, 64, nil),
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(
          gray: 1, alpha: 1),
      ])
    CTLineDraw(CTLineCreateWithAttributedString(text), context)
    let image = try #require(context.makeImage())
    let recognized = try await NativeSourceTextSessionWorker.textFrame(
      image: image, sample: .init(value: 0, timescale: 30), actual: .init(value: 0, timescale: 30),
      imageGenerationSeconds: 0, recognition: CPUTextRecognition.request())
    #expect(recognized.measurement.lines.contains { $0.text == "TEST" })
    let cropped = try FrameProbe.textLine(
      text: "TEST", confidence: 0.75,
      rawRectangle: CGRect(x: -1, y: 2, width: 12, height: 4),
      selected: CGRect(x: 20, y: 140, width: 30, height: 100), fullFrameHeight: 256)
    #expect(cropped.boundsClipped && cropped.region.x == 20 && cropped.region.y == 18)
  }
}
