import AVFoundation
import CoreImage
import Foundation
import Testing

@testable import ProAppsCore

private enum ForegroundTestFailure: Error { case expected }

struct ForegroundMeasurementTests {
  static func fixture() throws -> URL {
    try #require(
      Bundle.module.url(forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures")
    )
  }

  static func gray(_ bytes: [UInt8], width: Int, height: Int, stride: Int) throws -> CGImage {
    let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
    return try #require(
      CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: stride,
        space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
  }

  @Test(arguments: [
    ForegroundTime(value: -1, timescale: 1), .init(value: 0, timescale: 0),
    .init(value: 0, timescale: -1), .init(value: 0, timescale: 1_000_000_001),
    .init(value: 86401, timescale: 1), .init(value: Int64.max, timescale: 1),
  ])
  func invalidClock(_ sample: ForegroundTime) {
    #expect(throws: ProAppsError.self) { try sample.time() }
  }

  @Test func rationalBoundariesAndOrdering() throws {
    #expect(
      try ForegroundTime(value: 86_400_000_000_000, timescale: 1_000_000_000).time()
        == CMTime(value: 86400, timescale: 1))
    let tick = ForegroundTime(value: 1, timescale: 1_000_000_000)
    #expect(try tick.time().value == 1)
    #expect(try tick.time().timescale == 1_000_000_000)
    #expect(
      try ForegroundTime.validate((0..<8).map { .init(value: Int64($0), timescale: 7) }).count == 8)
    #expect(throws: ProAppsError.self) { try ForegroundTime.validate([]) }
    #expect(throws: ProAppsError.self) {
      try ForegroundTime.validate(Array(repeating: tick, count: 9))
    }
    #expect(throws: ProAppsError.self) {
      try ForegroundTime.validate([.init(value: 1, timescale: 7), .init(value: 2, timescale: 14)])
    }
    #expect(throws: ProAppsError.self) {
      try ForegroundTime.validate([tick, .init(value: 0, timescale: 1)])
    }
    #expect(
      try ForegroundTime.actual(
        CMTime(value: 2, timescale: 14), requested: CMTime(value: 1, timescale: 7))
        == .init(value: 2, timescale: 14))
  }

  @Test(arguments: [
    CMTime.invalid, .indefinite, .positiveInfinity, .negativeInfinity,
    CMTime(value: 1, timescale: 7, flags: [.valid, .hasBeenRounded], epoch: 0),
    CMTime(value: 1, timescale: 7, flags: .valid, epoch: 1),
    CMTime(value: 1, timescale: 8), CMTime(value: -1, timescale: 7),
    CMTime(value: 0, timescale: 1_000_000_001),
  ])
  func invalidActualAndRequested(_ time: CMTime) {
    #expect(throws: ProAppsError.self) {
      try ForegroundTime.actual(time, requested: CMTime(value: 1, timescale: 7))
    }
    #expect(throws: ProAppsError.self) {
      try ForegroundTime.actual(CMTime(value: 1, timescale: 7), requested: time)
    }
  }

  @Test(arguments: [(0, 3), (3, 0)])
  func coreImageRowsAreTopLeft(_ coreImageY: Int, _ expectedTop: Int) throws {
    let extent = CGRect(x: 0, y: 0, width: 5, height: 4)
    let background = CIImage(color: .black).cropped(to: extent)
    let marker = CIImage(color: .white).cropped(
      to: CGRect(x: 3, y: coreImageY, width: 1, height: 1))
    let context = CIContext()
    defer { context.clearCaches() }
    let mask = try ForegroundMaskStatistics.measure(
      marker.composited(over: background), width: 5, height: 4, context: context)
    #expect(mask.nonzeroPixels == 1)
    #expect(mask.nonzeroBounds == .init(x: 3, y: expectedTop, width: 1, height: 1))
  }

  @Test func l8RoundTripAndLayoutRejection() throws {
    let image = try Self.gray([0, 1, 255, 255, 0, 0, 0, 255], width: 3, height: 2, stride: 4)
    let context = CIContext()
    defer { context.clearCaches() }
    let mask = try ForegroundMaskStatistics.measure(
      CIImage(cgImage: image), width: 3, height: 2, context: context)
    #expect(mask.nonzeroPixels == 2)
    #expect(mask.nonzeroBounds == .init(x: 1, y: 0, width: 2, height: 1))
    #expect(throws: ProAppsError.self) {
      try ForegroundMaskStatistics.bitmap(image, width: 2, height: 2)
    }
    let rgba = try #require(
      context.createCGImage(
        CIImage(cgImage: image), from: CGRect(x: 0, y: 0, width: 3, height: 2),
        format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB()))
    #expect(throws: ProAppsError.self) {
      try ForegroundMaskStatistics.bitmap(rgba, width: 3, height: 2)
    }
    #expect(throws: ProAppsError.self) {
      try ForegroundMaskStatistics.measure(
        CIImage(cgImage: image).transformed(by: .init(translationX: 1, y: 0)),
        width: 3, height: 2, context: context)
    }
  }

  @Test(arguments: [IndexSet(integer: 0), IndexSet(integersIn: 1...257)])
  func invalidLabels(_ labels: IndexSet) {
    #expect(throws: ProAppsError.self) {
      try ForegroundObservation.collect(labels: labels) {
        Issue.record("Invalid labels reached mask generation")
        throw ForegroundTestFailure.expected
      }
    }
  }

  @Test func emptyLabelsDoNotGenerateAndFailuresPropagate() throws {
    let empty = try ForegroundObservation.collect(labels: []) {
      throw ForegroundTestFailure.expected
    }
    #expect(empty.labels.isEmpty && empty.mask == nil)
    #expect(throws: ForegroundTestFailure.self) {
      try ForegroundObservation.collect(labels: [1]) { throw ForegroundTestFailure.expected }
    }
  }

  @Test func absentObservationEncodesExplicitNulls() throws {
    let frame = ForegroundMeasurement(
      requested: .init(value: 1, timescale: 7), actual: .init(value: 2, timescale: 14),
      width: 2, height: 1, observation: nil)
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    #expect(
      String(decoding: try encoder.encode(frame), as: UTF8.self)
        == #"{"actual":{"timescale":14,"value":2},"captionPlacementApproved":false,"height":1,"instanceLabels":null,"maskThreshold8Bit":1,"observationPresent":false,"requested":{"timescale":7,"value":1},"unionMask":null,"width":2}"#
    )
  }

  @Test func realGeneratorReceivesOriginalRationalAndPreservesSource() async throws {
    let source = try Self.fixture()
    let original = try Data(contentsOf: source)
    let times = try ForegroundTime.validate([.init(value: 3000, timescale: 90000)])
    let time = try #require(times.first)
    #expect(time.value == 3000 && time.timescale == 90000)
    #expect(time.epoch == 0 && time.flags == .valid)
    let generator = try await FrameProbe().source(
      path: source.path, samples: [FrameSample(timeSeconds: time.seconds)])
    defer { generator.cancelAllCGImageGeneration() }
    #expect(generator.appliesPreferredTrackTransform)
    #expect(generator.requestedTimeToleranceBefore == .zero)
    #expect(generator.requestedTimeToleranceAfter == .zero)
    let decoded = try await generator.image(at: time)
    #expect(decoded.actualTime == CMTime(value: 1, timescale: 30))
    var operations = ForegroundOperations()
    operations.observe = { _, _, _ in nil }
    let frames = try await FrameProbe().measureForeground(
      path: source.path, samples: [.init(value: 3000, timescale: 90000)], operations: operations)
    let frame = try #require(frames.first)
    #expect(frame.requested == .init(value: 3000, timescale: 90000))
    #expect(try frame.actual.time() == CMTime(value: 1, timescale: 30))
    #expect(frame.width == 16 && frame.height == 16)
    #expect(frame.observation == nil)
    #expect(try Data(contentsOf: source) == original)
  }

  @Test func nonGridRequestCannotBecomeAnObservation() async throws {
    let source = try Self.fixture()
    let original = try Data(contentsOf: source)
    let times = try ForegroundTime.validate([.init(value: 1, timescale: 7)])
    let time = try #require(times.first)
    #expect(time.value == 1 && time.timescale == 7)
    var operations = ForegroundOperations()
    operations.observe = { _, _, _ in
      Issue.record("A mismatched frame reached inference")
      return nil
    }
    await #expect(throws: (any Error).self) {
      try await FrameProbe().measureForeground(
        path: source.path, samples: [.init(value: 1, timescale: 7)], operations: operations)
    }
    #expect(try Data(contentsOf: source) == original)
  }

  @Test(arguments: ["decode", "observe", "cancel"])
  func failuresPreserveSourceAndProbeRemainsUsable(_ stage: String) async throws {
    let source = try Self.fixture()
    let original = try Data(contentsOf: source)
    let probe = FrameProbe()
    var operations = ForegroundOperations()
    let samples: [ForegroundTime] =
      stage == "decode" ? [.init(value: 1, timescale: 7)] : [.init(value: 0, timescale: 7)]
    operations.observe = { _, _, _ in
      if stage == "cancel" { throw CancellationError() }
      throw ForegroundTestFailure.expected
    }
    do {
      _ = try await probe.measureForeground(
        path: source.path, samples: samples, operations: operations)
      Issue.record("Expected the decoding-clock or injected observation failure")
    } catch is CancellationError {
      #expect(stage == "cancel")
    } catch is ForegroundTestFailure {
      #expect(stage == "observe")
    } catch is ProAppsError {
      #expect(stage == "decode")
    } catch is AVError {
      #expect(stage == "decode")
    } catch {
      Issue.record(error)
    }
    var recovery = ForegroundOperations()
    recovery.observe = { _, _, _ in nil }
    let result = try await probe.measureForeground(
      path: source.path, samples: [.init(value: 0, timescale: 7)], operations: recovery)
    #expect(result.count == 1)
    #expect(try Data(contentsOf: source) == original)
  }

  @Test func cancellationBeforePreflightAndInvalidPublicRequest() async throws {
    let source = try Self.fixture()
    await withTaskGroup(of: Void.self) { group in
      group.cancelAll()
      group.addTask {
        await #expect(throws: CancellationError.self) {
          try await FrameProbe().measureForeground(
            path: source.path, samples: [.init(value: 0, timescale: 1)])
        }
      }
    }
    await #expect(throws: ProAppsError.self) {
      try await FrameProbe().measureForeground(path: source.path, samples: [])
    }
  }
}
