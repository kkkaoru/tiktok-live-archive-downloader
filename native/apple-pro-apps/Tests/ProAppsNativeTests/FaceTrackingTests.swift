import AVFoundation
import CoreML
import Foundation
import Testing
import Vision

@testable import ProAppsCore

struct FaceTrackingTests {
  @Test func rejectsMissingCPUComputeStages() throws {
    let request = VNDetectFaceRectanglesRequest()
    #expect(throws: ProAppsError.unavailable("Vision request exposes no CPU compute stages")) {
      try FaceTrackingSequence.configureCPU(request, supported: [:])
    }
    #expect(
      throws: ProAppsError.unavailable("Vision compute stage does not support CPU execution")
    ) {
      try FaceTrackingSequence.configureCPU(request, supported: [.main: []])
    }
  }

  @Test func explicitlyConfiguresEveryNativeDetectionAndTrackingStageForCPU() throws {
    let detection = VNDetectFaceRectanglesRequest()
    let tracking = VNTrackObjectRequest(
      detectedObjectObservation: VNDetectedObjectObservation(
        boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.3)))
    try FaceTrackingSequence.configureCPU(
      detection, supported: detection.supportedComputeStageDevices)
    try FaceTrackingSequence.configureCPU(
      tracking, supported: tracking.supportedComputeStageDevices)
    let detectionStages = try detection.supportedComputeStageDevices.keys
    let trackingStages = try tracking.supportedComputeStageDevices.keys
    #expect(!detectionStages.isEmpty && !trackingStages.isEmpty)
    #expect(
      detectionStages.allSatisfy {
        if case .cpu = detection.computeDevice(for: $0) { true } else { false }
      })
    #expect(
      trackingStages.allSatisfy {
        if case .cpu = tracking.computeDevice(for: $0) { true } else { false }
      })
  }
  @Test(arguments: [Float.nan, -.infinity, .infinity, -1, 0, 0.3, 1.1])
  func rejectsMissingLowOrInvalidConfidence(_ confidence: Float) {
    #expect(!FaceTrackingSequence.accepts(confidence))
  }

  @Test(arguments: [Float(0.31), 0.5, 1])
  func acceptsBoundedTrackingConfidence(_ confidence: Float) {
    #expect(FaceTrackingSequence.accepts(confidence))
  }

  private func image(checkerboard: Bool = true) throws -> CGImage {
    let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
    let context = try #require(
      CGContext(
        data: nil, width: 256, height: 256, bitsPerComponent: 8, bytesPerRow: 0,
        space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    for row in 0..<8 {
      for column in 0..<8 {
        let level: CGFloat = !checkerboard || (row + column).isMultiple(of: 2) ? 0 : 1
        context.setFillColor(CGColor(gray: level, alpha: 1))
        context.fill(CGRect(x: column * 32, y: row * 32, width: 32, height: 32))
      }
    }
    return try #require(context.makeImage())
  }

  @Test func mapsSourcePixelsAndExercisesNativeObjectTrackingWithoutClaimingFaceDetection()
    throws
  {
    let bitmap = try image()
    let observation = VNFaceObservation(
      boundingBox: CGRect(x: 0.125, y: 0.25, width: 0.25, height: 0.5))
    let measured = try FaceTrackingSequence.measure(observation, in: bitmap)
    #expect(measured.region.x == 32 && measured.region.y == 64)
    #expect(measured.region.width == 64 && measured.region.height == 128)
    let native = FaceTrackingSequence()
    #expect(try native.track(in: bitmap).isEmpty)
    #expect(try native.detect(in: bitmap).isEmpty)
    #expect(try native.track(in: bitmap).isEmpty)
    let sequence = FaceTrackingSequence(detectFaces: { _ in [observation] })
    #expect(try sequence.detect(in: bitmap).count == 1)
    let tracked = try sequence.track(in: bitmap)
    #expect(tracked.count == 1)
    let first = try #require(tracked.first)
    #expect(first.confidence > 0.3 && first.confidence <= 1)
    let blank = try image(checkerboard: false)
    #expect(try sequence.track(in: blank).isEmpty)
    #expect(try sequence.track(in: blank).isEmpty)
    let excessive = FaceTrackingSequence(detectFaces: { _ in
      Array(repeating: observation, count: 33)
    })
    #expect(throws: ProAppsError.outputLimit) { try excessive.detect(in: bitmap) }
    let invalid = FaceTrackingSequence(detectFaces: { _ in
      [VNFaceObservation(boundingBox: CGRect(x: 2, y: 0, width: 1, height: 1))]
    })
    #expect(try invalid.detect(in: bitmap).isEmpty)
    #expect(invalid.outOfFrameDetectionObservations == 1)
    #expect(throws: ProAppsError.self) {
      try FaceTrackingSequence.measure(
        VNFaceObservation(boundingBox: CGRect(x: 2, y: 0, width: 1, height: 1)), in: bitmap)
    }
  }

  @Test(arguments: [
    CGRect(x: -0.1, y: 0.2, width: 0.3, height: 0.3),
    CGRect(x: 0.8, y: 0.2, width: 0.3, height: 0.3),
    CGRect(x: 0.2, y: -0.1, width: 0.3, height: 0.3),
    CGRect(x: 0.2, y: 0.8, width: 0.3, height: 0.3),
  ])
  func terminatesOffImageTracksWithoutReturningClippedGeometry(_ box: CGRect) throws {
    let bitmap = try image()
    let observation = VNDetectedObjectObservation(boundingBox: box)
    let request = VNTrackObjectRequest(detectedObjectObservation: observation)
    let sequence = FaceTrackingSequence()
    #expect(try sequence.advance(request, observation: observation, in: bitmap) == nil)
    #expect(request.isLastFrame)
    #expect(sequence.outOfFrameTrackingObservations == 1)
    sequence.beginFrame()
    #expect(sequence.outOfFrameTrackingObservations == 0)
  }

  @Test func malformedTrackingExtentStillFailsRatherThanBecomingMissing() throws {
    let bitmap = try image()
    let observation = VNDetectedObjectObservation(
      boundingBox: CGRect(x: 0.2, y: 0.2, width: 0, height: 0.3))
    let request = VNTrackObjectRequest(detectedObjectObservation: observation)
    let sequence = FaceTrackingSequence()
    #expect(throws: ProAppsError.self) {
      try sequence.advance(request, observation: observation, in: bitmap)
    }
    #expect(!request.isLastFrame)
    #expect(sequence.outOfFrameTrackingObservations == 0)
  }

  @Test func offImageDetectionsKeepUnknownGeometryAndExplicitPerFrameRejections() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures")
    )
    let probe = FaceTrackProbe(makeSequence: {
      FaceTrackingSequence(detectFaces: { _ in
        [VNFaceObservation(boundingBox: CGRect(x: 0.8, y: 0.2, width: 0.3, height: 0.3))]
      })
    })
    let frames = try await probe.track(
      path: source.path, plan: .init(startSeconds: 0, durationSeconds: 1))
    #expect(frames.count == 30)
    #expect(frames.allSatisfy { $0.faces.isEmpty && $0.method == .missing })
    #expect(frames.allSatisfy { $0.outOfFrameTrackingObservations == 0 })
    let rejected = frames.filter { $0.outOfFrameDetectionObservations > 0 }
    #expect(rejected.map(\.presentationValue) == [0, 15000])
    #expect(
      rejected.allSatisfy { $0.detectionAttempted && $0.outOfFrameDetectionObservations == 1 })
  }

  @Test func invalidGeometryDiagnosticsRetainMeasuredValuesAndSourceClock() async throws {
    let bitmap = try image()
    let invalid = VNFaceObservation(boundingBox: CGRect(x: 2, y: 0, width: 1, height: 1))
    #expect(
      throws: ProAppsError.invalid(
        "Vision face observation exceeds the decoded image; method=tracking "
          + "normalizedBox=(2.0,0.0,1.0,1.0) confidence=1.0")
    ) {
      try FaceTrackingSequence.measure(invalid, in: bitmap, method: .tracking)
    }
    let source = try #require(
      Bundle.module.url(forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures")
    )
    let probe = FaceTrackProbe(makeSequence: {
      FaceTrackingSequence(detectFaces: { _ in
        [VNFaceObservation(boundingBox: CGRect(x: 2, y: 0, width: 0, height: 1))]
      })
    })
    await #expect(
      throws: ProAppsError.invalid(
        "Vision face observation exceeds the decoded image; method=detection "
          + "normalizedBox=(2.0,0.0,0.0,1.0) confidence=1.0; sourcePTS=0/30000")
    ) {
      try await probe.track(path: source.path, plan: .init(startSeconds: 0, durationSeconds: 1))
    }
  }

  @Test func retainsEveryDecodedClockAndMissingFaceWithoutChangingTheSource() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures")
    )
    let before = try Data(contentsOf: source)
    let frames = try await FaceTrackProbe().track(
      path: source.path, plan: .init(startSeconds: 0, durationSeconds: 1))
    #expect(frames.count == 30)
    #expect(frames.allSatisfy { $0.method == .missing && $0.faces.isEmpty })
    #expect(frames.filter(\.detectionAttempted).count == 2)
    #expect(frames.allSatisfy { $0.outOfFrameTrackingObservations == 0 })
    #expect(frames.allSatisfy { $0.width == 16 && $0.height == 16 })
    #expect(
      frames.map { Double($0.presentationValue) / Double($0.presentationTimescale) }
        == (0..<30).map { Double($0) / 30 })
    #expect(try Data(contentsOf: source) == before)
    let encoded = try JSONEncoder().encode(frames)
    #expect(try JSONDecoder().decode([FaceTrackingFrame].self, from: encoded).count == 30)
    await #expect(throws: ProAppsError.self) {
      try await FaceTrackProbe().track(
        path: source.path,
        plan: .init(startSeconds: 0, durationSeconds: 1, maximumFrames: 1))
    }
    await #expect(throws: CancellationError.self) {
      try await withThrowingTaskGroup(of: Void.self) { group in
        group.cancelAll()
        group.addTask {
          _ = try await FaceTrackProbe().track(
            path: source.path, plan: .init(startSeconds: 0, durationSeconds: 1))
        }
        try await group.waitForAll()
      }
    }
  }

  @Test(arguments: [
    NativeOperationFailure.Operation.trackingSeed, .trackingAdvance,
  ])
  func trackingFailuresDistinguishSeedFromAdvance(_ operation: NativeOperationFailure.Operation)
    throws
  {
    let bitmap = try image()
    let observation = VNFaceObservation(
      boundingBox: CGRect(x: 0.125, y: 0.25, width: 0.25, height: 0.5))
    var calls = 0
    let sequence = FaceTrackingSequence(
      detectFaces: { _ in [observation] },
      performTracking: { handler, requests, image in
        calls += 1
        if operation == .trackingSeed || calls == 2 {
          throw NSError(domain: VNErrorDomain, code: 9)
        }
        try handler.perform(requests, on: image)
      })
    do {
      _ = try sequence.detect(in: bitmap)
      _ = try sequence.track(in: bitmap)
      Issue.record("Expected injected framework failure")
    } catch let failure as NativeOperationFailure {
      #expect(failure.operation == operation && failure.code == 9)
      #expect(failure.domain == VNErrorDomain && failure.sourceTime == nil)
    }
  }

  @Test func exhaustedTrackingDoesNotSubmitTerminalRequestsAgainAndRequiresDetection() throws {
    let bitmap = try image()
    let blank = try image(checkerboard: false)
    let observation = VNFaceObservation(
      boundingBox: CGRect(x: 0.125, y: 0.25, width: 0.25, height: 0.5))
    var calls = 0
    let sequence = FaceTrackingSequence(
      detectFaces: { _ in [observation] },
      performTracking: { handler, requests, image in
        let hasActiveRequest = requests.contains { !$0.isLastFrame }
        #expect(hasActiveRequest)
        calls += 1
        try handler.perform(requests, on: image)
      })
    #expect(try sequence.detect(in: bitmap).count == 1)
    #expect(calls == 1)
    #expect(try sequence.track(in: blank).isEmpty)
    #expect(calls == 2)
    #expect(try sequence.track(in: blank).isEmpty)
    #expect(try sequence.track(in: bitmap).isEmpty)
    #expect(calls == 2)
    #expect(try sequence.detect(in: bitmap).count == 1)
    #expect(try sequence.track(in: bitmap).count == 1)
    #expect(calls == 4)
  }

  @Test func detectionFailureRetainsItsActualSourceClock() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures")
    )
    let probe = FaceTrackProbe(makeSequence: {
      FaceTrackingSequence(detectFaces: { _ in
        throw NSError(domain: VNErrorDomain, code: 9)
      })
    })
    do {
      _ = try await probe.track(
        path: source.path, plan: .init(startSeconds: 0.5, durationSeconds: 0.5))
      Issue.record("Expected injected detection failure")
    } catch let failure as NativeOperationFailure {
      #expect(failure.operation == .faceDetection && failure.code == 9)
      #expect(failure.sourceTime?.value == 15000 && failure.sourceTime?.timescale == 30000)
    }
  }

  @Test func selectsAnExplicitSourceClockWindow() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "quadrants-30", withExtension: "mp4", subdirectory: "Fixtures")
    )
    let frames = try await FaceTrackProbe().track(
      path: source.path, plan: .init(startSeconds: 0.5, durationSeconds: 0.5))
    #expect(
      frames.map { Double($0.presentationValue) / Double($0.presentationTimescale) }
        == (15..<30).map { Double($0) / 30 })
  }

  @Test func syntheticDetectionFeedsRealSequentialTrackingAndPeriodicRedetection() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    // Tracking needs texture, not editor/SwiftUI/LaunchServices integration.
    let fixture = try #require(
      Bundle.module.url(
        forResource: "tracking-object-30", withExtension: "mp4", subdirectory: "Fixtures"))
    let source = root.appendingPathComponent("tracking.mp4")
    try FileManager.default.copyItem(at: fixture, to: source)
    let decoded = try await MediaProbe().verifyShortVideo(path: source.path)
    try #require(decoded.decodedFrames == 30 && decoded.media.durationSeconds == 1)
    let asset = AVURLAsset(url: source)
    let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
    // Obtain the expected profile from the decoded surface, independently of
    // the bitmap renderer under test. CGColorSpace is SDK-declared Sendable.
    let sourceSpace = try TrackingFrameDecoder.withReader(
      asset: asset, track: track,
      timeRange: CMTimeRange(start: .zero, duration: CMTime(value: 1, timescale: 30))
    ) { _, output in
      let sample = try #require(output.copyNextSampleBuffer())
      let pixels = try #require(CMSampleBufferGetImageBuffer(sample))
      return try #require(CVImageBufferGetColorSpace(pixels)?.takeUnretainedValue())
    }
    let probe = FaceTrackProbe(makeSequence: {
      FaceTrackingSequence(
        detectFaces: { image in
          let actualSpace = try #require(image.colorSpace)
          #expect(CFEqual(sourceSpace, actualSpace))
          return [VNFaceObservation(boundingBox: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5))]
        },
        performTracking: { handler, requests, image in
          let actualSpace = try #require(image.colorSpace)
          #expect(CFEqual(sourceSpace, actualSpace))
          try handler.perform(requests, on: image)
        })
    })
    let frames = try await probe.track(
      path: source.path, plan: .init(startSeconds: 0, durationSeconds: 1))
    #expect(frames.count == 30)
    #expect(frames.first?.method == .detection)
    #expect(frames.contains { $0.method == .tracking })
    #expect(frames.filter { $0.method == .detection }.count >= 2)
    #expect(frames.allSatisfy { $0.faces.count == 1 && $0.width == 256 && $0.height == 256 })
    #expect(frames.allSatisfy { $0.faces.first.map { $0.confidence > 0.3 } == true })
  }

  @Test func refusesAudioOnlyAndWindowsWithoutASourceFrameBoundary() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let audio = root.appendingPathComponent("audio.wav")
    try CueSound(durationSeconds: 1, onsetSeconds: [], gain: 0).wave().write(to: audio)
    await #expect(throws: ProAppsError.self) {
      try await FaceTrackProbe().track(
        path: audio.path, plan: .init(startSeconds: 0, durationSeconds: 1))
    }
    let source = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    await #expect(throws: ProAppsError.self) {
      try await FaceTrackProbe().track(
        path: source.path, plan: .init(startSeconds: 0.5, durationSeconds: 0.25))
    }
  }

  @Test func preservesSourceClockWindowsWithoutRoundingToTheEditGrid() throws {
    let range = try FaceTrackingPlan(
      startSeconds: 1014.0 / 30000, durationSeconds: 28041.0 / 30000,
      sourceTimescale: 30000
    ).validate(sourceDuration: 2)
    #expect(range.start.value == 1014)
    #expect(range.start.timescale == 30000)
    #expect(range.duration.value == 28041)
    #expect(range.duration.timescale == 30000)
    let legacy = try FaceTrackingPlan(startSeconds: 1014.0 / 30000, durationSeconds: 1)
      .validate(sourceDuration: 2)
    #expect(legacy.start.value == 2028)
    #expect(legacy.start.timescale == 60000)
    #expect(throws: ProAppsError.self) {
      try FaceTrackingPlan(startSeconds: 1.0 / 90001, durationSeconds: 1)
        .validate(sourceDuration: 2)
    }
    #expect(throws: ProAppsError.self) {
      try FaceTrackingPlan(startSeconds: 0, durationSeconds: 1.0 / 90001, sourceTimescale: 30000)
        .validate(sourceDuration: 2)
    }
  }

  @Test(arguments: [Int32(0), -1, 1_000_000_001])
  func refusesInvalidSourceTimescales(_ timescale: Int32) {
    #expect(throws: ProAppsError.self) {
      try FaceTrackingPlan(startSeconds: 0, durationSeconds: 1, sourceTimescale: timescale)
        .validate(sourceDuration: 2)
    }
  }

  @Test func validatesBudgetsBeforeDecode() throws {
    try FaceTrackingPlan(startSeconds: 1, durationSeconds: 120).validate(sourceDuration: 121)
    #expect(throws: ProAppsError.self) {
      try FaceTrackingPlan(startSeconds: 1, durationSeconds: 1).validate(sourceDuration: 1)
    }
    #expect(throws: ProAppsError.self) {
      try FaceTrackingPlan(startSeconds: 0, durationSeconds: 121).validate(sourceDuration: 122)
    }
    #expect(throws: ProAppsError.self) {
      try FaceTrackingPlan(startSeconds: 0, durationSeconds: 1, maximumFrames: 7201)
        .validate(sourceDuration: 2)
    }
    #expect(throws: ProAppsError.self) {
      try FaceTrackingPlan(startSeconds: 0, durationSeconds: 1, redetectIntervalSeconds: 0)
        .validate(sourceDuration: 2)
    }
    #expect(throws: ProAppsError.self) {
      try FaceTrackingPlan(startSeconds: .nan, durationSeconds: 1).validate(sourceDuration: 2)
    }
  }
}
