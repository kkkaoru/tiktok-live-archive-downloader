import CoreMedia
import Foundation
import Testing

@testable import ProAppsCore

struct HeadTopProbeTests {
  @Test func batchedFacesUseTheirOwnExactDecodedFramesAndGeometry() async throws {
    let source = try ForegroundMeasurementTests.fixture()
    let original = try Data(contentsOf: source)
    let samples: [ForegroundTime] = (0..<8).map { .init(value: Int64($0), timescale: 30) }
    let faces = samples.enumerated().map { index, sample in
      HeadTopFaceRequest(
        presentation: .init(value: sample.value * 3000, timescale: 90000),
        x: 2, y: 2 + index, width: 2, height: 2, minimumFaceCoverage: 1)
    }
    var operations = ForegroundOperations()
    operations.observe = { _, _, _ in
      Issue.record("A batched face sample reached unassociated foreground inference")
      return nil
    }
    operations.head = { _, image, _, face in
      #expect(image.width == 16 && image.height == 16)
      var pixels = Data(repeating: 0, count: 256)
      for y in (face.y - 1)..<(face.y + face.height) {
        pixels[y * 16 + 2] = 255
        pixels[y * 16 + 3] = 255
      }
      let candidate = try HeadTopMeasurement.measure(
        masks: [.init(label: 1, pixels: pixels, bytesPerRow: 16)], width: 16, height: 16,
        face: face.bounds, minimumFaceCoverage: face.minimumFaceCoverage)
      return ForegroundHeadTopResult(observation: nil, candidate: candidate)
    }
    let result = try await FrameProbe().measureForeground(
      path: source.path, samples: samples, operations: operations, headFaces: faces)
    #expect(result.count == 8)
    #expect(result.map(\.requested) == samples)
    #expect(
      try result.map { try $0.actual.time() } == [
        CMTime(value: 0, timescale: 30), CMTime(value: 1, timescale: 30),
        CMTime(value: 2, timescale: 30), CMTime(value: 3, timescale: 30),
        CMTime(value: 4, timescale: 30), CMTime(value: 5, timescale: 30),
        CMTime(value: 6, timescale: 30), CMTime(value: 7, timescale: 30),
      ])
    #expect(result.map { $0.headTop?.topY } == [1, 2, 3, 4, 5, 6, 7, 8])
    #expect(result.allSatisfy { $0.headTop?.captionPlacementApproved == false })
    #expect(try Data(contentsOf: source) == original)
  }

  @Test(arguments: ["count", "order", "invalidClock", "both", "empty"])
  func batchedFaceContractFailsBeforeSourceRead(condition: String) async {
    let first = HeadTopFaceRequest(
      presentation: .init(value: 0, timescale: 30), x: 2, y: 2, width: 2, height: 2,
      minimumFaceCoverage: 1)
    let second = HeadTopFaceRequest(
      presentation: .init(value: 1, timescale: 30), x: 2, y: 2, width: 2, height: 2,
      minimumFaceCoverage: 1)
    let faces: [HeadTopFaceRequest]
    let expected: ProAppsError
    switch condition {
    case "count":
      faces = [first]
      expected = .invalid("headFaces must match every requested sample")
    case "order":
      faces = [second, first]
      expected = .invalid("Head-top face PTS differs from its requested sample")
    case "invalidClock":
      faces = [
        first,
        .init(
          presentation: .init(value: 1, timescale: 0), x: 2, y: 2, width: 2, height: 2,
          minimumFaceCoverage: 1),
      ]
      expected = .invalid("Invalid foreground clock")
    case "empty":
      faces = []
      expected = .invalid("headFaces must match every requested sample")
    default:
      faces = [first, second]
      expected = .invalid("headFace and headFaces are mutually exclusive")
    }
    await #expect(throws: expected) {
      try await FrameProbe().measureForeground(
        path: "/unavailable-source.mp4",
        samples: [.init(value: 0, timescale: 30), .init(value: 1, timescale: 30)],
        headFace: condition == "both" ? first : nil, headFaces: faces)
    }
  }

  @Test func rasterBudgetIncludesPaddingAndProviderBacking() throws {
    let bitmap = try ForegroundMeasurementTests.gray(
      [0, 1, 255, 255, 255, 255], width: 2, height: 1, stride: 4)
    var bytes = 0
    let result = try HeadTopRasterStorage.raster(
      label: 1, bitmap: bitmap, width: 2, height: 1, retainedBytes: &bytes)
    #expect(result.pixels.count == 4)
    #expect(bytes == 6)
    bytes = ForegroundMaskStatistics.maximumBytes - 5
    #expect(throws: ProAppsError.self) {
      try HeadTopRasterStorage.raster(
        label: 1, bitmap: bitmap, width: 2, height: 1, retainedBytes: &bytes)
    }
    #expect(bytes == ForegroundMaskStatistics.maximumBytes - 5)
    #expect(throws: ProAppsError.self) {
      try HeadTopRasterStorage.raster(
        label: 1, bitmap: nil, width: 2, height: 1, retainedBytes: &bytes)
    }
    #expect(throws: ProAppsError.self) { try HeadTopRasterStorage.validateLabels([0]) }
    #expect(throws: ProAppsError.self) {
      try HeadTopRasterStorage.validateLabels(IndexSet(integersIn: 1...257))
    }
    try HeadTopRasterStorage.validateLabels([1, 5])
    try HeadTopRasterStorage.validateExtent(
      .init(x: 0, y: 0, width: 2, height: 1), width: 2, height: 1)
    #expect(throws: ProAppsError.self) {
      try HeadTopRasterStorage.validateExtent(
        .init(x: 1, y: 0, width: 2, height: 1), width: 2, height: 1)
    }
  }

  @Test(.timeLimit(.minutes(1)), arguments: ["black", "tracking-object-30"])
  func systemVisionKeepsHeadCandidateUnapproved(fixture: String) async throws {
    let source = try #require(
      Bundle.module.url(forResource: fixture, withExtension: "mp4", subdirectory: "Fixtures"))
    let result = try await FrameProbe().measureForeground(
      path: source.path, samples: [.init(value: 0, timescale: 30)],
      headFace: .init(
        presentation: .init(value: 0, timescale: 30), x: 2, y: 2,
        width: 2, height: 2, minimumFaceCoverage: 0.5))
    let candidate = try #require(result.first?.headTop)
    #expect(!candidate.semanticHeadVerified)
    #expect(!candidate.captionPlacementApproved)
  }
  @Test func exactFaceClockIsRequiredBeforeSourceRead() async throws {
    let face = HeadTopFaceRequest(
      presentation: .init(value: 1, timescale: 30), x: 2, y: 2, width: 2, height: 2,
      minimumFaceCoverage: 0.5)
    await #expect(throws: ProAppsError.self) {
      try await FrameProbe().measureForeground(
        path: "/unused.mp4", samples: [.init(value: 0, timescale: 30)], headFace: face)
    }
    await #expect(throws: ProAppsError.self) {
      try await FrameProbe().measureForeground(
        path: "/unused.mp4",
        samples: [.init(value: 0, timescale: 30), .init(value: 1, timescale: 30)], headFace: face)
    }
  }

  @Test func injectedMaskUsesSameDecodedFrameAndPreservesRationalPTS() async throws {
    let source = try ForegroundMeasurementTests.fixture()
    let original = try Data(contentsOf: source)
    let face = HeadTopFaceRequest(
      presentation: .init(value: 3000, timescale: 90000), x: 2, y: 2, width: 2, height: 2,
      minimumFaceCoverage: 0.5)
    var operations = ForegroundOperations()
    operations.observe = { _, _, _ in nil }
    operations.head = { _, image, _, observed in
      #expect(image.width == 16 && image.height == 16)
      #expect(observed.presentation == .init(value: 3000, timescale: 90000))
      var pixels = Data(repeating: 0, count: 256)
      pixels[18] = 255
      pixels[34] = 255
      pixels[35] = 255
      pixels[50] = 255
      pixels[51] = 255
      let candidate = try HeadTopMeasurement.measure(
        masks: [.init(label: 1, pixels: pixels, bytesPerRow: 16)], width: 16, height: 16,
        face: observed.bounds, minimumFaceCoverage: observed.minimumFaceCoverage)
      return ForegroundHeadTopResult(observation: nil, candidate: candidate)
    }
    let result = try await FrameProbe().measureForeground(
      path: source.path, samples: [.init(value: 1, timescale: 30)], operations: operations,
      headFace: face)
    let frame = try #require(result.first)
    #expect(frame.headTop?.topY == 1)
    #expect(frame.requested == .init(value: 1, timescale: 30))
    #expect(try frame.actual.time() == face.presentation.time())
    let encoded = String(decoding: try JSONEncoder().encode(frame), as: UTF8.self)
    #expect(encoded.contains("\"headTop\""))
    #expect(encoded.contains("\"semanticHeadVerified\":false"))
    #expect(try Data(contentsOf: source) == original)
  }

  @Test func optionalRequestPreservesLegacyDecoding() throws {
    let legacy = try JSONDecoder().decode(
      ForegroundRequest.self,
      from: Data(#"{"path":"/unused.mp4","samples":[{"value":0,"timescale":30}]}"#.utf8))
    #expect(legacy.headFace == nil)
    #expect(legacy.headFaces == nil)
    let extended = try JSONDecoder().decode(
      ForegroundRequest.self,
      from: Data(
        #"{"path":"/unused.mp4","samples":[{"value":0,"timescale":30}],"headFace":{"presentation":{"value":0,"timescale":30},"x":2,"y":2,"width":2,"height":2,"minimumFaceCoverage":0.5}}"#
          .utf8))
    #expect(extended.headFace?.bounds == .init(x: 2, y: 2, width: 2, height: 2))
    #expect(extended.headFaces == nil)
    let batched = try JSONDecoder().decode(
      ForegroundRequest.self,
      from: Data(
        #"{"path":"/unused.mp4","samples":[{"value":0,"timescale":30}],"headFaces":[{"presentation":{"value":0,"timescale":30},"x":2,"y":2,"width":2,"height":2,"minimumFaceCoverage":0.5}]}"#
          .utf8))
    #expect(batched.headFace == nil)
    #expect(batched.headFaces?.count == 1)
    #expect(batched.headFaces?.first?.bounds == .init(x: 2, y: 2, width: 2, height: 2))
  }
}
