import CoreML
import Foundation
import Testing

@testable import ProAppsCore

struct DemucsModelTests {
  @Test func typedFeaturesAndTensorBounds() throws {
    let tensor = try DemucsModel.tensor([0, 0.5], shape: [1, 2])
    let features = DemucsFeatures(spectral: tensor, waveform: tensor)
    #expect(features.featureNames == ["spectral_magnitude", "audio_waveform"])
    #expect(features.featureValue(for: "unknown") == nil)
    #expect(features.featureValue(for: "audio_waveform")?.multiArrayValue?.count == 2)
    #expect(
      try DemucsModel.output(features, name: "spectral_magnitude", shape: [1, 2]) == [0, 0.5])
    #expect(throws: ProAppsError.self) { try DemucsModel.tensor([.nan], shape: [1]) }
    #expect(throws: ProAppsError.self) { try DemucsModel.tensor([70_000], shape: [1]) }
    #expect(throws: ProAppsError.self) { try DemucsModel.tensor([0], shape: [2]) }
    #expect(throws: ProAppsError.self) {
      try DemucsModel.output(features, name: "missing", shape: [1])
    }
    #expect(throws: ProAppsError.self) {
      try DemucsModel.output(features, name: "spectral_magnitude", shape: [2])
    }
  }

  @Test func missingModelAndInvalidPathAreExplicitErrors() async throws {
    let model = DemucsModel()
    await #expect(throws: ProAppsError.self) { try await model.vocals(left: [], right: []) }
    await #expect(throws: ProAppsError.self) {
      try await model.load(compiledModelPath: "relative.model")
    }
  }

  /// Explicit native integration fixture, never an implicit model download.
  @Test func approvedCompiledModelRunsWithFiniteBoundedSilenceOutput() async throws {
    let path = try #require(
      ProcessInfo.processInfo.environment["DEMUCS_TEST_MODEL"],
      "Set DEMUCS_TEST_MODEL to the approved compiled model; no download is performed by tests")
    let model = DemucsModel()
    try await model.load(compiledModelPath: path)
    let silence = [Float](repeating: 0, count: 343_980)
    let vocals = try await model.vocals(left: silence, right: silence)
    #expect(vocals.left.count == 343_980)
    #expect(vocals.right.count == 343_980)
    #expect(vocals.left.allSatisfy { $0.isFinite && abs($0) < 0.01 })
    #expect(vocals.right.allSatisfy { $0.isFinite && abs($0) < 0.01 })
  }
}
