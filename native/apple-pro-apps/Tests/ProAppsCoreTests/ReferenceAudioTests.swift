import Foundation
import Testing

@testable import ProAppsCore

struct ReferenceAudioTests {
  @Test func alignedReferencePreservesOrthogonalVoice() throws {
    let result = try ReferenceAudio.subtract(
      source: [0.75, 0.25, -0.25, -0.75], reference: [0.5, 0.5, -0.5, -0.5],
      search: .init(referenceStartSample: 0, searchRadiusSamples: 0, minimumCorrelation: 0.8))
    #expect(result.samples == [0.25, -0.25, 0.25, -0.25])
    #expect(result.match.gain == 1)
    #expect(result.match.referenceStartSample == 0)
    #expect(result.match.residualRMS == 0.25)
  }

  @Test func searchesExplicitOffsetAndAllowsInvertedReference() throws {
    let result = try ReferenceAudio.subtract(
      source: [-0.5, -0.25, 0.5, 0.25], reference: [0, 0, 0.5, 0.25, -0.5, -0.25, 0, 0],
      search: .init(referenceStartSample: 2, searchRadiusSamples: 2, minimumCorrelation: 0.99))
    #expect(result.match.referenceStartSample == 2)
    #expect(result.match.gain == -1)
    #expect(result.match.correlation == -1)
    #expect(result.samples == [0, 0, 0, 0])
  }

  @Test(arguments: [
    [0.0, 0, 0, 0], [0.5, -0.5, 0.5, -0.5], [0.001, 0.001, -0.001, -0.001],
  ])
  func rejectsSilentUnmatchedOrExcessiveGainReference(_ reference: [Double]) {
    #expect(throws: ProAppsError.self) {
      try ReferenceAudio.fit(
        source: [0.5, 0.5, -0.5, -0.5], reference: reference,
        search: .init(referenceStartSample: 0, searchRadiusSamples: 0, minimumCorrelation: 0.5))
    }
  }

  @Test(arguments: [
    ReferenceAudioSearch(referenceStartSample: -1, searchRadiusSamples: 0, minimumCorrelation: 0.5),
    ReferenceAudioSearch(referenceStartSample: 0, searchRadiusSamples: -1, minimumCorrelation: 0.5),
    ReferenceAudioSearch(referenceStartSample: 0, searchRadiusSamples: 1, minimumCorrelation: 0.5),
    ReferenceAudioSearch(referenceStartSample: 1, searchRadiusSamples: 0, minimumCorrelation: 0.5),
    ReferenceAudioSearch(referenceStartSample: 0, searchRadiusSamples: 0, minimumCorrelation: 0),
    ReferenceAudioSearch(referenceStartSample: 0, searchRadiusSamples: 0, minimumCorrelation: .nan),
  ])
  func rejectsInvalidSearch(_ search: ReferenceAudioSearch) {
    #expect(throws: ProAppsError.self) {
      try ReferenceAudio.fit(
        source: [0.5, 0.5, -0.5, -0.5],
        reference: [0.5, 0.5, -0.5, -0.5], search: search)
    }
  }

  @Test(arguments: [
    [Double.nan, 0, 0, 0], [Double.infinity, 0, 0, 0], [1.1, 0, 0, 0], [0, 0, 0, 0], [0, 0],
  ])
  func rejectsInvalidOrSilentSource(_ source: [Double]) {
    #expect(throws: ProAppsError.self) {
      try ReferenceAudio.fit(
        source: source, reference: [0.5, 0.5, -0.5, -0.5],
        search: .init(referenceStartSample: 0, searchRadiusSamples: 0, minimumCorrelation: 0.5))
    }
  }

  @Test func rejectsInvalidReferenceAndExcessiveWork() {
    #expect(throws: ProAppsError.self) {
      try ReferenceAudio.fit(
        source: [0.5, 0.5, -0.5, -0.5], reference: [.nan, 0, 0, 0],
        search: .init(referenceStartSample: 0, searchRadiusSamples: 0, minimumCorrelation: 0.5))
    }
    #expect(throws: ProAppsError.self) {
      try ReferenceAudio.fit(
        source: Array(repeating: 0.5, count: 10_000),
        reference: Array(repeating: 0.5, count: 14_000),
        search: .init(
          referenceStartSample: 2000, searchRadiusSamples: 2000, minimumCorrelation: 0.5))
    }
  }

  @Test func refusesOutputClippingInsteadOfNormalizingVoice() {
    #expect(throws: ProAppsError.self) {
      try ReferenceAudio.subtract(
        source: [1, 1, 1, -1], reference: [0.3, 0.3, 0.3, 0.15],
        search: .init(referenceStartSample: 0, searchRadiusSamples: 0, minimumCorrelation: 0.05))
    }
  }

  @Test func cancellationIsPropagated() async {
    let task = Task {
      while !Task.isCancelled { await Task.yield() }
      return try ReferenceAudio.fit(
        source: [0.5, 0.5, -0.5, -0.5],
        reference: [0.5, 0.5, -0.5, -0.5],
        search: .init(referenceStartSample: 0, searchRadiusSamples: 0, minimumCorrelation: 0.5))
    }
    task.cancel()
    await #expect(throws: CancellationError.self) { try await task.value }
  }
}
