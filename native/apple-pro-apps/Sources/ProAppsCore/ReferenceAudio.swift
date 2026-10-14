import Foundation

/// Explicit bounded sample-domain search, shared sample rate required by caller.
/// This is reference cancellation, not arbitrary music/voice source separation.
public struct ReferenceAudioSearch: Codable, Sendable {
  public let referenceStartSample: Int
  public let searchRadiusSamples: Int
  public let minimumCorrelation: Double

  public init(
    referenceStartSample: Int, searchRadiusSamples: Int, minimumCorrelation: Double
  ) {
    self.referenceStartSample = referenceStartSample
    self.searchRadiusSamples = searchRadiusSamples
    self.minimumCorrelation = minimumCorrelation
  }
}

public struct ReferenceAudioMatch: Codable, Sendable {
  public let referenceStartSample: Int
  public let gain: Double
  public let correlation: Double
  public let sourceRMS: Double
  public let residualRMS: Double
}

/// Pure bounded channel-level fitting. Run on the native audio worker, not UI.
/// Samples must be finite normalized linear PCM. No clipping or resampling occurs.
public enum ReferenceAudio {
  public static let maximumSamples = 1_000_000
  public static let maximumProducts = 16_000_000

  public static func fit(
    source: [Double], reference: [Double], search: ReferenceAudioSearch
  ) throws -> ReferenceAudioMatch {
    try Task.checkCancellation()
    let offsets = try validate(source: source, reference: reference, search: search)
    let sourceEnergy = source.reduce(0) { $0 + $1 * $1 }
    guard sourceEnergy > 1e-12 else {
      throw ProAppsError.invalid("Reference fitting requires non-silent source audio")
    }
    var best: ReferenceAudioMatch?
    for offset in offsets {
      try Task.checkCancellation()
      let candidate = match(
        source: source, reference: reference, offset: offset,
        sourceEnergy: sourceEnergy)
      if let candidate, abs(candidate.correlation) > abs(best?.correlation ?? 0) {
        best = candidate
      }
    }
    guard let best, abs(best.correlation) >= search.minimumCorrelation else {
      throw ProAppsError.invalid("Reference does not meet the requested correlation threshold")
    }
    guard abs(best.gain) <= 4 else {
      throw ProAppsError.invalid("Reference fit requires excessive gain")
    }
    return best
  }

  public static func subtract(
    source: [Double], reference: [Double], search: ReferenceAudioSearch
  ) throws -> (samples: [Double], match: ReferenceAudioMatch) {
    let match = try fit(source: source, reference: reference, search: search)
    var samples: [Double] = []
    samples.reserveCapacity(source.count)
    for index in source.indices {
      if index.isMultiple(of: 4096) { try Task.checkCancellation() }
      let residual = source[index] - match.gain * reference[index + match.referenceStartSample]
      guard residual.isFinite, abs(residual) <= 1 else {
        throw ProAppsError.invalid("Reference subtraction would exceed normalized PCM headroom")
      }
      samples.append(residual)
    }
    return (samples, match)
  }

  private static func validate(
    source: [Double], reference: [Double], search: ReferenceAudioSearch
  ) throws -> ClosedRange<Int> {
    guard source.count >= 4, source.count <= maximumSamples,
      reference.count >= source.count, reference.count <= maximumSamples,
      search.searchRadiusSamples >= 0, search.searchRadiusSamples <= maximumSamples,
      search.referenceStartSample >= search.searchRadiusSamples,
      search.referenceStartSample <= reference.count - source.count,
      search.searchRadiusSamples <= reference.count - source.count - search.referenceStartSample,
      search.minimumCorrelation.isFinite, (0.05...1).contains(search.minimumCorrelation)
    else { throw ProAppsError.invalid("Invalid reference search bounds or correlation threshold") }
    let candidates = 2 * search.searchRadiusSamples + 1
    guard source.count <= maximumProducts / candidates else {
      throw ProAppsError.invalid("Reference search operation budget exceeded; narrow the search")
    }
    guard source.allSatisfy({ $0.isFinite && abs($0) <= 1 }),
      reference.allSatisfy({ $0.isFinite && abs($0) <= 1 })
    else { throw ProAppsError.invalid("Reference fitting requires finite normalized PCM samples") }
    let lower = search.referenceStartSample - search.searchRadiusSamples
    let upper = search.referenceStartSample + search.searchRadiusSamples
    return lower...upper
  }

  private static func match(
    source: [Double], reference: [Double], offset: Int, sourceEnergy: Double
  ) -> ReferenceAudioMatch? {
    var dot = 0.0
    var energy = 0.0
    for index in source.indices {
      let sample = reference[index + offset]
      dot += source[index] * sample
      energy += sample * sample
    }
    guard energy > 1e-12 else { return nil }
    let gain = dot / energy
    let correlation = min(1, max(-1, dot / sqrt(sourceEnergy * energy)))
    let residualEnergy = max(0, sourceEnergy - dot * dot / energy)
    return ReferenceAudioMatch(
      referenceStartSample: offset, gain: gain, correlation: correlation,
      sourceRMS: sqrt(sourceEnergy / Double(source.count)),
      residualRMS: sqrt(residualEnergy / Double(source.count)))
  }
}
