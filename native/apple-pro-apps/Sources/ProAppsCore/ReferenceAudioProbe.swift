import Dispatch
import Foundation

/// Read-only analysis of explicitly prepared mono PCM16 WAVs, at equal rates.
/// This worker performs bounded file I/O and DSP outside the cooperative pool.
public actor ReferenceAudioProbe {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.reference-audio")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }

  public init() {}

  public func analyze(
    sourcePath: String, referencePath: String, search: ReferenceAudioSearch
  ) throws -> ReferenceAudioMatch {
    try Task.checkCancellation()
    let source = try PCMMeasurement.decodedSamples(
      Files.read(Files.existing(sourcePath, extensions: ["wav"])))
    let reference = try PCMMeasurement.decodedSamples(
      Files.read(Files.existing(referencePath, extensions: ["wav"])))
    guard source.sampleRate == reference.sampleRate else {
      throw ProAppsError.invalid(
        "Reference and source sample rates must match; no implicit resampling")
    }
    return try ReferenceAudio.fit(
      source: source.samples, reference: reference.samples, search: search)
  }
}
