import CoreML
import Dispatch
import Foundation

/// Typed Core ML provider; never passes an unchecked dictionary to the model.
final class DemucsFeatures: MLFeatureProvider {
  let spectral: MLMultiArray
  let waveform: MLMultiArray
  var featureNames: Set<String> { ["spectral_magnitude", "audio_waveform"] }

  init(spectral: MLMultiArray, waveform: MLMultiArray) {
    self.spectral = spectral
    self.waveform = waveform
  }

  func featureValue(for featureName: String) -> MLFeatureValue? {
    switch featureName {
    case "spectral_magnitude": MLFeatureValue(multiArray: spectral)
    case "audio_waveform": MLFeatureValue(multiArray: waveform)
    default: nil
    }
  }
}

public struct DemucsVocals: Sendable {
  public let left: [Float]
  public let right: [Float]
}

/// A single approved model and its predictions are confined to one serial worker.
/// Uses public shaped-array conversions; no handwritten raw tensor/audio pointers.
public actor DemucsModel {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.demucs")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }
  private var model: MLModel?

  public init() {}

  public func load(compiledModelPath: String) throws {
    try Task.checkCancellation()
    let url = URL(fileURLWithPath: compiledModelPath)
    guard url.pathExtension == "mlmodelc", compiledModelPath.hasPrefix("/"),
      try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey]).isDirectory == true,
      try url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true
    else { throw ProAppsError.invalid("Expected an existing local compiled Core ML directory") }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuOnly
    let loaded = try MLModel(contentsOf: url, configuration: configuration)
    try Self.validateDescription(loaded.modelDescription)
    try Task.checkCancellation()
    model = loaded
  }

  static func validateDescription(_ description: MLModelDescription) throws {
    let shapes = ["spectral_magnitude": [1, 4, 2048, 336], "audio_waveform": [1, 2, 343_980]]
    guard Set(description.inputDescriptionsByName.keys) == Set(shapes.keys) else {
      throw ProAppsError.invalid("Unsupported Demucs model input names")
    }
    for (name, shape) in shapes {
      guard let constraint = description.inputDescriptionsByName[name]?.multiArrayConstraint,
        constraint.shape.map(\.intValue) == shape, constraint.dataType == .float16
      else { throw ProAppsError.invalid("Unsupported Demucs model input shape or precision") }
    }
  }

  static func tensor(_ samples: [Float], shape: [Int]) throws -> MLMultiArray {
    guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 && $0 <= 1_000_000 }),
      shape.reduce(1.0, { $0 * Double($1) }) == Double(samples.count),
      samples.count <= 12_000_000,
      samples.allSatisfy({ $0.isFinite && abs($0) <= Float(Float16.greatestFiniteMagnitude) })
    else { throw ProAppsError.invalid("Invalid finite Float16 tensor dimensions or samples") }
    return MLMultiArray(MLShapedArray<Float16>(scalars: samples.map(Float16.init), shape: shape))
  }

  static func output(_ provider: any MLFeatureProvider, name: String, shape: [Int]) throws
    -> [Float]
  {
    guard let array = provider.featureValue(for: name)?.multiArrayValue,
      array.shape.map(\.intValue) == shape, array.dataType == .float16
    else { throw ProAppsError.invalid("Demucs output tensor shape or precision mismatch") }
    let values = MLShapedArray<Float16>(converting: array).scalars.map(Float.init)
    guard values.allSatisfy(\.isFinite) else {
      throw ProAppsError.invalid("Demucs inference produced nonfinite samples")
    }
    return values
  }

  public func vocals(left: [Float], right: [Float]) throws -> DemucsVocals {
    try Task.checkCancellation()
    guard let model else { throw ProAppsError.invalid("Load a validated Demucs model first") }
    let l = try DemucsSpectrum.forward(left)
    let r = try DemucsSpectrum.forward(right)
    let features = DemucsFeatures(
      spectral: try Self.tensor(
        l.real + l.imaginary + r.real + r.imaginary, shape: [1, 4, 2048, 336]),
      waveform: try Self.tensor(left + right, shape: [1, 2, 343_980]))
    let prediction = try model.prediction(from: features)
    try Task.checkCancellation()
    let frequency = try Self.output(prediction, name: "freq_output", shape: [1, 16, 2048, 336])
    let time = try Self.output(prediction, name: "time_output", shape: [1, 8, 343_980])
    return DemucsVocals(
      left: try Self.restore(frequency: frequency, time: time, channel: 0),
      right: try Self.restore(frequency: frequency, time: time, channel: 1))
  }

  private static func restore(frequency: [Float], time: [Float], channel: Int) throws -> [Float] {
    // Upstream source order is drums, bass, other, vocals. Vocals is stem 3.
    let plane = DemucsSpectrum.planeCount
    let realStart = (12 + 2 * channel) * plane
    let imaginaryStart = realStart + plane
    let spectral = try DemucsSpectrum.inverse(
      real: Array(frequency[realStart..<(realStart + plane)]),
      imaginary: Array(frequency[imaginaryStart..<(imaginaryStart + plane)]))
    let timeStart = (6 + channel) * DemucsSpectrum.sampleCount
    let restored = spectral.enumerated().map { $0.element + time[timeStart + $0.offset] }
    guard restored.allSatisfy(\.isFinite) else {
      throw ProAppsError.invalid("Demucs restoration overflowed")
    }
    return restored
  }
}
