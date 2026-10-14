import Dispatch
import Foundation

public struct DemucsRequest: Codable, Sendable {
  public let leftPath: String
  public let rightPath: String
  public let compiledModelPath: String
  public let outputDirectory: String
  public let outputName: String
}

public struct DemucsResult: Codable, Sendable {
  public let outputPath: String
  public let projectPath: String
  public let sampleRate: Int
  public let sampleCount: Int
  public let peak: Float
  public let humanReviewed: Bool
}

/// Bounded segment separation. Long recordings use explicit overlapping segments;
/// this operation neither downloads models nor modifies/overwrites any input.
public actor DemucsSeparator {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.demucs-files")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }
  private let writeArtifact: @Sendable (Data, String) throws -> URL

  public init(
    writeArtifact: @escaping @Sendable (Data, String) throws -> URL = {
      try Files.writeNew($0, to: $1, extensions: ["json", "wav"])
    }
  ) {
    self.writeArtifact = writeArtifact
  }

  public func separate(_ request: DemucsRequest) async throws -> DemucsResult {
    try Task.checkCancellation()
    guard URL(fileURLWithPath: request.outputName).pathExtension.lowercased() == "wav" else {
      throw ProAppsError.invalid("Separated vocals require a new WAV output")
    }
    let left = try PCMMeasurement.decodedSamples(
      Files.read(Files.existing(request.leftPath, extensions: ["wav"])))
    let right = try PCMMeasurement.decodedSamples(
      Files.read(Files.existing(request.rightPath, extensions: ["wav"])))
    guard left.sampleRate == 44100, right.sampleRate == 44100,
      left.samples.count == right.samples.count, left.samples.count <= DemucsSpectrum.sampleCount
    else {
      throw ProAppsError.invalid(
        "Demucs needs equal-length mono PCM16 WAV channels at 44100 Hz, at most 7.8 seconds")
    }
    let count = left.samples.count
    let model = DemucsModel()
    try await model.load(compiledModelPath: request.compiledModelPath)
    let padding = [Float](repeating: 0, count: DemucsSpectrum.sampleCount - count)
    let vocals = try await model.vocals(
      left: left.samples.map(Float.init) + padding,
      right: right.samples.map(Float.init) + padding)
    try Task.checkCancellation()
    let l = Array(vocals.left.prefix(count))
    let r = Array(vocals.right.prefix(count))
    let wave = try FloatWave.stereo(left: l, right: r, sampleRate: 44100)
    let output = try Files.reserveOutput(
      directory: request.outputDirectory, name: request.outputName, kind: .edit)
    do {
      let project = try writeArtifact(
        JSONEncoder().encode(request),
        output.deletingLastPathComponent().appendingPathComponent("separation-request.json").path)
      try Task.checkCancellation()
      _ = try writeArtifact(wave, output.path)
      let peak = zip(l, r).reduce(Float.zero) { max($0, abs($1.0), abs($1.1)) }
      return DemucsResult(
        outputPath: output.path, projectPath: project.path, sampleRate: 44100,
        sampleCount: count, peak: peak, humanReviewed: false)
    } catch {
      Cleanup.perform { try FileManager.default.removeItem(at: output.deletingLastPathComponent()) }
      throw error
    }
  }
}
