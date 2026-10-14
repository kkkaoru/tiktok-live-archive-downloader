import Foundation
import MCP
import ProAppsCore

extension ToolSpec {
  static let referenceTime = object(
    ["value": integer(0, Int.max), "timescale": integer(1, 1_000_000_000)],
    ["value", "timescale"])
  static let referenceRequest = object(
    [
      "rootDirectory": string(), "outputPath": string(),
      "timescale": integer(1, 1_000_000_000),
      "segments": array(
        object(
          [
            "sourcePath": string(), "sourceStart": referenceTime, "duration": referenceTime,
            "outputStart": referenceTime,
            "layer": integer(0, ReferenceMovieRequest.maximumLayers - 1),
          ], ["sourcePath", "sourceStart", "duration", "outputStart", "layer"]),
        maximum: ReferenceMovieRequest.maximumSegments),
    ], ["rootDirectory", "outputPath", "timescale", "segments"])
}

extension NativeService {
  func referenceLocalFile(_ path: String) async throws -> String {
    try Task.checkCancellation()
    let data = try Files.read(Files.existing(path, extensions: ["json"]))
    let value = try JSONDecoder().decode(Value.self, from: data)
    try validate(value, schema: ToolSpec.referenceRequest)
    let request = try decode(ReferenceMovieRequest.self, value)
    let result = try await ReferenceMovieWriter().write(request)
    return String(decoding: try JSONEncoder().encode(result), as: UTF8.self)
  }
}
