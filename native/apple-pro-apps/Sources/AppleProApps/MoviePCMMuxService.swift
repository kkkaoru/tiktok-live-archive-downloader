import Foundation
import MCP
import ProAppsCore

extension NativeService {
  func muxPCMLocalFile(_ path: String) async throws -> String {
    try Task.checkCancellation()
    let data = try Files.read(Files.existing(path, extensions: ["json"]))
    let value = try JSONDecoder().decode(Value.self, from: data)
    let properties = ["rootDirectory", "videoPath", "audioPath", "outputDirectory", "outputName"]
    let schema = ToolSpec.object(
      Dictionary(uniqueKeysWithValues: properties.map { ($0, ToolSpec.string()) }), properties)
    try validate(value, schema: schema)
    let request = try decode(MoviePCMMuxRequest.self, value)
    return String(
      decoding: try JSONEncoder().encode(try await MoviePCMMux().write(request)), as: UTF8.self)
  }
}
