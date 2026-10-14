import Foundation

@main
struct FontIdentityCommand {
  static func main() throws {
    // Bounded stdin/stdout only: no Foundation temporary-directory API is used.
    var input = Data()
    while input.count <= 64 * 1024,
      let chunk = try FileHandle.standardInput.read(upToCount: 64 * 1024 + 1 - input.count),
      !chunk.isEmpty
    {
      input.append(chunk)
    }
    try FileHandle.standardOutput.write(contentsOf: FontIdentityProbe.response(input))
  }
}
