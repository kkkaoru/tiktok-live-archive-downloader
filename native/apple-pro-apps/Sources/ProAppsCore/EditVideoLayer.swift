import Foundation

/// An output-timed video-only layer above the base clips. Later array entries
/// are above earlier entries. Source alpha is retained during compositing;
/// opacity defaults to one. Opaque sources cover the base. Global effects run
/// after composition; audio is ignored and requires an explicit audio layer.
public struct EditVideoLayer: Codable, Sendable {
  public let sourcePath: String
  public let selection: EditSelection
  public let offsetSeconds: Double
  public let geometry: EditGeometry?
  public let opacity: Double?

  public init(
    sourcePath: String, selection: EditSelection, offsetSeconds: Double,
    geometry: EditGeometry? = nil, opacity: Double? = nil
  ) {
    self.sourcePath = sourcePath
    self.selection = selection
    self.offsetSeconds = offsetSeconds
    self.geometry = geometry
    self.opacity = opacity
  }
}
