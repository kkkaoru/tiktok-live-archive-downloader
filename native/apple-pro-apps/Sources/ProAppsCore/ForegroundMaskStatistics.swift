import Foundation

/// Tight integer support of a quantized mask, in top-left source pixels.
/// Neither this rectangle nor its contents identify a head or approve placement.
public struct ForegroundPixelBounds: Encodable, Equatable, Sendable {
  public let x: Int
  public let y: Int
  public let width: Int
  public let height: Int
}

public struct ForegroundMaskStatistics: Encodable, Equatable, Sendable {
  static let maximumDimension = 16384
  static let maximumPixels = 16_777_216
  static let maximumBytes = maximumPixels * 4

  public let width: Int
  public let height: Int
  public let nonzeroPixels: Int
  public let nonzeroBounds: ForegroundPixelBounds?

  private enum CodingKeys: String, CodingKey {
    case width, height, nonzeroPixels, nonzeroBounds
  }

  public func encode(to encoder: any Encoder) throws {
    var fields = encoder.container(keyedBy: CodingKeys.self)
    try fields.encode(width, forKey: .width)
    try fields.encode(height, forKey: .height)
    try fields.encode(nonzeroPixels, forKey: .nonzeroPixels)
    try fields.encode(nonzeroBounds, forKey: .nonzeroBounds)
  }

  static func storageSize(width: Int, height: Int, bytesPerRow: Int) throws -> Int {
    guard (1...maximumDimension).contains(width), (1...maximumDimension).contains(height),
      width * height <= maximumPixels,
      bytesPerRow >= width, bytesPerRow <= maximumBytes / height
    else { throw ProAppsError.invalid("Invalid foreground mask storage or geometry") }
    return bytesPerRow * height
  }

  /// Scan validated L8 rows in top-left order. Padding is not mask support.
  /// The producer must independently establish the bitmap layout and orientation.
  static func scan(
    _ data: Data, width: Int, height: Int, bytesPerRow: Int,
    checkRow: (Int) throws -> Void = { _ in try Task.checkCancellation() }
  ) throws -> Self {
    try Task.checkCancellation()
    let size = try storageSize(width: width, height: height, bytesPerRow: bytesPerRow)
    guard data.count == size else {
      throw ProAppsError.invalid(
        "Foreground L8 storage count \(data.count) differs from \(size) "
          + "(width \(width), height \(height), stride \(bytesPerRow))")
    }

    var pixels = 0
    var left = width
    var top = height
    var right = -1
    var bottom = -1
    for y in 0..<height {
      try checkRow(y)
      let row = data.startIndex + y * bytesPerRow
      for x in 0..<width where data[row + x] >= 1 {
        pixels += 1
        left = min(left, x)
        top = min(top, y)
        right = max(right, x)
        bottom = max(bottom, y)
      }
    }
    try Task.checkCancellation()
    let bounds: ForegroundPixelBounds? =
      pixels == 0
      ? nil
      : .init(x: left, y: top, width: right - left + 1, height: bottom - top + 1)
    return Self(width: width, height: height, nonzeroPixels: pixels, nonzeroBounds: bounds)
  }
}
