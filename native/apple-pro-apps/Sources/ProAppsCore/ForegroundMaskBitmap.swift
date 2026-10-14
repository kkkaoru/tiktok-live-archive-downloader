import CoreImage
import Foundation

extension ForegroundMaskStatistics {
  /// Core Image providers can include allocation pages beyond the logical rows.
  /// Bound the entire provider before taking the exact raster view; do not count
  /// trailing storage as pixels or relax the scanner's complete-row contract.
  static func scanBacking(_ data: Data, width: Int, height: Int, bytesPerRow: Int) throws -> Self {
    try Task.checkCancellation()
    let size = try storageSize(width: width, height: height, bytesPerRow: bytesPerRow)
    guard data.count >= size, data.count <= maximumBytes else {
      throw ProAppsError.invalid("Incomplete or over-budget foreground provider storage")
    }
    return try scan(data.prefix(size), width: width, height: height, bytesPerRow: bytesPerRow)
  }

  static func bitmap(_ bitmap: CGImage, width: Int, height: Int) throws -> Self {
    try Task.checkCancellation()
    _ = try storageSize(width: width, height: height, bytesPerRow: bitmap.bytesPerRow)
    guard bitmap.width == width, bitmap.height == height,
      bitmap.bitsPerComponent == 8, bitmap.bitsPerPixel == 8,
      bitmap.bitmapInfo.rawValue == CGImageAlphaInfo.none.rawValue,
      bitmap.alphaInfo == .none, !bitmap.isMask, bitmap.decode == nil,
      bitmap.colorSpace?.model == .monochrome, bitmap.colorSpace?.numberOfComponents == 1
    else { throw ProAppsError.invalid("Expected source-sized, unmodified L8 bitmap layout") }
    try Task.checkCancellation()
    guard let storage = bitmap.dataProvider?.data else {
      throw ProAppsError.unavailable("Cannot copy foreground bitmap bytes")
    }
    return try scanBacking(
      storage as Data, width: width, height: height, bytesPerRow: bitmap.bytesPerRow)
  }

  static func measure(_ image: CIImage, width: Int, height: Int, context: CIContext) throws -> Self
  {
    try Task.checkCancellation()
    _ = try storageSize(width: width, height: height, bytesPerRow: width)
    guard image.extent == CGRect(x: 0, y: 0, width: width, height: height) else {
      throw ProAppsError.invalid("Foreground mask extent differs from source")
    }
    guard
      let rendered = context.createCGImage(
        image, from: image.extent, format: .L8, colorSpace: CGColorSpaceCreateDeviceGray())
    else { throw ProAppsError.unavailable("Cannot render foreground L8 mask") }
    return try bitmap(rendered, width: width, height: height)
  }
}
