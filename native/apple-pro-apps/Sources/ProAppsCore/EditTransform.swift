import CoreGraphics
import Foundation

/// Source-pixel crop plus display transform for AVFoundation layer instructions.
/// Crop coordinates in the recipe are top-left display pixels after rotation.
public struct EditTransform: Sendable {
  public let transform: CGAffineTransform
  public let sourceCrop: CGRect

  public static func make(
    naturalSize: CGSize, preferred: CGAffineTransform,
    geometry: EditGeometry?, canvas: EditVideoSettings
  ) throws -> EditTransform {
    try EditPlan.validate(canvas)
    let maximumCoordinate: CGFloat = 16384
    let epsilon: CGFloat = 0.000_001
    guard naturalSize.width.isFinite, naturalSize.height.isFinite,
      (1...maximumCoordinate).contains(naturalSize.width),
      (1...maximumCoordinate).contains(naturalSize.height),
      [preferred.a, preferred.b, preferred.c, preferred.d, preferred.tx, preferred.ty].allSatisfy({
        $0.isFinite && abs($0) <= maximumCoordinate
      }),
      abs(preferred.a * preferred.d - preferred.b * preferred.c) > epsilon
    else { throw ProAppsError.invalid("Invalid source dimensions or display transform") }
    let source = CGRect(origin: .zero, size: naturalSize)
    var display = normalized(preferred, source: source)
    let rotation = geometry?.rotation ?? .none
    let radians = Double(rotation.rawValue) * Double.pi / 180
    display = normalized(
      display.concatenating(CGAffineTransform(rotationAngle: radians)), source: source)
    let bounds = source.applying(display)
    guard bounds.width > 0, bounds.height > 0,
      bounds.width <= maximumCoordinate, bounds.height <= maximumCoordinate
    else { throw ProAppsError.invalid("Transformed source exceeds its geometry limit") }
    let crop: CGRect
    if let requested = geometry?.crop {
      // An axis-aligned source crop cannot exactly represent a display crop
      // after an arbitrary shear/angle. Refuse instead of cropping extra pixels.
      guard
        (abs(display.a) < epsilon && abs(display.d) < epsilon)
          || (abs(display.b) < epsilon && abs(display.c) < epsilon)
      else {
        throw ProAppsError.invalid("Display cropping requires a quarter-turn source transform")
      }
      guard requested.x >= 0, requested.y >= 0, requested.width > 0, requested.height > 0 else {
        throw ProAppsError.invalid("Crop requires nonnegative origin and positive size")
      }
      crop = CGRect(
        x: requested.x, y: requested.y, width: requested.width, height: requested.height)
      guard crop.minX.isFinite, crop.minY.isFinite, crop.width.isFinite, crop.height.isFinite,
        crop.minX >= 0, crop.minY >= 0, crop.width > 0, crop.height > 0,
        crop.maxX <= bounds.width + epsilon, crop.maxY <= bounds.height + epsilon
      else { throw ProAppsError.invalid("Crop lies outside the displayed source") }
    } else {
      crop = CGRect(origin: .zero, size: bounds.size)
    }
    let horizontal = CGFloat(canvas.width) / crop.width
    let vertical = CGFloat(canvas.height) / crop.height
    let scale: CGFloat
    switch canvas.resizeMode {
    case .fit: scale = min(horizontal, vertical)
    case .fill: scale = max(horizontal, vertical)
    }
    let destinationX = (CGFloat(canvas.width) - crop.width * scale) / 2
    let destinationY = (CGFloat(canvas.height) - crop.height * scale) / 2
    let sourceCrop = crop.applying(display.inverted())
    let transform =
      display
      .concatenating(CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
      .concatenating(CGAffineTransform(scaleX: scale, y: scale))
      .concatenating(CGAffineTransform(translationX: destinationX, y: destinationY))
    return EditTransform(transform: transform, sourceCrop: sourceCrop)
  }

  private static func normalized(_ transform: CGAffineTransform, source: CGRect)
    -> CGAffineTransform
  {
    let bounds = source.applying(transform)
    return transform.concatenating(CGAffineTransform(translationX: -bounds.minX, y: -bounds.minY))
  }
}
