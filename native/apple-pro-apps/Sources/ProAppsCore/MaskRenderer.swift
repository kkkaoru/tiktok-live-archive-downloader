import CoreGraphics
import CoreImage.CIFilterBuiltins
import Foundation

/// Managed image operations only. Callers validate settings with EditPlan first.
/// Each callback owns its filters; no mutable filter crosses concurrency domains.
enum MaskRenderer {
  static func requireOutput(_ image: CIImage?) throws -> CIImage {
    guard let image else {
      throw ProAppsError.unavailable("Mask filter produced no image")
    }
    return image
  }

  static func apply(
    _ masks: [EditMask], to source: CIImage, at seconds: Double, canvasHeight: Int
  ) throws -> CIImage {
    var image = source
    for mask in masks where mask.isActive(at: seconds) && mask.opacity > 0 {
      let region = mask.region
      let rectangle = CGRect(
        x: region.x, y: Double(canvasHeight) - region.y - region.height,
        width: region.width, height: region.height)
      let overlay: CIImage
      if let radius = mask.blurRadius {
        let blur = CIFilter.gaussianBlur()
        blur.inputImage = image.clampedToExtent()
        blur.radius = Float(radius)
        let blurred = try requireOutput(blur.outputImage)
        let alpha = CIFilter.colorMatrix()
        alpha.inputImage = blurred.cropped(to: rectangle)
        alpha.aVector = CIVector(x: 0, y: 0, z: 0, w: mask.opacity)
        let blended = try requireOutput(alpha.outputImage)
        overlay = blended.cropped(to: rectangle)
      } else {
        overlay = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: mask.opacity))
          .cropped(to: rectangle)
      }
      image = overlay.composited(over: image)
    }
    return image.cropped(to: source.extent)
  }
}
