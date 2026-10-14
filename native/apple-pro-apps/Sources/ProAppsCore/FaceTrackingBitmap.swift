import CoreGraphics
import CoreVideo
import Foundation

/// CPU bitmap conversion. No Core Image context or device enumeration is involved.
enum FaceTrackingBitmap {
  static func require<Value>(_ value: Value?, operation: String) throws -> Value {
    guard let value else { throw ProAppsError.unavailable(operation) }
    return value
  }

  static func validateStride(_ stride: Int, width: Int, height: Int) throws {
    guard width > 0, width <= Int.max / 4, height > 0,
      stride >= width * 4, stride <= Int.max / height
    else { throw ProAppsError.invalid("Tracking pixel row stride is invalid") }
  }

  static func validateLock(_ result: CVReturn) throws {
    guard result == kCVReturnSuccess else {
      throw ProAppsError.unavailable("Cannot lock tracking pixel buffer")
    }
  }

  static func render(pixels: CVPixelBuffer, transform: CGAffineTransform) throws -> CGImage {
    let width = CVPixelBufferGetWidth(pixels)
    let height = CVPixelBufferGetHeight(pixels)
    let size = CGSize(width: width, height: height)
    try FrameProbe.validateGeometry(size: size, transform: transform)
    guard CVPixelBufferGetPixelFormatType(pixels) == kCVPixelFormatType_32BGRA,
      !CVPixelBufferIsPlanar(pixels),
      let space = CVImageBufferGetColorSpace(pixels)?.takeUnretainedValue(), space.model == .rgb
    else {
      throw ProAppsError.invalid("Tracking requires packed BGRA with decoded RGB color space")
    }
    let stride = CVPixelBufferGetBytesPerRow(pixels)
    try validateStride(stride, width: width, height: height)
    try validateLock(CVPixelBufferLockBaseAddress(pixels, .readOnly))
    let data: Data
    do {
      defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
      let address = try require(
        CVPixelBufferGetBaseAddress(pixels), operation: "Tracking pixel buffer has no base address")
      data = Data(bytes: address, count: stride * height)
    }
    let info = CGBitmapInfo.byteOrder32Little.union(
      CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue))
    let provider = try require(
      CGDataProvider(data: data as CFData), operation: "Cannot construct tracking data provider")
    let source = try require(
      CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: stride, space: space, bitmapInfo: info, provider: provider,
        decode: nil, shouldInterpolate: false, intent: .defaultIntent),
      operation: "Cannot construct tracking source bitmap")
    let extent = CGRect(origin: .zero, size: size).applying(transform).integral
    let context = try require(
      CGContext(
        data: nil, width: Int(extent.width), height: Int(extent.height), bitsPerComponent: 8,
        bytesPerRow: 0, space: space, bitmapInfo: info.rawValue),
      operation: "Cannot allocate tracking display bitmap")
    context.interpolationQuality = .none
    context.translateBy(x: -extent.minX, y: -extent.minY)
    context.concatenate(transform)
    context.draw(source, in: CGRect(origin: .zero, size: size))
    return try require(context.makeImage(), operation: "Cannot finish tracking display bitmap")
  }
}
