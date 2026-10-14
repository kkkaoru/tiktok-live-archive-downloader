import CoreGraphics
import CoreVideo
import Foundation
import Testing

@testable import ProAppsCore

struct FaceTrackingBitmapTests {
  private func pixels(format: OSType = kCVPixelFormatType_32BGRA) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    try #require(CVPixelBufferCreate(nil, 13, 7, format, nil, &buffer) == kCVReturnSuccess)
    let pixels = try #require(buffer)
    let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
    CVBufferSetAttachment(pixels, kCVImageBufferCGColorSpaceKey, space, .shouldPropagate)
    if format == kCVPixelFormatType_32BGRA {
      try #require(CVPixelBufferLockBaseAddress(pixels, []) == kCVReturnSuccess)
      defer { CVPixelBufferUnlockBaseAddress(pixels, []) }
      let context = try #require(
        CGContext(
          data: CVPixelBufferGetBaseAddress(pixels), width: 13, height: 7,
          bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixels), space: space,
          bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue
            | CGImageAlphaInfo.premultipliedFirst.rawValue))
      context.setFillColor(try #require(CGColor(colorSpace: space, components: [0, 0, 1, 1])))
      context.fill(CGRect(x: 0, y: 0, width: 13, height: 7))
      context.setFillColor(try #require(CGColor(colorSpace: space, components: [1, 0, 0, 1])))
      context.fill(CGRect(x: 0, y: 0, width: 5, height: 3))
    }
    return pixels
  }

  private func bytes(_ image: CGImage) throws -> Data {
    let space = try #require(image.colorSpace)
    let context = try #require(
      CGContext(
        data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
        bytesPerRow: image.width * 4, space: space,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    return Data(bytes: try #require(context.data), count: image.width * image.height * 4)
  }

  @Test(arguments: [
    CGAffineTransform.identity,
    CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 7, ty: 0),
    CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: 13, ty: 7),
    CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: 13),
    CGAffineTransform(a: -1, b: 0, c: 0, d: 1, tx: 13, ty: 0),
    CGAffineTransform(translationX: -5, y: 11),
  ])
  func matchesAnalyticOrientationAndPaddedRows(_ transform: CGAffineTransform) throws {
    let pixels = try pixels()
    #expect(CVPixelBufferGetBytesPerRow(pixels) > 13 * 4)
    let image = try FaceTrackingBitmap.render(pixels: pixels, transform: transform)
    let extent = CGRect(x: 0, y: 0, width: 13, height: 7).applying(transform)
    #expect(image.width == Int(extent.width) && image.height == Int(extent.height))
    #expect(image.colorSpace == CGColorSpace(name: CGColorSpace.sRGB))
    let actual = try bytes(image)
    var expected = Data()
    for row in 0..<image.height {
      for column in 0..<image.width {
        let source = CGPoint(
          x: extent.minX + CGFloat(column) + 0.5,
          // Bitmap rows are top-first; affine geometry uses bottom-left coordinates.
          y: extent.maxY - CGFloat(row) - 0.5
        ).applying(transform.inverted())
        let red = source.x < 5 && source.y < 3
        expected.append(contentsOf: red ? [255, 0, 0, 255] : [0, 0, 255, 255])
      }
    }
    let actualRed = stride(from: 0, to: actual.count, by: 4).filter { actual[$0] == 255 }
    let expectedRed = stride(from: 0, to: expected.count, by: 4).filter { expected[$0] == 255 }
    #expect(actualRed == expectedRed)
    #expect(actual == expected)
  }

  @Test func rejectsUnsupportedPixelsAndInvalidGeometry() throws {
    let pixels = try pixels(format: kCVPixelFormatType_OneComponent8)
    #expect(throws: ProAppsError.self) {
      try FaceTrackingBitmap.render(pixels: pixels, transform: .identity)
    }
    #expect(throws: ProAppsError.self) {
      try FaceTrackingBitmap.render(pixels: pixels, transform: CGAffineTransform(scaleX: 0, y: 0))
    }
  }

  @Test func validatesAllocationLockAndStrideFailures() throws {
    let missing: CGImage? = nil
    #expect(throws: ProAppsError.unavailable("allocation")) {
      try FaceTrackingBitmap.require(missing, operation: "allocation")
    }
    #expect(throws: ProAppsError.self) { try FaceTrackingBitmap.validateLock(kCVReturnError) }
    #expect(throws: ProAppsError.self) {
      try FaceTrackingBitmap.validateStride(4, width: 2, height: 1)
    }
    #expect(throws: ProAppsError.self) {
      try FaceTrackingBitmap.validateStride(Int.max, width: 1, height: 2)
    }
  }
}
