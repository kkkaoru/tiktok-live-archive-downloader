import CoreGraphics
import Testing

@testable import ProAppsCore

struct EditTransformTests {
  @Test(arguments: [
    (EditVideoSettings.ResizeMode.fit, 80.0, 0.0, 560.0, 360.0), (.fill, 0.0, -60.0, 640.0, 420.0),
  ])
  func canvasFitAndFillHaveDifferentObservableBounds(
    _ mode: EditVideoSettings.ResizeMode,
    _ left: Double, _ top: Double, _ right: Double, _ bottom: Double
  ) throws {
    let result = try EditTransform.make(
      naturalSize: CGSize(width: 320, height: 240), preferred: .identity,
      geometry: nil, canvas: .init(width: 640, height: 360, frameRate: 30, resizeMode: mode))
    let origin = CGPoint.zero.applying(result.transform)
    let end = CGPoint(x: 320, y: 240).applying(result.transform)
    #expect(abs(origin.x - left) < 0.000001)
    #expect(abs(origin.y - top) < 0.000001)
    #expect(abs(end.x - right) < 0.000001)
    #expect(abs(end.y - bottom) < 0.000001)
    #expect(result.sourceCrop == CGRect(x: 0, y: 0, width: 320, height: 240))
  }

  @Test(arguments: [
    (EditGeometry.Rotation.none, 320, 240, 20.0, 30.0),
    (.clockwise90, 240, 320, 210.0, 20.0),
    (.clockwise180, 320, 240, 300.0, 210.0),
    (.clockwise270, 240, 320, 30.0, 300.0),
  ])
  func allQuarterTurnsMapKnownPixels(
    _ rotation: EditGeometry.Rotation, _ width: Int,
    _ height: Int, _ expectedX: Double, _ expectedY: Double
  ) throws {
    let result = try EditTransform.make(
      naturalSize: CGSize(width: 320, height: 240), preferred: .identity,
      geometry: .init(rotation: rotation),
      canvas: .init(width: width, height: height, frameRate: 30, resizeMode: .fit))
    let point = CGPoint(x: 20, y: 30).applying(result.transform)
    #expect(abs(point.x - expectedX) < 0.000001)
    #expect(abs(point.y - expectedY) < 0.000001)
  }

  @Test func cropIsMappedBackIntoSourceCoordinatesAfterRotation() throws {
    let result = try EditTransform.make(
      naturalSize: CGSize(width: 320, height: 240), preferred: .identity,
      geometry: .init(rotation: .clockwise90, crop: .init(x: 0, y: 0, width: 120, height: 160)),
      canvas: .init(width: 240, height: 320, frameRate: 30, resizeMode: .fit))
    #expect(abs(result.sourceCrop.minX) < 0.000001)
    #expect(abs(result.sourceCrop.minY - 120) < 0.000001)
    #expect(abs(result.sourceCrop.width - 160) < 0.000001)
    #expect(abs(result.sourceCrop.height - 120) < 0.000001)
    let corner = CGPoint(x: 0, y: 240).applying(result.transform)
    #expect(abs(corner.x) < 0.000001)
    #expect(abs(corner.y) < 0.000001)
  }

  @Test func nativeOrientationAndPlainCropArePreserved() throws {
    let native = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 240, ty: 0)
    let oriented = try EditTransform.make(
      naturalSize: CGSize(width: 320, height: 240), preferred: native,
      geometry: nil, canvas: .init(width: 240, height: 320, frameRate: 30, resizeMode: .fit))
    #expect(CGPoint(x: 20, y: 30).applying(oriented.transform) == CGPoint(x: 210, y: 20))
    let cropped = try EditTransform.make(
      naturalSize: CGSize(width: 320, height: 240), preferred: .identity,
      geometry: .init(rotation: .none, crop: .init(x: 80, y: 60, width: 160, height: 120)),
      canvas: .init(width: 320, height: 240, frameRate: 30, resizeMode: .fit))
    #expect(cropped.sourceCrop == CGRect(x: 80, y: 60, width: 160, height: 120))
    #expect(CGPoint(x: 80, y: 60).applying(cropped.transform) == .zero)
  }

  @Test(arguments: [
    (CGSize(width: 0, height: 100), CGAffineTransform.identity),
    (CGSize(width: CGFloat.nan, height: 100), CGAffineTransform.identity),
    (CGSize(width: 100, height: 100), CGAffineTransform(scaleX: 0, y: 1)),
    (CGSize(width: 100, height: 100), CGAffineTransform(translationX: .infinity, y: 0)),
    (CGSize(width: 100, height: 100), CGAffineTransform(scaleX: 1000, y: 1000)),
  ])
  func invalidSourceGeometryNeverReachesAVFoundation(_ size: CGSize, _ preferred: CGAffineTransform)
  {
    #expect(throws: (any Error).self) {
      try EditTransform.make(
        naturalSize: size, preferred: preferred, geometry: nil,
        canvas: .init(width: 320, height: 240, frameRate: 30, resizeMode: .fit))
    }
  }

  @Test(arguments: [
    EditCrop(x: 0, y: 0, width: 321, height: 240),
    EditCrop(x: 0, y: 0, width: -1, height: 10),
    EditCrop(x: 0, y: 0, width: .infinity, height: 10),
  ])
  func outOfSourceAndNegativeCropsAreNotSilentlyClamped(_ crop: EditCrop) {
    #expect(throws: (any Error).self) {
      try EditTransform.make(
        naturalSize: CGSize(width: 320, height: 240), preferred: .identity,
        geometry: .init(rotation: .none, crop: crop),
        canvas: .init(width: 320, height: 240, frameRate: 30, resizeMode: .fit))
    }
  }

  @Test func nonRectangularInverseCropsAreExplicitlyUnsupported() {
    #expect(throws: (any Error).self) {
      try EditTransform.make(
        naturalSize: CGSize(width: 320, height: 240),
        preferred: CGAffineTransform(rotationAngle: .pi / 4),
        geometry: .init(rotation: .none, crop: .init(x: 0, y: 0, width: 20, height: 20)),
        canvas: .init(width: 320, height: 240, frameRate: 30, resizeMode: .fit))
    }
  }
}
