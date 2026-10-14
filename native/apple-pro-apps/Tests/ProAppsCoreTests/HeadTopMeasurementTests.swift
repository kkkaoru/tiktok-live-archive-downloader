import Foundation
import Testing

@testable import ProAppsCore

struct HeadTopMeasurementTests {
  private struct EncodedSupport: Decodable {
    let status: String
    let minimumY: Int?
    let contourTouchesTopEdge: Bool
    let contourTouchesBottomEdge: Bool
    let contourTouchesSideEdge: Bool
    let componentTouchesTopEdge: Bool?
    let componentSideContactAtOrAboveFace: Bool?
  }

  private struct EncodedCandidate: Decodable {
    let status: String
    let topY: Int?
    let upperSupport: EncodedSupport
    let semanticHeadVerified: Bool
    let captionPlacementApproved: Bool
  }

  private let face = ForegroundPixelBounds(x: 2, y: 2, width: 2, height: 2)

  private func mask(_ rows: [String], label: Int = 1, padding: Int = 0) -> HeadTopMask {
    HeadTopMask(
      label: label,
      pixels: Data(
        rows.flatMap {
          $0.map { $0 == "#" ? UInt8(255) : 0 } + [UInt8](repeating: 255, count: padding)
        }),
      bytesPerRow: (rows.first?.count ?? 0) + padding)
  }

  @Test func disconnectedHUDDoesNotChangeFaceComponentTop() throws {
    let measured = try HeadTopMeasurement.measure(
      masks: [mask(["#.....", "..##..", "..##..", "..##..", "..##.."], padding: 2)],
      width: 6, height: 5, face: face, minimumFaceCoverage: 0.5)
    #expect(measured.status == .candidate)
    #expect(measured.topY == 1)
    #expect(measured.componentBounds == .init(x: 2, y: 1, width: 2, height: 4))
    #expect(measured.faceCoveredPixels == 4)
    #expect(measured.facePixels == 4)
    #expect(measured.upperContour == [.init(x: 2, y: 1), .init(x: 3, y: 1)])
    #expect(measured.upperSupport.status == .interiorMeasured)
    #expect(measured.upperSupport.minimumY == 1)
    #expect(!measured.semanticHeadVerified)
    #expect(!measured.captionPlacementApproved)
    #expect(!(try JSONEncoder().encode(measured)).isEmpty)
  }

  @Test func differentInstanceAwayFromFaceIsIgnored() throws {
    let result = try HeadTopMeasurement.measure(
      masks: [
        mask(["#.....", "#.....", "#.....", "#.....", "#....."], label: 9),
        mask(["......", "..##..", "..##..", "..##..", "......"], label: 4),
      ],
      width: 6, height: 5, face: face, minimumFaceCoverage: 1)
    #expect(result.instanceLabel == 4)
    #expect(result.topY == 1)
  }

  @Test func lowerTorsoAtFrameEdgesDoesNotClaimClippedHead() throws {
    let result = try HeadTopMeasurement.measure(
      masks: [mask(["......", "..##..", "..##..", "######", "######"])],
      width: 6, height: 5, face: face, minimumFaceCoverage: 0.5)
    #expect(result.status == .candidate)
    #expect(result.topY == 1)
    #expect(result.componentBounds == .init(x: 0, y: 1, width: 6, height: 4))
  }

  @Test func lateralClippingBelowApexRetainsActualInteriorSupport() throws {
    var pixels = Data(repeating: 0, count: 48 * 64)
    for y in 8..<48 {
      pixels.replaceSubrange(y * 48 + 16..<y * 48 + 32, with: repeatElement(UInt8(255), count: 16))
    }
    pixels.replaceSubrange(20 * 48..<20 * 48 + 16, with: repeatElement(UInt8(255), count: 16))
    let result = try HeadTopMeasurement.measure(
      masks: [.init(label: 1, pixels: pixels, bytesPerRow: 48)], width: 48, height: 64,
      face: .init(x: 16, y: 32, width: 16, height: 16), minimumFaceCoverage: 1)
    #expect(result.status == .clippedSupport)
    #expect(result.topY == nil)
    #expect(result.faceCoveredPixels == 256)
    #expect(result.upperContour.count == 16)
    #expect(result.upperContour.first == .init(x: 16, y: 8))
    #expect(result.upperContour.last == .init(x: 31, y: 8))
    #expect(result.upperContour.allSatisfy { $0.y == 8 })
    #expect(
      result.interiorRegions == [
        .init(x: 16, y: 16, width: 16, height: 16),
        .init(x: 16, y: 32, width: 16, height: 16),
      ])
    #expect(
      result.upperSupport
        == .init(
          status: .interiorMeasured, minimumY: 8,
          contourTouchesTopEdge: false, contourTouchesBottomEdge: false,
          contourTouchesSideEdge: false,
          componentTouchesTopEdge: false, componentSideContactAtOrAboveFace: true))
    let encoded = try JSONDecoder().decode(
      EncodedCandidate.self, from: JSONEncoder().encode(result))
    #expect(encoded.status == "clippedSupport")
    #expect(encoded.topY == nil)
    #expect(encoded.upperSupport.status == "interiorMeasured")
    #expect(encoded.upperSupport.minimumY == 8)
    #expect(!encoded.upperSupport.contourTouchesTopEdge)
    #expect(!encoded.upperSupport.contourTouchesBottomEdge)
    #expect(!encoded.upperSupport.contourTouchesSideEdge)
    #expect(encoded.upperSupport.componentTouchesTopEdge == false)
    #expect(encoded.upperSupport.componentSideContactAtOrAboveFace == true)
    #expect(!encoded.semanticHeadVerified)
    #expect(!encoded.captionPlacementApproved)
  }

  @Test func canvasClippedApexRemainsExplicitAndUnapproved() throws {
    let result = try HeadTopMeasurement.measure(
      masks: [mask(["..##..", "..##..", "..##..", "..##..", "......"])],
      width: 6, height: 5, face: face, minimumFaceCoverage: 1)
    #expect(result.status == .clippedSupport)
    #expect(result.topY == nil)
    #expect(result.upperContour == [.init(x: 2, y: 0), .init(x: 3, y: 0)])
    #expect(
      result.upperSupport
        == .init(
          status: .canvasClipped, minimumY: 0,
          contourTouchesTopEdge: true, contourTouchesBottomEdge: false,
          contourTouchesSideEdge: false,
          componentTouchesTopEdge: true, componentSideContactAtOrAboveFace: false))
    #expect(!result.semanticHeadVerified)
    #expect(!result.captionPlacementApproved)
  }

  @Test func contourAtSideEdgeIsNotInteriorSupport() throws {
    let result = try HeadTopMeasurement.measure(
      masks: [mask(["......", "##....", "##....", "##....", "......"])],
      width: 6, height: 5, face: .init(x: 0, y: 2, width: 2, height: 2), minimumFaceCoverage: 1)
    #expect(result.status == .clippedSupport)
    #expect(result.upperContour == [.init(x: 0, y: 1), .init(x: 1, y: 1)])
    #expect(
      result.upperSupport
        == .init(
          status: .canvasClipped, minimumY: 1,
          contourTouchesTopEdge: false, contourTouchesBottomEdge: false,
          contourTouchesSideEdge: true,
          componentTouchesTopEdge: false, componentSideContactAtOrAboveFace: true))
    #expect(result.topY == nil)
  }

  @Test func measuredContourAtCanvasBottomIsNotInteriorSupport() throws {
    let result = try HeadTopMeasurement.measure(
      masks: [mask(["......", "......", "......", "......", "..##.."])],
      width: 6, height: 5, face: .init(x: 2, y: 4, width: 2, height: 1), minimumFaceCoverage: 1)
    // The existing whole-component policy is unchanged; contour eligibility is separate.
    #expect(result.status == .candidate)
    #expect(result.topY == 4)
    #expect(result.upperContour == [.init(x: 2, y: 4), .init(x: 3, y: 4)])
    #expect(result.upperSupport.status == .canvasClipped)
    #expect(result.upperSupport.minimumY == 4)
    #expect(!result.upperSupport.contourTouchesTopEdge)
    #expect(result.upperSupport.contourTouchesBottomEdge)
    #expect(!result.upperSupport.contourTouchesSideEdge)
    #expect(!result.semanticHeadVerified)
    #expect(!result.captionPlacementApproved)
  }

  @Test func insufficientFaceSupportDoesNotEmitDiagnosticContour() throws {
    let result = try HeadTopMeasurement.measure(
      masks: [mask(["......", "..#...", "###...", "......", "......"])],
      width: 6, height: 5, face: face, minimumFaceCoverage: 1)
    #expect(result.status == .insufficientFaceSupport)
    #expect(result.upperContour.isEmpty)
    #expect(result.interiorRegions.isEmpty)
    #expect(result.upperSupport.status == .unavailable)
    #expect(result.upperSupport.minimumY == nil)
    #expect(result.upperSupport.componentSideContactAtOrAboveFace == true)
  }

  @Test(arguments: [
    (["......", "......", "......", "......", "......"], HeadTopCandidate.Status.noOverlap),
    (["......", "......", "..#...", "...#..", "......"], .ambiguousComponents),
    (["......", "......", "..#...", "......", "......"], .insufficientFaceSupport),
    (["..##..", "..##..", "..##..", "..##..", "......"], .clippedSupport),
    (["......", "......", "####..", "..##..", "......"], .clippedSupport),
    (["......", "......", "..####", "..##..", "......"], .clippedSupport),
  ])
  func unknownEvidenceDoesNotFabricateTop(input: ([String], HeadTopCandidate.Status)) throws {
    let result = try HeadTopMeasurement.measure(
      masks: [mask(input.0)], width: 6, height: 5, face: face, minimumFaceCoverage: 0.5)
    #expect(result.status == input.1)
    #expect(result.topY == nil)
  }

  @Test func overlappingInstancesAreAmbiguous() throws {
    let rows = ["......", "..##..", "..##..", "..##..", "......"]
    let result = try HeadTopMeasurement.measure(
      masks: [mask(rows), mask(rows, label: 2)], width: 6, height: 5,
      face: face, minimumFaceCoverage: 0.5)
    #expect(result.status == .ambiguousComponents)
    #expect(result.upperSupport == .unavailable)
    #expect(result.upperContour.isEmpty)
    #expect(result.interiorRegions.isEmpty)
  }

  @Test func checkerboardDoesNotRetainEveryComponent() throws {
    let pixels = Data(
      (0..<65536).map { index in
        (index / 256 + index % 256).isMultiple(of: 2) ? UInt8(255) : 0
      })
    let result = try HeadTopMeasurement.measure(
      masks: [.init(label: 1, pixels: pixels, bytesPerRow: 256)], width: 256, height: 256,
      face: .init(x: 0, y: 0, width: 256, height: 256), minimumFaceCoverage: 0.5)
    #expect(result.status == .ambiguousComponents)
    #expect(result.topY == nil)
  }

  @Test func contourUsesMeasuredColumnsAndIsBounded() throws {
    var pixels = Data(repeating: 0, count: 600 * 16)
    pixels.replaceSubrange(600 + 10..<600 + 590, with: repeatElement(UInt8(255), count: 580))
    pixels.replaceSubrange(1200 + 10..<1200 + 590, with: repeatElement(UInt8(255), count: 580))
    let result = try HeadTopMeasurement.measure(
      masks: [.init(label: 1, pixels: pixels, bytesPerRow: 600)], width: 600, height: 16,
      face: .init(x: 10, y: 2, width: 580, height: 1), minimumFaceCoverage: 1)
    #expect(result.upperContour.count == 256)
    #expect(result.upperContour.first == .init(x: 10, y: 1))
    #expect(result.upperContour.last == .init(x: 589, y: 1))
    #expect(result.upperContour.allSatisfy { $0.y == 1 })
  }

  @Test func preservesUnsampledApexAndOnlyCompleteInteriorTiles() throws {
    var pixels = Data(repeating: 0, count: 600 * 64)
    for y in 2..<48 {
      pixels.replaceSubrange(
        y * 600 + 10..<y * 600 + 590,
        with: repeatElement(UInt8(255), count: 580))
    }
    pixels[600 + 11] = 255
    pixels[20 * 600 + 20] = 0
    let result = try HeadTopMeasurement.measure(
      masks: [.init(label: 1, pixels: pixels, bytesPerRow: 600)], width: 600, height: 64,
      face: .init(x: 10, y: 32, width: 580, height: 16), minimumFaceCoverage: 1)
    #expect(result.upperContour.contains(.init(x: 11, y: 1)))
    #expect(result.upperContour.count == 256)
    #expect(!result.interiorRegions.isEmpty)
    #expect(!result.interiorRegions.contains(.init(x: 16, y: 16, width: 16, height: 16)))
    #expect(
      result.interiorRegions.allSatisfy { region in
        (region.y..<(region.y + region.height)).allSatisfy { y in
          (region.x..<(region.x + region.width)).allSatisfy { x in pixels[y * 600 + x] > 0 }
        }
      })
  }

  @Test(arguments: [0.0, -1.0, 1.1, Double.nan, Double.infinity])
  func invalidCoverageFails(coverage: Double) {
    #expect(throws: ProAppsError.self) {
      try HeadTopMeasurement.measure(
        masks: [], width: 6, height: 5, face: face, minimumFaceCoverage: coverage)
    }
  }

  @Test(arguments: [
    ForegroundPixelBounds(x: -1, y: 1, width: 2, height: 2),
    .init(x: 1, y: -1, width: 2, height: 2), .init(x: 1, y: 1, width: 0, height: 2),
    .init(x: 1, y: 1, width: 2, height: 0), .init(x: 6, y: 1, width: 1, height: 1),
    .init(x: 1, y: 5, width: 1, height: 1), .init(x: 1, y: 1, width: Int.max, height: 1),
    .init(x: 1, y: 1, width: 1, height: Int.max),
  ])
  func invalidFaceFails(face: ForegroundPixelBounds) {
    #expect(throws: ProAppsError.self) {
      try HeadTopMeasurement.measure(
        masks: [], width: 6, height: 5, face: face, minimumFaceCoverage: 0.5)
    }
  }

  @Test func malformedMaskFails() {
    #expect(throws: ProAppsError.self) {
      try HeadTopMeasurement.measure(
        masks: [.init(label: 1, pixels: Data([1]), bytesPerRow: 6)],
        width: 6, height: 5, face: face, minimumFaceCoverage: 0.5)
    }
    #expect(throws: ProAppsError.self) {
      try HeadTopMeasurement.measure(
        masks: [mask([], label: 0)], width: 6, height: 5, face: face, minimumFaceCoverage: 0.5)
    }
    #expect(throws: ProAppsError.self) {
      try HeadTopMeasurement.measure(
        masks: [mask([]), mask([])], width: 6, height: 5, face: face, minimumFaceCoverage: 0.5)
    }
  }

  @Test func cancelledScanPropagates() async {
    await withTaskGroup(of: Void.self) { group in
      group.cancelAll()
      group.addTask {
        #expect(throws: CancellationError.self) {
          try HeadTopMeasurement.measure(
            masks: [.init(label: 1, pixels: Data(repeating: 255, count: 30), bytesPerRow: 6)],
            width: 6, height: 5, face: face, minimumFaceCoverage: 0.5)
        }
      }
    }
  }
}
