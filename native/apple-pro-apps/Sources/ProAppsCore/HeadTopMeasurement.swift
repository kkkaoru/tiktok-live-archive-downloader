import Foundation

/// Source-sized instance raster. Labels are local to one observation, never identities.
struct HeadTopMask: Sendable {
  let label: Int
  let pixels: Data
  let bytesPerRow: Int
}

struct HeadTopContourPoint: Encodable, Equatable, Sendable {
  let x: Int
  let y: Int
}

/// Describes measured contour support, independently of whole-component eligibility.
struct HeadTopUpperSupport: Encodable, Equatable, Sendable {
  enum Status: String, Encodable, Sendable {
    case unavailable, interiorMeasured, canvasClipped
  }
  let status: Status
  let minimumY: Int?
  let contourTouchesTopEdge: Bool
  let contourTouchesBottomEdge: Bool
  let contourTouchesSideEdge: Bool
  let componentTouchesTopEdge: Bool?
  let componentSideContactAtOrAboveFace: Bool?

  static let unavailable = HeadTopUpperSupport(
    status: .unavailable, minimumY: nil,
    contourTouchesTopEdge: false, contourTouchesBottomEdge: false, contourTouchesSideEdge: false,
    componentTouchesTopEdge: nil, componentSideContactAtOrAboveFace: nil)
}

/// Geometric evidence only: a foreground component can include hats, hands or props.
struct HeadTopCandidate: Encodable, Equatable, Sendable {
  enum Status: String, Encodable, Sendable {
    case candidate, noOverlap, ambiguousComponents, insufficientFaceSupport, clippedSupport
  }
  let status: Status
  let instanceLabel: Int?
  let componentBounds: ForegroundPixelBounds?
  let topY: Int?
  let faceCoveredPixels: Int
  let facePixels: Int
  var upperContour: [HeadTopContourPoint] = []
  var interiorRegions: [ForegroundPixelBounds] = []
  var upperSupport: HeadTopUpperSupport = .unavailable
  var semanticHeadVerified: Bool { false }
  var captionPlacementApproved: Bool { false }

  private enum CodingKeys: String, CodingKey {
    case status, instanceLabel, componentBounds, topY, faceCoveredPixels, facePixels
    case semanticHeadVerified, captionPlacementApproved
    case upperContour, contourScope, interiorRegions, upperSupport
  }

  func encode(to encoder: any Encoder) throws {
    var fields = encoder.container(keyedBy: CodingKeys.self)
    try fields.encode(status, forKey: .status)
    try fields.encode(instanceLabel, forKey: .instanceLabel)
    try fields.encode(componentBounds, forKey: .componentBounds)
    try fields.encode(topY, forKey: .topY)
    try fields.encode(faceCoveredPixels, forKey: .faceCoveredPixels)
    try fields.encode(facePixels, forKey: .facePixels)
    try fields.encode(semanticHeadVerified, forKey: .semanticHeadVerified)
    try fields.encode(captionPlacementApproved, forKey: .captionPlacementApproved)
    try fields.encode(upperContour, forKey: .upperContour)
    try fields.encode(interiorRegions, forKey: .interiorRegions)
    try fields.encode(upperSupport, forKey: .upperSupport)
    try fields.encode("upper-component-in-face-x-range", forKey: .contourScope)
  }
}

/// Associates a face rectangle with four-connected support in individual L8 masks.
/// No face-height multiplier, manual apex, media-specific geometry or learned model.
enum HeadTopMeasurement {
  private struct Component {
    let label: Int
    let bounds: ForegroundPixelBounds
    let overlap: Int
    let upperClipped: Bool
    let contour: [HeadTopContourPoint]
    let regions: [ForegroundPixelBounds]
  }

  static func measure(
    masks: [HeadTopMask], width: Int, height: Int, face: ForegroundPixelBounds,
    minimumFaceCoverage: Double
  ) throws -> HeadTopCandidate {
    _ = try ForegroundMaskStatistics.storageSize(width: width, height: height, bytesPerRow: width)
    guard face.x >= 0, face.y >= 0, face.width > 0, face.height > 0,
      face.x < width, face.y < height, face.width <= width - face.x,
      face.height <= height - face.y,
      minimumFaceCoverage.isFinite, minimumFaceCoverage > 0, minimumFaceCoverage <= 1,
      masks.count <= 256, Set(masks.map(\.label)).count == masks.count,
      masks.allSatisfy({ $0.label > 0 })
    else { throw ProAppsError.invalid("Invalid head-top face, mask labels or coverage policy") }
    for mask in masks {
      let size = try ForegroundMaskStatistics.storageSize(
        width: width, height: height, bytesPerRow: mask.bytesPerRow)
      guard mask.pixels.count == size else {
        throw ProAppsError.invalid("Incomplete head-top mask raster")
      }
    }
    let facePixels = face.width * face.height
    var overlaps: [Component] = []
    for mask in masks {
      overlaps += try components(mask: mask, width: width, height: height, face: face)
      if overlaps.count > 1 { break }
    }
    guard overlaps.count == 1, let selected = overlaps.first else {
      return HeadTopCandidate(
        status: overlaps.isEmpty ? .noOverlap : .ambiguousComponents, instanceLabel: nil,
        componentBounds: nil, topY: nil, faceCoveredPixels: 0, facePixels: facePixels)
    }
    let bounds = selected.bounds
    let status: HeadTopCandidate.Status
    if Double(selected.overlap) / Double(facePixels) < minimumFaceCoverage {
      status = .insufficientFaceSupport
    } else if bounds.y == 0 || selected.upperClipped {
      // Torso contact below the face is not evidence that the head support is clipped.
      status = .clippedSupport
    } else {
      status = .candidate
    }
    // Clipping makes the complete head unknown, not the actual interior pixels absent.
    let retainsMeasuredSupport = status == .candidate || status == .clippedSupport
    let contour = retainsMeasuredSupport ? selected.contour : []
    return HeadTopCandidate(
      status: status, instanceLabel: selected.label, componentBounds: bounds,
      topY: status == .candidate ? bounds.y : nil,
      faceCoveredPixels: selected.overlap, facePixels: facePixels,
      upperContour: contour,
      interiorRegions: retainsMeasuredSupport ? selected.regions : [],
      upperSupport: upperSupport(
        contour: contour, component: selected, width: width, height: height))
  }

  private static func upperSupport(
    contour: [HeadTopContourPoint], component: Component, width: Int, height: Int
  ) -> HeadTopUpperSupport {
    let topContact = contour.contains { $0.y == 0 }
    let bottomContact = contour.contains { $0.y == height - 1 }
    let sideContact = contour.contains { $0.x == 0 || $0.x == width - 1 }
    let status: HeadTopUpperSupport.Status
    if contour.isEmpty {
      status = .unavailable
    } else if topContact || bottomContact || sideContact {
      status = .canvasClipped
    } else {
      status = .interiorMeasured
    }
    return HeadTopUpperSupport(
      status: status, minimumY: contour.map(\.y).min(),
      contourTouchesTopEdge: topContact, contourTouchesBottomEdge: bottomContact,
      contourTouchesSideEdge: sideContact,
      componentTouchesTopEdge: component.bounds.y == 0,
      componentSideContactAtOrAboveFace: component.upperClipped)
  }

  private static func components(
    mask: HeadTopMask, width: Int, height: Int, face: ForegroundPixelBounds
  ) throws -> [Component] {
    var visited = [Bool](repeating: false, count: width * height)
    var result: [Component] = []
    for index in 0..<(width * height) {
      if index % width == 0 { try Task.checkCancellation() }
      guard !visited[index], supported(index, mask: mask, width: width) else { continue }
      var queue = [index]
      visited[index] = true
      var cursor = 0
      var left = width
      var top = height
      var right = 0
      var bottom = 0
      var overlap = 0
      var upperClipped = false
      while cursor < queue.count {
        if cursor % 1024 == 0 { try Task.checkCancellation() }
        let current = queue[cursor]
        cursor += 1
        let x = current % width
        let y = current / width
        left = min(left, x)
        right = max(right, x)
        top = min(top, y)
        bottom = max(bottom, y)
        if y <= face.y, x == 0 || x == width - 1 { upperClipped = true }
        if x >= face.x, x < face.x + face.width, y >= face.y, y < face.y + face.height {
          overlap += 1
        }
        let neighbors = [
          x > 0 ? current - 1 : nil, x + 1 < width ? current + 1 : nil,
          y > 0 ? current - width : nil, y + 1 < height ? current + width : nil,
        ]
        for neighbor in neighbors {
          guard let neighbor, !visited[neighbor], supported(neighbor, mask: mask, width: width)
          else { continue }
          visited[neighbor] = true
          queue.append(neighbor)
        }
      }
      if overlap > 0 {
        result.append(
          Component(
            label: mask.label,
            bounds: .init(x: left, y: top, width: right - left + 1, height: bottom - top + 1),
            overlap: overlap, upperClipped: upperClipped,
            contour: componentContour(queue, width: width, height: height, face: face),
            regions: interiorRegions(queue, width: width, face: face)))
        if result.count == 2 { return result }
      }
    }
    return result
  }

  private static func componentContour(
    _ pixels: [Int], width: Int, height: Int, face: ForegroundPixelBounds
  ) -> [HeadTopContourPoint] {
    var top = [Int](repeating: height, count: face.width)
    for pixel in pixels {
      let x = pixel % width
      if x >= face.x, x < face.x + face.width {
        top[x - face.x] = min(top[x - face.x], pixel / width)
      }
    }
    return contour(top, originX: face.x, absent: height)
  }

  /// Only measured upper support pixels; no bbox corners or interpolated points.
  private static func contour(_ top: [Int], originX: Int, absent: Int) -> [HeadTopContourPoint] {
    let measured = top.enumerated().compactMap { column, y in
      y == absent ? nil : HeadTopContourPoint(x: originX + column, y: y)
    }
    let maximumPoints = 256
    guard measured.count > maximumPoints else { return measured }
    var indices = (0..<maximumPoints).map { $0 * (measured.count - 1) / (maximumPoints - 1) }
    let apex = measured.indices.reduce(0) { measured[$1].y < measured[$0].y ? $1 : $0 }
    if !indices.contains(apex) { indices[1] = apex }
    return indices.sorted().map { measured[$0] }
  }

  /// A region is emitted only when all its pixels belonged to this component.
  private static func interiorRegions(
    _ pixels: [Int], width: Int, face: ForegroundPixelBounds
  ) -> [ForegroundPixelBounds] {
    let side = 16
    let columns = (width + side - 1) / side
    var counts: [Int: Int] = [:]
    for pixel in pixels {
      let x = pixel % width
      let y = pixel / width
      if x >= face.x, x < face.x + face.width, y < face.y + face.height {
        counts[(y / side) * columns + x / side, default: 0] += 1
      }
    }
    let complete = counts.keys.filter { counts[$0] == side * side }.sorted()
    let count = min(256, complete.count)
    return (0..<count).map { index in
      let tile = complete[index * complete.count / count]
      return .init(
        x: (tile % columns) * side, y: (tile / columns) * side,
        width: side, height: side)
    }
  }

  private static func supported(_ index: Int, mask: HeadTopMask, width: Int) -> Bool {
    mask.pixels[mask.pixels.startIndex + (index / width) * mask.bytesPerRow + index % width] > 0
  }
}
