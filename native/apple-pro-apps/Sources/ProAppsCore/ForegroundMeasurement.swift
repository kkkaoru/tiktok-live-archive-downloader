import CoreMedia
import Foundation

/// Exact source clock; validation is repeated at the measurement boundary.
public struct ForegroundTime: Codable, Equatable, Sendable {
  public static let maximumSeconds: Int64 = 86400
  public static let maximumTimescale: Int32 = 1_000_000_000
  public let value: Int64
  public let timescale: Int32

  public init(value: Int64, timescale: Int32) {
    self.value = value
    self.timescale = timescale
  }

  func time() throws -> CMTime {
    guard (1...Self.maximumTimescale).contains(timescale), value >= 0,
      value <= Self.maximumSeconds * Int64(timescale)
    else { throw ProAppsError.invalid("Invalid foreground clock") }
    return CMTime(value: value, timescale: timescale)
  }

  static func validate(_ samples: [Self]) throws -> [CMTime] {
    guard (1...FrameProbe.maximumSamples).contains(samples.count) else {
      throw ProAppsError.invalid("Supply 1–8 foreground samples")
    }
    let times = try samples.map { try $0.time() }
    guard zip(times, times.dropFirst()).allSatisfy({ $0.0 < $0.1 }) else {
      throw ProAppsError.invalid("Foreground samples must be strictly ordered")
    }
    return times
  }

  static func actual(_ actual: CMTime, requested: CMTime) throws -> Self {
    guard requested.isNumeric, actual.isNumeric,
      requested.epoch == 0, actual.epoch == 0,
      !requested.flags.contains(.hasBeenRounded), !actual.flags.contains(.hasBeenRounded)
    else { throw ProAppsError.invalid("Invalid foreground presentation clock") }
    let request = Self(value: requested.value, timescale: requested.timescale)
    let result = Self(value: actual.value, timescale: actual.timescale)
    guard try request.time() == result.time() else {
      throw ProAppsError.invalid("Foreground presentation clock differs from request")
    }
    return result
  }
}

public struct ForegroundRequest: Decodable, Sendable {
  public let path: String
  public let samples: [ForegroundTime]
  public let headFace: HeadTopFaceRequest?
  public let headFaces: [HeadTopFaceRequest]?
}

/// Local labels have no persistent identity or semantic interpretation.
struct ForegroundObservation: Sendable {
  private static let maximumLabels = 256
  let labels: [Int]
  let mask: ForegroundMaskStatistics?

  static func collect(
    labels: IndexSet, union: () throws -> ForegroundMaskStatistics
  ) throws -> Self {
    try Task.checkCancellation()
    guard labels.count <= maximumLabels, labels.allSatisfy({ $0 > 0 }) else {
      throw ProAppsError.invalid("Invalid foreground instance labels")
    }
    return Self(labels: Array(labels), mask: labels.isEmpty ? nil : try union())
  }
}

/// Diagnostic union support only; never approves captions or establishes absence.
public struct ForegroundMeasurement: Encodable, Sendable {
  public let requested: ForegroundTime
  public let actual: ForegroundTime
  public let width: Int
  public let height: Int
  let observation: ForegroundObservation?
  var headTop: HeadTopCandidate? = nil

  private enum CodingKeys: String, CodingKey {
    case requested, actual, width, height, observationPresent, instanceLabels, unionMask
    case maskThreshold8Bit, captionPlacementApproved
    case headTop
  }

  public func encode(to encoder: any Encoder) throws {
    var fields = encoder.container(keyedBy: CodingKeys.self)
    try fields.encode(requested, forKey: .requested)
    try fields.encode(actual, forKey: .actual)
    try fields.encode(width, forKey: .width)
    try fields.encode(height, forKey: .height)
    try fields.encode(observation != nil, forKey: .observationPresent)
    try fields.encode(observation?.labels, forKey: .instanceLabels)
    try fields.encode(observation?.mask, forKey: .unionMask)
    try fields.encode(1, forKey: .maskThreshold8Bit)
    try fields.encode(false, forKey: .captionPlacementApproved)
    try fields.encodeIfPresent(headTop, forKey: .headTop)
  }
}
