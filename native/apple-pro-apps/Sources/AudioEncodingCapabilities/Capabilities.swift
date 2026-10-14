import AVFAudio
import AudioToolbox
import Foundation

enum CapabilityError: Error, Equatable {
  case arguments
  case invalidFormat
  case unavailableConverter
  case invalidRates
}

enum BitRateStrategy: String, Codable, CaseIterable {
  case constant
  case longTermAverage
  case constrainedVariable
  case variable

  var nativeValue: String {
    switch self {
    case .constant: AVAudioBitRateStrategy_Constant
    case .longTermAverage: AVAudioBitRateStrategy_LongTermAverage
    case .constrainedVariable: AVAudioBitRateStrategy_VariableConstrained
    case .variable: AVAudioBitRateStrategy_Variable
    }
  }
}

struct EncodingRequest: Equatable {
  let sampleRate: Double
  let channels: UInt32
  let strategy: BitRateStrategy

  init(sampleRate: Double, channels: UInt32, strategy: BitRateStrategy) throws {
    guard sampleRate.isFinite, sampleRate.rounded() == sampleRate,
      (8000...192000).contains(sampleRate), (1...8).contains(channels)
    else { throw CapabilityError.invalidFormat }
    self.sampleRate = sampleRate
    self.channels = channels
    self.strategy = strategy
  }

  init(arguments: [String]) throws {
    guard arguments.count == 4, let rate = Double(arguments[1]),
      let channels = UInt32(arguments[2]), let strategy = BitRateStrategy(rawValue: arguments[3])
    else { throw CapabilityError.arguments }
    try self.init(sampleRate: rate, channels: channels, strategy: strategy)
  }
}

struct EncodingCapabilities: Codable {
  let codec: String
  let sampleRate: Double
  let channels: UInt32
  let requestedStrategy: BitRateStrategy
  let observedStrategy: String?
  let availableBitRates: [Int]?
  let applicableBitRates: [Int]?
  let mediaEncoded: Bool
  let compressorAcceptanceVerified: Bool
}

enum AudioEncodingCapabilities {
  /// Codec availability only. No buffers are supplied and no encoder conversion occurs.
  static func inspect(_ request: EncodingRequest) throws -> EncodingCapabilities {
    guard
      let input = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: request.sampleRate,
        channels: request.channels, interleaved: false)
    else { throw CapabilityError.invalidFormat }
    let settings: [String: NSNumber] = [
      AVFormatIDKey: NSNumber(value: kAudioFormatMPEG4AAC),
      AVSampleRateKey: NSNumber(value: request.sampleRate),
      AVNumberOfChannelsKey: NSNumber(value: request.channels),
    ]
    guard let output = AVAudioFormat(settings: settings),
      let converter = AVAudioConverter(from: input, to: output)
    else { throw CapabilityError.unavailableConverter }
    converter.bitRateStrategy = request.strategy.nativeValue
    return try observation(
      request, observedStrategy: converter.bitRateStrategy,
      available: converter.availableEncodeBitRates,
      applicable: converter.applicableEncodeBitRates)
  }

  static func observation(
    _ request: EncodingRequest, observedStrategy: String?, available: [NSNumber]?,
    applicable: [NSNumber]?
  ) throws -> EncodingCapabilities {
    try EncodingCapabilities(
      codec: "aac-lc", sampleRate: request.sampleRate, channels: request.channels,
      requestedStrategy: request.strategy, observedStrategy: observedStrategy,
      availableBitRates: rates(available), applicableBitRates: rates(applicable),
      mediaEncoded: false, compressorAcceptanceVerified: false)
  }

  private static func rates(_ values: [NSNumber]?) throws -> [Int]? {
    guard let values else { return nil }
    guard values.count <= 1024 else { throw CapabilityError.invalidRates }
    return try values.map { value in
      let number = value.doubleValue
      guard number.isFinite, number > 0, number <= 100_000_000,
        number.rounded() == number
      else { throw CapabilityError.invalidRates }
      return Int(number)
    }
  }

  static func execute(
    arguments: [String], inspect: (EncodingRequest) throws -> EncodingCapabilities = inspect
  ) throws -> Data {
    let result = try inspect(EncodingRequest(arguments: arguments))
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return try encoder.encode(result)
  }
}
