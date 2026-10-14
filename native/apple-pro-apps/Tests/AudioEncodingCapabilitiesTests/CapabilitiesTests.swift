import AVFAudio
import Foundation
import Testing

@testable import audio_encoding_capabilities

struct AudioEncodingCapabilitiesTests {
  @Test func commandSuccessAndError() {
    var output = Data()
    var errors = Data()
    let success = AudioEncodingCapabilitiesCommand.run(
      arguments: ["probe"], execute: { _ in Data("{}".utf8) },
      output: { output.append($0) }, error: { errors.append($0) })
    #expect(success == 0)
    #expect(output == Data("{}\n".utf8))
    #expect(errors.isEmpty)
    let failure = AudioEncodingCapabilitiesCommand.run(
      arguments: ["probe"], output: { output.append($0) }, error: { errors.append($0) })
    #expect(failure == 1)
    #expect(output == Data("{}\n".utf8))
    #expect(String(decoding: errors, as: UTF8.self).contains("arguments"))
  }

  @Test func argumentsAndStrategies() throws {
    let request = try EncodingRequest(arguments: ["probe", "48000", "2", "constant"])
    #expect(request.sampleRate == 48000)
    #expect(request.channels == 2)
    #expect(request.strategy == .constant)
    #expect(BitRateStrategy.constant.nativeValue == AVAudioBitRateStrategy_Constant)
    #expect(BitRateStrategy.longTermAverage.nativeValue == AVAudioBitRateStrategy_LongTermAverage)
    #expect(
      BitRateStrategy.constrainedVariable.nativeValue
        == AVAudioBitRateStrategy_VariableConstrained)
    #expect(BitRateStrategy.variable.nativeValue == AVAudioBitRateStrategy_Variable)
  }

  @Test(arguments: [
    ["probe"], ["probe", "x", "2", "constant"],
    ["probe", "48000", "-1", "constant"], ["probe", "48000", "2", "other"],
  ])
  func badArguments(_ arguments: [String]) {
    #expect(throws: CapabilityError.arguments) { try EncodingRequest(arguments: arguments) }
  }

  @Test(arguments: [Double.nan, Double.infinity, 7999, 192001, 48000.5])
  func badRates(_ rate: Double) {
    #expect(throws: CapabilityError.invalidFormat) {
      try EncodingRequest(sampleRate: rate, channels: 2, strategy: .constant)
    }
  }

  @Test(arguments: [UInt32(0), 9]) func badChannels(_ channels: UInt32) {
    #expect(throws: CapabilityError.invalidFormat) {
      try EncodingRequest(sampleRate: 48000, channels: channels, strategy: .constant)
    }
  }

  @Test func absenceIsNotSupport() throws {
    let request = try EncodingRequest(sampleRate: 48000, channels: 2, strategy: .constant)
    let value = try AudioEncodingCapabilities.observation(
      request, observedStrategy: nil, available: nil, applicable: nil)
    #expect(value.availableBitRates == nil)
    #expect(value.applicableBitRates == nil)
    #expect(!value.mediaEncoded)
    #expect(!value.compressorAcceptanceVerified)
  }

  @Test func observedRatesAndEncoding() throws {
    let data = try AudioEncodingCapabilities.execute(
      arguments: ["probe", "48000", "2", "constant"]
    ) { request in
      try AudioEncodingCapabilities.observation(
        request, observedStrategy: "observed", available: [128000, 384000],
        applicable: [128000, 320000])
    }
    let value = try JSONDecoder().decode(EncodingCapabilities.self, from: data)
    #expect(value.codec == "aac-lc")
    #expect(value.availableBitRates == [128000, 384000])
    #expect(value.applicableBitRates == [128000, 320000])
    #expect(value.observedStrategy == "observed")
  }

  @Test(arguments: [Double.nan, Double.infinity, -1, 0, 0.5, 100_000_001])
  func badObservations(_ value: Double) throws {
    let request = try EncodingRequest(sampleRate: 48000, channels: 2, strategy: .constant)
    #expect(throws: CapabilityError.invalidRates) {
      try AudioEncodingCapabilities.observation(
        request, observedStrategy: nil, available: [NSNumber(value: value)], applicable: [])
    }
  }

  @Test func oversizedAndDependencyFailure() throws {
    let request = try EncodingRequest(sampleRate: 48000, channels: 2, strategy: .constant)
    #expect(throws: CapabilityError.invalidRates) {
      try AudioEncodingCapabilities.observation(
        request, observedStrategy: nil, available: [],
        applicable: Array(repeating: NSNumber(value: 128000), count: 1025))
    }
    #expect(throws: CapabilityError.unavailableConverter) {
      try AudioEncodingCapabilities.execute(arguments: ["probe", "48000", "2", "constant"]) { _ in
        throw CapabilityError.unavailableConverter
      }
    }
  }
}
