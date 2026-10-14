import AVFoundation
import Foundation
import Testing
import Vision

@testable import AppleProApps
@testable import ProAppsCore

struct NativeFailureMessageTests {
  @Test(arguments: [
    NSCocoaErrorDomain, NSPOSIXErrorDomain, NSOSStatusErrorDomain,
    AVFoundationErrorDomain, VNErrorDomain,
  ])
  func reportsFrameworkCodeWithoutDescriptionOrUserInfo(_ domain: String) {
    let error = NSError(
      domain: domain, code: -123,
      userInfo: [NSLocalizedDescriptionKey: "sensitive-source-path"])
    let result = NativeFailureMessage.describe(error)
    #expect(result.contains(domain) && result.contains("code -123"))
    #expect(!result.contains("sensitive-source-path"))
  }

  @Test func unknownDomainIsNotPrinted() {
    let result = NativeFailureMessage.describe(
      NSError(domain: "sensitive-custom-domain", code: 42))
    #expect(result.contains("unclassified, code 42"))
    #expect(!result.contains("sensitive-custom-domain"))
  }

  @Test func typedOperationsKeepSafeClockAndRedactUnknownDomains() {
    let timed = NativeOperationFailure(
      domain: VNErrorDomain, code: 9, operation: .trackingSeed,
      sourceTime: CMTime(value: 123, timescale: 90000))
    let result = NativeFailureMessage.describe(timed)
    #expect(result.contains("operation=trackingSeed; sourcePTS=123/90000"))
    #expect(result.contains("com.apple.Vision, code 9"))
    let unknown = NativeOperationFailure(
      domain: "sensitive-custom-domain", code: 9, operation: .trackingAdvance, sourceTime: nil)
    let redacted = NativeFailureMessage.describe(unknown)
    #expect(redacted.contains("unclassified, code 9"))
    #expect(redacted.contains("operation=trackingAdvance"))
    #expect(!redacted.contains("sensitive-custom-domain") && !redacted.contains("sourcePTS"))
  }

  @Test func preservesStructuredApplicationFailures() {
    #expect(
      NativeFailureMessage.describe(ProAppsError.invalid("Synthetic invalid input"))
        == "Invalid request: Synthetic invalid input")
  }
}
