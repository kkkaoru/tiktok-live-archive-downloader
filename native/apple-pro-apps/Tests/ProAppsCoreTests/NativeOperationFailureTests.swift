import CoreMedia
import Foundation
import Testing

@testable import ProAppsCore

struct NativeOperationFailureTests {
  @Test func preservesSuccessAndApplicationErrors() throws {
    #expect(try NativeOperationFailure.perform(.faceDetection) { 42 } == 42)
    #expect(throws: ProAppsError.invalid("Synthetic failure")) {
      try NativeOperationFailure.perform(.faceDetection) {
        throw ProAppsError.invalid("Synthetic failure")
      }
    }
    #expect(throws: CancellationError.self) {
      try NativeOperationFailure.perform(.trackingSeed) { throw CancellationError() }
    }
  }

  @Test func contextKeepsRootCodeAndOperationWithoutDescriptions() throws {
    do {
      try NativeOperationFailure.perform(.trackingSeed) {
        throw NSError(
          domain: "synthetic.framework", code: 9,
          userInfo: [NSLocalizedDescriptionKey: "sensitive-source-path"])
      }
      Issue.record("Expected framework failure")
    } catch let failure as NativeOperationFailure {
      #expect(failure.domain == "synthetic.framework" && failure.code == 9)
      #expect(failure.operation == .trackingSeed && failure.sourceTime == nil)
      let timed = failure.at(CMTime(value: 123, timescale: 90000))
      #expect(timed.sourceTime?.value == 123 && timed.sourceTime?.timescale == 90000)
      do {
        try NativeOperationFailure.perform(.faceDetection) { throw timed }
        Issue.record("Expected preserved framework failure")
      } catch let preserved as NativeOperationFailure {
        #expect(preserved.operation == .trackingSeed && preserved.code == 9)
        #expect(preserved.sourceTime == CMTime(value: 123, timescale: 90000))
      }
    }
  }
}
