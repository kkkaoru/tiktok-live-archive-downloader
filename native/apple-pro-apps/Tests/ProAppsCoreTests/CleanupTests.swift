import Foundation
import ProAppsCore
import Testing

struct CleanupTests {
  @Test func successfulCleanupIsReported() {
    var performed = false
    #expect(Cleanup.perform { performed = true })
    #expect(performed)
  }

  @Test func failedCleanupIsReportedWithoutReplacingThePrimaryError() {
    #expect(!Cleanup.perform { throw POSIXError(.EACCES) })
  }
}
