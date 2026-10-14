import Foundation
import OSLog

/// Explicit best-effort cleanup during unwinding. Preserve the primary error,
/// but record a bounded diagnostic instead of silently discarding cleanup errors.
public enum Cleanup {
  private static let logger = Logger(subsystem: "local.apple-pro-apps", category: "cleanup")

  @discardableResult
  public static func perform(_ operation: () throws -> Void) -> Bool {
    do {
      try operation()
      return true
    } catch {
      // Error descriptions may contain paths or native command data. Only the
      // numeric code reaches the unified log; callers retain the primary error.
      logger.error("Resource cleanup failed; error code: \((error as NSError).code)")
      return false
    }
  }
}
