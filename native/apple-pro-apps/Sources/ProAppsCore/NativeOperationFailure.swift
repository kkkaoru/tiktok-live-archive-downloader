import CoreMedia
import Foundation

/// Safe framework failure context; deliberately retains no descriptions or userInfo.
public struct NativeOperationFailure: Error, Sendable {
  public enum Operation: String, Sendable {
    case faceDetection, trackingSeed, trackingAdvance
  }

  public let domain: String
  public let code: Int
  public let operation: Operation
  public let sourceTime: CMTime?

  static func perform<Value>(_ operation: Operation, _ body: () throws -> Value) throws -> Value {
    do {
      return try body()
    } catch let failure as CancellationError {
      throw failure
    } catch let failure as ProAppsError {
      throw failure
    } catch let failure as NativeOperationFailure {
      throw failure
    } catch {
      let failure = error as NSError
      throw NativeOperationFailure(
        domain: failure.domain, code: failure.code, operation: operation, sourceTime: nil)
    }
  }

  func at(_ sourceTime: CMTime) -> NativeOperationFailure {
    NativeOperationFailure(
      domain: domain, code: code, operation: operation, sourceTime: sourceTime)
  }
}
