import AVFoundation
import Foundation
import ProAppsCore
import Vision

/// Framework domain/code only: never emit descriptions, paths or userInfo.
enum NativeFailureMessage {
  static func describe(_ error: any Error) -> String {
    if let failure = error as? ProAppsError { return failure.description }
    if let failure = error as? NativeOperationFailure {
      let clock = failure.sourceTime.map { "; sourcePTS=\($0.value)/\($0.timescale)" } ?? ""
      return frameworkMessage(domain: failure.domain, code: failure.code)
        + " operation=\(failure.operation.rawValue)\(clock)"
    }
    let failure = error as NSError
    return frameworkMessage(domain: failure.domain, code: failure.code)
  }

  private static func frameworkMessage(domain originalDomain: String, code: Int) -> String {
    let domain: String
    switch originalDomain {
    case NSCocoaErrorDomain, NSPOSIXErrorDomain, NSOSStatusErrorDomain,
      AVFoundationErrorDomain, VNErrorDomain:
      domain = originalDomain
    default:
      domain = "unclassified"
    }
    return
      "Native operation failed (\(domain), code \(code)); inspect the selected input and permissions."
  }
}
