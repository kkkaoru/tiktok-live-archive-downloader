import CoreGraphics
import CoreML
import Foundation
import Vision

/// CPU-only OCR preparation; no CoreImage context or implicit GPU request.
enum CPUTextRecognition {
  static func cpu(in devices: [MLComputeDevice]) throws -> MLComputeDevice {
    guard
      let device = devices.first(where: {
        if case .cpu = $0 { return true }
        return false
      })
    else { throw ProAppsError.unavailable("OCR compute stage has no CPU device") }
    return device
  }

  static func request() throws -> RecognizeTextRequest {
    var request = RecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = [
      Locale.Language(identifier: "ja-JP"), Locale.Language(identifier: "en-US"),
    ]
    request.usesLanguageCorrection = false
    let stages = request.supportedComputeStageDevices
    guard !stages.isEmpty else {
      throw ProAppsError.unavailable("OCR exposes no configurable CPU compute stages")
    }
    for (stage, devices) in stages {
      request.setComputeDevice(try cpu(in: devices), for: stage)
    }
    return request
  }

  /// Selection is in CoreImage bottom-left pixels; CGImage crops use top-left pixels.
  static func bitmap(_ image: CGImage, selected: CGRect) throws -> CGImage {
    let bounds = CGRect(x: 0, y: 0, width: image.width, height: image.height)
    guard !selected.isEmpty, bounds.contains(selected) else {
      throw ProAppsError.invalid("OCR crop exceeds the decoded image")
    }
    if selected == bounds { return image }
    let crop = CGRect(
      x: selected.minX, y: Double(image.height) - selected.maxY,
      width: selected.width, height: selected.height)
    guard let bitmap = image.cropping(to: crop) else {
      throw ProAppsError.unavailable("Cannot prepare the selected OCR frame")
    }
    return bitmap
  }
}
