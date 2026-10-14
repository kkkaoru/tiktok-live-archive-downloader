import CoreGraphics
import CoreText
import Foundation
import Testing
import Vision

@testable import ProAppsCore

struct CPUTextRecognitionTests {
  @Test func cpuRequestRetainsLanguagesAndRefusesMissingCPU() throws {
    let request = try CPUTextRecognition.request()
    #expect(request.recognitionLevel == .accurate)
    #expect(!request.usesLanguageCorrection)
    #expect(
      request.recognitionLanguages == [
        Locale.Language(identifier: "ja-JP"), Locale.Language(identifier: "en-US"),
      ])
    #expect(
      request.supportedComputeStageDevices.keys.allSatisfy { stage in
        if case .cpu = request.computeDevice(for: stage) { return true }
        return false
      })
    #expect(throws: ProAppsError.self) { try CPUTextRecognition.cpu(in: []) }
  }

  @Test func fullFrameAndCropUseDecodedPixelsWithCPURecognition() async throws {
    let context = try #require(
      CGContext(
        data: nil, width: 512, height: 256, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    context.setFillColor(CGColor(gray: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 512, height: 256))
    let text = NSAttributedString(
      string: "TEST",
      attributes: [
        NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName(
          "Helvetica-Bold" as CFString, 64, nil),
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(
          gray: 1, alpha: 1),
      ])
    context.textPosition = CGPoint(x: 40, y: 160)
    CTLineDraw(CTLineCreateWithAttributedString(text), context)
    let image = try #require(context.makeImage())
    let full = try CPUTextRecognition.bitmap(
      image, selected: CGRect(x: 0, y: 0, width: 512, height: 256))
    #expect(full === image)
    let crop = try CPUTextRecognition.bitmap(
      image, selected: CGRect(x: 20, y: 140, width: 300, height: 100))
    #expect(crop.width == 300 && crop.height == 100)
    let request = try CPUTextRecognition.request()
    let fullText = try await request.perform(on: full)
    let cropText = try await request.perform(on: crop)
    #expect(fullText.flatMap { $0.topCandidates(1).map(\.string) }.contains("TEST"))
    #expect(cropText.flatMap { $0.topCandidates(1).map(\.string) }.contains("TEST"))
    #expect(throws: ProAppsError.self) {
      try CPUTextRecognition.bitmap(image, selected: CGRect(x: -1, y: 0, width: 512, height: 256))
    }
  }

  @Test func fullFrameVideoOCRKeepsNativeClockAndGeometry() async throws {
    let source = try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures"))
    let result = try await FrameProbe().recognizeText(
      path: source.path, samples: [.init(timeSeconds: 0)])
    let frame = try #require(result.first)
    #expect(frame.width == 16 && frame.height == 16)
    #expect(frame.actualTimeSeconds == 0 && frame.requestedTimeSeconds == 0)
    #expect(frame.lines.isEmpty)
  }
}
