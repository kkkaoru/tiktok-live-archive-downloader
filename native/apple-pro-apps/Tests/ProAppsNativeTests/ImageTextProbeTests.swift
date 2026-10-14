import CoreGraphics
import CoreImage
import CoreText
import Foundation
import Testing

@testable import ProAppsCore

struct ImageTextProbeTests {
  @Test func recognizesSyntheticRegionsInOrderWithoutChangingSources() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "image-text-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let context = try #require(
      CGContext(
        data: nil, width: 512, height: 128, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    context.setFillColor(CGColor(gray: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 512, height: 128))
    let blankBitmap = try #require(context.makeImage())
    let blank = try #require(
      CIContext().pngRepresentation(
        of: CIImage(cgImage: blankBitmap), format: .RGBA8,
        colorSpace: CGColorSpaceCreateDeviceRGB()))
    let blankPath = root.appendingPathComponent("blank.png")
    try blank.write(to: blankPath, options: .withoutOverwriting)
    let text = NSAttributedString(
      string: "字幕 TEST",
      attributes: [
        NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName(
          "Helvetica-Bold" as CFString, 64, nil),
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(
          gray: 1, alpha: 1),
      ])
    context.textPosition = CGPoint(x: 40, y: 40)
    CTLineDraw(CTLineCreateWithAttributedString(text), context)
    let bitmap = try #require(context.makeImage())
    let data = try #require(
      CIContext().pngRepresentation(
        of: CIImage(cgImage: bitmap), format: .RGBA8,
        colorSpace: CGColorSpaceCreateDeviceRGB()))
    let path = root.appendingPathComponent("text.png")
    try data.write(to: path, options: .withoutOverwriting)
    let result = try await ImageTextProbe().recognize(paths: [path.path, blankPath.path])
    #expect(result.count == 2)
    let first = try #require(result.first)
    #expect(first.width == 512 && first.height == 128)
    #expect(first.lines.map(\.text).joined().contains("TEST"))
    #expect(first.lines.map(\.text).joined().contains("字幕"))
    #expect(first.lines.allSatisfy { $0.region.x >= 0 && $0.region.y >= 0 && !$0.boundsClipped })
    #expect(result.last?.lines.isEmpty == true)
    #expect(try Data(contentsOf: path) == data)
    let shared = ImageTextProbe()
    async let left = shared.recognize(paths: [path.path])
    async let right = shared.recognize(paths: [blankPath.path])
    let concurrent = try await (left, right)
    #expect(concurrent.0.first?.lines.map(\.text) == first.lines.map(\.text))
    #expect(concurrent.1.first?.lines.isEmpty == true)
    let request = root.appendingPathComponent("request.json")
    try JSONEncoder().encode(["paths": [path.path, blankPath.path]]).write(
      to: request, options: .withoutOverwriting)
    let products = Bundle(for: TestBundle.self).bundleURL.deletingLastPathComponent()
    var environment = ProcessInfo.processInfo.environment
    environment["LLVM_PROFILE_FILE"] =
      products.appendingPathComponent("codecov/image-text-%p-%m.profraw").path
    let process = try await Runner.run(
      products.appendingPathComponent("apple-pro-apps"),
      ["measure-media", "image_text_recognize_batch", request.path], environment: environment)
    #expect(process.status == 0)
    let measured = try JSONDecoder().decode(
      [ImageTextMeasurement].self, from: Data(process.stdout.utf8))
    #expect(measured.first?.lines.map(\.text) == first.lines.map(\.text))
    #expect(measured.last?.lines.isEmpty == true)
    await #expect(throws: ProAppsError.self) { try await ImageTextProbe().recognize(paths: []) }
    await #expect(throws: ProAppsError.self) {
      try await ImageTextProbe().recognize(paths: Array(repeating: path.path, count: 9))
    }
    let invalid = root.appendingPathComponent("invalid.png")
    try Data([1, 2, 3]).write(to: invalid, options: .withoutOverwriting)
    await #expect(throws: ProAppsError.self) {
      try await ImageTextProbe().recognize(paths: [invalid.path])
    }
  }
}
