import CoreGraphics
import CoreImage
import CoreText
import Foundation
import Testing

@testable import ProAppsCore

struct TextRectangleProbeTests {
  @Test func legacyMeasurementRemainsDecodable() throws {
    let data = Data(#"{"width":8,"height":8,"characterBoxes":[]}"#.utf8)
    let result = try JSONDecoder().decode(TextRectangleMeasurement.self, from: data)
    #expect(result.width == 8 && result.height == 8)
    #expect(result.characterBoxes.isEmpty)
    #expect(result.textBoxes == nil)
    #expect(result.diagnostics == nil)
  }

  @Test func coordinatesUseTopLeftPixelsAndRejectInvalidGeometry() throws {
    let box = try TextRectangleProbe.pixelBox(
      CGRect(x: 0.125, y: 0.25, width: 0.25, height: 0.5), width: 128, height: 64)
    #expect(box.x == 16 && box.y == 16 && box.width == 32 && box.height == 32)
    #expect(throws: ProAppsError.self) {
      try TextRectangleProbe.pixelBox(.infinite, width: 128, height: 64)
    }
    #expect(throws: ProAppsError.self) {
      try TextRectangleProbe.pixelBox(
        CGRect(x: -0.1, y: 0, width: 0.2, height: 0.2), width: 128, height: 64)
    }
  }

  @Test func zeroAreaTextIsExplicitButCharactersRemainStrict() throws {
    let empty = CGRect(x: 0.25, y: 1, width: 0.5, height: 0)
    #expect(try TextRectangleProbe.textBox(empty, width: 128, height: 64) == nil)
    #expect(throws: ProAppsError.self) {
      try TextRectangleProbe.pixelBox(empty, width: 128, height: 64)
    }
    let region = try TextRectangleProbe.textBox(
      CGRect(x: 0.125, y: 0.25, width: 0.25, height: 0.5), width: 128, height: 64)
    #expect(region?.x == 16 && region?.y == 16 && region?.width == 32 && region?.height == 32)
    #expect(throws: ProAppsError.self) {
      try TextRectangleProbe.textBox(
        CGRect(x: -0.1, y: 1, width: 0.5, height: 0), width: 128, height: 64)
    }
    #expect(throws: ProAppsError.self) {
      try TextRectangleProbe.textBox(
        CGRect(x: 0.5, y: 1.1, width: 0, height: 0), width: 128, height: 64)
    }
  }

  @Test func detectsIndividualSyntheticCharactersAndEmptyControl() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "text-boxes-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let context = try #require(
      CGContext(
        data: nil, width: 320, height: 128,
        bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    context.setFillColor(CGColor(gray: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 320, height: 128))
    let blank = CIImage(cgImage: try #require(context.makeImage()))
    let blankData = try #require(
      CIContext().pngRepresentation(
        of: blank, format: .RGBA8,
        colorSpace: CGColorSpaceCreateDeviceRGB()))
    let blankPath = root.appendingPathComponent("blank.png")
    try blankData.write(to: blankPath, options: .withoutOverwriting)
    let empty = try await TextRectangleProbe().detect(path: blankPath.path)
    #expect(empty.width == 320 && empty.height == 128 && empty.characterBoxes.isEmpty)
    #expect(empty.textBoxes?.isEmpty == true)
    let diagnostics = try #require(empty.diagnostics)
    #expect(diagnostics.observationCount == 0)
    #expect(diagnostics.charactersPerObservation.isEmpty)
    #expect(diagnostics.requestRevision > 0)
    #expect(diagnostics.bitsPerComponent > 0)
    #expect(diagnostics.bitsPerPixel >= 24)
    #expect(diagnostics.bytesPerRow >= 960)
    #expect(diagnostics.colorSpaceName != nil)
    let text = NSAttributedString(
      string: "TEST",
      attributes: [
        NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName(
          "Helvetica-Bold" as CFString, 64, nil),
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(
          gray: 1, alpha: 1),
      ])
    context.textPosition = CGPoint(x: 40, y: 40)
    CTLineDraw(CTLineCreateWithAttributedString(text), context)
    let image = CIImage(cgImage: try #require(context.makeImage()))
    let data = try #require(
      CIContext().pngRepresentation(
        of: image, format: .RGBA8,
        colorSpace: CGColorSpaceCreateDeviceRGB()))
    let path = root.appendingPathComponent("text.png")
    try data.write(to: path, options: .withoutOverwriting)
    let measured = try await TextRectangleProbe().detect(path: path.path)
    #expect(measured.characterBoxes.count >= 4)
    let measuredDiagnostics = try #require(measured.diagnostics)
    #expect(measuredDiagnostics.observationCount > 0)
    let textBoxes = try #require(measured.textBoxes)
    #expect(textBoxes.count == measuredDiagnostics.observationCount)
    #expect(textBoxes.allSatisfy { $0.width > 80 && $0.height > 0 })
    #expect(
      measuredDiagnostics.charactersPerObservation.reduce(0, +) == measured.characterBoxes.count)
    #expect(
      measured.characterBoxes.allSatisfy {
        $0.x >= 30 && $0.y >= 20 && $0.width < 80 && $0.height < 80
      })
    #expect(try Data(contentsOf: path) == data)
    let request = root.appendingPathComponent("request.json")
    try JSONEncoder().encode(["path": path.path]).write(to: request, options: .withoutOverwriting)
    let products = Bundle(for: TestBundle.self).bundleURL.deletingLastPathComponent()
    var environment = ProcessInfo.processInfo.environment
    environment["LLVM_PROFILE_FILE"] =
      products.appendingPathComponent("codecov/text-boxes-%p-%m.profraw").path
    let result = try await Runner.run(
      products.appendingPathComponent("apple-pro-apps"),
      ["measure-media", "image_text_rectangles", request.path], environment: environment)
    #expect(result.status == 0)
    let cli = try JSONDecoder().decode(
      TextRectangleMeasurement.self, from: Data(result.stdout.utf8))
    #expect(cli.characterBoxes.count >= 4)
    #expect(cli.textBoxes?.count == measuredDiagnostics.observationCount)
    let batchRequest = root.appendingPathComponent("batch.json")
    try JSONEncoder().encode(["paths": [path.path, blankPath.path]]).write(
      to: batchRequest, options: .withoutOverwriting)
    let batchProcess = try await Runner.run(
      products.appendingPathComponent("apple-pro-apps"),
      ["measure-media", "image_text_rectangles_batch", batchRequest.path], environment: environment)
    #expect(batchProcess.status == 0)
    let batch = try JSONDecoder().decode(
      [TextRectangleMeasurement].self, from: Data(batchProcess.stdout.utf8))
    #expect(batch.count == 2)
    #expect(batch.first?.characterBoxes.count == cli.characterBoxes.count)
    #expect(batch.last?.characterBoxes.isEmpty == true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    #expect(try encoder.encode(batch[0]) == encoder.encode(cli))
    await #expect(throws: ProAppsError.self) {
      try await TextRectangleProbe().detect(paths: [])
    }
    await #expect(throws: ProAppsError.self) {
      try await TextRectangleProbe().detect(paths: Array(repeating: path.path, count: 9))
    }
    let invalid = root.appendingPathComponent("invalid.png")
    try Data([1, 2, 3]).write(to: invalid, options: .withoutOverwriting)
    await #expect(throws: ProAppsError.self) {
      try await TextRectangleProbe().detect(path: invalid.path)
    }
  }
}
