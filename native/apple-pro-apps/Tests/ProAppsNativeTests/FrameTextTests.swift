import CoreGraphics
import CoreText
import Foundation
import ImageIO
import Testing

@testable import ProAppsCore

struct FrameTextTests {
  private func textVideo(in root: URL, frameRate: String = "30", offset: String = "0") async throws
    -> URL
  {
    let context = try #require(
      CGContext(
        data: nil, width: 640, height: 360, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    context.setFillColor(CGColor(gray: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 640, height: 360))
    let text = NSAttributedString(
      string: "TEST",
      attributes: [
        NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName(
          "Helvetica" as CFString, 64, nil),
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(
          gray: 1, alpha: 1),
      ])
    context.textPosition = CGPoint(x: 40, y: 100)
    CTLineDraw(CTLineCreateWithAttributedString(text), context)
    let image = try #require(context.makeImage())
    let png = root.appendingPathComponent("text.png")
    let destination = try #require(
      CGImageDestinationCreateWithURL(png as CFURL, "public.png" as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image, nil)
    try #require(CGImageDestinationFinalize(destination))
    let output = root.appendingPathComponent("text.mp4")
    let result = try await Runner.run(
      URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg"),
      [
        "-v", "error", "-nostdin", "-n", "-loop", "1", "-framerate", frameRate, "-i", png.path,
        "-t", "1",
        "-vf", "setpts=PTS+\(offset)/TB", "-copyts", "-fps_mode", "passthrough",
        "-enc_time_base", "1:60000", "-video_track_timescale", "60000",
        "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", output.path,
      ])
    try #require(result.status == 0)
    return output
  }
  @Test func integerMillisecondSourceFramesAreNotSelectedOneTickEarly() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "ocr-clock-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await textVideo(in: root, frameRate: "1000/33", offset: "0.165")
    let result = try await FrameProbe().recognizeText(
      path: source.path,
      samples: [
        .init(timeSeconds: 0.198), .init(timeSeconds: 0.264), .init(timeSeconds: 0.363),
      ])
    #expect(result.map(\.actualTimeSeconds) == [0.198, 0.264, 0.363])
    #expect(result.allSatisfy { $0.lines.contains { $0.text.contains("TEST") } })
    await #expect(throws: ProAppsError.self) {
      try await FrameProbe().recognizeText(
        path: source.path, samples: [.init(timeSeconds: 1.0 / 90001)])
    }
  }
  @Test func containedFractionalRectangleDoesNotAcquireAClippingFlag() throws {
    let rectangle = try FrameProbe.textRectangle(
      CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4),
      imageSize: CGSize(width: 100, height: 100))
    #expect(rectangle == CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4))
  }

  @Test func slightOCRBoxOverflowIsClippedAndInvalidBoxesAreRejected() throws {
    let clipped = try FrameProbe.textRectangle(
      CGRect(x: -0.25, y: 7, width: 20, height: 12), imageSize: CGSize(width: 100, height: 100))
    #expect(clipped == CGRect(x: 0, y: 7, width: 19.75, height: 12))
    #expect(throws: ProAppsError.self) {
      try FrameProbe.textRectangle(.infinite, imageSize: CGSize(width: 100, height: 100))
    }
    #expect(throws: ProAppsError.self) {
      try FrameProbe.textRectangle(
        CGRect(x: 120, y: 0, width: 10, height: 10), imageSize: CGSize(width: 100, height: 100))
    }
  }

  @Test func localOCRFindsVisibleTextAndMapsCroppedCoordinates() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "ocr-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let rendered = try await textVideo(in: root)
    let before = try Data(contentsOf: rendered)
    let samples: [FrameSample] = [
      .init(timeSeconds: 0.3),
      .init(timeSeconds: 0.3, region: .init(x: 20, y: 180, width: 300, height: 130)),
      .init(timeSeconds: 0.3, region: .init(x: 350, y: 20, width: 200, height: 100)),
    ]
    let results = try await FrameProbe().recognizeText(path: rendered.path, samples: samples)
    try #require(results.count == 3)
    let full = try #require(results[0].lines.first { $0.text.contains("TEST") })
    let cropped = try #require(results[1].lines.first { $0.text.contains("TEST") })
    #expect(results[0].actualTimeSeconds == 0.3)
    #expect(full.confidence > 0 && full.confidence <= 1)
    #expect(!full.boundsClipped)
    #expect(!cropped.boundsClipped)
    #expect(full.region.x > 30 && full.region.x < 80)
    #expect(full.region.y > 190 && full.region.y < 260)
    #expect(abs(full.region.x - cropped.region.x) < 5)
    #expect(abs(full.region.y - cropped.region.y) < 5)
    #expect(results[2].lines.isEmpty)
    struct Input: Codable {
      let path: String
      let samples: [FrameSample]
    }
    let request = root.appendingPathComponent("request.json")
    try JSONEncoder().encode(Input(path: rendered.path, samples: samples)).write(
      to: request, options: .withoutOverwriting)
    let products = Bundle(for: TestBundle.self).bundleURL.deletingLastPathComponent()
    var environment = ProcessInfo.processInfo.environment
    environment["LLVM_PROFILE_FILE"] =
      products.appendingPathComponent("codecov/ocr-%p-%m.profraw").path
    let process = try await Runner.run(
      products.appendingPathComponent("apple-pro-apps"),
      ["measure-media", "video_text_recognize", request.path], environment: environment)
    try #require(process.status == 0)
    let cli = try JSONDecoder().decode([FrameTextMeasurement].self, from: Data(process.stdout.utf8))
    #expect(cli.first?.lines.contains { $0.text.contains("TEST") } == true)
    #expect(try Data(contentsOf: rendered) == before)
    await #expect(throws: ProAppsError.self) {
      try await FrameProbe().recognizeText(path: rendered.path, samples: [])
    }
    await #expect(throws: ProAppsError.self) {
      try await FrameProbe().recognizeText(
        path: rendered.path, samples: [.init(timeSeconds: 2)])
    }
    await #expect(throws: ProAppsError.self) {
      try await FrameProbe().recognizeText(
        path: rendered.path,
        samples: [.init(timeSeconds: 0, region: .init(x: -1, y: 0, width: 20, height: 20))])
    }
  }
}
