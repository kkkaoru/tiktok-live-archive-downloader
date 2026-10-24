import AVFoundation
import CoreVideo
import Foundation

/// One `caption-alpha-render` invocation.
public enum MotionCaptionCommand: Equatable, Sendable {
  case render(MotionCaptionRequest)
  /// The part as read from its Motion document, for round-trip checks.
  case inspect(MotionCaptionPart)
  /// The skeleton text that pins a supported template version.
  case skeleton(String)
}

/// A caption part to draw natively: the same titles a Motion caption part reveals.
public struct MotionCaptionRequest: Codable, Equatable, Sendable {
  public let output: String
  public let width: Int
  public let height: Int
  public let rate: Int
  public let frames: Int
  public let style: MotionCaptionStyle
  public let titles: [MotionCaptionTitle]

  public init(
    output: String, width: Int, height: Int, rate: Int, frames: Int, style: MotionCaptionStyle,
    titles: [MotionCaptionTitle]
  ) {
    self.output = output
    self.width = width
    self.height = height
    self.rate = rate
    self.frames = frames
    self.style = style
    self.titles = titles
  }
}

public struct MotionCaptionReport: Codable, Equatable, Sendable {
  public let frames: Int
  /// Distinct (title, visible graphemes) bitmaps drawn.
  public let states: Int
  public let stateSeconds: Double
  public let totalSeconds: Double
}

/// Writes a caption part as a ProRes 4444 alpha movie (BT.709 tags, like Compressor's), so it
/// can be compared with the Motion render of the same part.
public enum MotionCaptionMovie {
  static let maximumFrames = 60 * 60 * 120
  static let writerPollInterval = Duration.milliseconds(2)

  static let usage =
    "usage: caption-alpha-render REQUEST.json | --moti PART.moti OUTPUT.mov"
    + " | --inspect PART.moti | --skeleton PART.moti"

  /// What `caption-alpha-render` was asked to do (arguments include the program name).
  public static func command(arguments: [String]) throws -> MotionCaptionCommand {
    switch arguments.dropFirst().first {
    case "--moti" where arguments.count == 4:
      let part = try MotionCaptionDocument.read(contentsOf: path(arguments[2]))
      return .render(
        MotionCaptionRequest(
          output: try path(arguments[3]).path, width: part.width, height: part.height,
          rate: part.rate, frames: part.frames, style: part.style, titles: part.titles))
    case "--inspect" where arguments.count == 3:
      return .inspect(try MotionCaptionDocument.read(contentsOf: path(arguments[2])))
    case "--skeleton" where arguments.count == 3:
      return .skeleton(try MotionCaptionDocument.skeleton(Data(contentsOf: path(arguments[2]))))
    case let file? where arguments.count == 2 && !file.hasPrefix("--"):
      let url = try path(file)
      do {
        return .render(
          try JSONDecoder().decode(MotionCaptionRequest.self, from: Data(contentsOf: url)))
      } catch {
        throw ProAppsError.invalid("Unreadable caption request \(url.path): \(error)")
      }
    default:
      throw ProAppsError.invalid(usage)
    }
  }

  static func path(_ argument: String) throws -> URL {
    try Files.absolute(
      argument.hasPrefix("/")
        ? argument : FileManager.default.currentDirectoryPath + "/" + argument)
  }

  public static func render(_ request: MotionCaptionRequest) async throws -> MotionCaptionReport {
    guard (16...8192).contains(request.width), (16...8192).contains(request.height),
      (1...120).contains(request.rate), (1...maximumFrames).contains(request.frames),
      request.titles.allSatisfy({ $0.endFrame <= request.frames })
    else { throw ProAppsError.invalid("Invalid caption movie request") }
    let destination = try Files.absolute(request.output)
    guard !FileManager.default.fileExists(atPath: destination.path) else {
      throw ProAppsError.invalid("Refusing to overwrite \(destination.path)")
    }
    let started = ContinuousClock.now
    let renderers = try request.titles.map {
      try MotionCaptionRenderer(
        $0, style: $0.style ?? request.style, canvasWidth: request.width,
        canvasHeight: request.height)
    }
    let writer = try AVAssetWriter(outputURL: destination, fileType: .mov)
    let input = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.proRes4444,
        AVVideoWidthKey: request.width, AVVideoHeightKey: request.height,
        AVVideoColorPropertiesKey: [
          AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
          AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
          AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
        ],
      ])
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: input,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: request.width,
        kCVPixelBufferHeightKey as String: request.height,
      ])
    try NativeEditor.require(writer.canAdd(input), "The encoder rejected ProRes 4444")
    writer.add(input)
    try NativeEditor.require(writer.startWriting(), "Cannot start the native encoder")
    writer.startSession(atSourceTime: .zero)
    var current: [Int: (visible: Int, bitmap: MotionCaptionBitmap)] = [:]
    var states = 0
    var stateTime = Duration.zero
    do {
      for frame in 0..<request.frames {
        try Task.checkCancellation()
        var shown: [MotionCaptionBitmap] = []
        for (index, title) in request.titles.enumerated() {
          let visible = title.visible(at: frame)
          guard visible > 0 else {
            current[index] = nil
            continue
          }
          if let state = current[index], state.visible == visible {
            shown.append(state.bitmap)
            continue
          }
          let before = ContinuousClock.now
          let bitmap = try renderers[index].bitmap(visible: visible)
          stateTime += ContinuousClock.now - before
          states += 1
          current[index] = (visible, bitmap)
          shown.append(bitmap)
        }
        let buffer = try frameBuffer(
          adaptor, bitmaps: shown, width: request.width,
          height: request.height)
        while !input.isReadyForMoreMediaData {
          try await Task.sleep(for: writerPollInterval)
        }
        try NativeEditor.require(
          adaptor.append(
            buffer,
            withPresentationTime: CMTime(
              value: CMTimeValue(frame),
              timescale: CMTimeScale(request.rate))),
          "The native encoder rejected a frame")
      }
    } catch {
      writer.cancelWriting()
      throw error
    }
    input.markAsFinished()
    await writer.finishWriting()
    try NativeEditor.require(writer.status == .completed, "The native encoder did not finish")
    return MotionCaptionReport(
      frames: request.frames, states: states, stateSeconds: seconds(stateTime),
      totalSeconds: seconds(ContinuousClock.now - started))
  }

  static func seconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
  }

  /// A cleared BGRA frame with the bitmaps composited over it in order (straight alpha).
  static func frameBuffer(
    _ adaptor: AVAssetWriterInputPixelBufferAdaptor, bitmaps: [MotionCaptionBitmap], width: Int,
    height: Int
  ) throws -> CVPixelBuffer {
    let pool = try NativeEditor.required(
      adaptor.pixelBufferPool, "The encoder provided no pixel buffer pool")
    var allocated: CVPixelBuffer?
    let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &allocated)
    let buffer = try NativeEditor.required(
      status == kCVReturnSuccess ? allocated : nil, "Cannot allocate an encoder frame")
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    let base = try NativeEditor.required(
      CVPixelBufferGetBaseAddress(buffer), "Cannot address an encoder frame")
    memset(base, 0, rowBytes * height)
    let pixels = base.bindMemory(to: UInt8.self, capacity: rowBytes * height)
    for bitmap in bitmaps {
      for row in 0..<bitmap.height {
        let y = bitmap.originY + row
        guard (0..<height).contains(y) else { continue }
        for column in 0..<bitmap.width {
          let x = bitmap.originX + column
          let source = (row * bitmap.width + column) * 4
          let alpha = bitmap.rgba[source + 3]
          guard (0..<width).contains(x), alpha > 0 else { continue }
          let target = y * rowBytes + x * 4
          guard pixels[target + 3] == 0 else {
            over(bitmap.rgba, at: source, into: pixels, at: target)
            continue
          }
          pixels[target] = bitmap.rgba[source + 2]
          pixels[target + 1] = bitmap.rgba[source + 1]
          pixels[target + 2] = bitmap.rgba[source]
          pixels[target + 3] = alpha
        }
      }
    }
    return buffer
  }

  /// Straight-alpha RGBA "over" straight-alpha BGRA.
  static func over(
    _ rgba: [UInt8], at source: Int, into pixels: UnsafeMutablePointer<UInt8>, at target: Int
  ) {
    let top = Float(rgba[source + 3]) / 255
    let under = Float(pixels[target + 3]) / 255 * (1 - top)
    let alpha = top + under
    for channel in 0..<3 {
      let slot = 2 - channel
      let value =
        (Float(rgba[source + channel]) * top + Float(pixels[target + slot]) * under)
        / alpha
      pixels[target + slot] = UInt8(min(max(value, 0), 255) + 0.5)
    }
    pixels[target + 3] = UInt8(min(alpha, 1) * 255 + 0.5)
  }
}
