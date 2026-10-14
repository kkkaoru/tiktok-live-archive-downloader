import AVFoundation
import CoreVideo
import VideoToolbox

/// The single approved AVAssetWriter/AVAssetReader settings boundary (see
/// NATIVE-BOUNDARIES.md, 2026-10-03). AVFoundation only accepts these as
/// `[String: Any]`; every dictionary here is built from fixed keys and typed values
/// (`EditEncoding`, canvas size and frame rate). Callers never supply dictionaries.
enum VideoWriterSettings {
  /// 30 fps output uses a 15360 track timescale, like the FFmpeg part renderer.
  static let timescalePerFrame: Int32 = 512

  static func output(_ encoding: EditEncoding, width: Int, height: Int, frameRate: Int)
    -> [String: Any]
  {
    let profile: String
    let codec: AVVideoCodecType
    switch encoding.codec {
    case .h264:
      codec = .h264
      profile = AVVideoProfileLevelH264HighAutoLevel
    case .hevc:
      codec = .hevc
      profile = kVTProfileLevel_HEVC_Main_AutoLevel as String
    }
    return [
      AVVideoCodecKey: codec,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      AVVideoColorPropertiesKey: [
        AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
        AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
        AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
      ],
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: encoding.averageBitRate,
        AVVideoAllowFrameReorderingKey: encoding.allowFrameReordering,
        AVVideoExpectedSourceFrameRateKey: frameRate,
        AVVideoProfileLevelKey: profile,
      ],
    ]
  }

  /// BGRA buffers for both the composition reader and the writer adaptor.
  static func pixelBuffers(width: Int, height: Int) -> [String: Any] {
    [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
    ]
  }

  static func readerPixels() -> [String: Any] {
    [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
  }
}
