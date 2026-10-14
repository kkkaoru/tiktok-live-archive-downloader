import AVFoundation
import CoreMedia
import Dispatch
import Foundation
import Speech
import WhisperKit

/// One disposable inference child owns the local model. No model/tokenizer
/// downloads, microphone capture, language detection or cloud fallback.
public enum WhisperProbe {
  public struct Request: Codable, Sendable {
    public let path: String
    public let modelDirectory: String
    public let tokenizerDirectory: String

    public init(path: String, modelDirectory: String, tokenizerDirectory: String) {
      self.path = path
      self.modelDirectory = modelDirectory
      self.tokenizerDirectory = tokenizerDirectory
    }
  }

  public static func transcribe(_ request: Request) async throws -> SpeechTranscript {
    try Task.checkCancellation()
    guard #available(macOS 26.0, *) else {
      throw ProAppsError.unavailable("Timed Japanese caption grouping requires macOS 26")
    }
    let input = try await WhisperInput().validate(request)
    let tokenizer = try await LocalWhisperTokenizer.load(directory: request.tokenizerDirectory)
    let configuration = WhisperKitConfig(
      modelFolder: input.model.path, verbose: false, prewarm: false, load: false, download: false)
    let engine = try await OfflineWhisper(configuration)
    engine.tokenizer = tokenizer
    do {
      try Task.checkCancellation()
      try await engine.loadModels()
      try Task.checkCancellation()
      let options = DecodingOptions(
        language: "ja", temperature: 0, temperatureFallbackCount: 0,
        detectLanguage: false, skipSpecialTokens: true, withoutTimestamps: false,
        wordTimestamps: true, suppressBlank: true, concurrentWorkerCount: 1)
      let results = try await engine.transcribe(
        audioPath: input.source.path, decodeOptions: options,
        callback: { _ in !Task.isCancelled })
      try Task.checkCancellation()
      let segments = try WhisperTimedText.segments(
        results.flatMap(\.segments), audioDuration: input.duration)
      let transcript = try SpeechTranscript(
        locale: "ja_JP", durationSeconds: input.duration, segments: segments)
      await engine.unloadModels()
      return transcript
    } catch {
      await engine.unloadModels()
      throw error
    }
  }
}

/// Upstream's convenience tokenizer loader has a network fallback. Never call it.
final class OfflineWhisper: WhisperKit {
  override func loadTokenizerIfNeeded() async throws {
    guard tokenizer != nil else {
      throw ProAppsError.unavailable(
        "A validated local tokenizer must be injected before model loading")
    }
  }
}

private actor WhisperInput {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.whisper-input")
  nonisolated var unownedExecutor: UnownedSerialExecutor { executor.asUnownedSerialExecutor() }

  struct Validated: Sendable {
    let source: URL
    let model: URL
    let duration: Double
  }

  func validate(_ request: WhisperProbe.Request) async throws -> Validated {
    let source = try Files.existing(request.path, extensions: ["wav", "m4a", "aif", "aiff", "caf"])
    let duration = try await AVURLAsset(url: source).load(.duration).seconds
    guard duration.isFinite, duration > 0, duration <= SpeechTranscript.maximumSeconds else {
      throw ProAppsError.invalid("Whisper input must contain at most 60 seconds of audio")
    }
    let model = try Files.absolute(request.modelDirectory).resolvingSymlinksInPath()
    for name in ["AudioEncoder.mlmodelc", "MelSpectrogram.mlmodelc", "TextDecoder.mlmodelc"] {
      let directory = model.appendingPathComponent(name)
      guard
        try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey]).isDirectory
          == true,
        directory.resolvingSymlinksInPath().deletingLastPathComponent() == model
      else {
        throw ProAppsError.invalid(
          "Expected local Whisper model components inside the model directory")
      }
    }
    return Validated(source: source, model: model, duration: duration)
  }
}

@available(macOS 26.0, *)
enum WhisperTimedText {
  static func segments(_ segments: [TranscriptionSegment], audioDuration: Double) throws
    -> [SpeechSegment]
  {
    let maximumWords = SpeechTranscript.maximumSegments
    var text = AttributedString()
    var count = 0
    var bytes = 0
    for segment in segments {
      try Task.checkCancellation()
      if segment.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
      guard let words = segment.words, !words.isEmpty else {
        throw ProAppsError.invalid("Whisper returned lexical text without word timing")
      }
      for word in words {
        count += 1
        bytes += word.word.utf8.count
        guard count <= maximumWords, bytes <= SpeechTranscript.maximumTextBytes else {
          throw ProAppsError.outputLimit
        }
        guard word.start.isFinite, word.end.isFinite, word.start >= 0, word.end >= word.start,
          Double(word.end) <= SpeechTranscript.maximumSeconds + 0.02,
          word.probability.isFinite, (0...1).contains(word.probability)
        else {
          throw ProAppsError.invalid("Whisper returned invalid native word timing or probability")
        }
        var fragment = AttributedString(word.word)
        // Recover integer 20-ms ticks from Float timestamps. CMTime(seconds:)
        // truncates values such as Float(0.9) to the preceding tick and warns.
        // Bounds above make integer conversion safe; no text interpolation.
        fragment.audioTimeRange = CMTimeRange(
          start: CMTime(value: Int64((Double(word.start) * 50).rounded()), timescale: 50),
          end: CMTime(value: Int64((Double(word.end) * 50).rounded()), timescale: 50))
        text += fragment
      }
    }
    return try SpeechTimedText.segments(from: text, audioDuration: audioDuration)
  }
}
