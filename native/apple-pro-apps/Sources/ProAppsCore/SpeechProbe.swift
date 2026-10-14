import AVFoundation
import Dispatch
import Foundation
import Speech

public struct SpeechLocaleReservation: Codable, Sendable {
  public let locale: String
  public let newlyReserved: Bool
  public let readyForTranscription: Bool
}

/// Transcription uses only prepared SpeechTranscriber models, with no installation,
/// microphone capture or cloud fallback. The separate explicit reserve operation
/// changes this application's locale reservation; it never downloads or releases.
public actor SpeechProbe {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.speech")
  nonisolated public var unownedExecutor: UnownedSerialExecutor {
    executor.asUnownedSerialExecutor()
  }

  public init() {}

  public func reserve(locale identifier: String) async throws -> SpeechLocaleReservation {
    try Task.checkCancellation()
    guard #available(macOS 26.0, *) else {
      throw ProAppsError.unavailable("Speech locale reservation requires macOS 26")
    }
    guard SpeechTranscriber.isAvailable,
      let locale = await SpeechTranscriber.supportedLocale(
        equivalentTo: Locale(identifier: identifier))
    else { throw ProAppsError.unavailable("Requested on-device speech locale is unsupported") }
    let newlyReserved = try await AssetInventory.reserve(locale: locale)
    let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
    let ready = await AssetInventory.status(forModules: [transcriber]) == .installed
    return SpeechLocaleReservation(
      locale: locale.identifier, newlyReserved: newlyReserved, readyForTranscription: ready)
  }

  public func transcribe(
    path: String, locale identifier: String, options: SpeechRecognitionOptions? = nil
  ) async throws -> SpeechTranscript {
    try Task.checkCancellation()
    guard #available(macOS 26.0, *) else {
      throw ProAppsError.unavailable("Local SpeechAnalyzer transcription requires macOS 26")
    }
    let source = try Files.existing(path, extensions: ["m4a", "wav", "aif", "aiff", "caf"])
    let asset = AVURLAsset(url: source)
    let duration = try await asset.load(.duration).seconds
    guard duration.isFinite, duration > 0, duration <= SpeechTranscript.maximumSeconds else {
      throw ProAppsError.invalid("Transcription requires an audio file of at most 60 seconds")
    }
    guard SpeechTranscriber.isAvailable,
      let locale = await SpeechTranscriber.supportedLocale(
        equivalentTo: Locale(identifier: identifier))
    else { throw ProAppsError.unavailable("Requested on-device speech locale is unsupported") }
    let transcriber =
      if options?.wordTiming == true {
        SpeechTranscriber(
          locale: locale, transcriptionOptions: [], reportingOptions: [],
          attributeOptions: [.audioTimeRange])
      } else {
        SpeechTranscriber(locale: locale, preset: .transcription)
      }
    guard await AssetInventory.status(forModules: [transcriber]) == .installed else {
      throw ProAppsError.unavailable(
        "Speech assets are not ready for this app; explicitly reserve the locale first. No automatic reservation or download was performed"
      )
    }
    try Task.checkCancellation()
    let work = NativeSpeechWork(
      source: source, transcriber: transcriber, audioDuration: duration, options: options)
    let segments = try await SpeechRun.collect(work)
    return try SpeechTranscript(
      locale: locale.identifier, durationSeconds: duration, segments: segments)
  }
}

@available(macOS 26.0, *)
actor NativeSpeechWork: SpeechWork {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.speech.file")
  nonisolated var unownedExecutor: UnownedSerialExecutor { executor.asUnownedSerialExecutor() }
  private let source: URL
  private let transcriber: SpeechTranscriber
  private let analyzer: SpeechAnalyzer
  private let audioDuration: Double
  private let options: SpeechRecognitionOptions?

  init(
    source: URL, transcriber: SpeechTranscriber, audioDuration: Double,
    options: SpeechRecognitionOptions? = nil
  ) {
    self.source = source
    self.transcriber = transcriber
    self.audioDuration = audioDuration
    self.options = options
    self.analyzer = SpeechAnalyzer(modules: [transcriber])
  }

  func analyze() async throws {
    try Task.checkCancellation()
    let audio = try AVAudioFile(forReading: source)
    if let vocabulary = options?.contextualStrings, !vocabulary.isEmpty {
      let context = AnalysisContext()
      context.contextualStrings = [.general: vocabulary]
      try await analyzer.setContext(context)
    }
    if let end = try await analyzer.analyzeSequence(from: audio) {
      try await analyzer.finalizeAndFinish(through: end)
    } else {
      await analyzer.cancelAndFinishNow()
    }
  }

  func collect() async throws -> [SpeechSegment] {
    var segments: [SpeechSegment] = []
    var bytes = 0
    for try await result in transcriber.results {
      try Task.checkCancellation()
      // No volatile/fast results are requested. Native run times remain ASR
      // estimates, not human-reviewed forced alignment or speaker identity.
      let text = String(result.text.characters)
      guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
      bytes += text.utf8.count
      guard bytes <= SpeechTranscript.maximumTextBytes,
        segments.count < SpeechTranscript.maximumSegments
      else {
        throw ProAppsError.outputLimit
      }
      if options?.wordTiming == true {
        segments += try SpeechTimedText.segments(from: result.text, audioDuration: audioDuration)
        guard segments.count <= SpeechTranscript.maximumSegments else {
          throw ProAppsError.outputLimit
        }
      } else {
        let segment = SpeechSegment(
          text: text, startSeconds: result.range.start.seconds,
          durationSeconds: result.range.duration.seconds)
        segments.append(
          try segment.normalizingNativeEnd(
            audioDuration: audioDuration,
            timeScale: result.range.end.timescale))
      }
    }
    return segments
  }

  func cancel() async { await analyzer.cancelAndFinishNow() }
}
