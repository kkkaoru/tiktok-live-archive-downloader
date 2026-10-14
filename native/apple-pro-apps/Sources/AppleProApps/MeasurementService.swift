import Foundation
import MCP
import ProAppsCore

extension ToolSpec {
  private static let headFaceMeasurementSchema = object(
    [
      "presentation": object(
        [
          "value": integer(
            0, Int(ForegroundTime.maximumSeconds * Int64(ForegroundTime.maximumTimescale))),
          "timescale": integer(1, Int(ForegroundTime.maximumTimescale)),
        ], ["value", "timescale"]),
      "x": integer(0, 16384), "y": integer(0, 16384),
      "width": integer(1, 16384), "height": integer(1, 16384),
      "minimumFaceCoverage": number,
    ], ["presentation", "x", "y", "width", "height", "minimumFaceCoverage"])

  // Explicit direct-CLI opt-in; not advertised through the unchanged MCP dispatcher.
  static let sourceTextSession: ToolSpec = .init(
    name: "video_text_recognize_session",
    description:
      "CLI-only explicit CPU accurate Japanese/English OCR session, 1–64 strictly ordered original rational PTS. Optional workers 1–2; omission preserves serial execution. Full canvas only, independent generator/request per worker. Private repository-local eight-frame measurement/checkpoint pairs retain exact actual clocks; 64 lines/frame and 32 KiB text/block, never truncation. Two-worker opt-in additionally limits the whole batch to 32 KiB text and publishes blocks only after the complete batch. Maximum 60 seconds with parent disposable-process deadline still required. No quality approval or cryptographic source validation; fresh parent boundaries remain mandatory. Creates new checkpoint files; old eight-frame tool unchanged.",
    properties: [
      "checkoutRoot": string(), "path": string(), "checkpointDirectory": string(),
      "width": integer(1, 16384), "height": integer(1, 16384),
      "workers": integer(1, 2),
      "allowedSourceRoot": string(),
      "maximumSeconds": .object([
        "type": .string("number"), "exclusiveMinimum": .double(0), "maximum": .int(60),
      ]),
      "samples": array(
        object(
          [
            "value": integer(
              0, Int(ForegroundTime.maximumSeconds * Int64(ForegroundTime.maximumTimescale))),
            "timescale": integer(1, Int(ForegroundTime.maximumTimescale)),
          ], ["value", "timescale"]), maximum: SourceTextSession.maximumSamples),
    ],
    required: [
      "checkoutRoot", "path", "checkpointDirectory", "width", "height", "maximumSeconds", "samples",
    ],
    readOnly: false)

  static let measurements: [ToolSpec] = [
    .init(
      name: "video_foreground_measure",
      description:
        "Opt-in foreground union-mask measurement at 1–8 exact rational source clocks. Modern Vision revision1; source-sized L8 support at threshold 1, top-left bounds. Nil observations, empty labels and empty quantized support remain distinct. No identity, head/hair geometry, subject absence or caption approval. Read-only; no playback, image export or source writes. 60-second disposable child deadline.",
      properties: [
        "path": string(),
        "headFace": headFaceMeasurementSchema,
        "headFaces": array(headFaceMeasurementSchema, maximum: FrameProbe.maximumSamples),
        "samples": array(
          object(
            [
              "value": integer(
                0, Int(ForegroundTime.maximumSeconds * Int64(ForegroundTime.maximumTimescale))),
              "timescale": integer(1, Int(ForegroundTime.maximumTimescale)),
            ],
            ["value", "timescale"]), maximum: FrameProbe.maximumSamples),
      ], required: ["path", "samples"], readOnly: true),
    .init(
      name: "caption_text_measure",
      description:
        "Measure native offscreen text bounds with an exact installed PostScript font and libass font sizing. One or two explicit lines, up to 256 texts/4096 UTF-16 units and 16 megapixels total. Bounds include outline/rim padding. No app launch, source writes, upload, automatic wrapping or animation. These are native font bounds, NOT verified Motion geometry; motionRenderVerified remains false.",
      properties: [
        "fontName": string(maximum: 128), "assFontSize": number,
        "border": number, "rim": number,
        "texts": array(
          // Schema lengths are UTF-8 bytes; each UTF-16 unit needs at most three.
          // The measurement implementation retains the exact UTF-16 budget.
          string(maximum: CaptionTextMeasurement.maximumCharacters * 3),
          maximum: CaptionTextMeasurement.maximumTexts),
      ], required: ["fontName", "assFontSize", "border", "rim", "texts"], readOnly: true),
    .init(
      name: "audio_separate_vocals",
      description:
        "Separate a bounded stereo audio segment with an explicitly installed, approved HTDemucs Core ML model. leftPath/rightPath are equal-length mono PCM16 WAVs at 44100 Hz, up to 343980 samples (7.8 seconds); shorter inputs are zero-padded for inference then trimmed. compiledModelPath is the local mlmodelc directory with the fixed Float16 spectral/waveform contract; CPU-only inference, no download/upload/playback. Output is a NEW private stereo Float32 WAV and request sidecar. No clipping or normalization; peak and humanReviewed:false are reported. Model inference is not proof of complete BGM removal or voice preservation. Long recordings require explicit overlapping segments and verified overlap-add. Bounded child operation, not a live app edit.",
      properties: [
        "leftPath": string(), "rightPath": string(), "compiledModelPath": string(),
        "outputDirectory": string(), "outputName": string(maximum: 180),
      ],
      required: ["leftPath", "rightPath", "compiledModelPath", "outputDirectory", "outputName"],
      readOnly: false),
    .init(
      name: "audio_sound_activity",
      description:
        "Classify local prepared mono PCM16 WAV audio (0.5–60 seconds) with Apple's built-in version1 SoundAnalysis classifier. Returns overlapping window start/duration and speech/music confidence, requested 0.5-second windows and 50% overlap. Actual window duration is reported. Scores are estimates, not exact speech boundaries or reviewed cut decisions; no transcription, speaker identity, stem separation or automatic cuts. Local file only, no playback, model download, upload or source writes. Runs synchronously in a bounded disposable child on a dedicated audio worker; cancellation is checked before/after native analysis.",
      properties: ["path": string()], required: ["path"], readOnly: true),
    .init(
      name: "audio_reference_analyze",
      description:
        "Fit an explicitly aligned local reference BGM against source audio with a bounded sample-offset search. Inputs must be prepared mono PCM16 WAVs, equal sample rates, at most 60 seconds and one million samples each. No implicit mixing/resampling, cloud or source writes. referenceStartSample is the center offset inside the reference, searchRadiusSamples scans on both sides; all candidates must fit. At most 16 million sample products. Returns gain, signed normalized correlation and estimated residual RMS; correlation is not proof of voice preservation. Unmatched/silent/excessive-gain inputs fail. Read-only analysis, not an exported separated track.",
      properties: [
        "sourcePath": string(), "referencePath": string(),
        "search": object(
          [
            "referenceStartSample": integer(0, ReferenceAudio.maximumSamples),
            "searchRadiusSamples": integer(0, ReferenceAudio.maximumSamples),
            "minimumCorrelation": number,
          ], ["referenceStartSample", "searchRadiusSamples", "minimumCorrelation"]),
      ], required: ["sourcePath", "referencePath", "search"], readOnly: true),
    .init(
      name: "audio_cue_track",
      description:
        "Generate one local mono PCM16/16kHz WAV track of up to 120 seconds with up to 120 ordered, nonoverlapping caption-onset sounds. Each sound is an 80ms 880Hz sine with a smooth envelope, gain 0–0.25. Output seconds round to PCM samples. No playback, recording or downloaded assets. Creates a private edit directory and cue-request.json without replacing files. Use one additionalAudio layer to mix it with source voice; allow headroom. This is sound generation, not audio measurement.",
      properties: [
        "outputDirectory": string(), "outputName": string(maximum: 180),
        "track": object(
          [
            "durationSeconds": number, "gain": number,
            "onsetSeconds": array(number, maximum: CueSound.maximumCues, minimum: 0),
          ], ["durationSeconds", "gain", "onsetSeconds"]),
      ],
      required: ["outputDirectory", "outputName", "track"], readOnly: false),
    .init(
      name: "media_verify_video",
      description:
        "Decode every video frame in a local clip. Defaults to 30 seconds/1800 frames; longer verification requires explicit maximumDurationSeconds (up to 120) and, if needed, maximumFrames (up to 7200). End-of-stream required. Audio, visual effects and editor import are not verified. Deadline-limited offline child; no playback or source writes.",
      properties: [
        "path": string(), "maximumFrames": integer(1, MediaProbe.maximumVerificationFrames),
        "maximumDurationSeconds": .object([
          "type": .string("number"), "exclusiveMinimum": .double(0),
          "maximum": .double(MediaProbe.maximumVerificationSeconds),
        ]),
      ], required: ["path"], readOnly: true),
    .init(
      name: "audio_transcribe_whisper",
      description:
        "Transcribe up to 60 seconds of an explicit local audio file in Japanese using installed local WhisperKit Core ML and tokenizer directories. Requires macOS 26 for timed lexical caption grouping. No model/tokenizer download, microphone, automatic language detection, previous-text prompt or cloud fallback. Returns unreviewed native word timing; missing lexical timing fails rather than guessing. First model specialization may be slow; a disposable native child has a 180-second hard deadline. No source writes.",
      properties: ["path": string(), "modelDirectory": string(), "tokenizerDirectory": string()],
      required: ["path", "modelDirectory", "tokenizerDirectory"], readOnly: true),
    Self(
      name: "speech_locale_reserve",
      description:
        "Explicitly reserve a supported SpeechTranscriber locale for this application on macOS 26+. Persistent app-scoped resource change: requires user approval. Does not download, install or release models/locales. Existing reservations are retained; full quota fails without eviction. Returns whether already-present assets are ready for this app. A locale listing alone does not prove readiness. Call before audio_transcribe when preparing an approved locale.",
      properties: ["locale": string(maximum: 64)], required: ["locale"], readOnly: false),
    .init(
      name: "audio_transcribe",
      description:
        "Transcribe an existing local M4A/WAV/AIFF/CAF audio file of at most 60 seconds using macOS 26 SpeechAnalyzer and an already-installed locale model. No download, microphone capture or cloud fallback. Returns final text with approximate phrase-level timings by default. Opt-in wordTiming returns native audio-time-indexed text runs instead of character-proportional timing; missing lexical timing is rejected. Optional contextualStrings supplies up to 128 explicit local vocabulary hints (8192 UTF-8 bytes total), not automatic text correction. No volatile/fast results, speaker identification, guaranteed word alignment or human-reviewed accuracy. Empty segments mean no recognized speech. 45-second analysis watchdog within a 60-second disposable child deadline.",
      properties: [
        "path": string(), "locale": string(maximum: 64), "wordTiming": boolean,
        "contextualStrings": array(string(maximum: 128), maximum: 128, minimum: 0),
      ], required: ["path", "locale"],
      readOnly: true),
    .init(
      name: "audio_measure",
      description:
        "Decode a local audio-bearing clip to temporary mono PCM16/16000 Hz using macOS afconvert. Default duration budget is 30 seconds; explicit maximumDurationSeconds permits up to 120 seconds, keeping fixed byte/sample and subprocess limits. Measure whole-clip and requested-window RMS, peak and zero crossings. Mono conversion is not per-channel verification, LUFS or a robust pitch estimate. No playback; temporary PCM is removed.",
      properties: [
        "path": string(),
        "maximumDurationSeconds": .object([
          "type": .string("number"), "exclusiveMinimum": .double(0),
          "maximum": .double(PCMMeasurement.maximumDurationSeconds),
        ]),
        "windows": array(
          object(
            ["startSeconds": number, "durationSeconds": number],
            ["startSeconds", "durationSeconds"]), maximum: PCMMeasurement.maximumWindows, minimum: 0
        ),
      ], required: ["path", "windows"], readOnly: true),
    .init(
      name: "image_text_recognize_batch",
      description:
        "Recognize Japanese/English text in 1–8 local PNG regions, in request order. At most 2 megapixels per image, 64 lines per image and 32 KiB text per batch. Returns observed context rectangles, not glyph masks or absence/repair approval. Read-only; no upload or source writes.",
      properties: ["paths": array(string(), maximum: 8)], required: ["paths"], readOnly: true),
    .init(
      name: "image_text_rectangles_batch",
      description:
        "Detect character rectangles in 1–8 local PNG subtitle ROIs, in request order, reusing one Vision context. Same per-image budgets and geometry contract as image_text_rectangles; sequential decoding bounds memory. No OCR, uploads, source writes, result truncation or quality approval.",
      properties: ["paths": array(string(), maximum: 8)], required: ["paths"], readOnly: true),
    .init(
      name: "image_text_rectangles",
      description:
        "Detect character rectangles in a local PNG subtitle ROI using VNDetectTextRectanglesRequest with reportCharacterBoxes. Top-left pixel coordinates, up to 2 megapixels, 64 text regions and 4096 characters. No OCR, upload or source writes. Missing character geometry fails; an empty result is not proof of subtitle absence or removal quality.",
      properties: ["path": string()], required: ["path"], readOnly: true),
    .init(
      name: "video_text_recognize",
      description:
        "Recognize Japanese/English visible text locally with Apple Vision in 1–8 selected video frames or regions. Returns actual frame times, OCR text/confidence and full-frame top-left pixel rectangles. Regions round outward to pixel boundaries. At most 64 text lines per frame and 32 KiB text per batch; excessive results fail rather than truncate. No video/image upload, screenshot, playback or source writes. OCR confidence is not speech accuracy; burned-in captions are corroboration, not ground truth. 60-second disposable child deadline.",
      properties: [
        "path": string(),
        "samples": array(
          object(
            [
              "timeSeconds": number,
              "region": object(
                ["x": number, "y": number, "width": number, "height": number],
                ["x", "y", "width", "height"]),
            ], ["timeSeconds"]), maximum: FrameProbe.maximumSamples),
      ],
      required: ["path", "samples"], readOnly: true),
    .init(
      name: "video_head_detect",
      description:
        "Detect face and whole-person rectangles locally with Apple Vision in 1–8 explicit video frames. Returns requested and actual decoded times, display dimensions and top-left pixel rectangles, at most 32 per category per frame. Optional compareFaceDetectors records legacy and modern raw confidences, unclipped rectangles and policy eligibility on the identical decoded image; exact requested/actual rational clocks are required. It does not change tracking or approve geometry. Empty arrays mean no detection; no subject identity, interpolation, hair boundary, caption placement or exhaustive coverage is inferred. Full frames only, 16-megapixel decoder budget, 60-second disposable child deadline. No upload, playback, screenshots or source writes.",
      properties: [
        "path": string(),
        "samples": array(
          object(["timeSeconds": number], ["timeSeconds"]), maximum: FrameProbe.maximumSamples),
        "compareFaceDetectors": boolean,
      ], required: ["path", "samples"], readOnly: true),
    .init(
      name: "video_face_track",
      description:
        "Sequential local face observation using VNDetectFaceRectanglesRequest, VNTrackObjectRequest and VNSequenceRequestHandler. Decode every frame of an explicit window up to 120 seconds/7200 frames, retaining rational source PTS and display-oriented pixel rectangles. Redetect at the configured interval and after tracking loss. Confidence at or below 0.3 is not accepted. Missing faces stay absent. Optional objectEvidence returns separate raw unclipped object observations before unchanged rejection/termination. Optional partialObjectEvidence with objectEvidence and an explicit partialObjectPlan independently initializes current visible-image object requests after rejection; derived seed inputs and actual per-frame raw results remain separate from faces. No identity, head Y or caption quality is approved. Read-only, no camera, upload, playback or image export; the existing disposable child deadline still applies.",
      properties: [
        "path": string(),
        "objectEvidence": boolean,
        "partialObjectEvidence": boolean,
        "partialObjectPlan": object(
          [
            "maximumSeeds": integer(1, PartialObjectTrackingPlan.reseedLimit),
            "minimumVisibleFraction": number,
            "boundaries": array(
              object(
                [
                  "presentation": object(
                    [
                      "value": integer(0, Int.max),
                      "timescale": integer(1, Int(FaceTrackingPlan.maximumTimescale)),
                    ], ["value", "timescale"]),
                  "reason": string(),
                ], ["presentation", "reason"]), maximum: PartialObjectTrackingPlan.boundaryLimit,
              minimum: 0),
          ], ["maximumSeeds", "minimumVisibleFraction", "boundaries"]),
        "plan": object(
          [
            "startSeconds": number, "durationSeconds": number,
            "maximumFrames": integer(1, FaceTrackingPlan.frameLimit),
            "redetectIntervalSeconds": number,
            "sourceTimescale": integer(1, Int(FaceTrackingPlan.maximumTimescale)),
          ], ["startSeconds", "durationSeconds", "maximumFrames", "redetectIntervalSeconds"]),
      ], required: ["path", "plan"], readOnly: true),
    .init(
      name: "video_frame_measure",
      description:
        "Measure mean device-RGB values (0–1) in 1–8 selected decoded video frames/regions. Regions use top-left display-oriented pixels. Returns actual sample times and dimensions. Limited to 16 megapixels; no screenshots, image export or playback. Selected-region evidence only, not full-file verification.",
      properties: [
        "path": string(),
        "samples": array(
          object(
            [
              "timeSeconds": number,
              "region": object(
                ["x": number, "y": number, "width": number, "height": number],
                ["x", "y", "width", "height"]),
            ], ["timeSeconds"]), maximum: FrameProbe.maximumSamples),
      ], required: ["path", "samples"], readOnly: true),
  ]
}

extension NativeService {
  func measureLocalFile(
    name: String, path: String,
    foreground: @Sendable (ForegroundRequest) async throws -> [ForegroundMeasurement] = {
      try await FrameProbe().measureForeground(
        path: $0.path, samples: $0.samples, headFace: $0.headFace, headFaces: $0.headFaces)
    },
    textSession: @Sendable (SourceTextSessionRequest) async throws -> SourceTextSessionReport = {
      try await SourceTextSession.run($0)
    }
  ) async throws -> String {
    guard
      let spec = (ToolSpec.measurements + [ToolSpec.sourceTextSession]).first(where: {
        $0.name == name
      })
    else {
      throw ProAppsError.invalid("Unknown measurement operation")
    }
    let data = try Files.read(Files.existing(path, extensions: ["json"]))
    let arguments = try JSONDecoder().decode(Value.self, from: data)
    try validate(arguments, schema: spec.tool.inputSchema)
    let measured: Value
    switch name {
    case "caption_text_measure":
      let request = try decode(CaptionTextMeasurementRequest.self, arguments)
      let result = try await CaptionTextMeasurement.measure(request)
      measured = try JSONDecoder().decode(Value.self, from: JSONEncoder().encode(result))
    case "media_verify_video":
      struct Input: Decodable {
        let path: String
        let maximumFrames: Int?
        let maximumDurationSeconds: Double?
      }
      let input = try decode(Input.self, arguments)
      measured = try await Value(
        MediaProbe().verifyShortVideo(
          path: input.path,
          maximumFrames: input.maximumFrames ?? MediaProbe.defaultVerificationFrames,
          maximumDurationSeconds: input.maximumDurationSeconds
            ?? MediaProbe.defaultVerificationSeconds))
    case "audio_cue_track":
      measured = try createCueTrack(arguments)
    case "audio_transcribe_whisper":
      let request = try decode(WhisperProbe.Request.self, arguments)
      measured = try await Value(WhisperProbe.transcribe(request))
    case "speech_locale_reserve":
      struct Input: Decodable { let locale: String }
      let input = try decode(Input.self, arguments)
      measured = try await Value(SpeechProbe().reserve(locale: input.locale))
    case "audio_transcribe":
      struct Input: Decodable {
        let path: String
        let locale: String
        let wordTiming: Bool?
        let contextualStrings: [String]?
      }
      let input = try decode(Input.self, arguments)
      let options = try SpeechRecognitionOptions(
        wordTiming: input.wordTiming ?? false, contextualStrings: input.contextualStrings ?? [])
      measured = try await Value(
        SpeechProbe().transcribe(path: input.path, locale: input.locale, options: options))
    case "audio_separate_vocals":
      let input = try decode(DemucsRequest.self, arguments)
      measured = try await Value(DemucsSeparator().separate(input))
    case "audio_sound_activity":
      struct Input: Decodable { let path: String }
      let input = try decode(Input.self, arguments)
      measured = try await Value(SoundActivityProbe().analyze(path: input.path))
    case "audio_reference_analyze":
      struct Input: Decodable {
        let sourcePath: String
        let referencePath: String
        let search: ReferenceAudioSearch
      }
      let input = try decode(Input.self, arguments)
      measured = try await Value(
        ReferenceAudioProbe().analyze(
          sourcePath: input.sourcePath, referencePath: input.referencePath, search: input.search))
    case "audio_measure":
      struct Input: Decodable {
        let path: String
        let windows: [AudioWindow]
        let maximumDurationSeconds: Double?
      }
      let input = try decode(Input.self, arguments)
      measured = try await Value(
        AudioProbe().measure(
          path: input.path, windows: input.windows,
          maximumDurationSeconds: input.maximumDurationSeconds
            ?? PCMMeasurement.defaultDurationSeconds))
    case "image_text_recognize_batch":
      struct Input: Decodable { let paths: [String] }
      let input = try decode(Input.self, arguments)
      measured = try await Value(ImageTextProbe().recognize(paths: input.paths))
    case "image_text_rectangles_batch":
      struct Input: Decodable { let paths: [String] }
      let input = try decode(Input.self, arguments)
      measured = try await Value(TextRectangleProbe().detect(paths: input.paths))
    case "image_text_rectangles":
      struct Input: Decodable { let path: String }
      let input = try decode(Input.self, arguments)
      measured = try await Value(TextRectangleProbe().detect(path: input.path))
    case "video_text_recognize":
      struct Input: Decodable {
        let path: String
        let samples: [FrameSample]
      }
      let input = try decode(Input.self, arguments)
      measured = try await Value(
        FrameProbe().recognizeText(path: input.path, samples: input.samples))
    case "video_foreground_measure":
      let input = try decode(ForegroundRequest.self, arguments)
      let result = try await foreground(input)
      measured = try JSONDecoder().decode(Value.self, from: JSONEncoder().encode(result))
    case "video_text_recognize_session":
      let request = try decode(SourceTextSessionRequest.self, arguments)
      measured = try JSONDecoder().decode(
        Value.self, from: JSONEncoder().encode(try await textSession(request)))
    case "video_head_detect":
      struct Input: Decodable {
        let path: String
        let samples: [FrameSample]
        let compareFaceDetectors: Bool?
      }
      let input = try decode(Input.self, arguments)
      measured = try await Value(
        FrameProbe().detectHeads(
          path: input.path, samples: input.samples,
          compareFaceDetectors: input.compareFaceDetectors ?? false))
    case "video_face_track":
      struct Input: Decodable {
        let path: String
        let plan: FaceTrackingPlan
        let objectEvidence: Bool?
        let partialObjectEvidence: Bool?
        let partialObjectPlan: PartialObjectTrackingPlan?
      }
      let input = try decode(Input.self, arguments)
      if input.partialObjectEvidence == true {
        guard input.objectEvidence == true, let partialPlan = input.partialObjectPlan else {
          throw ProAppsError.invalid("Partial objects require raw evidence and an explicit policy")
        }
        let result = try await FaceTrackProbe().trackPartialObjects(
          path: input.path, plan: input.plan, partialPlan: partialPlan)
        measured = try JSONDecoder().decode(Value.self, from: JSONEncoder().encode(result))
      } else if input.partialObjectPlan != nil {
        throw ProAppsError.invalid("Partial object policy requires explicit opt-in")
      } else if input.objectEvidence == true {
        let result = try await FaceTrackProbe().trackObjects(path: input.path, plan: input.plan)
        measured = try JSONDecoder().decode(Value.self, from: JSONEncoder().encode(result))
      } else {
        measured = try await Value(FaceTrackProbe().track(path: input.path, plan: input.plan))
      }
    case "video_frame_measure":
      struct Input: Decodable {
        let path: String
        let samples: [FrameSample]
      }
      let input = try decode(Input.self, arguments)
      measured = try await Value(FrameProbe().measure(path: input.path, samples: input.samples))
    default:
      throw ProAppsError.invalid("Unknown measurement operation")
    }
    return String(decoding: try JSONEncoder().encode(measured), as: UTF8.self)
  }

  private func createCueTrack(_ arguments: Value) throws -> Value {
    struct Input: Decodable {
      let outputDirectory: String
      let outputName: String
      let track: CueSound
    }
    let input = try decode(Input.self, arguments)
    guard URL(fileURLWithPath: input.outputName).pathExtension.lowercased() == "wav" else {
      throw ProAppsError.invalid("Cue track output requires a WAV extension")
    }
    let wave = try input.track.wave()
    let output = try Files.reserveOutput(
      directory: input.outputDirectory, name: input.outputName, kind: .edit)
    do {
      let project = output.deletingLastPathComponent().appendingPathComponent("cue-request.json")
      _ = try writeCueArtifact(JSONEncoder().encode(input.track), to: project.path)
      try Task.checkCancellation()
      _ = try writeCueArtifact(wave, to: output.path)
      return .object([
        "outputPath": .string(output.path), "projectPath": .string(project.path),
        "cueCount": .int(input.track.onsetSeconds.count), "generatedTone": .bool(true),
      ])
    } catch {
      Cleanup.perform { try FileManager.default.removeItem(at: output.deletingLastPathComponent()) }
      throw error
    }
  }

  func measurement(
    _ name: String, _ arguments: Value,
    execute: @Sendable (String, String) async throws -> Value
  ) async throws -> CallTool.Result {
    let request = try Files.reserveOutput(
      directory: Files.temporaryDirectory().path, name: "measurement.json", kind: .edit)
    defer {
      Cleanup.perform {
        try FileManager.default.removeItem(at: request.deletingLastPathComponent())
      }
    }
    _ = try Files.writeNew(JSONEncoder().encode(arguments), to: request.path, extensions: ["json"])
    return response([
      "measurement": try await execute(name, request.path), "sourceModified": .bool(false),
      "playbackPerformed": .bool(false),
    ])
  }
}
