import Foundation
import Testing

@testable import ProAppsCore

// Hardware codec sessions are a shared macOS resource. Serialize this native
// suite, matching the MCP's single-mutation admission; model tests stay parallel.
@Suite(.serialized)
struct NativeEditorTests {
  private func directory() throws -> URL {
    let url = try Files.temporaryDirectory().appendingPathComponent(
      "native-edit-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
    return url
  }

  private func black() throws -> String {
    try #require(
      Bundle.module.url(forResource: "black", withExtension: "mp4", subdirectory: "Fixtures")
    ).path
  }

  private func tone(in directory: URL) throws -> String {
    let sampleRate = 16000
    let frames = sampleRate * 2
    var data = Data("RIFF".utf8)
    func word(_ value: UInt32) -> [UInt8] {
      [
        UInt8(truncatingIfNeeded: value), UInt8(truncatingIfNeeded: value >> 8),
        UInt8(truncatingIfNeeded: value >> 16), UInt8(truncatingIfNeeded: value >> 24),
      ]
    }
    data.append(contentsOf: word(UInt32(36 + frames * 2)))
    data.append(Data("WAVEfmt ".utf8))
    data.append(contentsOf: word(16))
    data.append(contentsOf: [1, 0, 1, 0])
    data.append(contentsOf: word(UInt32(sampleRate)))
    data.append(contentsOf: word(UInt32(sampleRate * 2)))
    data.append(contentsOf: [2, 0, 16, 0])
    data.append(Data("data".utf8))
    data.append(contentsOf: word(UInt32(frames * 2)))
    for index in 0..<frames {
      let sample = Int16(
        (sin(2 * Double.pi * 440 * Double(index) / Double(sampleRate)) * 10000).rounded())
      let bits = UInt16(bitPattern: sample)
      data.append(contentsOf: [
        UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8),
      ])
    }
    let url = directory.appendingPathComponent("tone.wav")
    try data.write(to: url)
    return url.path
  }

  @Test func nativeVideoConcatenationAndRotationPreserveContinuousFrames() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root).path
    let sourceVerification = try await MediaProbe().verifyShortVideo(path: source)
    try #require(
      sourceVerification.decodedFrames == 30,
      "Continuous fixture duration=\(sourceVerification.media.durationSeconds), nominalRate=\(sourceVerification.media.frameRate)"
    )
    let recipe = EditRecipe(
      clips: [
        .init(sourcePath: source, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1)),
        .init(
          sourcePath: source, selection: .init(startSeconds: 0, durationSeconds: 0.5, rate: 1),
          geometry: .init(rotation: .clockwise90)),
      ], video: .init(width: 320, height: 240, frameRate: 30, resizeMode: .fit))
    let result = try await NativeEditor().render(recipe, directory: root.path, name: "result.mp4")
    #expect(result.expectedDurationSeconds == 1.5)
    #expect(abs(result.actualDurationSeconds - 1.5) < 0.04)
    #expect(result.videoTrackCount == 1)
    #expect(result.audioTrackCount == 0)
    #expect(result.outputPath.hasPrefix(root.path + "/edit-"))
    let frame = try await MediaProbe().inspect(path: result.outputPath)
    #expect(frame.width == 320)
    #expect(frame.height == 240)
    #expect(frame.firstFrameDecoded)
    let complete = try await MediaProbe().verifyShortVideo(path: result.outputPath)
    #expect(complete.fullVideoDecoded)
    #expect(complete.decodedFrames == 45)
    await #expect(throws: (any Error).self) {
      try await MediaProbe().verifyShortVideo(path: result.outputPath, maximumFrames: 1)
    }
    let colors = try await FrameProbe().measure(
      path: result.outputPath,
      samples: [
        .init(timeSeconds: 0.25, region: .init(x: 80, y: 40, width: 40, height: 40)),
        .init(timeSeconds: 1.25, region: .init(x: 80, y: 40, width: 40, height: 40)),
      ])
    try #require(colors.count == 2)
    #expect(colors[0].meanRed > 0.8 && colors[0].meanBlue < 0.1)
    #expect(colors[1].meanBlue > 0.8 && colors[1].meanRed < 0.1)
    #expect(
      !FileManager.default.fileExists(
        atPath: URL(fileURLWithPath: result.outputPath).deletingLastPathComponent()
          .appendingPathComponent(".rendering.mp4").path))
  }

  @Test(arguments: [
    (EditColor(brightness: -1, contrast: 1, saturation: 1), "black"),
    (EditColor(brightness: 1, contrast: 1, saturation: 1), "white"),
    (EditColor(brightness: 0, contrast: 1, saturation: 0), "gray"),
    (EditColor(brightness: 0, contrast: 0, saturation: 1), "gray"),
  ])
  func colorControlsChangeDecodedPixelsAndPreserveAudio(_ color: EditColor, _ expected: String)
    async throws
  {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let audio = try tone(in: root)
    let recipe = EditRecipe(
      clips: [
        .init(
          sourcePath: source.path, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
      ], video: .init(width: 32, height: 32, frameRate: 30, resizeMode: .fit, color: color),
      additionalAudio: [
        .init(
          sourcePath: audio, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
          offsetSeconds: 0)
      ])
    let rendered = try await NativeEditor().render(recipe, directory: root.path, name: "color.mp4")
    #expect(rendered.audioTrackCount == 1)
    #expect(try await MediaProbe().verifyShortVideo(path: rendered.outputPath).decodedFrames == 30)
    let measurements = try await FrameProbe().measure(
      path: rendered.outputPath,
      samples: [.init(timeSeconds: 0.5, region: .init(x: 4, y: 4, width: 8, height: 8))])
    let pixel = try #require(measurements.first)
    #expect(abs(pixel.meanRed - pixel.meanGreen) < 0.02)
    #expect(abs(pixel.meanRed - pixel.meanBlue) < 0.02)
    if expected == "black" {
      #expect(pixel.meanRed < 0.02)
    } else if expected == "white" {
      #expect(pixel.meanRed > 0.98)
    } else {
      #expect(pixel.meanRed > 0.05 && pixel.meanRed < 0.95)
    }
    let level = try await AudioProbe().measure(
      path: rendered.outputPath, windows: [.init(startSeconds: 0.25, durationSeconds: 0.5)])
    let middle = try #require(level.windows.first)
    #expect(middle.rms > 0.18 && middle.rms < 0.25)
    let saved = try JSONDecoder().decode(
      EditRequest.self, from: Data(contentsOf: URL(fileURLWithPath: rendered.projectPath)))
    #expect(saved.recipe.video?.color?.saturation == color.saturation)
    #expect(
      !FileManager.default.fileExists(
        atPath: URL(fileURLWithPath: rendered.outputPath).deletingLastPathComponent()
          .appendingPathComponent(".graded.mp4").path))
  }

  @Test(arguments: [false, true])
  func staticTitlesAppearAtTopLeftAfterOptionalColor(_ withColor: Bool) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source: String
    if withColor {
      source = try await ContinuousVideoFixture.make(in: root).path
    } else {
      source = try black()
    }
    let title = EditTitle(text: "TEST", x: 16, y: 20, fontSize: 32)
    let color: EditColor? = withColor ? .init(brightness: -1, contrast: 1, saturation: 1) : nil
    let recipe = EditRecipe(
      clips: [
        .init(sourcePath: source, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
      ],
      video: .init(
        width: 320, height: 240, frameRate: 30, resizeMode: .fit, color: color, titles: [title]))
    let result = try await NativeEditor().render(recipe, directory: root.path, name: "title.mp4")
    #expect(try await MediaProbe().verifyShortVideo(path: result.outputPath).fullVideoDecoded)
    let pixels = try await FrameProbe().measure(
      path: result.outputPath,
      samples: [
        .init(timeSeconds: 0.25, region: .init(x: 8, y: 8, width: 128, height: 64)),
        .init(timeSeconds: 0.75, region: .init(x: 8, y: 168, width: 128, height: 64)),
      ])
    try #require(pixels.count == 2)
    #expect(pixels[0].meanRed > 0.02)
    #expect(pixels[1].meanRed < 0.02)
    let saved = try JSONDecoder().decode(
      EditRequest.self, from: Data(contentsOf: URL(fileURLWithPath: result.projectPath)))
    #expect(saved.recipe.video?.titles?.first?.text == "TEST")
  }

  @Test(arguments: [false, true])
  func crossDissolvesBlendVideoAndDoNotDoubleCoherentAudio(_ audioOnly: Bool) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let audio = try tone(in: root)
    let source: String
    let canvas: EditVideoSettings? =
      audioOnly ? nil : .init(width: 32, height: 32, frameRate: 30, resizeMode: .fill)
    if audioOnly {
      source = audio
    } else {
      let video = try await ContinuousVideoFixture.make(in: root)
      let fixture = try await NativeEditor().render(
        .init(
          clips: [
            .init(
              sourcePath: video.path, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1)
            )
          ], video: .init(width: 16, height: 16, frameRate: 30, resizeMode: .fit),
          additionalAudio: [
            .init(
              sourcePath: audio, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
              offsetSeconds: 0)
          ]), directory: root.path, name: "with-tone.mp4")
      source = fixture.outputPath
    }
    let first = EditClip(
      sourcePath: source, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
      geometry: audioOnly
        ? nil : .init(rotation: .none, crop: .init(x: 2, y: 2, width: 4, height: 4)))
    let second = EditClip(
      sourcePath: source, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
      geometry: audioOnly
        ? nil : .init(rotation: .none, crop: .init(x: 2, y: 10, width: 4, height: 4)),
      transitionInSeconds: 0.5)
    let result = try await NativeEditor().render(
      .init(clips: [first, second], video: canvas), directory: root.path,
      name: audioOnly ? "crossfade.m4a" : "dissolve.mp4")
    #expect(abs(result.actualDurationSeconds - 1.5) < 0.04)
    if !audioOnly {
      #expect(try await MediaProbe().verifyShortVideo(path: result.outputPath).decodedFrames == 45)
      let colors = try await FrameProbe().measure(
        path: result.outputPath,
        samples: [.init(timeSeconds: 0.25), .init(timeSeconds: 0.75), .init(timeSeconds: 1.25)])
      try #require(colors.count == 3)
      #expect(colors[0].meanRed > 0.8 && colors[0].meanBlue < 0.1)
      #expect(colors[1].meanRed > 0.3 && colors[1].meanBlue > 0.3)
      #expect(colors[1].meanRed + colors[1].meanBlue > 0.9)
      #expect(colors[2].meanBlue > 0.8 && colors[2].meanRed < 0.1)
    }
    let levels = try await AudioProbe().measure(
      path: result.outputPath,
      windows: [
        .init(startSeconds: 0.25, durationSeconds: 0.1),
        .init(startSeconds: 0.75, durationSeconds: 0.1),
        .init(startSeconds: 1.25, durationSeconds: 0.1),
      ])
    try #require(levels.windows.count == 3)
    #expect(levels.windows[0].rms > 0.18 && levels.windows[0].rms < 0.25)
    #expect(levels.windows[1].rms > 0.18 && levels.windows[1].rms < 0.25)
    #expect(levels.windows[2].rms > 0.18 && levels.windows[2].rms < 0.25)
  }

  private enum ExportObservation: Equatable, Sendable {
    case stagingObserved, cancelled, completed, timedOut, observerCancelled
    case failed(String)
  }

  @Test func cancellationDuringAnActiveExportJoinsAndRemovesStaging() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let clip = EditClip(
      sourcePath: source.path, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 0.25))
    let recipe = EditRecipe(
      clips: Array(repeating: clip, count: 60),
      video: .init(width: 1920, height: 1080, frameRate: 30, resizeMode: .fit))
    let observations = await withTaskGroup(
      of: ExportObservation.self, returning: [ExportObservation].self
    ) { group in
      group.addTask {
        do {
          _ = try await NativeEditor().render(recipe, directory: root.path, name: "cancelled.mp4")
          return .completed
        } catch is CancellationError { return .cancelled } catch {
          return .failed(String(describing: error))
        }
      }
      group.addTask {
        do {
          let deadline = ContinuousClock.now.advanced(by: .seconds(10))
          while ContinuousClock.now < deadline {
            try Task.checkCancellation()
            let children = try FileManager.default.contentsOfDirectory(
              at: root, includingPropertiesForKeys: nil)
            if children.contains(where: {
              $0.lastPathComponent.hasPrefix("edit-")
                && FileManager.default.fileExists(
                  atPath: $0.appendingPathComponent(".rendering.mp4").path)
            }) {
              return .stagingObserved
            }
            try await Task.sleep(for: .milliseconds(2))
          }
          return .timedOut
        } catch is CancellationError { return .observerCancelled } catch {
          return .failed(String(describing: error))
        }
      }
      var result: [ExportObservation] = []
      if let first = await group.next() { result.append(first) }
      group.cancelAll()
      for await next in group { result.append(next) }
      return result
    }
    #expect(observations == [.stagingObserved, .cancelled])
    let children = try FileManager.default.contentsOfDirectory(
      at: root, includingPropertiesForKeys: nil)
    let renderedDirectories = children.filter { $0.lastPathComponent.hasPrefix("edit-") }
    #expect(renderedDirectories.count == 1)
    let output = try #require(renderedDirectories.first)
    #expect(
      !FileManager.default.fileExists(atPath: output.appendingPathComponent(".rendering.mp4").path))
    #expect(
      !FileManager.default.fileExists(atPath: output.appendingPathComponent("cancelled.mp4").path))
    #expect(
      !FileManager.default.fileExists(
        atPath: output.appendingPathComponent("edit-request.json").path))
  }

  @Test func fullDecodeRefusesLongVideoButMetadataInspectionRemainsAvailable() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let clip = EditClip(
      sourcePath: try black(), selection: .init(startSeconds: 0, durationSeconds: 1, rate: 0.25))
    let recipe = EditRecipe(
      clips: Array(repeating: clip, count: 8),
      video: .init(width: 32, height: 32, frameRate: 1, resizeMode: .fit))
    let result = try await NativeEditor().render(recipe, directory: root.path, name: "long.mp4")
    #expect(result.actualDurationSeconds == 32)
    #expect(try await MediaProbe().inspect(path: result.outputPath).firstFrameDecoded)
    await #expect(throws: (any Error).self) {
      try await MediaProbe().verifyShortVideo(path: result.outputPath)
    }
  }

  @Test func captionsAppearOnlyInsideTheirIntervalAndKeepAudio() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let audio = try tone(in: root)
    let clip = EditClip(
      sourcePath: source.path, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
    let recipe = EditRecipe(
      clips: [clip, clip],
      video: .init(
        width: 320, height: 240, frameRate: 30, resizeMode: .fit,
        captions: [.init(text: "Timed caption", startSeconds: 0.5, endSeconds: 1.2)]),
      additionalAudio: [
        .init(
          sourcePath: audio, selection: .init(startSeconds: 0, durationSeconds: 2, rate: 1),
          offsetSeconds: 0)
      ])
    let rendered = try await NativeEditor().render(
      recipe, directory: root.path, name: "captions.mp4")
    #expect(rendered.actualDurationSeconds == 2)
    #expect(try await MediaProbe().verifyShortVideo(path: rendered.outputPath).decodedFrames == 60)
    let bottom = EditCrop(x: 32, y: 188, width: 256, height: 48)
    let control = EditCrop(x: 8, y: 8, width: 32, height: 32)
    let frames = try await FrameProbe().measure(
      path: rendered.outputPath,
      samples: [
        .init(timeSeconds: 0.25, region: bottom), .init(timeSeconds: 0.5, region: bottom),
        .init(timeSeconds: 0.75, region: bottom), .init(timeSeconds: 1.2, region: bottom),
        .init(timeSeconds: 1.5, region: bottom),
        .init(timeSeconds: 0.25, region: control), .init(timeSeconds: 0.75, region: control),
      ])
    try #require(frames.count == 7)
    #expect(abs(frames[1].actualTimeSeconds - 0.5) < 0.000001)
    #expect(abs(frames[3].actualTimeSeconds - 1.2) < 0.000001)
    #expect(abs(frames[0].meanBlue - frames[1].meanBlue) > 0.02)
    #expect(abs(frames[0].meanBlue - frames[2].meanBlue) > 0.02)
    #expect(abs(frames[0].meanBlue - frames[3].meanBlue) < 0.015)
    #expect(abs(frames[0].meanBlue - frames[4].meanBlue) < 0.015)
    #expect(abs(frames[5].meanRed - frames[6].meanRed) < 0.015)
    #expect(try await AudioProbe().measure(path: rendered.outputPath, windows: []).whole.rms > 0.1)
  }

  @Test(arguments: [
    (seconds: 60, frames: 1800, audioFrames: 960000),
    (seconds: 90, frames: 2700, audioFrames: 1_440_000),
  ])
  func decoratedVideoPreservesVoiceAndTimesCueSoundsAtBothLengths(
    _ sample: (seconds: Int, frames: Int, audioFrames: Int)
  ) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let image = try await ContinuousVideoFixture.make(in: root)
    let voice = try tone(in: root)
    let seed = try await NativeEditor().render(
      .init(
        clips: [
          .init(
            sourcePath: image.path, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 0.5)
          )
        ],
        video: .init(width: 320, height: 240, frameRate: 30, resizeMode: .fill),
        additionalAudio: [
          .init(
            sourcePath: voice, selection: .init(startSeconds: 0, durationSeconds: 2, rate: 1),
            offsetSeconds: 0)
        ]),
      directory: root.path, name: "voiced-seed.mp4")
    let sourceBefore = try Data(contentsOf: URL(fileURLWithPath: seed.outputPath))
    let seconds = Double(sample.seconds)
    let sound = root.appendingPathComponent("cue.wav")
    try CueSound(durationSeconds: seconds, onsetSeconds: [1, 30, seconds - 1], gain: 0.2).wave()
      .write(to: sound, options: .withoutOverwriting)
    let clip = EditClip(
      sourcePath: seed.outputPath, selection: .init(startSeconds: 0, durationSeconds: 2, rate: 1),
      audio: .init(volume: 0.5, fadeInSeconds: 0, fadeOutSeconds: 0))
    let recipe = EditRecipe(
      clips: Array(repeating: clip, count: sample.seconds / 2),
      video: .init(
        width: 320, height: 240, frameRate: 30, resizeMode: .fill,
        captions: [
          .init(text: "TEST", startSeconds: 1, endSeconds: 3),
          .init(text: "TEST", startSeconds: 30, endSeconds: 32),
          .init(text: "TEST", startSeconds: seconds - 1, endSeconds: seconds - 0.5),
        ],
        masks: [.init(region: .init(x: 0, y: 140, width: 320, height: 70), opacity: 1)],
        captionStyle: .init(outlineWidth: 3, backgroundOpacity: 0, fontSize: 20, bottomMargin: 40)),
      additionalAudio: [
        .init(
          sourcePath: sound.path,
          selection: .init(startSeconds: 0, durationSeconds: seconds, rate: 1), offsetSeconds: 0)
      ])
    let result = try await NativeEditor().render(
      recipe, directory: root.path, name: "decorated.mp4")
    #expect(
      try await MediaProbe().verifyShortVideo(
        path: result.outputPath, maximumFrames: 3000, maximumDurationSeconds: 100
      ).decodedFrames == sample.frames)
    let region = EditCrop(x: 100, y: 145, width: 120, height: 60)
    let control = EditCrop(x: 240, y: 30, width: 30, height: 30)
    let frames = try await FrameProbe().measure(
      path: result.outputPath,
      samples: [
        .init(timeSeconds: 0.5, region: region), .init(timeSeconds: 1.5, region: region),
        .init(timeSeconds: 30.5, region: region),
        .init(timeSeconds: seconds - 0.75, region: region),
        .init(timeSeconds: seconds - 0.25, region: region),
        .init(timeSeconds: 0.5, region: control), .init(timeSeconds: 30.5, region: control),
        .init(timeSeconds: seconds - 0.25, region: control),
      ])
    try #require(frames.count == 8)
    #expect(frames[0].meanBlue < 0.01 && frames[4].meanBlue < 0.01)
    #expect(frames[1].meanBlue > 0.02 && frames[2].meanBlue > 0.02 && frames[3].meanBlue > 0.02)
    #expect(frames[5].meanGreen > 0.8 && frames[6].meanGreen > 0.8 && frames[7].meanGreen > 0.8)
    let audio = try await AudioProbe().measure(
      path: result.outputPath,
      windows: [
        .init(startSeconds: 0.5, durationSeconds: 0.08),
        .init(startSeconds: 1, durationSeconds: 0.08),
        .init(startSeconds: 30, durationSeconds: 0.08),
        .init(startSeconds: seconds - 1, durationSeconds: 0.08),
        .init(startSeconds: 1.2, durationSeconds: 0.08),
      ], maximumDurationSeconds: 100)
    try #require(audio.windows.count == 5)
    #expect(audio.whole.frames == sample.audioFrames && audio.whole.peak < 0.6)
    #expect(audio.windows[0].rms > 0.09 && audio.windows[0].rms < 0.13)
    #expect(audio.windows[1].rms > audio.windows[0].rms * 1.08)
    #expect(audio.windows[2].rms > audio.windows[0].rms * 1.08)
    #expect(audio.windows[3].rms > audio.windows[0].rms * 1.08)
    #expect(abs(audio.windows[4].rms - audio.windows[0].rms) < 0.005)
    #expect(try Data(contentsOf: URL(fileURLWithPath: seed.outputPath)) == sourceBefore)
  }

  @Test(arguments: [1800, 1801, 7200])
  func oneMinuteVerificationRequiresExplicitDurationBudget(_ frameBudget: Int) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root)
    let clip = EditClip(
      sourcePath: source.path, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
    let recipe = EditRecipe(
      clips: Array(repeating: clip, count: 60),
      video: .init(width: 32, height: 32, frameRate: 30, resizeMode: .fit))
    let rendered = try await NativeEditor().render(recipe, directory: root.path, name: "minute.mp4")
    #expect(rendered.actualDurationSeconds == 60)
    await #expect(throws: (any Error).self) {
      try await MediaProbe().verifyShortVideo(path: rendered.outputPath)
    }
    let verified = try await MediaProbe().verifyShortVideo(
      path: rendered.outputPath, maximumFrames: frameBudget, maximumDurationSeconds: 60)
    #expect(verified.decodedFrames == 1800)
    #expect(verified.fullVideoDecoded)
    await #expect(throws: (any Error).self) {
      try await MediaProbe().verifyShortVideo(
        path: rendered.outputPath, maximumFrames: 1799, maximumDurationSeconds: 60)
    }
  }

  @Test(arguments: [(0.5, 2.0), (2.0, 0.5)])
  func nativeVideoSpeedChangesTimelineWithoutAssumingConstantCadence(
    _ rate: Double, _ expected: Double
  ) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try await ContinuousVideoFixture.make(in: root).path
    let recipe = EditRecipe(
      clips: [
        .init(sourcePath: source, selection: .init(startSeconds: 0, durationSeconds: 1, rate: rate))
      ], video: .init(width: 320, height: 240, frameRate: 30, resizeMode: .fit))
    let result = try await NativeEditor().render(recipe, directory: root.path, name: "speed.mp4")
    #expect(abs(result.actualDurationSeconds - expected) < 0.04)
    #expect(try await MediaProbe().verifyShortVideo(path: result.outputPath).fullVideoDecoded)
  }

  @Test(arguments: [true, false])
  func audioOnlySpeedVolumeAndFadesExportWithoutPlayback(_ adjusted: Bool) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try tone(in: root)
    let recipe = EditRecipe(clips: [
      .init(
        sourcePath: source, selection: .init(startSeconds: 0, durationSeconds: 2, rate: 2),
        audio: adjusted ? .init(volume: 0.5, fadeInSeconds: 0.2, fadeOutSeconds: 0.2) : nil)
    ])
    let result = try await NativeEditor().render(recipe, directory: root.path, name: "audio.m4a")
    #expect(abs(result.actualDurationSeconds - 1) < 0.1)
    #expect(result.audioTrackCount == 1)
    #expect(result.videoTrackCount == 0)
    let levels = try await AudioProbe().measure(
      path: result.outputPath,
      windows: [
        .init(startSeconds: 0, durationSeconds: 0.15),
        .init(startSeconds: 0.4, durationSeconds: 0.15),
        .init(startSeconds: 0.8, durationSeconds: 0.15),
      ])
    try #require(levels.windows.count == 3)
    let middle = levels.windows[1].rms
    if adjusted {
      #expect(middle > 0.09 && middle < 0.13)
      #expect(levels.windows[0].rms < middle * 0.8)
      #expect(levels.windows[2].rms < middle * 0.85)
    } else {
      #expect(middle > 0.18 && middle < 0.25)
    }
  }

  @Test func timedReplacementAudioDoesNotShortenAnAudioOnlyTimeline() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = try tone(in: root)
    let recipe = EditRecipe(
      clips: [
        .init(sourcePath: source, selection: .init(startSeconds: 0, durationSeconds: 2, rate: 1))
      ],
      additionalAudio: [
        .init(
          sourcePath: source, selection: .init(startSeconds: 0, durationSeconds: 0.5, rate: 1),
          offsetSeconds: 0.5)
      ], muteOriginalAudio: true)
    let result = try await NativeEditor().render(
      recipe, directory: root.path, name: "replacement.m4a")
    #expect(abs(result.actualDurationSeconds - 2) < 0.1)
    #expect(result.audioTrackCount == 1)
  }

  @Test(arguments: [
    "extension", "missing", "duration", "video-track", "clip-audio", "additional-audio",
  ])
  func invalidNativeInputsFailWithoutPublishing(_ failure: String) async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let black = try black()
    let audio = try tone(in: root)
    let video = EditVideoSettings(width: 320, height: 240, frameRate: 30, resizeMode: .fit)
    let recipe: EditRecipe
    switch failure {
    case "missing":
      recipe = .init(
        clips: [
          .init(
            sourcePath: root.appendingPathComponent("missing.mp4").path,
            selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
        ], video: video)
    case "duration":
      recipe = .init(
        clips: [
          .init(sourcePath: black, selection: .init(startSeconds: 0, durationSeconds: 2, rate: 1))
        ], video: video)
    case "video-track":
      recipe = .init(
        clips: [
          .init(sourcePath: audio, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
        ], video: video)
    case "clip-audio":
      recipe = .init(
        clips: [
          .init(
            sourcePath: black, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
            audio: .unity)
        ], video: video)
    case "additional-audio":
      recipe = .init(
        clips: [
          .init(sourcePath: black, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
        ], video: video,
        additionalAudio: [
          .init(
            sourcePath: black, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1),
            offsetSeconds: 0)
        ])
    default:
      recipe = .init(
        clips: [
          .init(sourcePath: black, selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
        ], video: video)
    }
    let name = failure == "extension" ? "wrong.m4a" : "result.mp4"
    await #expect(throws: (any Error).self) {
      try await NativeEditor().render(recipe, directory: root.path, name: name)
    }
    let entries = try FileManager.default.contentsOfDirectory(
      at: root, includingPropertiesForKeys: nil)
    #expect(!entries.contains { $0.lastPathComponent.hasPrefix("edit-") })
  }

  @Test func cancellationIsObservedBeforeAssetWork() async throws {
    let root = try directory()
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let recipe = EditRecipe(clips: [
      .init(sourcePath: try black(), selection: .init(startSeconds: 0, durationSeconds: 1, rate: 1))
    ])
    await #expect(throws: CancellationError.self) {
      try await withThrowingTaskGroup(of: EditRenderResult.self) { group in
        group.cancelAll()
        group.addTask {
          try await NativeEditor().render(recipe, directory: root.path, name: "audio.m4a")
        }
        _ = try await group.next()
      }
    }
  }
}
