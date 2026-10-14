import Foundation
import SoundAnalysis
import Testing

@testable import ProAppsCore

struct SoundActivityTests {
  private final class UnexpectedResult: NSObject, SNResult {}

  @Test func observesTerminalStatesAndInvalidResults() throws {
    let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
    let observer = SoundActivityObserver()
    #expect(throws: ProAppsError.self) { try observer.results() }
    observer.request(request, didProduce: UnexpectedResult())
    observer.requestDidComplete(request)
    #expect(throws: ProAppsError.self) { try observer.results() }
    observer.request(request, didFailWithError: ProAppsError.invalid("Later error"))
    #expect(throws: ProAppsError.self) { try observer.results() }
  }

  @Test func validatesWindowsAndRefusesLateResults() throws {
    let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
    let observer = SoundActivityObserver()
    observer.append(
      .init(
        startSeconds: 0, durationSeconds: 0.5,
        speechConfidence: 0.8, musicConfidence: 0.1))
    observer.requestDidComplete(request)
    observer.append(
      .init(
        startSeconds: 1, durationSeconds: 0.5,
        speechConfidence: 0.2, musicConfidence: 0.9))
    let windows = try observer.results()
    #expect(windows.count == 1)
    #expect(windows.first?.speechConfidence == 0.8)
    let invalid = SoundActivityObserver()
    invalid.append(
      .init(
        startSeconds: -1, durationSeconds: 0.5,
        speechConfidence: 0.8, musicConfidence: 0.1))
    #expect(throws: ProAppsError.self) { try invalid.results() }
  }

  @Test func nativeAnalysisProducesBoundedWindowsWithoutSourceChanges() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "sound-activity-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = root.appendingPathComponent("tone.wav")
    let wave = try CueSound(durationSeconds: 2, onsetSeconds: [0, 1], gain: 0.2).wave()
    try wave.write(to: source)
    let result = try await SoundActivityProbe().analyze(path: source.path)
    #expect(result.durationSeconds == 2)
    #expect(result.windowDurationSeconds == 0.5)
    #expect(result.overlapFactor == 0.5)
    #expect(!result.humanReviewed)
    #expect(!result.windows.isEmpty)
    #expect(result.windows.count <= 8)
    #expect(try Data(contentsOf: source) == wave)
    let short = root.appendingPathComponent("short.wav")
    try CueSound(durationSeconds: 0.1, onsetSeconds: [], gain: 0).wave().write(to: short)
    await #expect(throws: ProAppsError.self) {
      try await SoundActivityProbe().analyze(path: short.path)
    }
  }
}
