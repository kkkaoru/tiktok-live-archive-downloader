import AVFoundation
import Foundation
import Testing

@testable import audio_clock_probe

/// A retained, one-shot rendezvous: cancellation is requested only after arrival.
private actor ProbeSignal {
  private var fired = false
  private var waiter: CheckedContinuation<Void, Never>?
  func wait() async {
    if fired { return }
    await withCheckedContinuation { waiter = $0 }
  }
  func signal() {
    fired = true
    waiter?.resume()
    waiter = nil
  }
}

private enum WriterFault: String, Sendable {
  case append, finalize, readiness, finalizeStall
}

private final class FaultWriter: StatefulWriting {
  let fault: WriterFault
  let arrived: ProbeSignal
  let release: ProbeSignal
  private(set) var status = AVAssetWriter.Status.writing
  var isReady: Bool { fault != .readiness }
  init(_ fault: WriterFault, arrived: ProbeSignal, release: ProbeSignal) {
    self.fault = fault
    self.arrived = arrived
    self.release = release
  }
  func append(_ sample: CMSampleBuffer) -> Bool { fault != .append }
  func finish() async {
    if fault == .finalizeStall {
      await arrived.signal()
      await release.wait()
      status = .completed
    } else {
      status = .failed
    }
  }
  func cancel() { status = .cancelled }
}

private func negativeRequest(root: URL, name: String) throws -> ClockProbeRequest {
  let work = root.appendingPathComponent(
    "edits/work/audio-clock-repair/negative-" + name + "-" + UUID().uuidString)
  try FileManager.default.createDirectory(at: work, withIntermediateDirectories: false)
  let source = work.appendingPathComponent("source.wav")
  try writeWave(source, frames: 48000)
  var request = ClockProbeRequest(
    source: source.path, outputDirectory: work.appendingPathComponent("result").path,
    startValue: 0, durationValue: 1, timescale: 1)
  request.statefulSelection = StatefulSelection(
    checkpoints: [SourceCheckpoint(startValue: 1, timescale: 48000, sampleCount: 17)],
    budget: StatefulBudget(
      maxRecords: 100, maxLedgerBytes: 100000, maxSavedBytes: 1_000_000,
      maxDecodedSamples: 48000, maxWallSeconds: 1, minimumFreeBytes: 0))
  return request
}

@Test(.timeLimit(.minutes(1))) func actualNativeCancellationAfterFirstAppendedBuffer() async throws
{
  let root = try testRoot()
  let request = try negativeRequest(root: root, name: "mid-cancel")
  let arrived = ProbeSignal()
  let release = ProbeSignal()
  await #expect(throws: CancellationError.self) {
    try await withThrowingTaskGroup(of: Void.self) { group in
      group.addTask {
        do {
          try await AudioClockProbe.run(
            request, root: root,
            progress: { count in
              if count == 1 {
                await arrived.signal()
                await release.wait()
              }
            })
        } catch {
          await arrived.signal()
          throw error
        }
        await arrived.signal()
      }
      await arrived.wait()
      group.cancelAll()
      await release.signal()
      try await group.waitForAll()
    }
  }
  let directory = URL(fileURLWithPath: request.outputDirectory)
  let ledger = try String(
    contentsOf: directory.appendingPathComponent("ledger.jsonl"), encoding: .utf8)
  #expect(ledger.split(separator: "\n").count == 1)
  let first = try #require(ledger.split(separator: "\n").first)
  let record = try JSONDecoder().decode(StatefulRecord.self, from: Data(first.utf8))
  #expect(record.samples > 0)
  #expect(record.bufferOrdinal == 0)
  // AVAssetWriter.cancelWriting may remove its unfinalized movie; the ledger is retained.
  #expect(
    !FileManager.default.fileExists(atPath: directory.appendingPathComponent("stateful.json").path))
}

@Test(arguments: [WriterFault.append, .finalize, .readiness])
private func typedWriterFailureRefusesCompletion(fault: WriterFault) async throws {
  let root = try testRoot()
  let request = try negativeRequest(root: root, name: fault.rawValue)
  let arrived = ProbeSignal()
  let release = ProbeSignal()
  await #expect {
    try await AudioClockProbe.run(
      request, root: root,
      writerBoundary: { _, _ in FaultWriter(fault, arrived: arrived, release: release) })
  } throws: { error in
    guard let failure = error as? ClockProbeError else { return false }
    switch failure {
    case .writerFailed: return fault == .append || fault == .finalize
    case .budgetExceeded: return fault == .readiness
    case .invalidRequest, .unsupportedAudio, .readerFailed: return false
    }
  }
  #expect(!FileManager.default.fileExists(atPath: request.outputDirectory + "/stateful.json"))
}

@Test(.timeLimit(.minutes(1))) func typedFinalizeStallRetainsCancellationAndRefusesCompletion()
  async throws
{
  let root = try testRoot()
  let request = try negativeRequest(root: root, name: "finalize-stall")
  let arrived = ProbeSignal()
  let release = ProbeSignal()
  await #expect(throws: CancellationError.self) {
    try await withThrowingTaskGroup(of: Void.self) { group in
      group.addTask {
        do {
          try await AudioClockProbe.run(
            request, root: root,
            writerBoundary: { _, _ in
              FaultWriter(.finalizeStall, arrived: arrived, release: release)
            })
        } catch {
          await arrived.signal()
          throw error
        }
        await arrived.signal()
      }
      await arrived.wait()
      group.cancelAll()
      // A framework callback that never returns still needs the outer process watchdog.
      await release.signal()
      try await group.waitForAll()
    }
  }
  #expect(!FileManager.default.fileExists(atPath: request.outputDirectory + "/stateful.json"))
}
