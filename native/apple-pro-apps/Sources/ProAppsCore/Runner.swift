import Darwin
import Dispatch
import Foundation
import Synchronization

public struct CommandResult: Sendable {
  public let stdout: String
  public let status: Int32
  public init(stdout: String, status: Int32) {
    self.stdout = stdout
    self.status = status
  }
}

/// One-shot callback bridge: completion-before-wait and wait-before-completion
/// share a lock, and each continuation is resumed exactly once outside that lock.
final class ExitSignal: Sendable {
  private struct State {
    var finished = false
    var waiters: [CheckedContinuation<Void, Never>] = []
  }
  private let state = Mutex(State())

  func finish() {
    let waiters = state.withLock { state -> [CheckedContinuation<Void, Never>] in
      guard !state.finished else { return [] }
      state.finished = true
      let waiters = state.waiters
      state.waiters.removeAll()
      return waiters
    }
    for waiter in waiters { waiter.resume() }
  }

  func wait() async {
    await withCheckedContinuation { continuation in
      let finished = state.withLock { state in
        if state.finished { return true }
        state.waiters.append(continuation)
        return false
      }
      if finished { continuation.resume() }
    }
  }
}

public enum Runner {
  public static func run(
    _ executable: URL, _ arguments: [String],
    environment: [String: String]? = nil, timeout: Duration = .seconds(30)
  ) async throws -> CommandResult {
    try await ProcessWorker().execute(
      executable, arguments, environment: environment, timeout: timeout)
  }

  public static func checked(
    _ executable: URL, _ arguments: [String], environment: [String: String]? = nil
  ) async throws -> String {
    let result = try await run(executable, arguments, environment: environment)
    guard result.status == 0 else { throw ProAppsError.commandFailed(result.status) }
    return result.stdout
  }
}

/// Blocking Foundation/process/file work runs on a dedicated Dispatch executor,
/// not MainActor or the cooperative pool. The MCP service bounds admission.
private actor ProcessWorker {
  nonisolated private let executor = DispatchSerialQueue(label: "apple-pro-apps.process")
  nonisolated var unownedExecutor: UnownedSerialExecutor { executor.asUnownedSerialExecutor() }
  private let captureLimit = 1_048_576

  private func validateCapture(_ url: URL) throws {
    // URL caches resource values. Polling a cached zero-byte size would miss
    // later stderr growth, including a process that exits between polls.
    var current = url
    current.removeCachedResourceValue(forKey: .fileSizeKey)
    guard let size = try current.resourceValues(forKeys: [.fileSizeKey]).fileSize else {
      throw ProAppsError.unavailable("Cannot read native capture size")
    }
    guard size <= captureLimit else { throw ProAppsError.outputLimit }
  }

  func execute(
    _ executable: URL, _ arguments: [String], environment: [String: String]?, timeout: Duration
  ) async throws -> CommandResult {
    try Task.checkCancellation()
    let directory = try Files.temporaryDirectory().appendingPathComponent(
      "pro-apps-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    // Report cleanup failures without replacing the primary command error.
    defer { Cleanup.perform { try FileManager.default.removeItem(at: directory) } }
    let output = directory.appendingPathComponent("stdout")
    let errors = directory.appendingPathComponent("stderr")
    for url in [output, errors] {
      // Keep staging beside the capture, avoiding Foundation's OS temp directory.
      _ = try Files.writeNew(Data(), to: url.path, extensions: [""])
    }
    let outHandle = try FileHandle(forWritingTo: output)
    defer { Cleanup.perform { try outHandle.close() } }
    let errHandle = try FileHandle(forWritingTo: errors)
    defer { Cleanup.perform { try errHandle.close() } }
    let process = Process()
    let exit = ExitSignal()
    process.executableURL = executable
    process.arguments = arguments
    process.environment = environment
    process.standardInput = FileHandle.nullDevice
    process.standardOutput = outHandle
    process.standardError = errHandle
    process.terminationHandler = { _ in exit.finish() }
    defer { process.terminationHandler = nil }
    try process.run()
    let deadline = ContinuousClock.now.advanced(by: timeout)
    do {
      while process.isRunning {
        try Task.checkCancellation()
        guard ContinuousClock.now < deadline else { throw ProAppsError.timedOut }
        for url in [output, errors] { try validateCapture(url) }
        try await Task.sleep(for: .milliseconds(20))
      }
      await exit.wait()
      try Task.checkCancellation()
    } catch {
      if process.isRunning { process.terminate() }
      for _ in 0..<25 where process.isRunning {
        // Cancellation-insensitive bounded grace at a callback boundary;
        // no blocking sleep or detached task on a cooperative executor.
        await withCheckedContinuation { continuation in
          DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(10)) {
            continuation.resume()
          }
        }
      }
      if process.isRunning { Darwin.kill(process.processIdentifier, SIGKILL) }
      await exit.wait()
      throw error
    }
    // A fast process can finish before the first polling iteration. Validate
    // both streams after exit too; stderr is bounded even though never returned.
    for url in [output, errors] { try validateCapture(url) }
    let bytes = try Files.read(output)
    guard bytes.count <= captureLimit else { throw ProAppsError.outputLimit }
    guard let text = String(data: bytes, encoding: .utf8) else {
      throw ProAppsError.unavailable("Native output was not UTF-8")
    }
    return CommandResult(stdout: text, status: process.terminationStatus)
  }
}
