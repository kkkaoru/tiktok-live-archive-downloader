import AVFoundation
import Darwin
import Dispatch
import Foundation
import Vision

/// Explicit session input. Samples are original source clocks, never an inferred CFR grid.
public struct SourceTextSessionRequest: Codable, Sendable {
  public let checkoutRoot: String
  public let path: String
  public let checkpointDirectory: String
  public let samples: [ForegroundTime]
  public let width: Int
  public let height: Int
  public let maximumSeconds: Double
  public let workers: Int?
  public let allowedSourceRoot: String?

  public init(
    checkoutRoot: String, path: String, checkpointDirectory: String,
    samples: [ForegroundTime], width: Int, height: Int, maximumSeconds: Double,
    workers: Int? = nil, allowedSourceRoot: String? = nil
  ) {
    self.checkoutRoot = checkoutRoot
    self.path = path
    self.checkpointDirectory = checkpointDirectory
    self.samples = samples
    self.width = width
    self.height = height
    self.maximumSeconds = maximumSeconds
    self.workers = workers
    self.allowedSourceRoot = allowedSourceRoot
  }
}

public struct SourceTextSessionIdentity: Codable, Equatable, Sendable {
  public let device: Int32
  public let inode: UInt64
  public let bytes: Int64
  public let links: UInt16
  public let modifiedSeconds: Int64
  public let modifiedNanoseconds: Int64
  public let changedSeconds: Int64
  public let changedNanoseconds: Int64
}

public struct SourceTextSessionTiming: Codable, Sendable {
  public var assetSetupSeconds = 0.0
  public var ocrSetupSeconds = 0.0
  public var imageGenerationSeconds = 0.0
  public var ocrSeconds = 0.0
  public var serializationSeconds = 0.0
  public var checkpointWriteSeconds = 0.0
  public var elapsedSeconds = 0.0
}

public struct SourceTextSessionBlock: Codable, Sendable {
  public let firstSample: Int
  public let requested: [ForegroundTime]
  public let actual: [ForegroundTime]
  public let measurementsPath: String
}

public struct SourceTextSessionReport: Encodable, Sendable {
  public let protocolVersion: String
  public let sourceModified: Bool
  public let qualityApproved: Bool
  /// Parent producer/consumer boundaries still own fresh cryptographic source checks.
  public let sourceHashVerified: Bool
  public let sourceIdentity: SourceTextSessionIdentity
  public let frames: Int
  public let blocks: [SourceTextSessionBlock]
  public let timing: SourceTextSessionTiming

  public init(
    sourceIdentity: SourceTextSessionIdentity, frames: Int,
    blocks: [SourceTextSessionBlock], timing: SourceTextSessionTiming
  ) {
    protocolVersion = "source-caption-ocr-session-v1"
    sourceModified = false
    qualityApproved = false
    sourceHashVerified = false
    self.sourceIdentity = sourceIdentity
    self.frames = frames
    self.blocks = blocks
    self.timing = timing
  }
}

struct SourceTextSessionFrame: Sendable {
  let measurement: FrameTextMeasurement
  let actual: ForegroundTime
  let imageGenerationSeconds: Double
  let ocrSeconds: Double
}

protocol SourceTextSessionWorker: Actor {
  func process(_ request: SourceTextSessionRequest, sink: SourceTextSessionSink) async throws
}

/// Decoder/request are session-local, never aliased across an actor suspension.
actor NativeSourceTextSessionWorker: SourceTextSessionWorker {
  nonisolated private let executor = DispatchSerialQueue(
    label: "apple-pro-apps.source-text-session")
  nonisolated var unownedExecutor: UnownedSerialExecutor { executor.asUnownedSerialExecutor() }

  func process(_ request: SourceTextSessionRequest, sink: SourceTextSessionSink) async throws {
    var timing = SourceTextSessionTiming()
    let clock = ContinuousClock()
    let assetStart = clock.now
    let generator = try await FrameProbe().sessionSource(
      path: request.path, samples: request.samples)
    timing.assetSetupSeconds = SourceTextSession.seconds(assetStart.duration(to: clock.now))
    let ocrStart = clock.now
    let recognition = try CPUTextRecognition.request()
    timing.ocrSetupSeconds = SourceTextSession.seconds(ocrStart.duration(to: clock.now))
    try await sink.begin(timing)
    for sample in request.samples {
      try await sink.beforeFrame()
      let imageStart = clock.now
      let requested = try sample.time()
      let frame = try await generator.image(at: requested)
      let actual = try ForegroundTime.actual(frame.actualTime, requested: requested)
      let imageSeconds = SourceTextSession.seconds(imageStart.duration(to: clock.now))
      try await sink.receive(
        Self.textFrame(
          image: frame.image, sample: sample, actual: actual,
          imageGenerationSeconds: imageSeconds, recognition: recognition))
    }
  }

  static func textFrame(
    image: CGImage, sample: ForegroundTime, actual: ForegroundTime,
    imageGenerationSeconds: Double, recognition: RecognizeTextRequest
  ) async throws -> SourceTextSessionFrame {
    let clock = ContinuousClock()
    let ocrStart = clock.now
    let observations = try await recognition.perform(on: image)
    guard observations.count <= SourceTextSession.maximumLinesPerFrame else {
      throw ProAppsError.outputLimit
    }
    var lines: [FrameTextLine] = []
    var textBytes = 0
    for observation in observations {
      try Task.checkCancellation()
      guard let candidate = observation.topCandidates(1).first else {
        throw ProAppsError.unavailable("OCR observation has no text candidate")
      }
      textBytes += candidate.string.utf8.count
      guard textBytes <= SourceTextSession.maximumTextBytesPerBlock else {
        throw ProAppsError.outputLimit
      }
      let size = CGSize(width: image.width, height: image.height)
      let raw = observation.boundingBox.toImageCoordinates(size, origin: .upperLeft)
      lines.append(
        try FrameProbe.textLine(
          text: candidate.string, confidence: candidate.confidence,
          rawRectangle: raw, selected: CGRect(origin: .zero, size: size),
          fullFrameHeight: image.height))
    }
    return SourceTextSessionFrame(
      measurement: FrameTextMeasurement(
        requestedTimeSeconds: try sample.time().seconds,
        actualTimeSeconds: try actual.time().seconds,
        width: image.width, height: image.height, lines: lines), actual: actual,
      imageGenerationSeconds: imageGenerationSeconds,
      ocrSeconds: SourceTextSession.seconds(ocrStart.duration(to: clock.now)))
  }
}

/// Writes each completed eight-frame block exclusively. Failure preserves completed
/// blocks; reuse/resume requires the parent's original immutable receipts and hashes.
public enum SourceTextSession {
  public static let maximumSamples = 64
  public static let blockSamples = 8
  public static let maximumLinesPerFrame = 64
  public static let maximumTextBytesPerBlock = 32768

  public static func run(_ request: SourceTextSessionRequest) async throws
    -> SourceTextSessionReport
  {
    let workers = request.workers ?? 1
    guard (1...2).contains(workers) else { throw ProAppsError.invalid("Invalid OCR worker count") }
    if workers == 1 {
      return try await run(request, worker: NativeSourceTextSessionWorker())
    }
    try Task.checkCancellation()
    let paths = try SessionPaths(request)
    defer { paths.close() }
    let sink = SourceTextSessionSink(request: request, paths: paths)
    let result = try await OCRFrameSession.run(
      OCRFrameSessionRequest(
        path: request.path, samples: request.samples, width: request.width, height: request.height,
        maximumSeconds: request.maximumSeconds),
      factory: { NativeOCRFrameSessionWorker() },
      beforeFrame: { try await sink.beforeFrame() })
    try await sink.begin(result.setup)
    for frame in result.frames { try await sink.receive(frame) }
    return try await sink.finish()
  }

  static func seconds(_ duration: Duration) -> Double {
    let parts = duration.components
    return Double(parts.seconds) + Double(parts.attoseconds) / 1_000_000_000_000_000_000
  }

  static func run(_ request: SourceTextSessionRequest, worker: any SourceTextSessionWorker)
    async throws -> SourceTextSessionReport
  {
    try Task.checkCancellation()
    let paths = try SessionPaths(request)
    defer { paths.close() }
    let sink = SourceTextSessionSink(request: request, paths: paths)
    try await worker.process(request, sink: sink)
    return try await sink.finish()
  }

  static func deadline(
    start: ContinuousClock.Instant, now: ContinuousClock.Instant,
    maximumSeconds: Double
  ) throws {
    try Task.checkCancellation()
    guard seconds(start.duration(to: now)) < maximumSeconds else { throw ProAppsError.timedOut }
  }

  static func validate(
    _ frame: SourceTextSessionFrame, requested: ForegroundTime,
    width: Int, height: Int
  ) throws {
    let row = frame.measurement
    let requestedTime = try requested.time()
    let actualTime = try frame.actual.time()
    guard actualTime == requestedTime,
      row.requestedTimeSeconds == requestedTime.seconds,
      row.actualTimeSeconds == actualTime.seconds,
      row.width == width, row.height == height,
      frame.imageGenerationSeconds.isFinite, frame.imageGenerationSeconds >= 0,
      frame.ocrSeconds.isFinite, frame.ocrSeconds >= 0
    else { throw ProAppsError.invalid("OCR session clock, canvas or timing differs") }
    guard row.lines.count <= maximumLinesPerFrame else { throw ProAppsError.outputLimit }
    for line in row.lines {
      guard line.confidence.isFinite, (0...1).contains(line.confidence) else {
        throw ProAppsError.invalid("Invalid OCR confidence")
      }
      _ = try FrameProbe.region(line.region, width: width, height: height)
    }
  }
}

/// Serial eight-frame accumulator: checks before decode and after each native result.
actor SourceTextSessionSink {
  private let request: SourceTextSessionRequest
  private let paths: SessionPaths
  private let clock = ContinuousClock()
  private let start = ContinuousClock().now
  private var prepared = false
  private var timing = SourceTextSessionTiming()
  private var blocks: [SourceTextSessionBlock] = []
  private var measurements: [FrameTextMeasurement] = []
  private var actual: [ForegroundTime] = []
  private var processed = 0
  private var textBytes = 0

  init(request: SourceTextSessionRequest, paths: SessionPaths) {
    self.request = request
    self.paths = paths
  }

  func begin(_ setup: SourceTextSessionTiming) throws {
    guard !prepared, setup.assetSetupSeconds.isFinite, setup.assetSetupSeconds >= 0,
      setup.ocrSetupSeconds.isFinite, setup.ocrSetupSeconds >= 0
    else { throw ProAppsError.invalid("Invalid or repeated OCR setup") }
    timing.assetSetupSeconds = setup.assetSetupSeconds
    timing.ocrSetupSeconds = setup.ocrSetupSeconds
    prepared = true
    try beforeFrame()
  }

  func beforeFrame() throws {
    try paths.check()
    try SourceTextSession.deadline(
      start: start, now: clock.now, maximumSeconds: request.maximumSeconds)
  }

  func receive(_ frame: SourceTextSessionFrame) throws {
    guard prepared, processed < request.samples.count else {
      throw ProAppsError.invalid("Unexpected session frame")
    }
    try beforeFrame()
    let sample = request.samples[processed]
    try SourceTextSession.validate(
      frame, requested: sample, width: request.width, height: request.height)
    textBytes += frame.measurement.lines.reduce(0) { $0 + $1.text.utf8.count }
    guard textBytes <= SourceTextSession.maximumTextBytesPerBlock else {
      throw ProAppsError.outputLimit
    }
    measurements.append(frame.measurement)
    actual.append(frame.actual)
    timing.imageGenerationSeconds += frame.imageGenerationSeconds
    timing.ocrSeconds += frame.ocrSeconds
    processed += 1
    if measurements.count == SourceTextSession.blockSamples || processed == request.samples.count {
      try flush()
    }
  }

  private func flush() throws {
    let first = processed - measurements.count
    let number = first / SourceTextSession.blockSamples
    let destination = paths.directory.appendingPathComponent("block-\(number).json")
    let block = SourceTextSessionBlock(
      firstSample: first,
      requested: Array(request.samples[first..<processed]), actual: actual,
      measurementsPath: destination.path)
    let serializationStart = clock.now
    let data = try JSONEncoder().encode(measurements)
    let receipt = try JSONEncoder().encode(block)
    timing.serializationSeconds += SourceTextSession.seconds(
      serializationStart.duration(to: clock.now))
    try paths.check()
    let writeStart = clock.now
    try paths.write(data, name: "block-\(number).json")
    try paths.check()
    try paths.write(receipt, name: "block-\(number)-clocks.json")
    timing.checkpointWriteSeconds += SourceTextSession.seconds(writeStart.duration(to: clock.now))
    blocks.append(block)
    measurements.removeAll(keepingCapacity: true)
    actual.removeAll(keepingCapacity: true)
    textBytes = 0
  }

  func finish() throws -> SourceTextSessionReport {
    guard prepared, processed == request.samples.count, measurements.isEmpty else {
      throw ProAppsError.invalid("OCR session did not deliver every original PTS")
    }
    try beforeFrame()
    timing.elapsedSeconds = SourceTextSession.seconds(start.duration(to: clock.now))
    let report = SourceTextSessionReport(
      sourceIdentity: paths.sourceIdentity,
      frames: processed, blocks: blocks, timing: timing)
    try paths.write(JSONEncoder().encode(report), name: "complete.json")
    return report
  }
}

struct SessionPaths {

  let source: URL
  let directory: URL
  let sourceIdentity: SourceTextSessionIdentity
  private let directoryDevice: Int32
  private let directoryInode: UInt64
  private let directoryDescriptor: Int32
  private let sourceRoot: URL
  private let sourceRootDevice: Int32
  private let sourceRootInode: UInt64

  init(_ request: SourceTextSessionRequest) throws {
    guard (1...SourceTextSession.maximumSamples).contains(request.samples.count),
      request.maximumSeconds.isFinite, request.maximumSeconds > 0, request.maximumSeconds <= 60,
      request.width > 0, request.height > 0, request.width <= 16384, request.height <= 16384,
      request.width * request.height <= 16_777_216
    else { throw ProAppsError.invalid("Invalid OCR session budget") }
    let times = try request.samples.map { try $0.time() }
    guard zip(times, times.dropFirst()).allSatisfy({ $0.0 < $0.1 }) else {
      throw ProAppsError.invalid("Session original PTS must be strictly ordered")
    }
    let root = try Self.canonical(request.checkoutRoot)
    guard
      FileManager.default.fileExists(
        atPath: root.appendingPathComponent("native/apple-pro-apps/Package.swift").path)
    else { throw ProAppsError.invalid("Expected an explicit checkout with the native package") }
    sourceRoot = try Self.canonical(request.allowedSourceRoot ?? root.path)
    let sourceParent = try Self.metadata(sourceRoot, kind: S_IFDIR)
    sourceRootDevice = sourceParent.st_dev
    sourceRootInode = sourceParent.st_ino
    guard sourceRoot.path != "/", request.path.hasPrefix(sourceRoot.path + "/"),
      request.checkpointDirectory.hasPrefix(root.path + "/edits/work/")
    else {
      throw ProAppsError.invalid("Session paths escape their declared source or checkout root")
    }
    source = try Self.canonical(request.path)
    directory = try Self.canonical(request.checkpointDirectory)
    let parent = try Self.metadata(directory, kind: S_IFDIR)
    guard parent.st_uid == getuid(), parent.st_mode & 0o777 == 0o700,
      try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty
    else { throw ProAppsError.invalid("Checkpoint directory must be private and empty") }
    directoryDevice = parent.st_dev
    directoryInode = parent.st_ino
    sourceIdentity = try Self.identity(source)
    if request.allowedSourceRoot != nil && sourceIdentity.links != 1 {
      throw ProAppsError.invalid("Explicit source roots require a single-link source")
    }
    // Native framework/process scratch is explicit; never silently fall back to OS temporary storage.
    guard let scratch = ProcessInfo.processInfo.environment["TMPDIR"],
      scratch.hasPrefix(root.path + "/edits/work/")
    else { throw ProAppsError.invalid("Session requires repository-local process scratch") }
    _ = try Self.metadata(Self.canonical(scratch), kind: S_IFDIR)
    let descriptor = Darwin.open(directory.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW)
    guard descriptor >= 0 else { throw Files.systemError(errno) }
    var opened = stat()
    guard Darwin.fstat(descriptor, &opened) == 0,
      opened.st_dev == directoryDevice, opened.st_ino == directoryInode
    else {
      Darwin.close(descriptor)
      throw ProAppsError.invalid("Checkpoint parent changed while opening")
    }
    directoryDescriptor = descriptor
  }

  func check() throws {
    let sourceParent = try Self.metadata(Self.canonical(sourceRoot.path), kind: S_IFDIR)
    let parent = try Self.metadata(Self.canonical(directory.path), kind: S_IFDIR)
    var opened = stat()
    guard sourceParent.st_dev == sourceRootDevice, sourceParent.st_ino == sourceRootInode,
      parent.st_dev == directoryDevice, parent.st_ino == directoryInode,
      Darwin.fstat(directoryDescriptor, &opened) == 0,
      opened.st_dev == directoryDevice, opened.st_ino == directoryInode,
      parent.st_uid == getuid(), parent.st_mode & 0o777 == 0o700,
      try Self.identity(Self.canonical(source.path)) == sourceIdentity
    else { throw ProAppsError.invalid("OCR session source or checkpoint parent changed") }
  }

  func close() { Darwin.close(directoryDescriptor) }

  /// Relative descriptor operations cannot redirect publication through a raced symlink.
  func write(_ data: Data, name: String) throws {
    try check()
    guard data.count <= Files.maximumBytes, !name.contains("/"), name.hasSuffix(".json") else {
      throw ProAppsError.invalid("Invalid checkpoint publication")
    }
    let temporary = ".source-text-session-\(UUID().uuidString)"
    let descriptor = Darwin.openat(
      directoryDescriptor, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o600)
    guard descriptor >= 0 else { throw Files.systemError(errno) }
    defer {
      Darwin.close(descriptor)
      // Only our exclusive staging name; completed blocks are never removed.
      _ = Darwin.unlinkat(directoryDescriptor, temporary, 0)
    }
    try FileHandle(fileDescriptor: descriptor, closeOnDealloc: false).write(contentsOf: data)
    guard Darwin.fsync(descriptor) == 0 else { throw Files.systemError(errno) }
    guard Darwin.linkat(directoryDescriptor, temporary, directoryDescriptor, name, 0) == 0 else {
      throw Files.systemError(errno)
    }
    try check()
  }

  static func canonical(_ path: String) throws -> URL {
    let url = try Files.absolute(path)
    guard url.path == path, url.resolvingSymlinksInPath().path == path else {
      throw ProAppsError.invalid("Session paths must be canonical without symlinks")
    }
    return url
  }

  static func metadata(_ path: URL, kind: mode_t) throws -> stat {
    var value = stat()
    guard lstat(path.path, &value) == 0 else { throw Files.systemError(errno) }
    guard value.st_mode & mode_t(S_IFMT) == mode_t(kind) else {
      throw ProAppsError.invalid("Unexpected session file type")
    }
    return value
  }

  static func identity(_ path: URL) throws -> SourceTextSessionIdentity {
    let value = try metadata(path, kind: S_IFREG)
    guard value.st_size > 0 else { throw ProAppsError.invalid("Session source must be nonempty") }
    return SourceTextSessionIdentity(
      device: value.st_dev, inode: value.st_ino, bytes: value.st_size, links: value.st_nlink,
      modifiedSeconds: Int64(value.st_mtimespec.tv_sec),
      modifiedNanoseconds: Int64(value.st_mtimespec.tv_nsec),
      changedSeconds: Int64(value.st_ctimespec.tv_sec),
      changedNanoseconds: Int64(value.st_ctimespec.tv_nsec))
  }
}
