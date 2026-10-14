import CryptoKit
import Foundation

/// Offline, fingerprint-bound Save planning; no app launch, AX or file writes.
public enum FCPSaveSnapshotFile {
  private struct Input: Decodable {
    let context: FCPSaveContext
    let snapshotPath: String
    let snapshotSHA256: String
    let historicalInspection: Bool
  }
  private struct Snapshot: Decodable { let elements: [UIElementSummary] }

  public static func plan(path: String, now: Date = Date()) throws -> FCPSavePlan {
    let input = try JSONDecoder().decode(Input.self, from: Files.read(Files.existing(path)))
    let source = try Files.existing(input.snapshotPath)
    guard source.path == input.snapshotPath else {
      throw ProAppsError.invalid("Snapshot must have a canonical path")
    }
    let bytes = try Files.read(source)
    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    guard digest == input.snapshotSHA256 else {
      throw ProAppsError.invalid("Save snapshot fingerprint differs")
    }
    guard
      let modified = try source.resourceValues(forKeys: [.contentModificationDateKey])
        .contentModificationDate,
      input.historicalInspection
        || (now.timeIntervalSince(modified) >= 0 && now.timeIntervalSince(modified) <= 30)
    else { throw ProAppsError.invalid("Save snapshot is stale; obtain a fresh observation") }
    let snapshot = try JSONDecoder().decode(Snapshot.self, from: bytes)
    let result = try FCPSavePlanner.plan(input.context, elements: snapshot.elements)
    if input.historicalInspection {
      return FCPSavePlan(
        outputPath: result.outputPath, next: nil,
        state: "historical-inspection-\(result.state)", directoryBinding: result.directoryBinding)
    }
    return result
  }
}
