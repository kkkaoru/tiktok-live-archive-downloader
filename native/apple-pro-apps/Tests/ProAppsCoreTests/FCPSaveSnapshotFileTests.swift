import CryptoKit
import Foundation
import Testing

@testable import ProAppsCore

struct FCPSaveSnapshotFileTests {
  private struct Request: Encodable {
    let context: FCPSaveContext
    let snapshotPath: String
    let snapshotSHA256: String
    let historicalInspection: Bool
  }

  @Test(arguments: [false, true]) func freshAndHistoricalSnapshot(_ historical: Bool) throws {
    let directory = try CoreTests().temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let snapshot = directory.appendingPathComponent("snapshot.json")
    let bytes = Data(
      """
      {"elements":[{"path":[0],"role":"AXPopUpButton","identifier":"where popup",
        "value":"\(directory.lastPathComponent)","actions":["AXShowMenu"],"valueSettable":false,"unreadable":[]},
        {"path":[1],"role":"AXTextField","identifier":"saveAsNameTextField",
        "value":"old.mov","actions":[],"valueSettable":true,"unreadable":[]}]}
      """.utf8)
    try bytes.write(to: snapshot)
    let pin = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    let context = FCPSaveContext(
      directory: directory.path, fileName: "new.mov",
      observedDirectory: directory.path, window: .init(title: "Save"), bundleID: nil,
      saveDispatched: false)
    let request = directory.appendingPathComponent("request.json")
    func write(_ path: String, _ digest: String) throws {
      try JSONEncoder().encode(
        Request(
          context: context, snapshotPath: path,
          snapshotSHA256: digest, historicalInspection: historical)
      ).write(to: request)
    }
    try write(snapshot.path, pin)
    let planned = try FCPSaveSnapshotFile.plan(path: request.path)
    #expect((planned.next == nil) == historical)
    if historical {
      #expect(planned.state == "historical-inspection-set-name-and-observe")
    } else {
      #expect(throws: ProAppsError.self) {
        try FCPSaveSnapshotFile.plan(path: request.path, now: Date().addingTimeInterval(60))
      }
    }
    try write(snapshot.path, String(repeating: "0", count: 64))
    #expect(throws: ProAppsError.self) { try FCPSaveSnapshotFile.plan(path: request.path) }
    let link = directory.appendingPathComponent("link.json")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: snapshot)
    try write(link.path, pin)
    #expect(throws: ProAppsError.self) { try FCPSaveSnapshotFile.plan(path: request.path) }
  }

  @Test func cliParsesOnlyExplicitLocalRequest() throws {
    #expect(
      try Command.parse(["measure-media", "audio_measure", "/example/request.json"])
        == .measureMedia("audio_measure", "/example/request.json"))
    #expect(throws: ProAppsError.self) { try Command.parse(["measure-media"]) }
    #expect(
      try Command.parse(["mux-pcm", "/example/request.json"]) == .muxPCM("/example/request.json"))
    #expect(throws: ProAppsError.self) { try Command.parse(["mux-pcm"]) }
    #expect(
      try Command.parse(["fcp-save-plan", "/example/request.json"])
        == .fcpSavePlan("/example/request.json"))
    #expect(throws: ProAppsError.self) { try Command.parse(["fcp-save-plan"]) }
    #expect(throws: ProAppsError.self) { try Command.parse(["fcp-save-plan", "relative.json"]) }
  }
}
