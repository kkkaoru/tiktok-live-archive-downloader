import Foundation
import Testing

@testable import ProAppsCore

struct CoreTests {
  func temporary() throws -> URL {
    let url = try Files.temporaryDirectory().appendingPathComponent(
      "pro-apps-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: url, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    return url
  }

  @Test func nativeErrorMappingPreservesKnownAndUnknownFailures() {
    #expect(Files.systemError(2).code == .ENOENT)
    #expect(Files.systemError(Int32.max).code == .EIO)
  }

  @Test func absolutePathsAndRegularFiles() throws {
    #expect(throws: (any Error).self) { try Files.absolute("relative/file") }
    #expect(throws: (any Error).self) { try Files.absolute("/tmp/a\0b") }
    #expect(throws: (any Error).self) {
      try Files.absolute("/" + String(repeating: "x", count: 5000))
    }
    let directory = try temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    #expect(throws: (any Error).self) { try Files.existing(directory.path) }
    let file = try Files.writeNew(
      Data("test".utf8), to: directory.appendingPathComponent("test.fcpxml").path,
      extensions: ["fcpxml"])
    #expect(try Files.read(file) == Data("test".utf8))
    #expect(try Files.existing(file.path, extensions: ["fcpxml"]) == file)
    #expect(throws: (any Error).self) { try Files.existing(file.path, extensions: ["mid"]) }
    #expect(throws: (any Error).self) {
      try Files.writeNew(Data(), to: file.path, extensions: ["fcpxml"])
    }
    #expect(try Files.read(file) == Data("test".utf8))
    let empty = try Files.writeNew(
      Data(), to: directory.appendingPathComponent("empty.mid").path, extensions: ["mid"])
    #expect(try Files.read(empty).isEmpty)
    let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
    #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
    let link = directory.appendingPathComponent("dangling.fcpxml")
    try FileManager.default.createSymbolicLink(
      atPath: link.path, withDestinationPath: directory.appendingPathComponent("missing").path)
    #expect(throws: (any Error).self) {
      try Files.writeNew(Data(), to: link.path, extensions: ["fcpxml"])
    }
    #expect(throws: (any Error).self) {
      try Files.writeNew(
        Data(), to: directory.appendingPathComponent("no.txt").path, extensions: ["mid"])
    }
    #expect(throws: (any Error).self) {
      try Files.writeNew(Data(count: Files.maximumBytes + 1), to: file.path, extensions: ["fcpxml"])
    }
  }

  @Test func xmlRejectsEntitiesAndWrongRoots() throws {
    let good = Data(
      "<?xml version=\"1.0\"?><!DOCTYPE fcpxml><fcpxml version=\"1.12\"><resources/></fcpxml>".utf8)
    let summary = try Interchange.inspect(good, kind: .fcpxml)
    #expect(summary.elementCount == 2)
    #expect(summary.version == "1.12")
    #expect(throws: (any Error).self) {
      try Interchange.inspect(Data([0xff, 0xfe]), kind: .motion)
    }
    #expect(
      try Interchange.inspect(Data("<ozml version=\"6\"/>".utf8), kind: .motion).root == "ozml")
  }

  @Test func xmlPatchCopiesAndRequiresUniqueLeaf() throws {
    let directory = try temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let source =
      "<fcpxml version=\"1.12\"><resources><asset name=\"old\"/><asset name=\"other\"/></resources></fcpxml>"
    let input = try Interchange.write(
      source, kind: .fcpxml, output: directory.appendingPathComponent("input.fcpxml").path,
      allowUndocumented: false)
    let output = directory.appendingPathComponent("output.fcpxml")
    _ = try Interchange.patch(
      input: input.path, output: output.path, kind: .fcpxml,
      changes: [.init(xpath: "/fcpxml/resources/asset[1]/@name", value: "new<&")],
      allowUndocumented: false)
    let fragments = try Interchange.query(
      Files.read(output), kind: .fcpxml, xpath: "//asset[1]/@name", limit: 1)
    #expect(fragments.count == 1)
    #expect(try Files.read(input) == Data(source.utf8))
    #expect(throws: (any Error).self) {
      try Interchange.patch(
        input: input.path, output: output.path, kind: .fcpxml, changes: [], allowUndocumented: false
      )
    }
    #expect(throws: (any Error).self) {
      try Interchange.write(
        "<ozml/>", kind: .motion, output: directory.appendingPathComponent("copy.motn").path,
        allowUndocumented: false)
    }
    _ = try Interchange.write(
      "<ozml><value>1</value></ozml>", kind: .motion,
      output: directory.appendingPathComponent("copy.motn").path, allowUndocumented: true)
    #expect(throws: (any Error).self) {
      try Interchange.query(Data(source.utf8), kind: .fcpxml, xpath: "//asset", limit: 0)
    }
  }

  @Test func patchesEmptyAndTextOnlyLeavesWithoutAddingStructure() throws {
    let directory = try temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let input = try Interchange.write(
      "<fcpxml><title/><label>old</label><comment><!--keep--></comment></fcpxml>", kind: .fcpxml,
      output: directory.appendingPathComponent("leaves.fcpxml").path, allowUndocumented: false)
    let output = try Interchange.patch(
      input: input.path, output: directory.appendingPathComponent("changed.fcpxml").path,
      kind: .fcpxml,
      changes: [
        .init(xpath: "/fcpxml/title", value: "hello"), .init(xpath: "/fcpxml/label", value: "new"),
      ], allowUndocumented: false)
    #expect(
      try Interchange.query(Files.read(output), kind: .fcpxml, xpath: "/fcpxml/title", limit: 1)
        == ["<title>hello</title>"])
    #expect(
      try Interchange.query(Files.read(output), kind: .fcpxml, xpath: "/fcpxml/label", limit: 1)
        == ["<label>new</label>"])
  }

  @Test func motionPatchesRequireTheirOwnExplicitOptIn() throws {
    let directory = try temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let input = try Interchange.write(
      "<ozml><value>old</value></ozml>", kind: .motion,
      output: directory.appendingPathComponent("synthetic.motn").path, allowUndocumented: true)
    let destination = directory.appendingPathComponent("changed.motn")
    #expect(throws: (any Error).self) {
      try Interchange.patch(
        input: input.path, output: destination.path, kind: .motion,
        changes: [.init(xpath: "/ozml/value", value: "new")], allowUndocumented: false)
    }
    #expect(!FileManager.default.fileExists(atPath: destination.path))
    let output = try Interchange.patch(
      input: input.path, output: destination.path, kind: .motion,
      changes: [.init(xpath: "/ozml/value", value: "new")], allowUndocumented: true)
    #expect(
      try Interchange.query(Files.read(output), kind: .motion, xpath: "/ozml/value", limit: 1) == [
        "<value>new</value>"
      ])
  }

  @Test func rejectsCommentBearingLeavesWithoutRemovingComments() throws {
    let directory = try temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let input = try Interchange.write(
      "<fcpxml><comment><!--keep--></comment></fcpxml>", kind: .fcpxml,
      output: directory.appendingPathComponent("comment.fcpxml").path, allowUndocumented: false)
    #expect(throws: (any Error).self) {
      try Interchange.patch(
        input: input.path, output: directory.appendingPathComponent("changed.fcpxml").path,
        kind: .fcpxml, changes: [.init(xpath: "/fcpxml/comment", value: "replace")],
        allowUndocumented: false)
    }
    #expect(
      try Interchange.query(
        Files.read(input), kind: .fcpxml, xpath: "/fcpxml/comment/comment()", limit: 1) == [
          "<!--keep-->"
        ])
  }

  @Test func midiFileEncodingAndBounds() throws {
    #expect(try MIDIFile.variableLength(0) == [0])
    #expect(try MIDIFile.variableLength(127) == [127])
    #expect(try MIDIFile.variableLength(128) == [0x81, 0])
    #expect(try MIDIFile.variableLength(0x0fff_ffff) == [0xff, 0xff, 0xff, 0x7f])
    #expect(throws: (any Error).self) { try MIDIFile.variableLength(-1) }
    #expect(throws: (any Error).self) { try MIDIFile.variableLength(0x1000_0000) }
    let notes = [MIDINote(note: 60, velocity: 80, startTick: 0, durationTick: 480)]
    let bytes = try MIDIFile.create(notes: notes, bpm: 120)
    #expect(Array(bytes.prefix(14)) == [77, 84, 104, 100, 0, 0, 0, 6, 0, 0, 0, 1, 1, 224])
    #expect(Array(bytes[14..<22]) == [77, 84, 114, 107, 0, 0, 0, 20])
    #expect(Array(bytes.suffix(4)) == [0, 0xff, 0x2f, 0])
    #expect(bytes.range(of: Data([0x90, 60, 80])) != nil)
    #expect(bytes.range(of: Data([0x80, 60, 0])) != nil)
    #expect(throws: (any Error).self) { try MIDIFile.create(notes: [], bpm: 120) }
    #expect(throws: (any Error).self) { try MIDIFile.create(notes: notes, bpm: .nan) }
    #expect(throws: (any Error).self) {
      try MIDIFile.create(notes: notes, bpm: 120, ticksPerQuarter: 0)
    }
    #expect(throws: (any Error).self) {
      try MIDIFile.create(
        notes: [.init(note: 128, velocity: 80, startTick: 0, durationTick: 10)], bpm: 120)
    }
  }

  @Test func midiMessagesAreTypedAndBounded() throws {
    #expect(
      try MIDIControl.message(kind: .controlChange, channel: 1, number: 7, value: 100)
        == 0x20b0_0764)
    #expect(
      try MIDIControl.message(kind: .programChange, channel: 16, number: 2, value: 0) == 0x20cf_0200
    )
    #expect(
      try MIDIControl.message(kind: .pitchBend, channel: 1, number: 0, value: 8192) == 0x20e0_0040)
    #expect(throws: (any Error).self) {
      try MIDIControl.message(kind: .controlChange, channel: 1, number: 7, value: 128)
    }
    #expect(throws: (any Error).self) {
      try MIDIControl.message(kind: .programChange, channel: 1, number: 7, value: 1)
    }
    #expect(throws: (any Error).self) {
      try MIDIControl.message(kind: .pitchBend, channel: 1, number: 0, value: 16384)
    }
  }

  @Test func compressorInspectionUsesAnEncodedFileURL() {
    #expect(
      Compressor.inspectionArguments(source: URL(fileURLWithPath: "/tmp/a b#c.mp4")) == [
        "-checkstream", "file:///tmp/a%20b%23c.mp4",
      ])
  }

  @Test func oscPacketAndAddressValidation() throws {
    #expect(Array(try OSC.packet(path: "/gain", value: -1).suffix(4)) == [191, 128, 0, 0])
    #expect(
      try OSC.packet(path: "/gain", value: 1)
        == Data([47, 103, 97, 105, 110, 0, 0, 0, 44, 102, 0, 0, 63, 128, 0, 0]))
    #expect(throws: (any Error).self) { try OSC.packet(path: "/gain", value: .infinity) }
    #expect(throws: (any Error).self) { try OSC.send(port: 80, path: "/gain", value: 1) }
  }

  @Test(arguments: [
    (0, 5, "00:00:00;00", "00:00:05;00"), (3661, 599, "01:01:01;00", "01:11:00;00"),
  ])
  func compressorRangesAreBoundedSourceTimecodes(
    _ start: Int, _ duration: Int, _ lower: String, _ upper: String
  ) throws {
    let range = Compressor.TimeRange(startSeconds: start, durationSeconds: duration)
    #expect(try range.arguments() == ["-in", lower, "-out", upper])
    let decoded = try JSONDecoder().decode(
      Compressor.TimeRange.self, from: JSONEncoder().encode(range))
    #expect(decoded.startSeconds == start)
    let arguments = try Compressor.submissionArguments(
      source: URL(fileURLWithPath: "/tmp/source.mov"),
      preset: URL(fileURLWithPath: "/tmp/apple.setting"),
      output: URL(fileURLWithPath: "/tmp/out.mp4"), batchName: "synthetic", range: range)
    #expect(Array(arguments.dropFirst(4).prefix(4)) == ["-in", lower, "-out", upper])
  }

  @Test(arguments: [(Int.max, 5), (-1, 5), (0, 0), (0, 601), (0, Int.max)])
  func compressorRangesRejectInvalidValuesBeforeArithmetic(_ start: Int, _ duration: Int) {
    #expect(throws: (any Error).self) {
      try Compressor.TimeRange(startSeconds: start, durationSeconds: duration).arguments()
    }
  }

  @Test func compressorPlansNeverUseShellOrGlobalControl() throws {
    #expect(
      try Compressor.monitoringArguments(id: "job-123", job: true) == [
        "-monitor", "-jobid", "job-123", "-once", "-timeout", "10", "-outputformat", "json",
      ])
    #expect(
      try Compressor.monitoringArguments(id: "batch-123", job: false, control: .cancel).prefix(3)
        == ["-kill", "-batchid", "batch-123"])
    #expect(
      try Compressor.monitoringArguments(id: "123", job: false, control: .pause).first == "-pause")
    #expect(
      try Compressor.monitoringArguments(id: "123", job: false, control: .resume).first == "-resume"
    )
    let args = try Compressor.submissionArguments(
      source: URL(fileURLWithPath: "/tmp/a ; b.mov"),
      preset: URL(fileURLWithPath: "/tmp/p.cmprstng"), output: URL(fileURLWithPath: "/tmp/out.mov"),
      batchName: "a ; b")
    #expect(args[3] == "/tmp/a ; b.mov")
    #expect(args[1] == "a ; b")
    #expect(throws: (any Error).self) {
      try Compressor.submissionArguments(
        source: URL(fileURLWithPath: "/a"), preset: URL(fileURLWithPath: "/b"),
        output: URL(fileURLWithPath: "/c"), batchName: "")
    }
    let directory = try temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let first = try Compressor.reserveOutput(directory: directory.path, name: "out.mov")
    let second = try Compressor.reserveOutput(directory: directory.path, name: "out.mov")
    #expect(first != second)
    #expect(!FileManager.default.fileExists(atPath: first.path))
    #expect(throws: (any Error).self) {
      try Compressor.reserveOutput(directory: directory.path, name: "../existing.mov")
    }
  }

  @Test(arguments: ["stdout", "stderr"])
  func captureLimitsAlsoApplyToFastExitingCommands(_ stream: String) async throws {
    let binary = URL(fileURLWithPath: stream == "stdout" ? "/usr/bin/head" : "/usr/bin/awk")
    let arguments =
      stream == "stdout"
      ? ["-c", "1048577", "/dev/zero"]
      : ["BEGIN { for(i=0;i<1100;i++) printf \"%1024s\", \"\" > \"/dev/stderr\" }"]
    do {
      let result = try await Runner.run(binary, arguments, timeout: .seconds(5))
      Issue.record(
        "Oversized native capture must fail; status=\(result.status), stdoutBytes=\(result.stdout.utf8.count)"
      )
    } catch let error as ProAppsError {
      #expect(
        error.description
          == "Native output exceeded its bounded capture limit. Do not automatically retry.")
    }
  }

  @Test func nativeRunnerCapturesAndTerminates() async throws {
    let result = try await Runner.run(
      URL(fileURLWithPath: "/usr/bin/printf"), ["%s", "literal ; $(not-a-shell)"])
    #expect(result.stdout == "literal ; $(not-a-shell)")
    #expect(result.status == 0)
    await #expect(throws: (any Error).self) {
      try await Runner.checked(URL(fileURLWithPath: "/usr/bin/false"), [])
    }
    await #expect(throws: (any Error).self) {
      try await Runner.run(URL(fileURLWithPath: "/bin/sleep"), ["5"], timeout: .milliseconds(30))
    }
    await #expect(throws: (any Error).self) {
      try await Runner.run(URL(fileURLWithPath: "/nonexistent/pro-apps-command"), [])
    }
  }

  @Test(arguments: [
    "<other/>", "<fcpxml>", "<!DOCTYPE fcpxml SYSTEM 'file:///etc/passwd'><fcpxml/>",
    "<!DOCTYPE fcpxml [<!ENTITY e SYSTEM 'https://example.com'>]><fcpxml>&e;</fcpxml>",
  ])
  func rejectsUnsafeXML(_ xml: String) {
    #expect(throws: (any Error).self) { try Interchange.inspect(Data(xml.utf8), kind: .fcpxml) }
  }

  @Test(arguments: ["//asset/@name", "//missing", "/fcpxml/resources", "///["])
  func rejectsAmbiguousOrStructuralPatches(_ xpath: String) throws {
    let directory = try temporary()
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let input = try Interchange.write(
      "<fcpxml><resources><asset name=\"a\"/><asset name=\"b\"/></resources></fcpxml>",
      kind: .fcpxml, output: directory.appendingPathComponent("input.fcpxml").path,
      allowUndocumented: false)
    #expect(throws: (any Error).self) {
      try Interchange.patch(
        input: input.path, output: directory.appendingPathComponent("output.fcpxml").path,
        kind: .fcpxml, changes: [.init(xpath: xpath, value: "bad")], allowUndocumented: false)
    }
  }

  @Test(arguments: [0, 17])
  func rejectsInvalidMIDIChannels(_ channel: Int) {
    #expect(throws: (any Error).self) {
      try MIDIControl.message(kind: .controlChange, channel: channel, number: 7, value: 100)
    }
  }

  @Test(arguments: ["gain", "/a*", "/a\0b", "/a b", "/{a,b}"])
  func rejectsUnsafeOSCPaths(_ path: String) {
    #expect(throws: (any Error).self) { try OSC.packet(path: path, value: 1) }
  }

  @Test(arguments: ["", "-resetBackgroundProcessing", "x; touch /tmp/no", "a\nb"])
  func rejectsUnsafeCompressorIDs(_ id: String) {
    #expect(throws: (any Error).self) { try Compressor.monitoringArguments(id: id, job: false) }
  }

  @Test func editionsDoNotGuessLogicBundleID() {
    #expect(ProApp.logicPro.bundleIDs[0] == "com.apple.mobilelogic")
    #expect(ProApp.mainStage.bundleIDs.contains("com.apple.mainstage3"))
    #expect(ProApp.motion.documentExtensions.contains("motn"))
    #expect(!ProApp.logicPro.documentExtensions.contains("concert"))
  }
}
