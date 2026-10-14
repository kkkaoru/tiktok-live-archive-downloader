import Foundation
import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

struct MotionTextServiceTests {
  @Test @MainActor func textToolsEditACopyWithoutOpeningMotion() async throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let source = root.appendingPathComponent("source.motn")
    let output = root.appendingPathComponent("copy.motn")
    let data = Data(
      """
      <ozml version="4.0"><scenenode id="7" name="Title"><style id="9"/><text>A</text>
      <styleRun style="9" offset="0" length="1"/>
      <object value="65"><parameter name="Kerning" id="1" value="0"/></object>
      </scenenode></ozml>
      """.utf8)
    try data.write(to: source)
    let hash = try MotionText.inspect(data).sourceSHA256
    let service = NativeService()
    let inspection = await service.call(
      .init(name: "motion_text_inspect", arguments: ["path": .string(source.path)]))
    #expect(inspection.isError == false)
    let missingArguments = await service.call(.init(name: "motion_text_inspect"))
    #expect(missingArguments.isError == true)
    let args: [String: Value] = [
      "inputPath": .string(source.path), "outputPath": .string(output.path),
      "expectedSHA256": .string(hash), "allowUndocumentedFormat": .bool(true),
      "changes": .array([
        .object(["layerID": .int(7), "expectedText": .string("A"), "replacement": .string("検証")])
      ]),
    ]
    var withoutOptIn = args
    withoutOptIn.removeValue(forKey: "allowUndocumentedFormat")
    withoutOptIn["outputPath"] = .string(root.appendingPathComponent("unapproved.motn").path)
    let unapproved = await service.call(.init(name: "motion_text_copy", arguments: withoutOptIn))
    #expect(unapproved.isError == true)
    #expect(
      !FileManager.default.fileExists(atPath: root.appendingPathComponent("unapproved.motn").path))
    let copy = await service.call(.init(name: "motion_text_copy", arguments: args))
    #expect(copy.isError == false)
    #expect(try MotionText.inspect(Data(contentsOf: output)).layers.first?.text == "検証")
    #expect(try Data(contentsOf: source) == data)
    let again = await service.call(.init(name: "motion_text_copy", arguments: args))
    #expect(again.isError == true)
    let spec = try #require(ToolSpec.all.first { $0.name == "motion_text_inspect" })
    #expect(spec.readOnly)
    let copySpec = try #require(ToolSpec.all.first { $0.name == "motion_text_copy" })
    #expect(!copySpec.readOnly)
  }
}
