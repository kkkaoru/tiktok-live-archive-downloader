import Foundation
import Testing

@testable import ProAppsCore

struct MotionInterchangeTests {
  @Test(arguments: [
    "<!DOCTYPE ozxmlscene><ozml version=\"4.0\"><value>1</value></ozml>",
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE ozxmlscene>\n<ozml version=\"4.0\"><value>1</value></ozml>",
    "\u{FEFF} \n<!DOCTYPE ozxmlscene>\n<ozml version=\"4.0\"><value>1</value></ozml>",
  ])
  func recognizesOnlyTheNativeEmptyDeclaration(xml: String) throws {
    let summary = try Interchange.inspect(Data(xml.utf8), kind: .motion)
    #expect(summary.root == "ozml")
    #expect(summary.version == "4.0")
    #expect(summary.elementCount == 2)
    #expect(summary.validation.contains("undocumented"))
  }

  @Test(arguments: [
    "<!DOCTYPE ozxmlscene SYSTEM 'file:///etc/passwd'><ozml/>",
    "<!DOCTYPE ozxmlscene PUBLIC 'example' 'https://example.invalid/dtd'><ozml/>",
    "<!DOCTYPE ozxmlscene [<!ENTITY e SYSTEM 'file:///etc/passwd'>]><ozml>&e;</ozml>",
    "<!DOCTYPE ozxmlscene [<!ENTITY e 'expanded'>]><ozml>&e;</ozml>",
    "<!DOCTYPE ozxmlscene []><ozml/>",
    "<!DOCTYPE ozxmlscene><!DOCTYPE ozxmlscene><ozml/>",
    "<!DOCTYPE ozxmlscene><!ENTITY e 'expanded'><ozml/>",
    "<!DOCTYPE fcpxml><ozml/>",
    "<!doctype ozxmlscene><ozml/>",
    "<!DOCTYPE ozml><ozml/>",
    "<ozml><![CDATA[<!DOCTYPE ozxmlscene>]]></ozml>",
    "<!-- <!DOCTYPE ozxmlscene> --><ozml/>",
    "<ozml><value>1</value></ozml><!DOCTYPE ozxmlscene>",
  ])
  func refusesExternalInternalMisplacedAndDuplicateDeclarations(xml: String) throws {
    #expect(throws: (any Error).self) {
      try Interchange.inspect(Data(xml.utf8), kind: .motion)
    }
  }

  @Test func doesNotAcceptTheMotionDeclarationForFinalCut() throws {
    #expect(throws: (any Error).self) {
      try Interchange.inspect(
        Data("<!DOCTYPE ozxmlscene><fcpxml/>".utf8), kind: .fcpxml)
    }
  }

  @Test func patchesNativeDeclaredCopiesWithoutMutatingTheTemplate() throws {
    let directory = try Files.temporaryDirectory().appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: directory) } catch { Issue.record(error) } }
    let xml = """
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE ozxmlscene>
      <ozml version="4.0"><value>1</value></ozml>
      """
    let source = try Interchange.write(
      xml, kind: .motion, output: directory.appendingPathComponent("source.motn").path,
      allowUndocumented: true)
    let original = try Data(contentsOf: source)
    let output = try Interchange.patch(
      input: source.path, output: directory.appendingPathComponent("copy.motn").path,
      kind: .motion, changes: [XMLChange(xpath: "/ozml/value", value: "2")],
      allowUndocumented: true)
    #expect(try Data(contentsOf: source) == original)
    #expect(
      try Interchange.query(Data(contentsOf: output), kind: .motion, xpath: "/ozml/value", limit: 1)
        == ["<value>2</value>"])
    #expect(try Interchange.inspect(Data(contentsOf: output), kind: .motion).version == "4.0")
    #expect(throws: (any Error).self) {
      try Interchange.patch(
        input: source.path, output: source.path, kind: .motion,
        changes: [XMLChange(xpath: "/ozml/value", value: "3")], allowUndocumented: true)
    }
    #expect(try Data(contentsOf: source) == original)
  }
}
