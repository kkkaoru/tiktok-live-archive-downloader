import Foundation
import Testing

@testable import ProAppsCore

struct FCPSaveDirectoryTests {
  @Test func literalJapaneseAndEnglishChainsRemainWhole() throws {
    let variants = try FCPSaveDirectory.variants(
      names: ["Project", "Users"], localized: [false, true], volumeName: "System Volume",
      tables: ["en": ["Users": "Users"], "ja": ["Users": "ユーザ"]], computerName: "Computer")
    let candidate = FCPSaveDirectoryCandidate(
      directory: "/Users/Project", volumeURL: "/", variants: variants)
    let japanese = try FCPSaveDirectory.prove(
      ["Project", "ユーザ", "System Volume", "Computer"], candidates: [candidate])
    #expect(japanese.directory == "/Users/Project")
    #expect(japanese.languages == ["ja"])
    #expect(japanese.volumeURL == "/")
    #expect(
      try FCPSaveDirectory.prove(["Project", "Users", "System Volume"], candidates: [candidate])
        .languages == ["en"])
    #expect(try FCPSaveDirectory.prove(["Project"], candidates: [candidate]).directory == nil)
    #expect(
      try FCPSaveDirectory.prove(["Project", "Users", "Other Volume"], candidates: [candidate])
        .directory == nil)
  }

  @Test func localizationsAreConsistentAndMarkersHaveMeaning() throws {
    let variants = try FCPSaveDirectory.variants(
      names: ["Users", "Applications"], localized: [true, true], volumeName: "Volume",
      tables: [
        "en": ["Users": "Users", "Applications": "Applications"],
        "ja": ["Users": "ユーザ", "Applications": "アプリケーション"],
      ], computerName: nil)
    let candidate = FCPSaveDirectoryCandidate(directory: "/a", volumeURL: "/", variants: variants)
    #expect(
      try FCPSaveDirectory.prove(["ユーザ", "Applications", "Volume"], candidates: [candidate])
        .directory == nil)
    let plain = try FCPSaveDirectory.variants(
      names: ["Users"], localized: [false], volumeName: "Volume",
      tables: ["en": [:], "ja": ["Users": "ユーザ"]], computerName: "")
    #expect(plain.count == 1)
    #expect(plain[0].languages == ["en", "ja"])
    #expect(plain[0].titles == ["Users", "Volume"])
    let root = try FCPSaveDirectory.variants(
      names: [], localized: [], volumeName: "Volume", tables: ["en": [:]], computerName: nil)
    #expect(root[0].titles == ["Volume"])
  }

  @Test func ambiguityAndInvalidEvidenceNeverBecomeAPath() throws {
    let variant = FCPSaveDirectoryVariant(languages: ["en"], titles: ["Folder", "Volume"])
    let candidates = [
      FCPSaveDirectoryCandidate(directory: "/a", volumeURL: "/", variants: [variant]),
      FCPSaveDirectoryCandidate(directory: "/b", volumeURL: "/", variants: [variant]),
    ]
    #expect(throws: ProAppsError.self) {
      try FCPSaveDirectory.prove(variant.titles, candidates: candidates)
    }
    for invalid in [
      [], [""], ["\u{0}"], [String(repeating: "x", count: 513)], Array(repeating: "a", count: 35),
    ] {
      #expect(throws: ProAppsError.self) {
        try FCPSaveDirectory.prove(invalid, candidates: candidates)
      }
    }
    #expect(throws: ProAppsError.self) {
      try FCPSaveDirectory.variants(
        names: ["a"], localized: [], volumeName: "V", tables: ["en": [:]], computerName: nil)
    }
    #expect(throws: ProAppsError.self) {
      try FCPSaveDirectory.variants(
        names: [], localized: [], volumeName: "", tables: ["en": [:]], computerName: nil)
    }
    #expect(throws: ProAppsError.self) {
      try FCPSaveDirectory.variants(
        names: ["a"], localized: [true], volumeName: "V", tables: ["ja": ["a": ""]],
        computerName: nil)
    }
  }

  @Test func ancestorMustBeAnExactWholeSuffixInTheSameVolumeAndLanguage() throws {
    let ancestor = FCPSaveDirectoryCandidate(
      directory: "/a", volumeURL: "/",
      variants: [.init(languages: ["ja"], titles: ["Localized A", "Volume"])])
    let current = FCPSaveDirectoryCandidate(
      directory: "/a/b", volumeURL: "/",
      variants: [.init(languages: ["ja"], titles: ["B", "Localized A", "Volume"])])
    let proof = try FCPSaveDirectory.prove(["B", "Localized A", "Volume"], candidates: [current])
    #expect(ancestor.ancestorLabels(of: proof) == ["Localized A"])
    let recent = FCPSaveDirectoryCandidate(
      directory: "/other", volumeURL: "/", variants: ancestor.variants)
    #expect(recent.ancestorLabels(of: proof).isEmpty)
    var differentLanguage = proof
    differentLanguage.languages = ["en"]
    #expect(ancestor.ancestorLabels(of: differentLanguage).isEmpty)
    let sameBasename = FCPSaveDirectoryCandidate(
      directory: "/a", volumeURL: "/",
      variants: [.init(languages: ["ja"], titles: ["Localized A", "Different Volume"])])
    #expect(sameBasename.ancestorLabels(of: proof).isEmpty)
    let unproved = FCPSaveDirectoryObservation(ancestorTitles: proof.ancestorTitles)
    #expect(ancestor.ancestorLabels(of: unproved).isEmpty)
  }

  @Test func mountedVolumeCandidateIsNavigationAuthorityOnlyForItsTargetVolume() throws {
    let boot = FCPSaveDirectoryCandidate(
      directory: "/Boot", volumeURL: "/Boot",
      variants: [.init(languages: ["ja"], titles: ["Boot volume"])])
    let proof = try FCPSaveDirectory.prove(["Boot volume"], candidates: [boot])
    let external = FCPSaveDirectoryCandidate(
      directory: "/External", volumeURL: "/External",
      variants: [.init(languages: ["ja"], titles: ["External volume", "Computer"])])
    #expect(external.ancestorLabels(of: proof).isEmpty)
    #expect(external.navigationLabels(of: proof, toward: "/External/Movies") == ["External volume"])
    #expect(external.navigationLabels(of: proof, toward: "/Other/Movies").isEmpty)
    #expect(boot.navigationLabels(of: proof, toward: "/Boot/Movies") == ["Boot volume"])
    let nonVolume = FCPSaveDirectoryCandidate(
      directory: "/External/Movies", volumeURL: "/External", variants: external.variants)
    #expect(nonVolume.navigationLabels(of: proof, toward: "/External/Movies").isEmpty)
    let invalidChain = FCPSaveDirectoryCandidate(
      directory: "/External", volumeURL: "/External",
      variants: [.init(languages: ["ja"], titles: ["Unproved parent", "Volume", "Computer"])])
    #expect(invalidChain.navigationLabels(of: proof, toward: "/External/Movies").isEmpty)
    let unproved = FCPSaveDirectoryObservation(ancestorTitles: ["Boot volume"])
    #expect(external.navigationLabels(of: unproved, toward: "/External/Movies").isEmpty)
  }

  @Test func liveSupplierReadsOnlyOwnedDirectoryMetadataAndOSLocalization() throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "directory-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    let candidate = try #require(FCPSaveDirectoryEnvironment.live.describe([root.path]).first)
    #expect(candidate.directory == root.path)
    #expect(
      root.pathComponents.starts(with: URL(fileURLWithPath: candidate.volumeURL).pathComponents))
    #expect(!candidate.variants.isEmpty)
    #expect(candidate.variants.allSatisfy { $0.titles.first == root.lastPathComponent })
  }

  @Test func liveSupplierRejectsOverdeepAncestryAndKeepsLiteralMarkedNames() throws {
    let root = try Files.temporaryDirectory().appendingPathComponent(
      "directory-depth-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { do { try FileManager.default.removeItem(at: root) } catch { Issue.record(error) } }
    try Data().write(to: root.appendingPathComponent(".localized"))
    let candidate = try #require(FCPSaveDirectoryEnvironment.live.describe([root.path]).first)
    #expect(candidate.variants.allSatisfy { $0.titles.first == root.lastPathComponent })
    var deep = root
    for _ in 0..<33 { deep.appendPathComponent("Nested") }
    try FileManager.default.createDirectory(at: deep, withIntermediateDirectories: true)
    #expect(throws: ProAppsError.self) {
      try FCPSaveDirectoryEnvironment.live.describe([deep.path])
    }
    let fallback = try FCPSaveDirectory.variants(
      names: ["Literal"], localized: [true], volumeName: "Volume",
      tables: ["ja": [:]], computerName: nil)
    #expect(fallback[0].titles == ["Literal", "Volume"])
  }

  @Test func requestBoundsAndLegacyCanonicalGuardStayEnforced() throws {
    let root = try Files.temporaryDirectory()
    func request(
      pid: Int32 = 42, steps: Int = 16, seconds: Double = 15, paths: [String]? = nil,
      title: String = "Save", name: String = "new.mov"
    ) -> FCPSaveNavigationRequest {
      .init(
        directory: root.path, fileName: name, candidateDirectories: paths ?? [root.path],
        window: .init(title: title), expectedPID: pid, maximumSteps: steps, maximumSeconds: seconds)
    }
    #expect(try request().validatedDirectories() == [root.path])
    for invalid in [
      request(pid: 0), request(steps: 0), request(steps: 17), request(seconds: 0),
      request(seconds: 16), request(seconds: .infinity), request(paths: []),
      request(paths: [root.path, root.path]),
      request(title: ""), request(name: "/wrong.mov"), request(name: "missing-extension"),
    ] {
      #expect(throws: ProAppsError.self) { try invalid.validatedDirectories() }
    }
  }
}
