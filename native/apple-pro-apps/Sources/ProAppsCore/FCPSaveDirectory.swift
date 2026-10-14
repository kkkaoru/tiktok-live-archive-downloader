import Foundation

/// Explicit destinations are inputs; an AX basename never creates a filesystem identity.
public struct FCPSaveNavigationRequest: Codable, Equatable, Sendable {
  public let directory: String
  public let fileName: String
  public let candidateDirectories: [String]
  public let window: UIWindowLocator
  public let expectedPID: Int32
  public let maximumSteps: Int
  public let maximumSeconds: Double
  public let saveDispatched: Bool

  public init(
    directory: String, fileName: String, candidateDirectories: [String], window: UIWindowLocator,
    expectedPID: Int32, maximumSteps: Int = 16, maximumSeconds: Double = 15,
    saveDispatched: Bool = false
  ) {
    self.directory = directory
    self.fileName = fileName
    self.candidateDirectories = candidateDirectories
    self.window = window
    self.expectedPID = expectedPID
    self.maximumSteps = maximumSteps
    self.maximumSeconds = maximumSeconds
    self.saveDispatched = saveDispatched
  }

  public func validatedDirectories() throws -> [String] {
    guard expectedPID > 0, (1...16).contains(maximumSteps), maximumSeconds.isFinite,
      maximumSeconds > 0, maximumSeconds <= 15,
      let title = window.title, !title.isEmpty, title.utf8.count <= 512, window.index == nil,
      !candidateDirectories.isEmpty, candidateDirectories.count <= 16,
      Set(candidateDirectories).count == candidateDirectories.count,
      candidateDirectories.contains(directory),
      candidateDirectories.reduce(0, { $0 + $1.utf8.count }) <= 32_000,
      !fileName.isEmpty, fileName != ".", fileName != "..", !fileName.contains("/"),
      !fileName.contains("\0"), fileName.utf8.count <= 180,
      !URL(fileURLWithPath: fileName).pathExtension.isEmpty
    else { throw ProAppsError.invalid("Bounded explicit Save navigation inputs are required") }
    for path in candidateDirectories { _ = try FCPSavePlanner.canonicalDirectoryForPlanning(path) }
    return candidateDirectories
  }
}

public struct FCPSaveDirectoryVariant: Codable, Equatable, Sendable {
  public var languages: [String]
  public let titles: [String]
  public init(languages: [String], titles: [String]) {
    self.languages = languages
    self.titles = titles
  }
}

public struct FCPSaveDirectoryCandidate: Codable, Equatable, Sendable {
  public let directory: String
  public let volumeURL: String
  public let variants: [FCPSaveDirectoryVariant]
  public init(directory: String, volumeURL: String, variants: [FCPSaveDirectoryVariant]) {
    self.directory = directory
    self.volumeURL = volumeURL
    self.variants = variants
  }

  /// Only a complete suffix of a currently proved OS chain identifies an ancestor.
  /// A matching title elsewhere in recents is never sufficient evidence.
  func ancestorLabels(of proof: FCPSaveDirectoryObservation) -> Set<String> {
    guard let current = proof.directory, proof.volumeURL == volumeURL,
      URL(fileURLWithPath: current).pathComponents.starts(
        with: URL(fileURLWithPath: directory).pathComponents)
    else { return [] }
    var labels: Set<String> = []
    for variant in variants {
      if !Set(variant.languages).isDisjoint(with: proof.languages),
        Array(proof.ancestorTitles.suffix(variant.titles.count)) == variant.titles,
        let first = variant.titles.first
      {
        labels.insert(first)
      }
    }
    return labels
  }

  /// A mounted-volume label selects a navigation candidate, never proves the
  /// resulting directory. The executor must freshly prove its entire chain.
  func navigationLabels(
    of proof: FCPSaveDirectoryObservation, toward targetDirectory: String
  ) -> Set<String> {
    let ancestor = ancestorLabels(of: proof)
    if !ancestor.isEmpty { return ancestor }
    guard proof.directory != nil, let currentVolume = proof.volumeURL,
      currentVolume != volumeURL, directory == volumeURL,
      URL(fileURLWithPath: targetDirectory).pathComponents.starts(
        with: URL(fileURLWithPath: directory).pathComponents)
    else { return [] }
    var labels: Set<String> = []
    for variant in variants {
      if (1...2).contains(variant.titles.count),
        !Set(variant.languages).isDisjoint(with: proof.languages), let first = variant.titles.first
      {
        labels.insert(first)
      }
    }
    return labels
  }
}

public struct FCPSaveDirectoryObservation: Codable, Equatable, Sendable {
  public let ancestorTitles: [String]
  public var directory: String?
  public var volumeURL: String?
  public var languages: [String] = []
  public init(ancestorTitles: [String]) { self.ancestorTitles = ancestorTitles }
}

public enum FCPSaveNavigationStatus: String, Codable, Sendable {
  case directoryVerified, observeExistingDispatch, unprovedDirectory, processChanged, focusChanged
  case stepBudgetExceeded, deadlineExceeded, navigationUnavailable, uncertainEffect, backendFailure
  case cancelled
}

public enum FCPSaveNavigationBasis: String, Codable, Sendable {
  case verifiedCurrentAncestor, mountedVolumeCandidate, verifiedChild, verifiedParentChild
}

public struct FCPSaveNavigationStep: Codable, Equatable, Sendable {
  public let fromDirectory: String
  public let expectedDirectory: String
  public var action: UIRequest?
  public var selectedChildrenPath: [Int]?
  public var selectedChildPath: [Int]?
  public var selectionBasis: FCPSaveNavigationBasis?
  public var actionAttempted = false
  public var actionError: String?
  public var after: FCPSaveDirectoryObservation?
}

public enum FCPSaveSelectionState: String, Codable, Sendable {
  case missing, elements, unsupported
}

public struct FCPSaveSelectionLabel: Codable, Equatable, Sendable {
  public let role: String?
  public let title: String?
  public let value: String?
  public let selected: Bool?
  public let url: String?
}

public struct FCPSaveSelectedChild: Codable, Equatable, Sendable {
  public let element: FCPSaveSelectionLabel
  public let directChildren: [FCPSaveSelectionLabel]
}

public struct FCPSaveColumnObservation: Codable, Equatable, Sendable {
  public let path: [Int]
  public let rowCount: Int
  public let directoryURL: String?
  public let selectedURLs: [String]
  public let selectedChildrenState: FCPSaveSelectionState
  public let selectedChildrenCount: Int?
  public let selectedChildren: [FCPSaveSelectedChild]
}

public struct FCPSaveNavigationReceipt: Codable, Equatable, Sendable {
  public var menuAttachments: [FCPSaveMenuAttachment] = []
  public var menuAttachmentFailure: String?
  public var status: FCPSaveNavigationStatus = .unprovedDirectory
  public let requestedDirectory: String
  public var candidates: [FCPSaveDirectoryCandidate] = []
  public var observations: [FCPSaveDirectoryObservation] = []
  public var inspections: [FCPSaveNavigationInspection] = []
  public var columns: [FCPSaveColumnObservation] = []
  public var steps: [FCPSaveNavigationStep] = []
  public var menuShowsAttempted = 0
  public var menuClosesAttempted = 0
  public var elapsedSeconds: Double = 0
  public var failure: String?
  public var directoryVerified: Bool { status == .directoryVerified }
  public var nameAttempted: Bool { false }
  public var saveAttempted: Bool { false }

  public init(requestedDirectory: String) { self.requestedDirectory = requestedDirectory }
}

/// Direct-child diagnostic only. A complete observation does not prove that a
/// menu belongs to the Save popup or authorize selecting any of its children.
public struct FCPSaveMenuAttachment: Codable, Equatable, Sendable {
  public enum Scope: String, Codable, Sendable { case popup, saveContainer, application }
  public struct Child: Codable, Equatable, Sendable {
    public let index: Int
    public let role: String?
    public let identifier: String?
  }
  public let scope: Scope
  public let path: [Int]?
  public let role: String?
  public let identifier: String?
  public let childCount: Int
  public var children: [Child] = []
  public var complete = false
}

/// Completeness applies only to the named scope, never the entire Save window.
public struct FCPSaveNavigationInspection: Codable, Equatable, Sendable {
  public enum Scope: String, Codable, Sendable {
    case controls, ancestorMenu, columnBrowser
  }
  public let scope: Scope
  public var elementCount = 0
  public var complete = false
  public init(scope: Scope) { self.scope = scope }
}

/// The production supplier reads OS volume metadata and the OS's folder localization tables.
/// Synthetic tests inject whole canonical chains rather than operating a desktop.
public struct FCPSaveDirectoryEnvironment: Sendable {
  public let describe: @Sendable ([String]) throws -> [FCPSaveDirectoryCandidate]
  public init(describe: @escaping @Sendable ([String]) throws -> [FCPSaveDirectoryCandidate]) {
    self.describe = describe
  }
  public static let live = Self(describe: FCPSaveDirectory.liveCandidates)
}

public enum FCPSaveDirectory {
  /// Match the entire ordered chain in one OS language, never individual substrings/recents.
  public static func prove(
    _ titles: [String], candidates: [FCPSaveDirectoryCandidate]
  ) throws -> FCPSaveDirectoryObservation {
    guard !titles.isEmpty, titles.count <= 34,
      titles.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 512 && !$0.contains("\0") })
    else { throw ProAppsError.invalid("A complete bounded location ancestor chain is required") }
    var observation = FCPSaveDirectoryObservation(ancestorTitles: titles)
    let matches = candidates.filter { $0.variants.contains { $0.titles == titles } }
    guard matches.count <= 1 else {
      throw ProAppsError.invalid("Location ancestors identify multiple canonical candidates")
    }
    if let candidate = matches.first {
      observation.directory = candidate.directory
      observation.volumeURL = candidate.volumeURL
      observation.languages = candidate.variants.filter { $0.titles == titles }.flatMap(\.languages)
    }
    return observation
  }

  static func liveCandidates(_ paths: [String]) throws -> [FCPSaveDirectoryCandidate] {
    let tables = try localizationTables()
    let computer = Host.current().localizedName
    let candidates = try paths.map { path in
      try candidate(path, tables: tables, computerName: computer)
    }
    guard try JSONEncoder().encode(candidates).count <= 384 * 1024 else {
      throw ProAppsError.invalid("OS directory evidence exceeds its bounded capture budget")
    }
    return candidates
  }

  /// Pure localization over actual metadata. Raw names are normal OS fallback only within
  /// a complete language's table; independently mixed segment languages are not generated.
  static func variants(
    names: [String], localized: [Bool], volumeName: String, tables: [String: [String: String]],
    computerName: String?
  ) throws -> [FCPSaveDirectoryVariant] {
    guard names.count <= 32, names.count == localized.count,
      !volumeName.isEmpty, !tables.isEmpty, tables.count <= 128
    else { throw ProAppsError.invalid("OS directory metadata is absent or over budget") }
    var variants: [FCPSaveDirectoryVariant] = []
    for (language, table) in tables.sorted(by: { $0.key < $1.key }) {
      var titles = names.enumerated().map { index, name in
        localized[index] ? table[name] ?? name : name
      }
      titles.append(volumeName)
      try appendVariant(titles, language: language, to: &variants)
      if let computerName, !computerName.isEmpty {
        try appendVariant(titles + [computerName], language: language, to: &variants)
      }
    }
    return variants
  }

  private static func appendVariant(
    _ titles: [String], language: String, to variants: inout [FCPSaveDirectoryVariant]
  ) throws {
    guard titles.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 512 && !$0.contains("\0") }) else {
      throw ProAppsError.invalid("OS directory display names are invalid")
    }
    if let index = variants.firstIndex(where: { $0.titles == titles }) {
      variants[index].languages.append(language)
    } else {
      variants.append(.init(languages: [language], titles: titles))
    }
  }

  private static func candidate(
    _ path: String, tables: [String: [String: String]], computerName: String?
  ) throws -> FCPSaveDirectoryCandidate {
    var current = try FCPSavePlanner.canonicalDirectoryForPlanning(path)
    var names: [String] = []
    var marked: [Bool] = []
    while names.count <= 32 {
      let metadata = try current.resourceValues(forKeys: [.isVolumeKey, .localizedNameKey])
      if metadata.isVolume == true {
        guard let volumeName = metadata.localizedName, !volumeName.isEmpty else {
          throw ProAppsError.unavailable("OS volume display name is unavailable")
        }
        // The first visible volume ancestor also handles APFS logical paths; no
        // hard-coded /Volumes, root disk label or hidden Data-volume alias is used.
        return FCPSaveDirectoryCandidate(
          directory: path, volumeURL: current.path,
          variants: try variants(
            names: names, localized: marked, volumeName: volumeName,
            tables: tables, computerName: computerName))
      }
      guard current.path != "/" else {
        throw ProAppsError.unavailable("Canonical ancestry has no observed OS volume boundary")
      }
      names.append(current.lastPathComponent)
      marked.append(try localizedMarker(current))
      current.deleteLastPathComponent()
    }
    throw ProAppsError.invalid("Canonical directory ancestry exceeds 32 levels")
  }

  private static func localizedMarker(_ directory: URL) throws -> Bool {
    do {
      let values = try directory.appendingPathComponent(".localized").resourceValues(
        forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
      return values.isRegularFile == true && values.isSymbolicLink != true
    } catch CocoaError.fileReadNoSuchFile {
      return false
    }
  }

  private static func localizationTables() throws -> [String: [String: String]] {
    let coreServices = try FileManager.default.url(
      for: .coreServiceDirectory, in: .systemDomainMask, appropriateFor: nil, create: false)
    let root = coreServices.appendingPathComponent("SystemFolderLocalizations")
    let languages = try FileManager.default.contentsOfDirectory(
      at: root, includingPropertiesForKeys: [.isDirectoryKey]
    ).filter { $0.pathExtension == "lproj" }
    guard !languages.isEmpty, languages.count <= 128 else {
      throw ProAppsError.unavailable(
        "OS folder localization languages are unavailable or over budget")
    }
    var tables: [String: [String: String]] = [:]
    var bytes = 0
    for language in languages {
      let url = try Files.existing(
        language.appendingPathComponent("SystemFolderLocalizations.strings").path)
      let data = try Files.read(url)
      bytes += data.count
      guard bytes <= 4 * 1024 * 1024,
        let table = try PropertyListSerialization.propertyList(from: data, format: nil)
          as? [String: String], table.count <= 1024
      else { throw ProAppsError.invalid("OS folder localization tables exceed their budget") }
      tables[language.deletingPathExtension().lastPathComponent] = table
    }
    return tables
  }
}
