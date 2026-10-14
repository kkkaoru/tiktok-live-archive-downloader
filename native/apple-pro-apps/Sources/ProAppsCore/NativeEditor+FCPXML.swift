import AVFoundation
import Foundation

extension NativeEditor {
  /// Every video render also writes `<name>.fcpxml` beside the media so the edit
  /// can be opened in Final Cut Pro. It is DTD-validated against the installed
  /// Final Cut Pro when present. Recipes FCPXML would misrepresent are reported
  /// with a reason instead of written.
  func exportTimeline(_ recipe: EditRecipe, plan: EditPlan, directory: URL, name: String)
    async throws -> FCPXMLExport?
  {
    guard let video = recipe.video else { return nil }
    if let reason = FCPXMLTimeline.unsupported(recipe) {
      return FCPXMLExport(
        path: nil, written: false, validDTD: nil, unrepresented: [], reason: reason)
    }
    var sources: [String: FCPXMLSource] = [:]
    for clip in recipe.clips where sources[clip.sourcePath] == nil {
      let asset = AVURLAsset(url: try Files.existing(clip.sourcePath))
      let duration = try await asset.load(.duration).seconds
      let audio = try await asset.loadTracks(withMediaType: .audio)
      sources[clip.sourcePath] = FCPXMLSource(durationSeconds: duration, hasAudio: !audio.isEmpty)
    }
    let font = try video.captionAppearance.map(StyledCaptionRenderer.resolve)
    let document = try FCPXMLTimeline.document(
      recipe, plan: plan, sources: sources, font: font, name: name)
    let url = try Files.writeNew(
      Data(document.xml.utf8), to: directory.appendingPathComponent("\(name).fcpxml").path,
      extensions: ["fcpxml"])
    return FCPXMLExport(
      path: url.path, written: true, validDTD: try await Self.validate(url),
      unrepresented: document.unrepresented, reason: nil)
  }

  /// DTD validity against the installed Final Cut Pro, or nil when none is installed.
  static func validate(_ url: URL) async throws -> Bool? {
    let installed: InstalledApp
    do {
      installed = try await Applications.resolve(.finalCutPro, bundleID: nil)
    } catch ProAppsError.unavailable {
      return nil
    }
    let directory = URL(fileURLWithPath: installed.path).appendingPathComponent(
      "Contents/Frameworks/Interchange.framework/Versions/A/Resources")
    do {
      return try await FCPXMLValidator(dtdDirectory: directory).validate(path: url.path).validDTD
    } catch ProAppsError.invalid {
      return false
    }
  }
}
