import Foundation
import Testing

@testable import ProAppsCore

struct MotionCaptionDocumentTests {
  static func fixture() throws -> String {
    let url = try #require(
      Bundle.module.url(
        forResource: "motion-caption-part", withExtension: "moti", subdirectory: "Fixtures"))
    return try String(contentsOf: url, encoding: .utf8)
  }

  static func read(_ text: String) throws -> MotionCaptionPart {
    try MotionCaptionDocument.read(Data(text.utf8))
  }

  /// The reader's refusal message for a mutated copy of the fixture.
  static func refusal(_ change: (String) -> String) throws -> String {
    do {
      _ = try read(change(try fixture()))
    } catch let error as ProAppsError {
      return "\(error)"
    }
    return "accepted"
  }

  static func replacingFirst(_ text: String, _ target: String, _ replacement: String) -> String {
    guard let range = text.range(of: target) else { return text }
    return text.replacingCharacters(in: range, with: replacement)
  }

  @Test func readsTheTitlesStyleAndClockThatTheMotionDocumentHolds() throws {
    let part = try Self.read(try Self.fixture())
    #expect((part.width, part.height, part.rate, part.frames) == (1080, 1920, 30, 60))
    #expect(
      part.titles == [
        MotionCaptionTitle(
          text: "テスト\nです", startFrame: 0, endFrame: 30, x: 0, y: 300,
          graphemeFrames: [0, 3, 6, 10, 10, 15]),
        MotionCaptionTitle(
          text: "はい", startFrame: 40, endFrame: 60, x: -12.5, y: -200, graphemeFrames: [40, 44]),
      ])
    #expect(part.style.fontName == "NotoSansJP-Thin_Bold")
    #expect((part.style.gradientTop, part.style.gradientBottom) == ("#97D8F7", "#2FB0EE"))
    #expect((part.style.topLocation, part.style.bottomLocation) == (0.2688, 0.3944))
    #expect(abs(part.style.whiteBorder - 9.502209944751382) < 1e-12)
    #expect(abs(part.style.shadowSigma - 7.541436464088398 / 2.0.squareRoot()) < 1e-12)
  }

  @Test func theSkeletonOfTheFixtureIsThePinnedOne() throws {
    let text = try MotionCaptionDocument.skeleton(Data(try Self.fixture().utf8))
    let lines = text.split(separator: "\n").map(String.init)
    let document = Array(lines[1..<(lines.firstIndex(of: "[Caption Keyline Face]") ?? 0)])
    #expect(document == MotionCaptionSkeleton.document())
    #expect(!MotionCaptionSkeleton.title("Caption Outer Rim").isEmpty)
  }

  @Test func refusesAnEnabledEffectTheRendererDoesNotDraw() throws {
    // The Glow group's flags differ when it is switched on.
    let message = try Self.refusal {
      Self.replacingFirst(
        $0, #"name="Glow" id="38" flags="8589973520""#,
        #"name="Glow" id="38" flags="8589938704""#)
    }
    #expect(message.contains("differs from the supported template"))
    #expect(message.contains("parameter[38]@flags=8589938704"))
  }

  @Test func refusesAnUnknownElement() throws {
    let message = try Self.refusal {
      Self.replacingFirst($0, "<aspectRatio>", "<newFeature>1</newFeature><aspectRatio>")
    }
    #expect(message.contains("/newFeature#text=1"))
  }

  @Test func refusesAnotherKeyframeInterpolation() throws {
    let message = try Self.refusal {
      $0.replacingOccurrences(
        of: #"<keypoint interpolation="1""#, with: #"<keypoint interpolation="2""#)
    }
    #expect(message.contains("Type On End keys must be linear"))
  }

  @Test func refusesObjectsOfOneTitleThatDisagree() throws {
    let message = try Self.refusal { Self.replacingFirst($0, "<text>はい", "<text>いいえ") }
    #expect(message.contains("its three objects differ"))
  }

  @Test func refusesAColourThatIsNotAnEightBitValue() throws {
    let message = try Self.refusal {
      Self.replacingFirst($0, #"value="0.592156862745098""#, #"value="0.5922""#)
    }
    #expect(message.contains("not an 8-bit sRGB value"))
  }

  @Test func refusesAFrameTheRendererCannotPlace() throws {
    let message = try Self.refusal {
      $0.replacingOccurrences(of: #"in="40 30 1 0""#, with: #"in="81 60 1 0""#)
    }
    #expect(message.contains("not a whole frame"))
  }

  @Test func refusesASidewaysShadow() throws {
    let message = try Self.refusal {
      $0.replacingOccurrences(of: #"value="4.71238898038469""#, with: #"value="4.0""#)
    }
    #expect(message.contains("only a downward shadow"))
  }

  @Test func refusesAGreyOutline() throws {
    let message = try Self.refusal {
      Self.replacingFirst(
        $0, #"<parameter name="Red" id="1" flags="80" default="1" value="0" />"#,
        #"<parameter name="Red" id="1" flags="80" default="1" value="0.5" />"#)
    }
    #expect(message.contains("black or white"))
  }

  @Test func refusesSomethingThatIsNotACaptionPart() throws {
    #expect(throws: ProAppsError.self) { try Self.read("<ozml version=\"5.12\"/>") }
    #expect(throws: ProAppsError.self) { try Self.read("not xml") }
  }

  @Test func theCommandReadsTheMotionDocumentForEveryMode() throws {
    let url = try #require(
      Bundle.module.url(
        forResource: "motion-caption-part", withExtension: "moti", subdirectory: "Fixtures"))
    let part = try MotionCaptionDocument.read(contentsOf: url)
    let program = "caption-alpha-render"
    #expect(
      try MotionCaptionMovie.command(arguments: [program, "--inspect", url.path]) == .inspect(part))
    guard
      case .render(let request) = try MotionCaptionMovie.command(
        arguments: [program, "--moti", url.path, "/synthetic/out.mov"])
    else {
      Issue.record("--moti must render")
      return
    }
    #expect(
      (request.output, request.titles, request.style) == (
        "/synthetic/out.mov", part.titles, part.style
      ))
    guard
      case .skeleton(let text) = try MotionCaptionMovie.command(arguments: [
        program, "--skeleton", url.path,
      ])
    else {
      Issue.record("--skeleton must print the skeleton")
      return
    }
    #expect(text.hasPrefix("[document]\n"))
  }

  @Test func aTitleRecolouredInMotionKeepsItsOwnStyle() throws {
    let text = try Self.fixture()
    let target = #"value="0.592156862745098""#
    // The second title's keyline gradient top: its red control is the last occurrence.
    let last = try #require(text.range(of: target, options: .backwards))
    let part = try Self.read(
      text.replacingCharacters(in: last, with: #"value="0.6901960784313725""#))
    #expect(part.titles[0].style == nil)
    #expect(part.titles[1].style?.gradientTop == "#B0D8F7")
    #expect(part.style.gradientTop == "#97D8F7")
  }
}
