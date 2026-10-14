import CoreMedia
import SoundAnalysis
import Testing

struct SoundClassifierTests {
  @Test func builtInClassifierExposesSpeechWithoutCustomModelInstallation() throws {
    let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
    #expect(request.knownClassifications.contains("speech"))
    #expect(request.knownClassifications.contains("music"))
    request.windowDuration = CMTime(seconds: 0.5, preferredTimescale: 16000)
    #expect(request.windowDuration.seconds > 0)
    #expect(request.windowDuration.seconds <= 1)
    #expect(request.overlapFactor == 0.5)
  }
}
