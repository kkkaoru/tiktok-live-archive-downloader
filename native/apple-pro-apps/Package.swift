// swift-tools-version: 6.2
import PackageDescription

let package = Package(
  name: "AppleProApps",
  platforms: [.macOS(.v15)],
  products: [.executable(name: "apple-pro-apps", targets: ["AppleProApps"])],
  dependencies: [
    .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", exact: "0.12.1"),
    .package(url: "https://github.com/argmaxinc/argmax-oss-swift.git", exact: "1.1.0"),
  ],
  targets: [
    .target(
      name: "ProAppsCore",
      dependencies: [.product(name: "WhisperKit", package: "argmax-oss-swift")]),
    .executableTarget(
      name: "AppleProApps",
      dependencies: ["ProAppsCore", .product(name: "MCP", package: "swift-sdk")]
    ),
    .executableTarget(name: "font-identity-probe", path: "Sources/FontIdentityProbe"),
    .testTarget(name: "FontIdentityProbeTests", dependencies: ["font-identity-probe"]),
    .executableTarget(
      name: "caption-alpha-render", dependencies: ["ProAppsCore"],
      path: "Sources/CaptionAlphaRender"),
    .executableTarget(name: "audio-clock-probe", path: "Sources/AudioClockProbe"),
    .testTarget(name: "AudioClockProbeTests", dependencies: ["audio-clock-probe"]),
    .executableTarget(
      name: "audio-encoding-capabilities", path: "Sources/AudioEncodingCapabilities"),
    .testTarget(
      name: "AudioEncodingCapabilitiesTests", dependencies: ["audio-encoding-capabilities"]),
    // Synthetic accessibility fixture used only by native UI backend tests.
    .executableTarget(name: "apple-pro-apps-ui-fixture", path: "Tests/UIFixture"),
    .testTarget(name: "ProAppsCoreTests", dependencies: ["ProAppsCore"]),
    .testTarget(
      name: "ProAppsNativeTests", dependencies: ["ProAppsCore"], resources: [.copy("Fixtures")]),
    .testTarget(
      name: "AppleProAppsTests",
      dependencies: ["AppleProApps", "ProAppsCore", .product(name: "MCP", package: "swift-sdk")]
    ),
  ],
  swiftLanguageModes: [.v6]
)
