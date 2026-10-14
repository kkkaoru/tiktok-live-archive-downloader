// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "TranscriptViewer",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "TranscriptCore", targets: ["TranscriptCore"]),
        .executable(name: "TranscriptViewer", targets: ["TranscriptViewer"]),
        .executable(name: "transcript-check", targets: ["TranscriptCheck"]),
        .executable(name: "transcript-summary", targets: ["TranscriptSummaryCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/nalexn/ViewInspector", exact: "0.10.5")
    ],
    targets: [
        .target(name: "TranscriptCore"),
        .target(name: "TranscriptSummaries", dependencies: ["TranscriptCore"]),
        .executableTarget(
            name: "TranscriptViewer", dependencies: ["TranscriptCore", "TranscriptSummaries"]),
        .executableTarget(name: "TranscriptSummaryCLI", dependencies: ["TranscriptSummaries"]),
        .executableTarget(
            name: "TranscriptCheck", dependencies: ["TranscriptCore", "TranscriptSummaries"]),
        .testTarget(
            name: "TranscriptCoreTests",
            dependencies: [
                "TranscriptCore", "TranscriptViewer", "TranscriptSummaries",
                .product(name: "ViewInspector", package: "ViewInspector"),
            ]),
    ]
)
