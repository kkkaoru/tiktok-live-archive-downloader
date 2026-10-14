import Foundation
import Testing

@testable import ProAppsCore

struct ForegroundMaskStatisticsTests {
  @Test func sparseRowsIgnorePaddingAndKeepThresholdOne() throws {
    let measured = try ForegroundMaskStatistics.scan(
      Data([0, 0, 1, 255, 0, 255, 0, 255, 0, 0, 0, 255]),
      width: 3, height: 3, bytesPerRow: 4)
    #expect(
      measured
        == .init(
          width: 3, height: 3, nonzeroPixels: 2,
          nonzeroBounds: .init(x: 1, y: 0, width: 2, height: 2)))
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    #expect(
      String(decoding: try encoder.encode(measured), as: UTF8.self)
        == #"{"height":3,"nonzeroBounds":{"height":2,"width":2,"x":1,"y":0},"nonzeroPixels":2,"width":3}"#
    )
  }

  @Test func emptySupportRemainsEmpty() throws {
    let measured = try ForegroundMaskStatistics.scan(
      Data([0, 255]), width: 1, height: 1, bytesPerRow: 2)
    #expect(
      measured == .init(width: 1, height: 1, nonzeroPixels: 0, nonzeroBounds: nil))
  }

  @Test func slicedDataRetainsItsIndexOrigin() throws {
    let data = Data([99, 99, 255, 1, 1, 255]).dropFirst(2)
    let measured = try ForegroundMaskStatistics.scan(
      data, width: 2, height: 2, bytesPerRow: 2)
    #expect(
      measured
        == .init(
          width: 2, height: 2, nonzeroPixels: 4,
          nonzeroBounds: .init(x: 0, y: 0, width: 2, height: 2)))
  }

  @Test(arguments: [
    (0, 1, 1), (-1, 1, 1), (16385, 1, 16385), (1, 0, 1),
    (1, 16385, 1), (16384, 16384, 16384), (2, 1, 1),
    (1, 1, 0), (1, 1, -1), (1, 2, Int.max), (1, 1, 67_108_865),
  ])
  func invalidStorageFails(dimensions: (Int, Int, Int)) {
    #expect(throws: ProAppsError.self) {
      try ForegroundMaskStatistics.scan(
        Data(), width: dimensions.0, height: dimensions.1, bytesPerRow: dimensions.2)
    }
  }

  @Test(arguments: [Data(), Data([0, 0])])
  func shortOrExtraStorageFails(data: Data) {
    #expect(throws: ProAppsError.self) {
      try ForegroundMaskStatistics.scan(data, width: 1, height: 1, bytesPerRow: 1)
    }
  }

  @Test func providerTrailingPagesAreNotPixels() throws {
    let measured = try ForegroundMaskStatistics.scanBacking(
      Data([0, 1, 255, 255, 255]), width: 2, height: 1, bytesPerRow: 2)
    #expect(
      measured
        == .init(
          width: 2, height: 1, nonzeroPixels: 1,
          nonzeroBounds: .init(x: 1, y: 0, width: 1, height: 1)))
    #expect(throws: ProAppsError.self) {
      try ForegroundMaskStatistics.scanBacking(Data([0]), width: 2, height: 1, bytesPerRow: 2)
    }
    #expect(throws: ProAppsError.self) {
      try ForegroundMaskStatistics.scanBacking(
        Data(repeating: 0, count: 67_108_865), width: 1, height: 1, bytesPerRow: 1)
    }
  }

  @Test func cancellationBetweenRowsStopsBeforeFollowingRows() {
    var rows: [Int] = []
    #expect(throws: CancellationError.self) {
      try ForegroundMaskStatistics.scan(
        Data(repeating: 1, count: 6), width: 2, height: 3, bytesPerRow: 2
      ) { row in
        rows.append(row)
        if row == 1 { throw CancellationError() }
      }
    }
    #expect(rows == [0, 1])
  }

  @Test func cancellationPropagatesBeforeScanning() async {
    await withTaskGroup(of: Void.self) { group in
      group.cancelAll()
      group.addTask {
        #expect(throws: CancellationError.self) {
          try ForegroundMaskStatistics.scan(
            Data([255]), width: 1, height: 1, bytesPerRow: 1)
        }
      }
    }
  }
}
