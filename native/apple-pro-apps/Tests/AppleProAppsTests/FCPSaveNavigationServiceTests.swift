import Foundation
import MCP
import ProAppsCore
import Testing

@testable import AppleProApps

struct FCPSaveNavigationServiceTests {
  private func input() throws -> (FCPSaveNavigationRequest, [String: Value]) {
    let root = try Files.temporaryDirectory()
    let navigation = FCPSaveNavigationRequest(
      directory: root.path, fileName: "new.mov", candidateDirectories: [root.path],
      window: .init(title: "Save"), expectedPID: 42)
    let fields = try #require(
      try JSONDecoder().decode(Value.self, from: JSONEncoder().encode(navigation)).objectValue)
    return (navigation, fields)
  }

  @Test(arguments: [
    FCPSaveNavigationStatus.directoryVerified, .observeExistingDispatch, .unprovedDirectory,
    .processChanged, .focusChanged, .stepBudgetExceeded, .deadlineExceeded, .navigationUnavailable,
    .uncertainEffect, .backendFailure, .cancelled,
  ])
  func typedServiceKeepsAttemptEvidenceAndCompletionSeparate(_ status: FCPSaveNavigationStatus)
    async throws
  {
    let (navigation, fields) = try input()
    var interfaces = NativeInterfaces()
    interfaces.ui = { request, timeout in
      #expect(request == .saveNavigate(.init(app: .finalCutPro), navigation))
      #expect(timeout == .seconds(15))
      var response = UIResponse(
        focus: .init(frontmostBefore: "terminal", frontmostAfter: "terminal"))
      var receipt = FCPSaveNavigationReceipt(requestedDirectory: navigation.directory)
      receipt.status = status
      response.saveNavigation = receipt
      return response
    }
    let result = await NativeService(interfaces: interfaces).call(
      .init(name: "fcp_save_navigate", arguments: fields))
    #expect(result.isError == (status != .directoryVerified && status != .observeExistingDispatch))
    let value = try #require(result.structuredContent?.objectValue)
    #expect(value["completionVerified"] == .bool(false))
    #expect(value["retrySafe"] == .bool(false))
    #expect(value["nameAttempted"] == .bool(false))
    #expect(value["saveAttempted"] == .bool(false))
    #expect(value["saveNavigation"]?.objectValue?["status"] == .string(status.rawValue))
  }

  @Test func missingReceiptFailsAndUnknownProofFieldsNeverBecomeAuthority() async throws {
    let (_, fields) = try input()
    var interfaces = NativeInterfaces()
    interfaces.ui = { _, _ in UIResponse(focus: .init(frontmostBefore: nil, frontmostAfter: nil)) }
    let service = NativeService(interfaces: interfaces)
    let result = await service.call(.init(name: "fcp_save_navigate", arguments: fields))
    #expect(result.isError == true)
    var forged = fields
    forged["verifiedDisplayName"] = .string("Users")
    let rejected = await service.call(.init(name: "fcp_save_navigate", arguments: forged))
    #expect(rejected.isError == true)
    #expect(ToolSpec.fcpSave.first { $0.name == "fcp_save_navigate" }?.readOnly == false)
    #expect(ToolSpec.fcpSave.first { $0.name == "fcp_save_plan" }?.readOnly == true)
  }

  @Test func invalidDeadlineRejectedBeforeUI() async throws {
    let (_, original) = try input()
    var fields = original
    fields["maximumSeconds"] = .double(16)
    var interfaces = NativeInterfaces()
    interfaces.ui = { _, _ in
      Issue.record("Invalid input reached the UI boundary")
      return UIResponse(focus: .init(frontmostBefore: nil, frontmostAfter: nil))
    }
    #expect(
      await NativeService(interfaces: interfaces).call(
        .init(name: "fcp_save_navigate", arguments: fields)
      ).isError == true)
  }
}
