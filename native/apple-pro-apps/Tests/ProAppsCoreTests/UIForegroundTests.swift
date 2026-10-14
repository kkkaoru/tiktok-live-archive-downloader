import AppKit
import Foundation
import Testing

@testable import ProAppsCore

@MainActor
struct UIForegroundTests {
  let target = UITarget(app: .finalCutPro)
  let process = UIRunningProcess(pid: 42, bundleID: "com.apple.FinalCutApp")

  func request(_ path: [String]? = nil) -> UIRequest {
    .activate(target, expectedPID: 42, allowForeground: true, menuPath: path)
  }

  @Test func activatesOnceAndLeavesTheExactProcessForeground() async throws {
    let backend = FakeUIBackend.standard()
    let result = try await UIAutomation(backend: backend).run(request())
    #expect(backend.activated == [process])
    #expect(backend.launched.isEmpty)
    #expect(backend.performed.isEmpty)
    #expect(backend.written.isEmpty)
    #expect(result.activation?.status == .foregroundVerified)
    #expect(result.activation?.activationAccepted == true)
    #expect(result.activation?.observedFrontmostPID == 42)
    #expect(result.activation?.observedFrontmostBundleID == "com.apple.FinalCutApp")
    #expect(result.dispatched == true)
    #expect(result.effectVerified == true)
    #expect(result.focus.frontmostBefore == "dev.terminal")
    #expect(result.focus.frontmostAfter == "com.apple.FinalCutApp")
    #expect(result.focus.frontmostChanged)
  }

  @Test func alreadyForegroundNeedsNoActivationButCanPressOneMenu() async throws {
    let backend = FakeUIBackend.standard()
    backend.foregroundProcess = process
    let result = try await UIAutomation(backend: backend).run(
      request(["File", "New", "Library…"]))
    #expect(backend.activated.isEmpty)
    #expect(backend.performed == ["AXPress@39"])
    #expect(result.activation?.activationAttempted == false)
    #expect(result.activation?.menuAttempted == true)
    #expect(result.activation?.menuDispatched == true)
    #expect(result.activation?.status == .menuDispatched)
  }

  @Test func activationAndMenuShareOneVerifiedRequest() async throws {
    let backend = FakeUIBackend.standard()
    backend.attributes[39]?[.enabled] = .bool(false)
    backend.onActivate = { backend, process in
      backend.foregroundProcess = process
      backend.attributes[39]?[.enabled] = .bool(true)
    }
    let result = try await UIAutomation(backend: backend).run(
      request(["File", "New", "Library…"]))
    #expect(backend.activated == [process])
    #expect(backend.performed == ["AXPress@39"])
    #expect(result.activation?.status == .menuDispatched)
    #expect(result.activation?.foregroundVerified == true)
  }

  @Test func refusedActivationRetainsAttemptAndNeverPresses() async throws {
    let backend = FakeUIBackend.standard()
    backend.activationAccepted = false
    let result = try await UIAutomation(backend: backend).run(request(["File", "New"]))
    #expect(result.activation?.status == .activationRefused)
    #expect(result.activation?.activationAttempted == true)
    #expect(result.activation?.activationAccepted == false)
    #expect(result.effectVerified == false)
    #expect(backend.performed.isEmpty)
  }

  @Test func acceptanceWithoutExactForegroundTimesOutWithoutRetry() async throws {
    let backend = FakeUIBackend.standard()
    backend.onActivate = { backend, _ in
      backend.foregroundProcess = UIRunningProcess(pid: 43, bundleID: "com.apple.FinalCutApp")
    }
    let result = try await UIAutomation(backend: backend).run(request(["File", "New"]))
    #expect(result.activation?.status == .foregroundTimeout)
    #expect(result.activation?.observedFrontmostPID == 43)
    #expect(result.activation?.foregroundVerified == false)
    #expect(backend.clock >= 2 && backend.clock < 2.3)
    #expect(backend.activated.count == 1)
    #expect(backend.performed.isEmpty)
  }

  @Test func delayedForegroundIsObservedAndProcessReplacementRefused() async throws {
    let delayed = FakeUIBackend.standard()
    delayed.onActivate = { _, _ in }
    delayed.onPause = { backend in
      backend.foregroundProcess = UIRunningProcess(pid: 42, bundleID: "com.apple.FinalCutApp")
    }
    let success = try await UIAutomation(backend: delayed).run(request())
    #expect(success.effectVerified == true)
    #expect(delayed.clock == 0.2)

    let replaced = FakeUIBackend.standard()
    replaced.onActivate = { _, _ in }
    replaced.onPause = { backend in
      backend.processes[FakeUIBackend.fcp] = UIRunningProcess(pid: 99, bundleID: FakeUIBackend.fcp)
    }
    let rejected = try await UIAutomation(backend: replaced).run(request(["File", "New"]))
    #expect(rejected.activation?.status == .processChanged)
    #expect(rejected.running == false)
    #expect(replaced.performed.isEmpty)
  }

  @Test(arguments: [
    UIRequest.activate(
      .init(app: .finalCutPro), expectedPID: 42, allowForeground: false, menuPath: nil),
    .activate(.init(app: .finalCutPro), expectedPID: 0, allowForeground: true, menuPath: nil),
    .activate(.init(app: .finalCutPro), expectedPID: 99, allowForeground: true, menuPath: nil),
    .activate(.init(app: .motion), expectedPID: 42, allowForeground: true, menuPath: nil),
    .activate(.init(app: .finalCutPro), expectedPID: 42, allowForeground: true, menuPath: []),
    .activate(.init(app: .finalCutPro), expectedPID: 42, allowForeground: true, menuPath: [""]),
    .activate(
      .init(app: .finalCutPro), expectedPID: 42, allowForeground: true,
      menuPath: Array(repeating: "File", count: 7)),
    .activate(
      .init(app: .finalCutPro), expectedPID: 42, allowForeground: true,
      menuPath: [String(repeating: "x", count: 257)]),
  ])
  func validatesBeforeActivation(_ request: UIRequest) async throws {
    let backend = FakeUIBackend.standard()
    await #expect(throws: ProAppsError.self) {
      try await UIAutomation(backend: backend).run(request)
    }
    #expect(backend.activated.isEmpty)
    #expect(backend.performed.isEmpty)
  }

  @Test(arguments: [["File", "Disabled"], ["File", "Dup"], ["File", "Missing"]])
  func menuFailureRetainsSuccessfulActivation(_ path: [String]) async throws {
    let backend = FakeUIBackend.standard()
    let result = try await UIAutomation(backend: backend).run(request(path))
    #expect(result.activation?.status == .menuFailed)
    #expect(result.activation?.foregroundVerified == true)
    #expect(result.activation?.failure?.isEmpty == false)
    #expect(result.activation?.menuAttempted == false)
    #expect(backend.performed.isEmpty)
  }

  @Test func unknownEnablementAndFocusLossRefuseTheMenu() async throws {
    let unknown = FakeUIBackend.standard()
    unknown.attributes[39]?[.enabled] = nil
    let unknownResult = try await UIAutomation(backend: unknown).run(
      request(["File", "New", "Library…"]))
    #expect(unknownResult.activation?.status == .menuFailed)
    #expect(unknown.performed.isEmpty)

    let lost = FakeUIBackend.standard()
    lost.onActivate = { backend, process in backend.foregroundProcess = process }
    lost.onForegroundRead = { backend in
      if backend.foregroundReads >= 3 { backend.foregroundProcess = nil }
    }
    let lostResult = try await UIAutomation(backend: lost).run(
      request(["File", "New", "Library…"]))
    #expect(lostResult.activation?.status == .menuFailed)
    #expect(lostResult.effectVerified == false)
    #expect(lost.performed.isEmpty)
  }

  @Test func pressErrorRecordsTheAttemptAndDoesNotReplay() async throws {
    let backend = FakeUIBackend.standard()
    backend.performFailure = .accessibility(-25204)
    let result = try await UIAutomation(backend: backend).run(
      request(["File", "New", "Library…"]))
    #expect(result.activation?.menuAttempted == true)
    #expect(result.activation?.menuDispatched == false)
    #expect(result.activation?.status == .menuFailed)
    #expect(backend.activated.count == 1)
  }

  @Test func cancellationPropagatesInsteadOfReportingSuccess() async throws {
    let backend = FakeUIBackend.standard()
    backend.onActivate = { _, _ in }
    backend.pauseFailure = CancellationError()
    await #expect(throws: CancellationError.self) {
      try await UIAutomation(backend: backend).run(request(["File", "New"]))
    }
    #expect(backend.activated.count == 1)
    #expect(backend.performed.isEmpty)
  }

  @Test func activationReceiptRoundTrips() throws {
    var receipt = UIActivationReceipt(expectedPID: 42)
    receipt.status = .menuFailed
    receipt.activationAttempted = true
    receipt.failure = "disabled"
    let encoded = try JSONEncoder().encode(receipt)
    #expect(try JSONDecoder().decode(UIActivationReceipt.self, from: encoded) == receipt)
  }
}

@MainActor
private final class ForegroundApplicationFixture: UIForegroundApplication {
  var processIdentifier: Int32 = 42
  var bundleIdentifier: String? = "com.apple.FinalCutApp"
  var isTerminated = false
  var accepted = true
  var activationOptions: [NSApplication.ActivationOptions] = []
  func activate(options: NSApplication.ActivationOptions) -> Bool {
    activationOptions.append(options)
    return accepted
  }
}

@MainActor
private struct ForegroundSystemFixture: UIForegroundSystem {
  let running: ForegroundApplicationFixture?
  let frontmost: ForegroundApplicationFixture?
  func application(pid: Int32) -> (any UIForegroundApplication)? { running }
  func frontmostApplication() -> (any UIForegroundApplication)? { frontmost }
}

@MainActor
struct UIForegroundBoundaryTests {
  @Test func injectedPublicBoundaryUsesExactIdentityAndNoActivationOptions() {
    let application = ForegroundApplicationFixture()
    let boundary = UIForegroundActivation(
      system: ForegroundSystemFixture(running: application, frontmost: application))
    #expect(boundary.activate(.init(pid: 42, bundleID: "com.apple.FinalCutApp")))
    #expect(application.activationOptions == [[]])
    #expect(boundary.frontmostProcess() == .init(pid: 42, bundleID: "com.apple.FinalCutApp"))
    application.accepted = false
    #expect(!boundary.activate(.init(pid: 42, bundleID: "com.apple.FinalCutApp")))
  }

  @Test(arguments: [0, 1, 2, 3, 4, 5])
  func rejectsMissingTerminatedAndDifferentIdentities(_ condition: Int) {
    let application = ForegroundApplicationFixture()
    if condition == 1 { application.isTerminated = true }
    if condition == 2 { application.bundleIdentifier = "other" }
    if condition == 3 { application.processIdentifier = 99 }
    let boundary = UIForegroundActivation(
      system: ForegroundSystemFixture(running: condition == 0 ? nil : application, frontmost: nil))
    let pid: Int32 = condition == 4 ? 0 : 42
    let id = condition == 5 ? "other" : "com.apple.FinalCutApp"
    #expect(!boundary.activate(.init(pid: pid, bundleID: id)))
    #expect(application.activationOptions.isEmpty)
    #expect(boundary.frontmostProcess() == nil)
  }

  @Test func missingBundleAndTerminatedForegroundStayUnknown() {
    let application = ForegroundApplicationFixture()
    let boundary = UIForegroundActivation(
      system: ForegroundSystemFixture(running: nil, frontmost: application))
    application.bundleIdentifier = nil
    #expect(boundary.frontmostProcess() == nil)
    application.bundleIdentifier = "com.apple.FinalCutApp"
    application.isTerminated = true
    #expect(boundary.frontmostProcess() == nil)
  }

  @Test func liveReadOnlyBoundaryAndMissingPIDNeverActivateAnApp() {
    let system = WorkspaceForegroundSystem()
    #expect(system.application(pid: -1) == nil)
    _ = system.frontmostApplication()
    let live = LiveUIBackend()
    #expect(!live.activate(process: .init(pid: -1, bundleID: "com.apple.FinalCutApp")))
    _ = live.frontmostProcess()
  }
}
