import AppKit
import Foundation

/// The narrow public AppKit boundary is injectable without desktop interaction.
@MainActor
protocol UIForegroundApplication {
  var processIdentifier: Int32 { get }
  var bundleIdentifier: String? { get }
  var isTerminated: Bool { get }
  func activate(options: NSApplication.ActivationOptions) -> Bool
}

extension NSRunningApplication: UIForegroundApplication {}

@MainActor
protocol UIForegroundSystem {
  func application(pid: Int32) -> (any UIForegroundApplication)?
  func frontmostApplication() -> (any UIForegroundApplication)?
}

@MainActor
struct WorkspaceForegroundSystem: UIForegroundSystem {
  func application(pid: Int32) -> (any UIForegroundApplication)? {
    NSRunningApplication(processIdentifier: pid)
  }

  func frontmostApplication() -> (any UIForegroundApplication)? {
    NSWorkspace.shared.frontmostApplication
  }
}

@MainActor
struct UIForegroundActivation<System: UIForegroundSystem> {
  let system: System

  func activate(_ process: UIRunningProcess) -> Bool {
    guard process.pid > 0,
      ProApp.allCases.contains(where: { $0.bundleIDs.contains(process.bundleID) }),
      let application = system.application(pid: process.pid),
      !application.isTerminated, application.processIdentifier == process.pid,
      application.bundleIdentifier == process.bundleID
    else { return false }
    return application.activate(options: [])
  }

  func frontmostProcess() -> UIRunningProcess? {
    guard let application = system.frontmostApplication(), !application.isTerminated,
      let bundleID = application.bundleIdentifier
    else { return nil }
    return UIRunningProcess(pid: application.processIdentifier, bundleID: bundleID)
  }
}

extension LiveUIBackend {
  public func activate(process: UIRunningProcess) -> Bool {
    UIForegroundActivation(system: WorkspaceForegroundSystem()).activate(process)
  }

  public func frontmostProcess() -> UIRunningProcess? {
    UIForegroundActivation(system: WorkspaceForegroundSystem()).frontmostProcess()
  }
}
