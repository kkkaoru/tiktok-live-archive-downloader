import CoreGraphics
import Foundation

/// Separate foreground event boundary. It cannot type a path or request access.
@MainActor
public struct GoToFolderEventBackend {
  let permission: () -> Bool
  let post: (Int32) throws -> Void
  let postForeground: () throws -> Void

  public init(
    permission: @escaping () -> Bool, post: @escaping (Int32) throws -> Void,
    postForeground: @escaping () throws -> Void = {
      throw ProAppsError.unavailable("Explicit foreground event delivery is unsupported")
    }
  ) {
    self.permission = permission
    self.post = post
    self.postForeground = postForeground
  }

  /// Construct only Command-Shift-G. Construction does not post either event.
  static func shortcut(
    make: (Bool) -> CGEvent? = { CGEvent(keyboardEventSource: nil, virtualKey: 5, keyDown: $0) }
  ) throws -> [CGEvent] {
    guard let down = make(true), let up = make(false)
    else { throw ProAppsError.unavailable("Cannot create Go to Folder shortcut") }
    down.flags = [.maskCommand, .maskShift]
    up.flags = [.maskCommand, .maskShift]
    return [down, up]
  }

  /// Complete modifier transitions for explicit HID only. Allocate the entire
  /// chord before returning anything to the posting loop; never leave a partial
  /// constructed sequence for a caller to send. See CGEvent.h's keyboard example.
  static func foregroundShortcut(
    make: (CGKeyCode, Bool) -> CGEvent? = {
      CGEvent(keyboardEventSource: nil, virtualKey: $0, keyDown: $1)
    },
    permission: () -> Bool = CGPreflightPostEventAccess
  ) throws -> [CGEvent] {
    let chord: [(CGKeyCode, Bool, CGEventFlags)] = [
      (55, true, [.maskCommand]),
      (56, true, [.maskCommand, .maskShift]),
      (5, true, [.maskCommand, .maskShift]),
      (5, false, [.maskCommand, .maskShift]),
      (56, false, [.maskCommand]),
      (55, false, []),
    ]
    var events: [CGEvent] = []
    for (key, down, flags) in chord {
      guard let event = make(key, down) else {
        throw ProAppsError.unavailable("Cannot allocate complete HID Go to Folder chord")
      }
      event.flags = flags
      events.append(event)
    }
    guard permission() else {
      throw ProAppsError.unavailable("Event access unavailable after HID chord construction")
    }
    return events
  }
}
