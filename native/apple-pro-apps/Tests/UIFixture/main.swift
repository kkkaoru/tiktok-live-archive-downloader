import AppKit

/// Synthetic accessibility fixture for native UI backend tests. It is never an
/// Apple production app: an accessory-policy process (no Dock icon, never
/// activates itself) exposing a button, a text field, a table, a sheet and a menu.
@MainActor
final class FixtureController: NSObject, NSTableViewDataSource, NSTableViewDelegate {
  static let rows = ["alpha.fcpbundle", "beta.fcpbundle"]
  let window = NSWindow(
    contentRect: NSRect(x: 40, y: 40, width: 360, height: 240),
    styleMask: [.titled], backing: .buffered, defer: false)
  let status = NSTextField(labelWithString: "idle")
  let button = NSButton(title: "Press Me", target: nil, action: nil)
  let field = NSTextField(string: "")
  let checkbox = NSButton(checkboxWithTitle: "Option", target: nil, action: nil)
  let table = NSTableView()
  let scratchStatus = NSTextField(labelWithString: "not-configured")
  var scratchHandle: FileHandle?
  let sheet = NSWindow(
    contentRect: NSRect(x: 0, y: 0, width: 200, height: 80), styleMask: [.titled],
    backing: .buffered, defer: false)

  func build() {
    window.title = "UI Fixture"
    status.setAccessibilityIdentifier("fixture-status")
    status.frame = NSRect(x: 16, y: 200, width: 320, height: 20)
    scratchStatus.setAccessibilityIdentifier("fixture-scratch")
    scratchStatus.frame = NSRect(x: 16, y: 224, width: 320, height: 16)
    button.setAccessibilityIdentifier("fixture-button")
    button.target = self
    button.action = #selector(pressed)
    button.frame = NSRect(x: 16, y: 160, width: 120, height: 30)
    field.setAccessibilityIdentifier("fixture-field")
    field.frame = NSRect(x: 16, y: 120, width: 320, height: 24)
    checkbox.setAccessibilityIdentifier("fixture-checkbox")
    checkbox.frame = NSRect(x: 160, y: 160, width: 120, height: 30)
    let column = NSTableColumn(identifier: NSUserInterfaceItemIdentifier("name"))
    column.width = 300
    table.addTableColumn(column)
    table.headerView = nil
    table.setAccessibilityIdentifier("fixture-table")
    table.dataSource = self
    table.delegate = self
    let scroll = NSScrollView(frame: NSRect(x: 16, y: 16, width: 320, height: 90))
    scroll.documentView = table
    let sheetButton = NSButton(title: "Close Sheet", target: self, action: #selector(closeSheet))
    sheetButton.frame = NSRect(x: 16, y: 20, width: 140, height: 30)
    sheet.contentView?.addSubview(sheetButton)
    sheet.title = "Fixture Sheet"
    for view in [status, scratchStatus, button, field, checkbox, scroll] {
      window.contentView?.addSubview(view)
    }
    NSApp.mainMenu = menu()
    window.orderFrontRegardless()
  }

  func configureScratch() throws {
    guard let path = ProcessInfo.processInfo.environment["UI_FIXTURE_SCRATCH"] else { return }
    let directory = URL(fileURLWithPath: path, isDirectory: true)
    guard directory.path == directory.resolvingSymlinksInPath().path,
      FileManager.default.changeCurrentDirectoryPath(directory.path)
    else { throw CocoaError(.fileWriteInvalidFileName) }
    let marker = directory.appendingPathComponent("fixture-open-descriptor.txt")
    try Data("synthetic fixture only\n".utf8).write(to: marker, options: .withoutOverwriting)
    scratchHandle = try FileHandle(forWritingTo: marker)
    scratchStatus.stringValue = [
      directory.path, FileManager.default.currentDirectoryPath,
      FileManager.default.temporaryDirectory.path,
      ProcessInfo.processInfo.environment["LLVM_PROFILE_FILE"] ?? "missing",
    ].joined(separator: "\n")
  }

  func menu() -> NSMenu {
    let main = NSMenu()
    let appItem = NSMenuItem()
    appItem.submenu = NSMenu(title: "Fixture")
    let fileItem = NSMenuItem(title: "File", action: nil, keyEquivalent: "")
    let file = NSMenu(title: "File")
    let newItem = NSMenuItem(title: "New", action: nil, keyEquivalent: "")
    let new = NSMenu(title: "New")
    let thing = NSMenuItem(title: "Thing", action: #selector(menuThing), keyEquivalent: "")
    thing.target = self
    let sheetItem = NSMenuItem(title: "Sheet", action: #selector(openSheet), keyEquivalent: "")
    sheetItem.target = self
    let disabled = NSMenuItem(title: "Disabled", action: nil, keyEquivalent: "")
    new.addItem(thing)
    new.addItem(sheetItem)
    file.addItem(disabled)
    newItem.submenu = new
    file.addItem(newItem)
    fileItem.submenu = file
    main.addItem(appItem)
    main.addItem(fileItem)
    return main
  }

  @objc func pressed() {
    button.title = "Pressed"
    status.stringValue = "button-pressed"
  }

  @objc func menuThing() { status.stringValue = "menu-selected" }

  @objc func openSheet() { window.beginSheet(sheet) }

  @objc func closeSheet() { window.endSheet(sheet) }

  func numberOfRows(in tableView: NSTableView) -> Int { Self.rows.count }

  func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView?
  {
    NSTextField(labelWithString: Self.rows[row])
  }

  func tableViewSelectionDidChange(_ notification: Notification) {
    let row = table.selectedRow
    status.stringValue = row >= 0 ? "selected-\(Self.rows[row])" : "selected-none"
  }
}

let application = NSApplication.shared
application.setActivationPolicy(.accessory)
let controller = FixtureController()
try controller.configureScratch()
controller.build()
application.run()
