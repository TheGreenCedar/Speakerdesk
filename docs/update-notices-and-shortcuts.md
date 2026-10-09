# Updates and keyboard shortcuts

The installed macOS app checks for an update once when it opens, using the existing signed updater. An available version appears above the workspace with a **Download update…** action and a dismiss button. Downloading opens Settings → Updates; installation and restart still require the separate **Install and restart** action. Recordings, including paused recordings, block downloading and installation.

Dismissing a notice stores that version in the existing local preferences database. It stays dismissed across restarts and changing loopback ports; a newer version gets its own notice. The Updates settings panel remains available for a dismissed version. Background checks that find no update or encounter an error leave the workspace open. Use **Speakerdesk → Check for Updates…** or Settings → Updates for a manual result.

| Shortcut | Action |
| --- | --- |
| ⌘, | Open Settings |
| ⌘N | Open the New Meeting form |
| ⌘O | Open the existing recording import picker |
| ⌘F | Find in the current transcript, or search meetings on the New Meeting screen |
| ⌘S | Save current transcript edits |
| Escape | Close Settings; clear an active transcript search |
| Tab / Shift-Tab | Move through controls; remain inside Settings while it is open |
| Left / Right, Home / End | Navigate the focused Settings tabs |

The native Edit menu retains standard Undo, Redo, Cut, Copy, Paste and Select All. Standard macOS Hide, Hide Others, Minimize and Quit are native menu items. App shortcuts use the same action dispatcher as matching buttons, respect disabled controls, and refuse navigation through an open dialog or protected update. Browser builds use the same Command/Control keyboard actions without changing text-editing shortcuts.

New Meeting opens a form rather than starting recording. It refuses while a recording is active, an identity edit or passage save is pending, or an Undo removal is outstanding. It saves normal transcript edits first, preserves a failed save unless the existing discard confirmation is accepted, and asks before discarding unsaved or recovered passage drafts. It rechecks recording/modal state after an asynchronous save.

Validation uses synthetic fixtures and the real editor/updater HTTP handlers. The tests exercise available/current/error states, explicit download, version-specific dismissal, persistence through runtime recreation, capture guards, modal/disabled controls, native-versus-browser dispatch, exact transcript saves and standard editing keys. Native menu compilation alone does not establish actual WebKit accelerator, file picker or accessibility behavior; those require a coordinated native fixture check.

Clickable controls show small keycaps for their existing bindings. The same
frontend map supplies keyboard dispatch, visible hints, tooltips and
`aria-keyshortcuts`; a regression checks native menu parity. Mac uses Command,
other browser platforms use Control. Keycaps are decorative and add no focus
stops. Find's hint follows its contextual target: meeting search on the start
screen and transcript search with a meeting open. Save retains its hint through
Saved, Save now and Saving states. Specialized model/voice settings destinations,
Save name and passage-draft commits have no equivalent global binding and gain
no invented shortcut.

`shortcut_hints.test.cjs` covers mapping, accessibility, platform and state
behavior. `shortcut_hints_browser.cjs` is an offline synthetic browser harness
for desktop/narrow/icon-only/dark/Control layout and screenshot checks; no app
backend, recording, audio source or model is used. Browser policy and runtime
availability must permit that harness; source tests do not establish visual or
native accessibility qualification.
