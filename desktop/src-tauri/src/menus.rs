//! Native accelerators call the same guarded page actions as buttons.
//! Predefined editing items retain WebKit's standard text editing behavior.
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::{App, AppHandle, Manager, Runtime};

pub fn install(app: &App) -> tauri::Result<()> {
    let settings = MenuItem::with_id(app, "settings", "Settings…", true, Some("Cmd+,"))?;
    let updates = MenuItem::with_id(
        app,
        "check-updates",
        "Check for Updates…",
        true,
        None::<&str>,
    )?;
    let application = Submenu::with_items(
        app,
        "Speakerdesk",
        true,
        &[
            &PredefinedMenuItem::about(app, None, None)?,
            &PredefinedMenuItem::separator(app)?,
            &settings,
            &updates,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::services(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::hide(app, None)?,
            &PredefinedMenuItem::hide_others(app, None)?,
            &PredefinedMenuItem::show_all(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::quit(app, None)?,
        ],
    )?;
    let new = MenuItem::with_id(app, "new-meeting", "New Meeting", true, Some("Cmd+N"))?;
    let import = MenuItem::with_id(
        app,
        "import-recording",
        "Import Recording…",
        true,
        Some("Cmd+O"),
    )?;
    let save = MenuItem::with_id(app, "save", "Save", true, Some("Cmd+S"))?;
    let file = Submenu::with_items(
        app,
        "File",
        true,
        &[&new, &import, &PredefinedMenuItem::separator(app)?, &save],
    )?;
    let find = MenuItem::with_id(app, "find", "Find…", true, Some("Cmd+F"))?;
    let edit = Submenu::with_items(
        app,
        "Edit",
        true,
        &[
            &PredefinedMenuItem::undo(app, None)?,
            &PredefinedMenuItem::redo(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::cut(app, None)?,
            &PredefinedMenuItem::copy(app, None)?,
            &PredefinedMenuItem::paste(app, None)?,
            &PredefinedMenuItem::select_all(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &find,
        ],
    )?;
    let window = Submenu::with_items(
        app,
        "Window",
        true,
        &[
            &PredefinedMenuItem::minimize(app, None)?,
            &PredefinedMenuItem::maximize(app, None)?,
        ],
    )?;
    app.set_menu(Menu::with_items(
        app,
        &[&application, &file, &edit, &window],
    )?)?;
    // Opt-in local diagnostics contain fixed action names and route states only.
    let tracing = std::env::var("SPEAKERDESK_MENU_TRACE").as_deref() == Ok("1");
    let trace_count = Arc::new(AtomicUsize::new(0));
    app.on_menu_event(move |app, event| {
        let trace = tracing && trace_count.fetch_add(1, Ordering::Relaxed) < 64;
        if route_menu_action(app, event.id().as_ref(), trace).is_err() {
            eprintln!("Speakerdesk could not send the menu action to its window.");
        }
    });
    Ok(())
}

// A native menu selection is the action source. NSWindow.isKeyWindow is a
// transient focus observation (for example while a menu/panel owns focus), not
// permission to discard that selection. Page modal, recording, disabled-control
// and update guards remain the authority for whether the action can run.
fn route_menu_action<R: Runtime>(
    app: &AppHandle<R>,
    action: &str,
    trace: bool,
) -> tauri::Result<bool> {
    if !matches!(
        action,
        "settings" | "check-updates" | "new-meeting" | "import-recording" | "save" | "find"
    ) {
        return Ok(false);
    }
    let Some(window) = app.get_webview_window("main") else {
        if trace {
            eprintln!("SPEAKERDESK_MENU action={action} phase=no-window");
        }
        return Ok(false);
    };
    if trace {
        let focus = match window.is_focused() {
            Ok(true) => "true",
            Ok(false) => "false",
            Err(_) => "unknown",
        };
        eprintln!("SPEAKERDESK_MENU action={action} phase=callback key_window={focus}");
    }
    // This acknowledges JavaScript receiver readiness, rather than silently
    // accepting an optional call to a missing receiver. No actions are queued
    // across navigation and no browser fallback is enabled alongside native.
    let encoded = serde_json::to_string(action).expect("fixed menu action");
    let script = format!("(() => {{ if (typeof window.speakerdeskAction !== 'function') return 0; void window.speakerdeskAction({encoded}); return 1; }})()");
    if trace {
        let action = action.to_owned();
        window.eval_with_callback(script, move |result| {
            let phase = if result == "1" {
                "receiver-ready"
            } else if result == "0" {
                "receiver-not-ready"
            } else {
                "eval-unconfirmed"
            };
            eprintln!("SPEAKERDESK_MENU action={action} phase={phase}");
        })?;
    } else {
        window.eval(script)?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tauri::test::{mock_app, MockRuntime};
    use tauri::{WebviewUrl, WebviewWindowBuilder};

    #[test]
    fn native_selection_reaches_page_when_main_window_is_not_key_focused() {
        let app = mock_app();
        let window = WebviewWindowBuilder::new(&app, "main", WebviewUrl::default())
            .build()
            .unwrap();
        assert!(
            !window.is_focused().unwrap(),
            "The real Tauri mock supplies the previously rejected focus condition."
        );
        for action in [
            "settings",
            "new-meeting",
            "import-recording",
            "find",
            "save",
            "check-updates",
        ] {
            assert!(
                route_menu_action::<MockRuntime>(app.handle(), action, false).unwrap(),
                "Native selection was discarded before page dispatch: {action}"
            );
        }
        assert!(!route_menu_action::<MockRuntime>(app.handle(), "paste", false).unwrap());
    }

    #[test]
    fn missing_main_window_does_not_queue_a_menu_action_for_later() {
        let app = mock_app();
        assert!(!route_menu_action::<MockRuntime>(app.handle(), "new-meeting", false).unwrap());
    }
}
