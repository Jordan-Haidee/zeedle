# Custom Window Title Bar for Zeedle

## Problem
Zeedle uses the native OS window title bar, which looks inconsistent with the custom-styled UI. A custom title bar provides a more cohesive visual experience.

## Design (Approved)
Remove the native window frame (`no-frame: true`) and replace it with a custom `TitleBar` component at the top of the window, inspired by [wsl-dashboard](https://github.com/owu/wsl-dashboard)'s implementation.

### Layout
```
┌──────────────────────────────────────────────────────────────┐
│  [drag region — "Zeedle"]        [🌙/☀️] [—] [□] [✕]  │
└──────────────────────────────────────────────────────────────┘
```

Left: window drag area with "Zeedle" label.
Right: theme toggle, minimize, maximize/restore, close buttons.

### Changes

| File | Change |
|------|--------|
| `ui/app.slint` | Add `no-frame: true`; import and place `TitleBar`; add window action callbacks |
| `ui/title_bar.slint` | **New** — custom title bar component with drag handling, buttons, theme toggle |
| `ui/sidebar.slint` | Remove "Zeedle" title text from toggle area |
| `src/main.rs` | Wire window drag, minimize, maximize, close callbacks |

### Key Details
- **no-frame** with outer `Rectangle` having `border-radius: 8px; clip: true` for rounded corners
- **Window dragging** via `TouchArea` pointer events → `window_drag_delta(dx, dy)` callback → `slint::Window::set_position()` in Rust
- **Window controls**: use `Window::minimized`, `Window::maximized`, `Window::close()` in Rust callbacks
- **Theme toggle**: move toggle from settings panel into title bar; keep the settings panel switch as well
- The existing sidebar "Zeedle" label is removed (only the collapse icon remains)

### Callbacks Added
- `window_minimize()` — set `Window::minimized`
- `window_maximize()` — toggle `Window::maximized`
- `window_close()` — call `Window::hide()` + `Window::close()`
- `window_drag_delta(float, float)` — move window via `set_position()`
- `toggle_theme()` — toggle light/dark UI
