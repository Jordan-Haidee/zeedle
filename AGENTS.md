# AGENTS

Instructions for AI coding agents working in this repository.

## Quick Start
- Dev run (with console): `cargo run`
- Release build (no console): `cargo build --release`
- Tests: `cargo test`
- fmt (requires nightly): `cargo +nightly fmt`
- clippy: `cargo clippy --all-targets --all-features -- -D warnings`
- Pre-commit hooks: `cargo +nightly fmt && cargo clippy --all-targets --all-features -- -D warnings`

## Distribution
- Windows NSIS: `./packager/pack-nsis.ps1`
- Linux deb: `./packager/pack-deb.sh`
- Linux AppImage: `./packager/pack-appimage.sh`
- All use `cargo packager` under the hood — see `packager/` for configs.

## Project Map
- Single binary crate, no workspace. Entrypoint: [src/main.rs](src/main.rs)
- Backend (worker thread, rodio playback): [src/main.rs](src/main.rs#L201)
- UI bridge types (generated from Slint): [src/slint_types.rs](src/slint_types.rs)
- Config load/save (TOML): [src/config.rs](src/config.rs)
- Logger (stdout + file): [src/logger.rs](src/logger.rs)
- Media metadata/lyrics/sorting: [src/utils.rs](src/utils.rs)
- FFT spectrum analysis: [src/spectrum.rs](src/spectrum.rs)
- Slint UI root: [ui/app.slint](ui/app.slint)
- Slint UI types: [ui/types.slint](ui/types.slint) (structs, enums)
- Slint custom theme: [ui/theme.slint](ui/theme.slint) (light/dark palette)
- Reusable Slint components: `ui/control_panel.slint`, `ui/song_list.slint`, `ui/lyrics_panel.slint`, `ui/settings_panel.slint`, `ui/sidebar.slint`, `ui/title_bar.slint`, `ui/spectrum.slint`
- Example: [examples/rodio_usage.rs](examples/rodio_usage.rs)

## Architecture Rules
- UI → backend communication: `mpsc::channel<PlayerCommand>` [src/main.rs](src/main.rs#L42). Variants: `Play`, `Pause`, `ChangeProgress`, `PlayNext`, `PlayPrev`, `SwitchMode`, `RefreshSongList`, `SortSongList`, `SetLang`, `ChangeVolume`, `SetShowSpectrum`.
- Backend → UI updates: must use `slint::invoke_from_event_loop` — never block the UI path.
- Audio playback lives in the worker thread via `rodio::Player` (wrapped in `Arc<Mutex<...>>`).
- Spectrum: `TapSource` taps audio samples in the playback thread, sends chunks to a separate FFT worker thread via `mpsc::sync_channel`, results polled by a Slint `Timer`.
- Window is `no-frame: true` — custom `TitleBar` with native OS drag via `winit::window().drag_window()`, with manual `set_position()` fallback.
- 5-page navigation: 0=Gallery (song list), 1=Search, 2=Lyrics, 3=Settings, 4=About.
- Two-phase init: (1) `set_start_ui_state` restores UI from config, (2) `set_start_player_state` starts playback.

## Hotkeys (defined in `ui/app.slint` FocusScope)
| Key | Action |
|-----|--------|
| `Space` | Play/pause toggle |
| `←` | Seek backward 15s; on lyrics page (page 2), snap to previous lyric line |
| `→` | Seek forward 15s; on lyrics page (page 2), snap to next lyric line |
| `↑` | Previous song |
| `↓` | Next song |
| `F1` | Gallery (song list, page 0) |
| `F2` | Search (page 1) |
| `F3` | Lyrics (page 2) |
| `F4` | Settings (page 3) |
| `F5` | About (page 4) |

## Theme
- Custom `Theme` global in [ui/theme.slint](ui/theme.slint) with 11 fully opaque color tokens (panel, surface, text, accent, etc.).
- Light/dark controlled by `Theme.is-dark`, set via `set_light_theme()` alongside `Palette.color-scheme` for std widgets.
- `dark-light` crate polls system theme every 1s (timer); `dark_light::Mode::Light` checks if light.

## Conventions
- Logging: `env_logger` with level `Info` by default, writes to stdout + `~/.zeedle.log` (auto-trim at 10MB).
- Config: `~/.config/zeedle/config.toml` (serde TOML).
- Supported audio extensions: `mp3`, `flac`, `wav`, `ogg`.
- Metadata: `lofty` for reading tags + cover art; `pinyin` for Chinese sorting.
- Album cover: falls back to `ui/cover.svg`.
- Font: auto-selects per platform (Windows: `Microsoft YaHei UI`, Linux: `Noto Sans CJK SC`, macOS: `PingFang SC`).

## Pitfalls
- Slint pinned to `=1.17.1` in both `[dependencies]` and `[build-dependencies]` — keep versions aligned.
- Build-time: `build.rs` uses `slint_build::CompilerConfiguration` with style `fluent` and `with_bundled_translations("lang")`. UI/i18n changes may fail if translations inconsistent.
- Windows icon embedding via `winresource` in `build.rs` (Windows-only build dep).
- Debug builds have a visible console window; release builds use `#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]` (Windows only).
- `single-instance` prevents multiple instances — do not remove.
- Release profile: `lto=true`, `codegen-units=1`, `strip=true`, `panic="abort"`.
- Rust edition is `2024` — `cargo +nightly fmt` is required for formatting (`use_small_heuristics = "Off"`, `imports_granularity = "Crate"`, `group_imports = "StdExternalCrate"`).
- Slint types are auto-generated via `slint::include_modules!()` in `src/slint_types.rs`. Add `#[allow(unused_imports)]` as needed.

## Change Playbooks
- **Add a player command:** 1) Add variant to `PlayerCommand` enum, 2) Handle in `start_player_backend_thread` match, 3) Wire UI callback sender in `register_ui_callbacks`.
- **Add audio format:** Update glob in `src/utils.rs` line 62.
- **Add/edit UI text:** Update Slint files, then run the i18n-update workflow (skill at `.agents/skills/i18n-update/SKILL.md`).
- **Add Slint component:** Create file in `ui/`, import in `ui/app.slint`, add slint_types export and Rust wiring.

## Existing Docs
- [README.md](README.md) — English overview
- [README-zh.md](README-zh.md) — Chinese overview
- [.agents/skills/i18n-update/SKILL.md](.agents/skills/i18n-update/SKILL.md) — i18n workflow
