# AGENTS

Instructions for AI coding agents working in this repository.

## Scope
- Keep changes small and focused.
- Prefer editing existing modules in `src/` and `ui/` over adding new abstractions.
- Do not change dependency versions (especially Slint) unless explicitly requested.

## Quick Start
- Dev run: `cargo run`
- Release build: `cargo build --release`
- Tests: `cargo test`

## Project Map
- App entry and runtime orchestration: [src/main.rs](src/main.rs)
- Config load/save: [src/config.rs](src/config.rs)
- Logger setup and file target: [src/logger.rs](src/logger.rs)
- Media metadata, lyrics, sorting helpers: [src/utils.rs](src/utils.rs)
- Slint UI root: [ui/app.slint](ui/app.slint)
- Reusable UI components: [ui/button.slint](ui/button.slint), [ui/song.slint](ui/song.slint), [ui/lyric.slint](ui/lyric.slint)
- Build-time Slint and Windows icon resource setup: [build.rs](build.rs)

## Architecture Rules
- UI to backend communication uses message passing via `PlayerCommand` in [src/main.rs](src/main.rs#L23).
- Backend to UI updates must use Slint event-loop callbacks (`invoke_from_event_loop`) in [src/main.rs](src/main.rs).
- Keep audio playback logic in `main.rs` worker-thread flow; do not block the UI path.

## Conventions
- Rust naming: snake_case functions, PascalCase types, SCREAMING_SNAKE_CASE constants.
- Song scanning currently supports `mp3`, `flac`, `wav`, `ogg` in [src/utils.rs](src/utils.rs#L52).
- Config persistence path is `~/.config/zeedle/config.toml` from [src/config.rs](src/config.rs#L6).
- Default logger writes to stdout and `~/.zeedle.log` from [src/logger.rs](src/logger.rs#L23).

## Pitfalls
- Slint is pinned to `=1.13.1` in [Cargo.toml](Cargo.toml#L22) and [Cargo.toml](Cargo.toml#L32); keep versions aligned.
- `build.rs` compiles UI and bundles translations from `lang/`; UI/i18n changes may fail build if resources are inconsistent.
- On Windows, icon resource embedding is handled in [build.rs](build.rs#L5); avoid changing icon paths without updating resources.
- App uses single-instance behavior in [src/main.rs](src/main.rs#L355); do not remove this unless requested.

## Change Playbooks
- Add a new player command:
  1. Add enum variant in `PlayerCommand` in [src/main.rs](src/main.rs#L23).
  2. Handle it in backend command match in [src/main.rs](src/main.rs).
  3. Wire UI callback sender path in [src/main.rs](src/main.rs).
- Add new audio extension support:
  1. Update glob in [src/utils.rs](src/utils.rs#L52).
  2. Validate metadata and playback behavior on sample files.
- Add/adjust UI strings:
  1. Update Slint UI files in [ui/app.slint](ui/app.slint) and related components.
  2. Ensure translation assets under [lang/](lang/) stay in sync.

## Existing Docs
- English overview: [README.md](README.md)
- Chinese overview: [README-zh.md](README-zh.md)
