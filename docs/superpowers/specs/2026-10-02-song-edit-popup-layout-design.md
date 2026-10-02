# Song Edit Popup Layout Design

## Goal

Improve the visual hierarchy and spacing of the existing song metadata popup while preserving its edit, upload, save, and error behavior. The user selected preview A, the compact form layout.

## Current Problems

- Song title, artist, and album each consume a full vertical row, leaving the form taller than necessary.
- The cover preview and cover action are loosely aligned with the form and have no clear section boundary.
- The LRC button, existing-file status, and lyric preview read as separate controls rather than one editing area.
- The popup has substantial unused vertical space around its content, so the actions do not feel anchored to the form.

## Selected Layout

Keep the existing `SongEditPopup` in `ui/song_list.slint` and retain the application’s Theme tokens and standard Slint controls.

1. **Header:** Keep the existing title and add a short secondary description to establish that the dialog edits the selected song’s metadata.
2. **Title field:** Keep the song title label and input on a full-width row.
3. **Artist and album:** Place their label/input pairs in a balanced two-column row.
4. **Media cards:** Place cover and lyrics cards side by side. Give both cards the same background, border, corner radius, and internal spacing.
   - The cover card groups a square preview (or empty-cover state), selected filename/retention status, and the existing file picker button.
   - The lyrics card groups a short heading, the existing LRC picker button, filename/retention status, and the existing compact lyrics preview.
5. **Validation and actions:** Keep all existing error and required-field messages. Separate the bottom action row with the Theme divider token, align Cancel and Save to the end, and keep the actions visible at the popup’s supported height.

Use a popup width of 520 logical pixels and reduce its height from 510 to approximately 470–480 logical pixels after checking the rendered layout. Keep margins and gaps consistent; avoid introducing custom controls or new Slint files. Use only existing colors (`Theme.surface`, `Theme.panel`, `Theme.border`, `Theme.divider`, `Theme.text`, and `Theme.text-secondary`) so dark and light themes remain matched to the app.

## Behavior and Scope

- Keep title, artist, album bindings and save callback unchanged.
- Keep the cover and LRC file picker callbacks and their file validation/write behavior unchanged.
- Keep cover preview, lyrics preview, empty states, saving-disabled controls, required-field validation, and operation errors available.
- Add localization entries for any new visible copy in the POT template and all supported catalogs: zh_CN, fr, de, es, and ru.
- Make no changes to metadata persistence or playback behavior as part of this layout work.

## Verification

- Compile and run the existing unit tests, Clippy, and Rust formatter.
- Validate all gettext catalogs with `msgfmt --check`.
- Render the popup in dark and light themes and inspect that the two cards align, errors do not cover the footer, and all controls remain visible.
- Open both file pickers from the redesigned popup and confirm the popup remains open after a file is selected.
- Reopen the popup after save to confirm the visual changes did not disturb existing metadata display.

## Self-Review

- **Scope:** The change is limited to the existing song edit popup and its new visible strings; no backend or persistence refactor is needed.
- **Consistency:** All sections use current Theme tokens and native Slint controls, and both media cards share the same visual treatment.
- **Responsive fit:** The proposed height is provisional within the existing fixed popup size; final height is determined from an actual render with error text both hidden and visible.
- **No placeholders:** The selected direction, affected file, preserved behavior, translation scope, and verification steps are explicit.
