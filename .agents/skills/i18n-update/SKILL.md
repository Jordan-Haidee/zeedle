---
name: i18n-update
description: "Update and validate Zeedle i18n changes for Slint/gettext. Use when adding or editing UI text, changing labels/messages, syncing lang/zeedle.pot and lang/*/LC_MESSAGES/zeedle.po, or reviewing translation completeness."
argument-hint: "Describe the UI text changes and target languages, for example: add 2 playback labels and update zh_CN, fr"
---

# Zeedle i18n Update

## When to Use
- You changed user-visible text in Slint UI files under ui/.
- You added a new control, dialog, tooltip, or message string.
- You need to sync translation sources and language catalogs.
- You need a translation regression check before merge.

Typical triggers:
- i18n
- translation
- gettext
- po pot
- localization
- string-sync

## Inputs
- What strings changed and in which files.
- Which locales must be updated now, and which can be deferred.
- Whether this is a full sync or a focused hotfix.

## Procedure
1. Locate all user-visible text changes.
- Check changed Slint files in ui/ and any Rust-side user-facing text in src/.
- Build a list of added, removed, and edited messages.

2. Decide update scope.
- If changes are broad or structural: run a full catalog sync.
- If changes are small and urgent: do a focused update for required locales first.
- Always ensure English source strings are final before touching translations.

3. Sync message source.
- Update message template at [lang/zeedle.pot](../../../lang/zeedle.pot).
- Keep msgid stable when possible; avoid unnecessary rewrites that invalidate existing translations.

4. Update locale catalogs.
- Update impacted locale files under [lang](../../../lang/) such as:
  - [lang/zh_CN/LC_MESSAGES/zeedle.po](../../../lang/zh_CN/LC_MESSAGES/zeedle.po)
  - [lang/fr/LC_MESSAGES/zeedle.po](../../../lang/fr/LC_MESSAGES/zeedle.po)
  - [lang/de/LC_MESSAGES/zeedle.po](../../../lang/de/LC_MESSAGES/zeedle.po)
  - [lang/es/LC_MESSAGES/zeedle.po](../../../lang/es/LC_MESSAGES/zeedle.po)
  - [lang/ru/LC_MESSAGES/zeedle.po](../../../lang/ru/LC_MESSAGES/zeedle.po)
- For each changed msgid, ensure msgstr is translated or intentionally marked pending.

5. Verify runtime wiring and packaging assumptions.
- Confirm bundled translation setup still points to lang in [build.rs](../../../build.rs).
- Confirm runtime language selection path remains valid in [src/main.rs](../../../src/main.rs).

6. Run validation checks.
- Build/test cycle:
  - cargo run
  - cargo test
- Manually switch language in app and confirm:
  - New strings appear in target locales.
  - No fallback-to-source where translation should exist.
  - No obvious truncation or overflow in UI controls.

7. Produce change report.
- Summarize:
  - Which msgid entries changed.
  - Which locale files were updated.
  - Which locales are still pending and why.
- Include risk notes for partial locale rollout.

## Decision Points
- Full sync vs focused update:
  - Choose full sync for broad UI edits.
  - Choose focused update for urgent patch, but document pending locales.
- Keep or rewrite msgid:
  - Keep existing msgid for punctuation-only or minor edits when semantics are unchanged.
  - Rewrite msgid when meaning changes to avoid mistranslation drift.
- Block merge or allow partial:
  - Block if default locale or required release locales are broken.
  - Allow partial only with explicit pending list and follow-up issue.

## Completion Criteria
- All changed UI source strings are reflected in [lang/zeedle.pot](../../../lang/zeedle.pot).
- Required locale .po files are updated with no missing required entries.
- App can switch languages and render updated text correctly.
- Build and tests pass.
- Final report includes changed locales plus pending-work notes.

## Project References
- Overview and usage: [README.md](../../../README.md), [README-zh.md](../../../README-zh.md)
- Build-time translation bundling: [build.rs](../../../build.rs)
- App runtime orchestration: [src/main.rs](../../../src/main.rs)
- UI source: [ui/app.slint](../../../ui/app.slint)
