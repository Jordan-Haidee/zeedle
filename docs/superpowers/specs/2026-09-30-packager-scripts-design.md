# Packager scripts design

## Goal

Make the platform packaging scripts easier to maintain, remove duplicate app version values from packager configuration, and use the release asset naming style from DBX v0.6.28.

## Naming

Final artifacts use the DBX capitalization and separators, with Zeedle's supported target architectures:

- `Zeedle_<version>_amd64.AppImage`
- `Zeedle_<version>_amd64.deb`
- `Zeedle_<version>_x64-setup.exe`

## Version source

`Cargo.toml` remains the only manually maintained application version. Remove `version` from both `Packager.linux.toml` and `Packager.windows.toml`, allowing cargo-packager to read the package version from the Cargo manifest. Packaging scripts should identify the generated artifact rather than separately parsing `Cargo.toml`.

## Script responsibilities

- Keep separate Linux AppImage, Linux deb, and Windows NSIS entry points.
- Preserve the AppImage desktop-file patch and rebuild step.
- Preserve Debian maintainer-script injection.
- Quote paths, stop on command failures, and report the final artifact only after confirming it exists.
- Rename generated artifacts to the agreed DBX-style names.
- Preserve the existing custom NSIS template configuration and user-owned working changes.

## Validation

Check shell syntax and PowerShell parsing, validate both TOML files, and inspect generated naming behavior. Run platform packaging commands where the corresponding packaging tools and host platform are available; otherwise record the unavailable platform-specific checks.
