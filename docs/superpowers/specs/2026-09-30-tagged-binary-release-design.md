# Tagged Binary Release Design

## Goal

When a version tag is pushed to Zeedle, build and publish the Windows and Linux installation packages on a GitHub Release.

## Repository evidence

- The crate version is maintained in `Cargo.toml` (`0.6.2` in the reviewed worktree).
- Existing release tags use the unprefixed form, such as `0.6.2`; the latest GitHub Release contains Windows NSIS, Debian, and AppImage assets.
- `packager/pack-nsis.ps1`, `packager/pack-deb.sh`, and `packager/pack-appimage.sh` already build those formats and derive their package version from `Cargo.toml`.
- The repository has no existing `.github/workflows` workflow.

## Workflow

Add `.github/workflows/release.yml`, triggered by a pushed tag. Accept both the established `0.6.3` form and the common `v0.6.3` form, but require the normalized tag to equal the crate version in `Cargo.toml`. A mismatch fails before packaging.

Build on native GitHub-hosted runners in parallel: Windows runs `packager/pack-nsis.ps1`; Ubuntu 22.04 runs `packager/pack-deb.sh` followed by `packager/pack-appimage.sh`. Install the Rust stable toolchain, `cargo-packager`, and Linux packaging prerequisites on the relevant runners. Each platform uploads its completed files as a workflow artifact.

A final job waits for both platform builds, downloads their artifacts, and creates one GitHub Release for the pushed tag with generated release notes and all three packages. Use the built-in `GITHUB_TOKEN`; grant `contents: write` only to this publishing job. A failed package build prevents the release job from publishing partial assets.

## Validation

Review the workflow syntax and the tag/version and asset-path handling locally. The first version-matching tag pushed after this change is the end-to-end validation of hosted Windows/Linux packaging and Release upload; do not create a test tag as part of implementation.

## Scope

This change adds the release workflow and does not alter application code or packaging scripts. It does not add separate branch or pull-request CI jobs.
