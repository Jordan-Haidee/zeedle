# Windows MSI Packaging Design

## Goal

Add a Windows MSI installer for Zeedle using the existing `cargo-packager` setup, and publish it alongside the current NSIS installer on tagged GitHub releases.

## Repository evidence

- `packager/Packager.windows.toml` contains the shared Windows package metadata and currently selects the `nsis` format.
- `packager/pack-nsis.ps1` reads the package version from `Cargo.toml`, writes a temporary versioned config, packages into a staging directory, verifies the expected artifact, and moves it to `target/release` under a stable name.
- `.github/workflows/release.yml` builds the NSIS installer on Windows, uploads one Windows artifact, and expects three total installer files across Windows and Linux.
- `README.md`, `README-zh.md`, and `AGENTS.md` document the current NSIS build command.
- `cargo-packager` supports MSI generation through the `wix` package format.

## Packaging workflow

Add `packager/pack-msi.ps1` following the existing NSIS script's version and staging behavior. Reuse `Packager.windows.toml` and pass `--formats wix` to `cargo packager`, keeping the NSIS script and its selected format unchanged. WiX requires numeric MSI versions, so map `alpha.N`, `beta.N`, and `rc.N` prereleases in the temporary config to separate numeric prerelease ranges; pass stable and already numeric versions through. Keep the original Cargo version in the output filename. Require exactly one generated `.msi`, then move it to `target/release/Zeedle_<version>_x64.msi`; fail clearly if packaging fails or the artifact is absent or ambiguous. Remove transient config and staging files in a `finally` block.

## Release workflow and documentation

Run both Windows packaging scripts in the Windows release job. Upload both Windows installers as a single workflow artifact and include them in the final GitHub Release. Update the release artifact glob and expected total from three to four files. Add the MSI build command alongside the NSIS command in both READMEs and in the repository's distribution quick start.

## Validation

- Parse the PowerShell script and confirm its failure/cleanup paths.
- Run the MSI packaging command on Windows using the repository's current `0.6.3-alpha.1` version and confirm it creates exactly one valid `.msi` at the stable release path.
- Check that the release workflow's build steps, artifact globs, and file-count check agree with the four published files.
- Review the updated build instructions and packaging configuration for consistency.

## Scope

This adds MSI packaging and release distribution while preserving the existing NSIS installer and Linux package workflows. It does not change application code or add a custom WiX installer template; numeric version mapping is limited to the temporary config required by WiX.
