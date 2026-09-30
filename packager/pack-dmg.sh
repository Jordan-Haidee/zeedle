#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RELEASE_DIR="$PROJECT_ROOT/target/release"
CONFIG="$SCRIPT_DIR/Packager.macos.toml"
cd "$PROJECT_ROOT"

VERSION="$(awk -F= '
    /^\[package\][[:space:]]*$/ { in_package = 1; next }
    /^\[/ { in_package = 0 }
    in_package && $1 ~ /^[[:space:]]*version[[:space:]]*$/ {
        gsub(/[[:space:]\"]/, "", $2)
        print $2
        exit
    }
' Cargo.toml)"
if [[ -z "$VERSION" ]]; then
    printf 'Could not read the package version from Cargo.toml\n' >&2
    exit 1
fi

case "$(uname -m)" in
    arm64|aarch64) PACKAGE_ARCH="aarch64" ;;
    x86_64) PACKAGE_ARCH="x64" ;;
    *)
        printf 'Unsupported macOS architecture: %s\n' "$(uname -m)" >&2
        exit 1
        ;;
esac

TEMP_CONFIG_BASE=""
VERSIONED_CONFIG=""
STAGING_DIR=""
cleanup() {
    if [[ -n "$TEMP_CONFIG_BASE" ]]; then
        rm -f -- "$TEMP_CONFIG_BASE"
    fi
    if [[ -n "$VERSIONED_CONFIG" ]]; then
        rm -f -- "$VERSIONED_CONFIG"
    fi
    if [[ -n "$STAGING_DIR" ]]; then
        rm -rf -- "$STAGING_DIR"
    fi
}
trap cleanup EXIT

TEMP_CONFIG_BASE="$(mktemp "$SCRIPT_DIR/.Packager.macos.XXXXXX")"
VERSIONED_CONFIG="${TEMP_CONFIG_BASE}.toml"
mv -- "$TEMP_CONFIG_BASE" "$VERSIONED_CONFIG"
TEMP_CONFIG_BASE=""
{
    printf 'version = "%s"\n' "$VERSION"
    cat "$CONFIG"
} > "$VERSIONED_CONFIG"

mkdir -p -- "$RELEASE_DIR"
STAGING_DIR="$(mktemp -d "$RELEASE_DIR/.dmg-staging.XXXXXX")"
OUTPUT_DMG="$RELEASE_DIR/Zeedle_${VERSION}_${PACKAGE_ARCH}.dmg"

printf 'Building Zeedle %s macOS DMG (%s)...\n' "$VERSION" "$PACKAGE_ARCH"
cargo packager --config "$VERSIONED_CONFIG" --out-dir "$STAGING_DIR" --formats dmg

shopt -s nullglob
GENERATED_DMGS=("$STAGING_DIR"/*.dmg)
if [[ "${#GENERATED_DMGS[@]}" -ne 1 ]]; then
    printf 'Expected exactly one DMG in %s, found %s\n' \
        "$STAGING_DIR" "${#GENERATED_DMGS[@]}" >&2
    exit 1
fi
if [[ ! -s "${GENERATED_DMGS[0]}" ]]; then
    printf 'Generated DMG is empty: %s\n' "${GENERATED_DMGS[0]}" >&2
    exit 1
fi

mv -f -- "${GENERATED_DMGS[0]}" "$OUTPUT_DMG"
if [[ ! -s "$OUTPUT_DMG" ]]; then
    printf 'DMG was not created: %s\n' "$OUTPUT_DMG" >&2
    exit 1
fi

printf 'Package ready: %s\n' "$OUTPUT_DMG"
