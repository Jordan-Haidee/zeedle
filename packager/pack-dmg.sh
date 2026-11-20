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
PACKAGER_LOG=""
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
PACKAGER_LOG="$STAGING_DIR/cargo-packager.log"

printf 'Building Zeedle %s macOS DMG (%s)...\n' "$VERSION" "$PACKAGE_ARCH"

MAX_DMG_ATTEMPTS=5
for attempt in $(seq 1 "$MAX_DMG_ATTEMPTS"); do
    : > "$PACKAGER_LOG"
    set +e
    cargo packager --config "$VERSIONED_CONFIG" --out-dir "$STAGING_DIR" --formats dmg \
        2>&1 | tee "$PACKAGER_LOG"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    packager_status="${pipeline_status[0]}"
    tee_status="${pipeline_status[1]}"

    if [[ "$tee_status" -ne 0 && "$packager_status" -eq 0 ]]; then
        exit "$tee_status"
    fi
    if [[ "$packager_status" -eq 0 ]]; then
        break
    fi

    # GitHub macOS runners can transiently have Spotlight/XProtect hold the
    # temporary image while create-dmg invokes hdiutil.
    if [[ "$attempt" -lt "$MAX_DMG_ATTEMPTS" ]] \
        && grep -Fq 'hdiutil: create failed - Resource busy' "$PACKAGER_LOG"; then
        printf 'DMG creation hit a transient hdiutil resource conflict; retrying (%s/%s)...\n' \
            "$((attempt + 1))" "$MAX_DMG_ATTEMPTS" >&2
        rm -f -- "$STAGING_DIR"/*.dmg
        sleep "$((attempt * 5))"
        continue
    fi

    exit "$packager_status"
done

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
