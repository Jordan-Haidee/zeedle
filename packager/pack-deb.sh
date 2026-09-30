#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RELEASE_DIR="$PROJECT_ROOT/target/release"
CONFIG="$SCRIPT_DIR/Packager.linux.toml"
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

VERSIONED_CONFIG="$(mktemp --suffix=.toml "$SCRIPT_DIR/.Packager.linux.XXXXXX")"
WORK_DIR=""
cleanup() {
    rm -f -- "$VERSIONED_CONFIG"
    if [[ -n "$WORK_DIR" ]]; then
        rm -rf -- "$WORK_DIR"
    fi
}
trap cleanup EXIT
{
    printf 'version = "%s"\n' "$VERSION"
    cat "$CONFIG"
} > "$VERSIONED_CONFIG"

DEB_FILE="$RELEASE_DIR/zeedle_${VERSION}_amd64.deb"
OUTPUT_FILE="$RELEASE_DIR/Zeedle_${VERSION}_amd64.deb"

printf 'Building Zeedle %s Debian package...\n' "$VERSION"
cargo packager --config "$VERSIONED_CONFIG" --formats deb

if [[ ! -f "$DEB_FILE" ]]; then
    printf 'Expected Debian package was not created: %s\n' "$DEB_FILE" >&2
    exit 1
fi

WORK_DIR="$(mktemp -d "$RELEASE_DIR/.deb-inject.XXXXXX")"

printf 'Adding Debian maintainer scripts...\n'
dpkg-deb --raw-extract "$DEB_FILE" "$WORK_DIR/package"
cp "$SCRIPT_DIR/debian/postinst" "$WORK_DIR/package/DEBIAN/postinst"
cp "$SCRIPT_DIR/debian/postrm" "$WORK_DIR/package/DEBIAN/postrm"
chmod 755 "$WORK_DIR/package/DEBIAN/postinst" "$WORK_DIR/package/DEBIAN/postrm"
dpkg-deb --build "$WORK_DIR/package" "$WORK_DIR/package.deb"

mv -f -- "$WORK_DIR/package.deb" "$OUTPUT_FILE"
if [[ "$DEB_FILE" != "$OUTPUT_FILE" ]]; then
    rm -- "$DEB_FILE"
fi

printf 'Package ready: %s\n' "$OUTPUT_FILE"
