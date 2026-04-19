#!/bin/bash
# Build deb package with maintainer scripts
set -e

cd "$(dirname "$0")/.."

# Get version from Cargo.toml
VERSION=$(grep -m1 '^version = ' Cargo.toml | sed 's/version = "\(.*\)"/\1/')
DEB_FILE="target/release/Zeedle_${VERSION}_amd64.deb"
WORK_DIR="target/release/.cargo-packager/deb-inject"

echo "Building deb package..."
cargo packager --config packager/Packager.linux.toml --release

echo "Injecting maintainer scripts..."
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

dpkg-deb -R "$DEB_FILE" "$WORK_DIR/package"

cp packager/debian/postinst "$WORK_DIR/package/DEBIAN/"
cp packager/debian/postrm "$WORK_DIR/package/DEBIAN/"
chmod 755 "$WORK_DIR/package/DEBIAN/postinst"
chmod 755 "$WORK_DIR/package/DEBIAN/postrm"

dpkg-deb -b "$WORK_DIR/package" "$DEB_FILE"
rm -rf "$WORK_DIR"

echo "✓ Package ready: $DEB_FILE"
