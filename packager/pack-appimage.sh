#!/bin/bash
# Build AppImage package
set -e

cd "$(dirname "$0")/.."

# Get version from Cargo.toml
VERSION=$(grep -m1 '^version = ' Cargo.toml | sed 's/version = "\(.*\)"/\1/')
APPIMAGE_FILE="target/release/Zeedle_${VERSION}_x86_64.AppImage"

echo "Building AppImage package..."
# Unset all_proxy to avoid SOCKS proxy issues (cargo-packager doesn't support SOCKS)
unset all_proxy ALL_PROXY
cargo packager --config packager/Packager.linux.toml --release --formats appimage

echo "✓ Package ready: $APPIMAGE_FILE"
