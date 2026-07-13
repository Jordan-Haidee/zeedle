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

# Patch .desktop file: add StartupNotify and StartupWMClass so the taskbar
# icon appears (without StartupWMClass the desktop environment can't link the
# running window to its .desktop entry).
DESKTOP_STAGING="target/release/.cargo-packager/appimage_deb/data/usr/share/applications/Zeedle.desktop"
DESKTOP_APPDIR="target/release/.cargo-packager/appimage/Zeedle.AppDir/usr/share/applications/Zeedle.desktop"
for f in "$DESKTOP_STAGING" "$DESKTOP_APPDIR"; do
    if [ -f "$f" ] && ! grep -q "StartupWMClass" "$f"; then
        printf "StartupNotify=true\nStartupWMClass=Zeedle\n" >> "$f"
    fi
done

# Rebuild AppImage with the patched .desktop file
cd target/release/.cargo-packager/appimage
bash build_appimage.sh
cd "$OLDPWD"

echo "✓ Package ready: $APPIMAGE_FILE"
