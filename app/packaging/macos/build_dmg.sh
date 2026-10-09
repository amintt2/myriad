#!/usr/bin/env bash
# Put dist/Myriad.app in a compressed disk image with a shortcut to /Applications.
#   app/packaging/macos/build_dmg.sh VERSION ARCH        (ARCH: arm64 or x86_64)
set -euo pipefail
VERSION="$1"
ARCH="$2"
APP_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DIST="$APP_DIR/dist"
test -d "$DIST/Myriad.app"
STAGE="$(mktemp -d)"
cp -R "$DIST/Myriad.app" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
OUT="$DIST/Myriad-$VERSION-$ARCH.dmg"
rm -f "$OUT"
hdiutil create -volname "Myriad $VERSION" -srcfolder "$STAGE" -fs HFS+ -format UDZO -ov "$OUT"
rm -rf "$STAGE"
echo "$OUT"
