#!/usr/bin/env bash
# Package dist/Myriad (PyInstaller output) for Linux: .tar.gz, AppImage and .deb.
#   app/packaging/linux/build_linux.sh VERSION [APPIMAGETOOL]
# APPIMAGETOOL: path of appimagetool (the AppImage is skipped without it).
set -euo pipefail
VERSION="$1"
APPIMAGETOOL="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
APP="$(cd "$HERE/../.." && pwd)"
DIST="$APP/dist"
ARCH="$(uname -m)"            # x86_64, aarch64
DEB_ARCH="$(dpkg --print-architecture 2>/dev/null || echo amd64)"
test -x "$DIST/Myriad/Myriad"

# 1. Portable archive
tar -C "$DIST" -czf "$DIST/Myriad-$VERSION-linux-$ARCH.tar.gz" Myriad

# 2. AppImage
if [ -n "$APPIMAGETOOL" ]; then
  APPDIR="$DIST/Myriad.AppDir"
  rm -rf "$APPDIR"
  mkdir -p "$APPDIR/usr/lib" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
  cp -a "$DIST/Myriad" "$APPDIR/usr/lib/myriad"
  sed 's/^Exec=myriad/Exec=Myriad/' "$HERE/myriad.desktop" > "$APPDIR/myriad.desktop"
  cp "$HERE/myriad.desktop" "$APPDIR/usr/share/applications/myriad.desktop"
  cp "$APP/packaging/icons/myriad-256.png" "$APPDIR/myriad.png"
  cp "$APP/packaging/icons/myriad-256.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/myriad.png"
  cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/lib/myriad/Myriad" "$@"
EOF
  chmod +x "$APPDIR/AppRun"
  ARCH="$ARCH" "$APPIMAGETOOL" --no-appstream "$APPDIR" "$DIST/Myriad-$VERSION-$ARCH.AppImage"
fi

# 3. Debian package (installs to /opt/myriad, a `myriad` launcher in /usr/bin)
PKG="$DIST/deb/myriad_${VERSION}_${DEB_ARCH}"
rm -rf "$DIST/deb"
mkdir -p "$PKG/DEBIAN" "$PKG/opt" "$PKG/usr/bin" "$PKG/usr/share/applications" "$PKG/usr/share/icons/hicolor/256x256/apps"
cp -a "$DIST/Myriad" "$PKG/opt/myriad"
ln -s /opt/myriad/Myriad "$PKG/usr/bin/myriad"
cp "$HERE/myriad.desktop" "$PKG/usr/share/applications/myriad.desktop"
cp "$APP/packaging/icons/myriad-256.png" "$PKG/usr/share/icons/hicolor/256x256/apps/myriad.png"
SIZE_KB="$(du -sk "$PKG" | cut -f1)"
cat > "$PKG/DEBIAN/control" <<EOF
Package: myriad
Version: $VERSION
Section: net
Priority: optional
Architecture: $DEB_ARCH
Installed-Size: $SIZE_KB
Depends: libc6, libgl1, libegl1, libxkbcommon0, libfontconfig1, libdbus-1-3
Recommends: libvulkan1
Maintainer: Myriad contributors <noreply@github.com>
Homepage: https://github.com/amintt2/myriad
Description: Myriad - a myriad of small models, one answer
 A decentralised LLM: every computer runs a small open model and lends it to the
 network; questions go to several model families at once and their answers are
 fused by a weighted vote.
EOF
dpkg-deb --root-owner-group --build "$PKG" "$DIST/myriad_${VERSION}_${DEB_ARCH}.deb"
ls -la "$DIST"
