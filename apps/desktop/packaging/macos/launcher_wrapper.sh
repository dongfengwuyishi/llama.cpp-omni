#!/bin/bash
# macOS .app launcher — uses bundled standalone Python

# Critical for code-signed bundles: never let Python write .pyc files into
# Resources/, otherwise `codesign --verify --deep --strict` would fail on the
# next launch ("file added: ...__pycache__/...") and Gatekeeper would refuse
# to open the app ("已损坏 / 无法打开"). Redirecting the cache outside the
# bundle is also fine, but the simplest fix is to disable bytecode writing
# entirely; the app starts fine without it.
export PYTHONDONTWRITEBYTECODE=1

DIR="$(cd "$(dirname "$0")" && pwd)"
RESOURCES="$DIR/../Resources"
APP_SUPPORT="$HOME/Library/Application Support/Comni"
LOG_FILE="$APP_SUPPORT/comni_service.log"

mkdir -p "$APP_SUPPORT"

PYTHON="$RESOURCES/python/bin/python3"

if [ ! -x "$PYTHON" ]; then
    osascript -e 'display alert "Bundled Python Missing" message "The app bundle appears corrupted.\nPlease re-download Comni." as critical'
    exit 1
fi

cd "$RESOURCES"
exec "$PYTHON" apps/desktop/menubar_app.py 2>>"$LOG_FILE"
