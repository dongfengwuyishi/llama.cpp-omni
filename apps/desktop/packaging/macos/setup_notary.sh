#!/bin/bash
# One-time setup: store Apple notary credentials in the macOS keychain so
# build_dmg.sh can later run `notarytool submit` non-interactively.
#
# Usage:
#   bash apps/desktop/packaging/macos/setup_notary.sh \
#       --apple-id you@example.com \
#       --app-password xxxx-xxxx-xxxx-xxxx \
#       [--team-id VGXUYC2C5K] \
#       [--profile comni-notarytool]
#
# What it does:
#   xcrun notarytool store-credentials <profile> \
#       --apple-id <APPLE_ID> --team-id <TEAM_ID> --password <APP_PASSWORD>
#
# After this runs once, build_dmg.sh can notarize without re-prompting.
# The keychain item is named after <profile> (default: comni-notarytool).
#
# Prerequisites:
#   - Apple Developer Program membership ($99/year)
#   - "Developer ID Application" certificate already imported into the
#     login keychain (verify with: security find-identity -p codesigning -v)
#   - App-specific password generated at https://appleid.apple.com
#     (NOT your regular Apple ID password)

set -e

PROFILE="comni-notarytool"
APPLE_ID=""
APP_PASSWORD=""
TEAM_ID=""

while [ $# -gt 0 ]; do
    case "$1" in
        --apple-id)     APPLE_ID="$2"; shift 2 ;;
        --app-password) APP_PASSWORD="$2"; shift 2 ;;
        --team-id)      TEAM_ID="$2"; shift 2 ;;
        --profile)      PROFILE="$2"; shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$APPLE_ID" ] || [ -z "$APP_PASSWORD" ]; then
    echo "ERROR: --apple-id and --app-password are required"
    echo "Run with --help for usage"
    exit 1
fi

# Auto-detect Team ID from Developer ID Application certificate if not given.
if [ -z "$TEAM_ID" ]; then
    TEAM_ID="$(security find-identity -p codesigning -v 2>/dev/null \
        | grep "Developer ID Application" \
        | head -1 \
        | sed -E 's/.*\(([A-Z0-9]{10})\)".*/\1/')"
    if [ -z "$TEAM_ID" ]; then
        echo "ERROR: could not auto-detect Team ID."
        echo "Either pass --team-id, or import a 'Developer ID Application' certificate first."
        exit 1
    fi
    echo "Auto-detected Team ID from keychain: $TEAM_ID"
fi

echo "Storing notary credentials..."
echo "  profile  = $PROFILE"
echo "  apple-id = $APPLE_ID"
echo "  team-id  = $TEAM_ID"
echo

xcrun notarytool store-credentials "$PROFILE" \
    --apple-id "$APPLE_ID" \
    --team-id "$TEAM_ID" \
    --password "$APP_PASSWORD"

echo
echo "Verifying credentials..."
if xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1; then
    echo "Verified — build_dmg.sh can now notarize automatically."
else
    echo "WARNING: credentials stored but history call failed; check inputs."
fi
