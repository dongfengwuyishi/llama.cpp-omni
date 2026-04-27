#!/bin/bash
# Wait for a previously submitted notarization job to finish, then staple
# the ticket onto the existing DMG.  Use this when build_dmg.sh's inline
# `notarytool submit --wait` had to be killed (e.g. Apple's queue is jammed
# and you don't want to block the build) but the submission itself is fine.
#
# The DMG is *already signed*; this script does not rebuild anything.
#
# Usage:
#   bash apps/desktop/packaging/macos/staple_pending.sh \
#       <submission-id> \
#       <path/to/Comni-macOS-arm64-X.Y.Z.dmg> \
#       [--profile comni-notarytool]
#
# Or auto-discover the most recent pending submission:
#   bash apps/desktop/packaging/macos/staple_pending.sh \
#       --auto path/to/Comni-macOS-arm64-X.Y.Z.dmg

set -e

PROFILE="comni-notarytool"
SUB_ID=""
DMG_PATH=""
AUTO=0

while [ $# -gt 0 ]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --auto)    AUTO=1; DMG_PATH="$2"; shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            if [ -z "$SUB_ID" ]; then
                SUB_ID="$1"
            elif [ -z "$DMG_PATH" ]; then
                DMG_PATH="$1"
            else
                echo "Unexpected argument: $1"; exit 1
            fi
            shift ;;
    esac
done

if [ "$AUTO" = "1" ]; then
    SUB_ID="$(xcrun notarytool history --keychain-profile "$PROFILE" 2>/dev/null \
        | awk '/^[[:space:]]*id:/ {print $2; exit}')"
    if [ -z "$SUB_ID" ]; then
        echo "ERROR: could not auto-detect a submission ID; pass it explicitly."
        exit 1
    fi
    echo "Auto-detected most recent submission: $SUB_ID"
fi

if [ -z "$SUB_ID" ] || [ -z "$DMG_PATH" ]; then
    echo "Usage: $(basename "$0") <submission-id> <dmg-path> [--profile NAME]"
    echo "       $(basename "$0") --auto <dmg-path>"
    exit 1
fi

if [ ! -f "$DMG_PATH" ]; then
    echo "ERROR: DMG not found at $DMG_PATH"
    exit 1
fi

echo "Submission ID : $SUB_ID"
echo "DMG           : $DMG_PATH"
echo "Profile       : $PROFILE"
echo

echo "Waiting for Apple to finish processing (will block until done)..."
xcrun notarytool wait "$SUB_ID" --keychain-profile "$PROFILE"

echo
echo "Fetching submission status..."
INFO_PLIST="$(dirname "$DMG_PATH")/notarytool-staple-${SUB_ID}.plist"
xcrun notarytool info "$SUB_ID" --keychain-profile "$PROFILE" \
    --output-format plist > "$INFO_PLIST"

STATUS="$(/usr/libexec/PlistBuddy -c 'Print :status' "$INFO_PLIST" 2>/dev/null || echo Unknown)"
echo "Final status: $STATUS"

if [ "$STATUS" != "Accepted" ]; then
    echo
    echo "Apple did not accept this submission. Pulling detailed log..."
    DETAIL="$(dirname "$DMG_PATH")/notarytool-staple-${SUB_ID}-detail.json"
    xcrun notarytool log "$SUB_ID" --keychain-profile "$PROFILE" "$DETAIL"
    echo "See: $DETAIL"
    exit 1
fi

echo
echo "Stapling notarization ticket onto DMG..."
xcrun stapler staple "$DMG_PATH"

echo
echo "Verifying with Gatekeeper..."
spctl -a -t open --context context:primary-signature -vv "$DMG_PATH"

echo
echo "Done. DMG is now notarized + stapled and works offline."
