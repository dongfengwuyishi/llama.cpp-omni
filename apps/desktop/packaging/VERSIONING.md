# Comni — Versioning Policy

## TL;DR

Comni ships **one version per release**, covering both macOS and Windows. The version is a normal semantic version `X.Y.Z` and only ever moves forward.

| 你看到的版本号 | 是什么 |
|---|---|
| GitHub release tag — e.g. `v1.0.19` | The release. Always the latest is `releases/latest`. |
| dmg / exe filename suffix — e.g. `1.0.19` | Same number as the release tag. |
| Info dialog inside the app | Same number; reflects the binary that was bundled. |

If two version numbers ever look out of sync, the **release tag wins** as the canonical "which Comni is this".

## Rules

1. **One number per release.** macOS and Windows do not have separate version sequences. A new release bumps the shared counter by one (patch by default).
2. **Strictly increasing.** A newer release always has a higher tag than every older release. We never reuse or backdate a tag.
3. **Per-release, not per-platform.** A release may only update one platform's binary (e.g. a macOS-only fix). The other platform's previous binary is re-published under the new release tag with the new version suffix; its content (and SHA-256) is unchanged.
4. **Latest is permanent.** `releases/latest/download/<file>` always points to the highest-numbered release.

## Asset names

Each release publishes four downloadable files on GitHub:

| File | Stable for archival | Permanent latest alias |
|---|---|---|
| macOS | `Comni-macOS-arm64-X.Y.Z.dmg` | `Comni-macOS-arm64.dmg` |
| Windows | `Comni-Setup-X.Y.Z-win64.exe` | `Comni-Setup-win64.exe` |

The version-suffixed names freeze a specific release for users who want a stable, never-mutating download. The non-suffixed aliases always point to the most recent release.

The same downloads are mirrored on ModelScope under `app/`:

| File | Direct URL |
|---|---|
| `Comni-macOS-arm64.dmg` | `https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf/resolve/master/app/Comni-macOS-arm64.dmg` |
| `Comni-Windows-x64.exe` | `https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf/resolve/master/app/Comni-Windows-x64.exe` |

ModelScope only carries the latest build (a single canonical filename per platform); the version is recorded in `app/VERSION.txt`.

## How to tell which version you have

* **macOS**: right-click `Comni.app` → **Get Info** → Version field, or open Comni → menu bar → **About**.
* **Windows**: right-click `Comni-*.exe` → **Properties** → **Details** → Product version.
* **Inside the app**: the main window's status bar shows the version next to the app name.

## Why one number across platforms

A separate `mac=1.0.x` / `win=1.0.y` scheme caused two problems in practice:

1. The two sequences drifted, and the larger of the two would visually appear "newer" in the GitHub release list even when it wasn't the latest.
2. People had to remember which platform they were on before reading a version number.

A single shared number removes both problems and matches how most cross-platform desktop apps version themselves.
