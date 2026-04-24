#!/usr/bin/env python3
"""
Shared version manager for Comni macOS/Windows packaging.

State lives in apps/desktop/packaging/VERSION_LOG.json so both platforms
write to the same log and testers can tell at a glance which build any
installer in the wild corresponds to.

Subcommands
-----------
peek <platform> [--bump {patch,minor,major}] [--set X.Y.Z]
    Print the version the next build *would* produce.
    Does NOT modify VERSION_LOG.json. Used by the build scripts to stamp
    artifact names before the build has succeeded.

record <platform> --version X.Y.Z [--commit H] [--arch A] [--note TEXT]
    Commit a successful build: update "current" and append a history entry.
    Called at the very end of the build scripts so failed builds do not
    waste version numbers.

Platforms: macos | windows
Default bump part: patch
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_LOG = _SCRIPT_DIR / "VERSION_LOG.json"
_DEFAULT_VERSION = "1.0.0"
_PLATFORMS = ("macos", "windows")


def _load(path: Path) -> dict:
    if not path.exists():
        return {"current": {}, "history": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("current", {})
    data.setdefault("history", [])
    return data


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _parse_version(s: str) -> tuple[int, int, int]:
    parts = s.strip().split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Invalid semver (expected X.Y.Z): {s!r}")
    a, b, c = (int(p) for p in parts)
    return a, b, c


def _bump(ver: str, part: str) -> str:
    a, b, c = _parse_version(ver)
    if part == "major":
        return f"{a + 1}.0.0"
    if part == "minor":
        return f"{a}.{b + 1}.0"
    return f"{a}.{b}.{c + 1}"


def _next_version(data: dict, platform: str, bump: str, explicit: str | None) -> str:
    if explicit:
        _parse_version(explicit)  # validate
        return explicit
    current = data["current"].get(platform)
    if current is None:
        # First ever build on this platform: start at default, do not bump.
        return _DEFAULT_VERSION
    return _bump(current, bump)


def _cmd_peek(args: argparse.Namespace) -> int:
    data = _load(Path(args.log))
    print(_next_version(data, args.platform, args.bump, args.explicit))
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    _parse_version(args.version)  # validate before touching the log
    log_path = Path(args.log)
    data = _load(log_path)
    data["current"][args.platform] = args.version
    entry = {
        "platform": args.platform,
        "version": args.version,
        "commit": args.commit or "unknown",
        "arch": args.arch or "",
        "built_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if args.note:
        entry["note"] = args.note
    data["history"].append(entry)
    _save(log_path, data)
    print(f"Recorded {args.platform} {args.version} -> {log_path}", file=sys.stderr)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    data = _load(Path(args.log))
    print(json.dumps(data.get("current", {}), indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(_DEFAULT_LOG),
                    help=f"Path to VERSION_LOG.json (default: {_DEFAULT_LOG})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_peek = sub.add_parser("peek", help="Print next version, do not write.")
    p_peek.add_argument("platform", choices=_PLATFORMS)
    p_peek.add_argument("--bump", default="patch", choices=("patch", "minor", "major"))
    p_peek.add_argument("--set", dest="explicit", default=None,
                        help="Override the computed version with an explicit X.Y.Z.")
    p_peek.set_defaults(func=_cmd_peek)

    p_rec = sub.add_parser("record", help="Record a successful build.")
    p_rec.add_argument("platform", choices=_PLATFORMS)
    p_rec.add_argument("--version", required=True)
    p_rec.add_argument("--commit", default="")
    p_rec.add_argument("--arch", default="")
    p_rec.add_argument("--note", default="")
    p_rec.set_defaults(func=_cmd_record)

    p_show = sub.add_parser("show", help="Print current versions as JSON.")
    p_show.set_defaults(func=_cmd_show)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
