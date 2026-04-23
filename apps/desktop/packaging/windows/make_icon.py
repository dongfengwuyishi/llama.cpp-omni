#!/usr/bin/env python3
"""Convert apps/desktop/packaging/macos/Comni.png to a Windows .ico
with all the sizes Windows expects (16, 20, 24, 32, 40, 48, 64, 128, 256).

Usage:
    python apps/desktop/packaging/windows/make_icon.py

Requires:
    pip install Pillow
"""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent.parent
PNG_SRC = REPO_ROOT / "apps" / "desktop" / "packaging" / "macos" / "Comni.png"
ICO_DST = HERE / "Comni.ico"


def main() -> int:
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow is required. Run:  pip install Pillow")
        return 2

    if not PNG_SRC.is_file():
        print(f"ERROR: source PNG not found: {PNG_SRC}")
        return 3

    img = Image.open(PNG_SRC).convert("RGBA")
    sizes = [(16, 16), (20, 20), (24, 24), (32, 32), (40, 40),
             (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(ICO_DST, format="ICO", sizes=sizes)
    print(f"OK: wrote {ICO_DST}")
    print(f"    ({img.size[0]}x{img.size[1]} source → "
          f"{len(sizes)} embedded sizes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
