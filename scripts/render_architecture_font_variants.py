"""Render five typography choices and a compact comparison of their top sections."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nas-figure-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "font_variants"
FONTS = [
    ("Times New Roman", "01-times-new-roman"),
    ("Baskerville", "02-baskerville"),
    ("Charter", "03-charter"),
    ("Helvetica Neue", "04-helvetica-neue"),
    ("Avenir Next", "05-avenir-next"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (family, stem) in enumerate(FONTS):
        command = [sys.executable, str(ROOT / "scripts" / "render_architecture_comparison.py"),
                   "--font-family", family, "--output-dir", str(OUT), "--stem", stem]
        subprocess.run(command, cwd=ROOT, check=True)
        verification = json.loads((OUT / f"{stem}.verification.json").read_text())
        records.append({"number": index + 1, "font_family": family, "stem": stem,
                        "preview": verification["preview"],
                        "png_dimensions": verification["png_dimensions"],
                        "codex_accent_color": verification["codex_accent_color"]})

    # Each strip shows actual rendered headings, both model widths, and the first
    # block labels. Full versions retain every row and the original dimensions.
    width, row_height = 2000, 810
    sheet = plt.figure(figsize=(width / 100, len(records) * row_height / 100), facecolor="white")
    for index, record in enumerate(records):
        top = index * row_height
        sheet.text(0.035, 1 - (top + 37) / (len(records) * row_height),
                   f"{record['number']}. {record['font_family']}", fontsize=26,
                   family=record["font_family"], weight="bold", va="center", color="#17232B")
        with Image.open(record["preview"]) as preview:
            # The preview is 90 dpi; original chart coordinates are 100 dpi.
            crop = preview.crop((0, 0, preview.width, 900))
            ax = sheet.add_axes([0.01, 1 - (top + row_height) / (len(records) * row_height),
                                0.98, (row_height - 70) / (len(records) * row_height)])
            ax.imshow(crop)
            ax.axis("off")
    sheet.savefig(OUT / "comparison.png", dpi=100, facecolor="white")
    plt.close(sheet)
    (OUT / "manifest.json").write_text(json.dumps({
        "comparison": "comparison.png", "comparison_content": "top sections of each full figure",
        "variants": records,
    }, indent=2) + "\n")
    print(f"Five full figures and comparison sheet saved in {OUT}", flush=True)


if __name__ == "__main__":
    main()
