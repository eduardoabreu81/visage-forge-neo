"""Check the optional SFW filter on a folder of images. Prints scores only; images are never shown.

    <forge>/venv/Scripts/python.exe tools/sfw_check.py <folder> [--models <forge>/models/visage]
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from lib_visage import models, safety  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--models", default=os.path.join("models", "visage"))
    a = ap.parse_args()
    models.set_models_dir(a.models)

    files = sorted(f for ext in ("png", "jpg", "jpeg", "webp") for f in glob.glob(os.path.join(a.folder, f"*.{ext}")))
    flagged = 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        v = safety.votes(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        hit = sum(v.values()) >= 2
        flagged += hit
        print(f"{'NSFW' if hit else 'ok  '}  {sum(v.values())}/3  {os.path.basename(f)}")
    print(f"\n{flagged} of {len(files)} flagged")


if __name__ == "__main__":
    main()
