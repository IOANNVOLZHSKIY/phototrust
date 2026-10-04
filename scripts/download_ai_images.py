"""Download AI-generated e-commerce photos (CC0) from Hugging Face.

Source: prajwalkothwal/ai-generated-ecommerce-images — 6,031 images made with
Google Gemini (gemini-3.1-flash-image) and OpenAI gpt-image-2. We take electronics-related
categories only.  Usage: python scripts/download_ai_images.py --per-category 40
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from photo_trust.config import AI_DIR  # noqa: E402

REPO = "https://huggingface.co/datasets/prajwalkothwal/ai-generated-ecommerce-images/resolve/main"
TREE = "https://huggingface.co/api/datasets/prajwalkothwal/ai-generated-ecommerce-images/tree/main"
CATEGORIES = ["damaged_electronics", "damaged_phone_screen", "missing_parts", "package_damage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=40)
    args = ap.parse_args()
    out = AI_DIR / "gemini_gptimage2"
    out.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        # file indices are not contiguous -> take names from the repo listing
        listing = json.load(urllib.request.urlopen(f"{TREE}/{cat}"))
        for name in [x["path"].split("/")[-1] for x in listing][:args.per_category]:
            dst = out / f"{cat}__{name}"
            if dst.exists():
                continue
            try:
                urllib.request.urlretrieve(f"{REPO}/{cat}/{name}", dst)
                print("ok", dst.name, flush=True)
            except Exception as e:  # missing index in a category -> skip
                print("skip", cat, name, e, flush=True)


if __name__ == "__main__":
    main()
