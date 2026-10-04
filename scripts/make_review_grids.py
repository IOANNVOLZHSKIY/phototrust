"""Build image grids for manual review of 'honest' Shopee photos that look suspiciously similar
to a photo in the base (SSCD >= 0.5 or pHash Hamming <= 12).

Each pair was labelled by eye in annotations/manual_review_labels.csv:
  S         - the same source picture (often with another seller's logo, frame or crop) = re-upload
  D         - a different photo of the same / similar product = honest seller
  ambiguous - collages, re-arranged templates; excluded from both sets
Run after the first pass of eval_copy.py (needs cache/pHash_* and cache/SSCD_*).

Usage: python scripts/make_review_grids.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from photo_trust.config import CACHE, EVAL_DIR, SHOPEE_IMAGES  # noqa: E402
from photo_trust.data import resolve  # noqa: E402
from photo_trust.embedders import PHash  # noqa: E402

TILE, PER_GRID = 140, 20


def top1(sim_fn, q, g, chunk=200):
    s = np.concatenate([sim_fn(q[i:i + chunk], g) for i in range(0, len(q), chunk)])
    return s.argmax(1), s.max(1)


def main():
    gal = pd.read_csv(EVAL_DIR / "gallery.csv")
    qs = pd.read_csv(EVAL_DIR / "queries.csv").fillna("")
    neg = qs.type.isin(["hard_neg", "easy_neg"]).values
    sub = qs[neg].reset_index(drop=True)
    ip, sp = top1(PHash.similarity, np.load(CACHE / "pHash_queries.npy")[neg], np.load(CACHE / "pHash_gallery.npy"))
    i_s, ss = top1(lambda a, b: a @ b.T, np.load(CACHE / "SSCD_queries.npy")[neg], np.load(CACHE / "SSCD_gallery.npy"))
    sub["sscd"], sub["ham"] = ss, (1 - sp) * 64
    sub["match"] = np.where(ss >= 0.5, gal.image.values[i_s], gal.image.values[ip])
    cand = sub[(sub.sscd >= 0.5) | (sub.ham <= 12)].sort_values("sscd", ascending=False).reset_index(drop=True)
    cand["rid"] = range(len(cand))
    out = EVAL_DIR / "review"
    out.mkdir(exist_ok=True)
    cand.to_csv(out / "candidates.csv", index=False)

    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 18)
    for gi in range(0, len(cand), PER_GRID):
        grid = Image.new("RGB", (4 * (2 * TILE + 20), 5 * (TILE + 24)), "white")
        d = ImageDraw.Draw(grid)
        for j, (_, r) in enumerate(cand.iloc[gi:gi + PER_GRID].iterrows()):
            x, y = (j % 4) * (2 * TILE + 20), (j // 4) * (TILE + 24)
            grid.paste(Image.open(resolve(r["query"])).convert("RGB").resize((TILE, TILE)), (x, y + 22))
            grid.paste(Image.open(SHOPEE_IMAGES / r["match"]).convert("RGB").resize((TILE, TILE)), (x + TILE + 2, y + 22))
            d.text((x, y), f"#{r.rid}  s={r.sscd:.2f} h={int(r.ham)}", font=font, fill=(200, 0, 0))
        grid.save(out / f"grid_{gi // PER_GRID:02d}.jpg", quality=85)
    print(f"{len(cand)} pairs -> {out}")


if __name__ == "__main__":
    main()
