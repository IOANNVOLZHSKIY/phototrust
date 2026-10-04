"""Build evaluation sets for the 'copy' signal from Shopee (Electronics subset).

Gallery  = photos already published on the platform (one photo per product group).
Gallery also gets 3000 non-electronics distractors.
Queries:
  reuse_same_photo - same product, identical picture already in the base (stock photo); reported apart
  positive  - a gallery photo after a fraudster edit (11 attack types)  -> must be found
  hard_neg  - another photo of the SAME product from a different listing -> honest seller
  easy_neg  - photo of a product that is not in the gallery at all       -> honest seller
Also writes data/eval/shopee_reuse_stats.json: how often listings reuse identical photos.

Usage: python scripts/build_eval_sets.py --n-sources 200
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from photo_trust import attacks  # noqa: E402
from photo_trust.config import ELECTRONICS_KEYWORDS, EVAL_DIR, SEED, SHOPEE_CSV, SHOPEE_IMAGES  # noqa: E402


def is_electronics(title):
    t = f" {str(title).lower()} "
    return any(k in t for k in ELECTRONICS_KEYWORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sources", type=int, default=200, help="gallery photos to attack")
    ap.add_argument("--n-easy", type=int, default=300, help="groups held out of the gallery")
    ap.add_argument("--n-distractors", type=int, default=3000, help="non-electronics photos in the base")
    args = ap.parse_args()
    rng = random.Random(SEED)

    df = pd.read_csv(SHOPEE_CSV)
    stats = {"shopee_postings": len(df), "shopee_groups": int(df.label_group.nunique())}
    el = df[df.title.map(is_electronics)].copy()
    stats.update(electronics_postings=len(el), electronics_groups=int(el.label_group.nunique()),
                 electronics_unique_images=int(el.image.nunique()))

    # Real-world photo reuse: the same image file used by several postings.
    per_img = el.groupby("image").agg(n=("posting_id", "size"), groups=("label_group", "nunique"))
    stats["postings_with_reused_image_share"] = float((el.image.map(per_img.n) > 1).mean())
    stats["images_used_in_3plus_postings"] = int((per_img.n >= 3).sum())
    stats["postings_with_identical_phash_share"] = float(el.duplicated("image_phash", keep=False).mean())

    # One row per unique image; product groups with their distinct photos.
    el = el.drop_duplicates("image")
    groups = {g: list(sub.image) for g, sub in el.groupby("label_group")}
    gids = sorted(groups)
    rng.shuffle(gids)
    easy_groups, gallery_groups = gids[:args.n_easy], gids[args.n_easy:]

    phash_of = dict(zip(el.image, el.image_phash))
    gallery, hard_neg, reuse = [], [], []
    for g in gallery_groups:
        imgs = groups[g][:]
        rng.shuffle(imgs)
        query = imgs.pop() if len(imgs) > 1 else None
        gallery += [(i, g) for i in imgs]          # all other photos of the product stay in the base
        if query is None:
            continue
        # Same perceptual hash as a photo already in the base = the same picture re-uploaded
        # (stock / manufacturer photo), not the seller's own photo -> separate bucket.
        if phash_of[query] in {phash_of[i] for i in imgs}:
            reuse.append((query, g))
        else:
            hard_neg.append((query, g))
    easy_neg = [(rng.choice(groups[g]), g) for g in easy_groups]

    el_gallery = gallery[:]
    # Distractors from other categories make the search base closer to a real platform.
    other = df[~df.image.isin(el.image)].drop_duplicates("image")
    gallery += [(i, g) for i, g in other.sample(args.n_distractors, random_state=SEED)[["image", "label_group"]]
                .itertuples(index=False)]

    rows = [{"query": f"shopee/{img}", "type": "hard_neg", "attack": "", "source": "", "group": g}
            for img, g in hard_neg]
    rows += [{"query": f"shopee/{img}", "type": "reuse_same_photo", "attack": "", "source": "", "group": g}
             for img, g in reuse]
    rows += [{"query": f"shopee/{img}", "type": "easy_neg", "attack": "", "source": "", "group": g}
             for img, g in easy_neg]

    out_dir = EVAL_DIR / "attacked"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = rng.sample(el_gallery, args.n_sources)
    for k, (img, g) in enumerate(sources):
        src = Image.open(SHOPEE_IMAGES / img).convert("RGB")
        for j, name in enumerate(attacks.ATTACKS):
            p = out_dir / f"{k:04d}_{name}.jpg"
            if not p.exists():
                attacks.apply(name, src, seed=SEED + 1000 * k + j).save(p, quality=92)
            rows.append({"query": f"attacked/{p.name}", "type": "positive", "attack": name,
                         "source": img, "group": g})
        if k % 20 == 0:
            print(f"attacked {k}/{len(sources)}", flush=True)

    pd.DataFrame({"image": [i for i, _ in gallery], "group": [g for _, g in gallery]}).to_csv(
        EVAL_DIR / "gallery.csv", index=False)
    pd.DataFrame(rows).to_csv(EVAL_DIR / "queries.csv", index=False)
    stats.update(gallery=len(gallery), gallery_electronics=len(el_gallery), reuse_same_photo=len(reuse),
                 positives=args.n_sources * len(attacks.ATTACKS),
                 hard_neg=len(hard_neg), easy_neg=len(easy_neg))
    (EVAL_DIR / "shopee_reuse_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
