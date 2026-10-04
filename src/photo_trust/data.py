"""Small helpers for loading eval images and caching embeddings."""
from pathlib import Path

import numpy as np
from PIL import Image

from .config import AI_DIR, CACHE, EVAL_DIR, SHOPEE_IMAGES


def resolve(ref):
    """'shopee/<file>' -> Shopee image, 'ai/<path>' -> AI image, 'attacked/<file>' -> generated query."""
    kind, name = ref.split("/", 1)
    return {"shopee": SHOPEE_IMAGES, "ai": AI_DIR}.get(kind, EVAL_DIR / kind) / name


def open_image(path):
    return Image.open(path).convert("RGB")


def cached_embed(embedder, paths, tag, chunk=64):
    """Embed images once and store them in cache/<model>_<tag>.npy."""
    f = CACHE / f"{embedder.name}_{tag}.npy"
    if f.exists():
        return np.load(f)
    out = []
    for i in range(0, len(paths), chunk):
        out.append(embedder.embed([open_image(p) for p in paths[i:i + chunk]]))
        print(f"  {embedder.name} {tag}: {min(i + chunk, len(paths))}/{len(paths)}", flush=True)
    arr = np.concatenate(out)
    np.save(f, arr)
    return arr


def paths_of(refs):
    return [Path(resolve(r)) for r in refs]
