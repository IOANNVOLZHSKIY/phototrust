"""Latency and throughput of every model in the pipeline on this machine (CPU or CUDA).
FAISS search is measured separately in benchmark_faiss.py (faiss and CUDA torch load
conflicting OpenMP runtimes in one process on Windows).

Usage: python scripts/benchmark.py
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from photo_trust import embedders  # noqa: E402
from photo_trust.ai_detector import load_detector, normalize_image  # noqa: E402
from photo_trust.config import AI_DETECTORS, DEVICE, EVAL_DIR, RESULTS, SHOPEE_IMAGES  # noqa: E402
from photo_trust.data import open_image  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, items, warmup=3):
    for it in items[:warmup]:
        fn(it)
    ts = []
    for it in items:
        t = time.perf_counter()
        fn(it)
        _sync()
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


def throughput(fn, images, batch=16):
    t = time.perf_counter()
    for i in range(0, len(images), batch):
        fn(images[i:i + batch])
    return len(images) / (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--detector", default=list(AI_DETECTORS)[0])
    args = ap.parse_args()

    files = pd.read_csv(EVAL_DIR / "gallery.csv").image[:args.n_images]
    imgs = [open_image(SHOPEE_IMAGES / f) for f in files]
    res = {"machine": {"cpu": platform.processor(), "threads": torch.get_num_threads(),
                       "torch": torch.__version__, "device": DEVICE,
                       "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None}}

    for name in ["pHash", "SSCD-288", "DINOv2"]:
        m = embedders.load(name)
        res[name] = timeit(lambda im: m.embed([im]), imgs)
        res[name]["images_per_sec_batch16"] = throughput(m.embed, imgs)
        print(name, res[name], flush=True)

    det = load_detector(AI_DETECTORS[args.detector])
    norm = [normalize_image(im) for im in imgs]
    res["ai_detector"] = timeit(lambda im: det.score([im]), norm)
    res["ai_detector"]["images_per_sec_batch16"] = throughput(det.score, norm)
    res["ai_detector"]["name"] = args.detector
    print("ai", res["ai_detector"], flush=True)

    out = RESULTS / f"benchmark_{DEVICE}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
