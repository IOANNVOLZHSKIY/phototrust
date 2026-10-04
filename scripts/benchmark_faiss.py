"""FAISS exact-search latency (IndexFlatIP) on synthetic bases of L2-normalised 512-d vectors
(SSCD size). Exact search is the upper bound; IVF/HNSW indexes are faster but the faiss-cpu
1.8 Windows wheel crashes on them, so they are left for the Linux production setup.

Usage: python scripts/benchmark_faiss.py --sizes 100000 1000000
"""
import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    d, res = 512, {}
    for n in args.sizes:
        base = rng.standard_normal((n, d), dtype=np.float32)
        faiss.normalize_L2(base)
        q = base[:100] + 0.05 * rng.standard_normal((100, d), dtype=np.float32)
        faiss.normalize_L2(q)
        index = faiss.IndexFlatIP(d)
        index.add(base)
        for v in q[:5]:
            index.search(v[None], 5)
        ts = []
        for v in q:
            t = time.perf_counter()
            index.search(v[None], 5)
            ts.append((time.perf_counter() - t) * 1000)
        res[str(n)] = {"p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
                       "index_ram_mb": n * d * 4 / 1e6}
        print(n, res[str(n)], flush=True)
        del base, index
    (RESULTS / "benchmark_faiss.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
