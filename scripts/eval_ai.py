"""Evaluate off-the-shelf AI-image detectors on real seller photos vs modern AI images.

Sets
  real        - real Shopee electronics photos (2021, before mass image generation)
  real_edited - the same photos after typical phone 'beautify' processing
                (sharpen + saturation + resize + JPEG) -> honest, must NOT be flagged
  ai          - Gemini 3.1 Flash Image / gpt-image-2 product photos (2026 generators,
                newer than all detectors -> 'unseen generator' test)
Every image passes the same normalisation (max side 512, JPEG q90) so a detector cannot
separate the sets by file format or resolution alone.

Usage: python scripts/eval_ai.py --n-real 160
"""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from photo_trust.ai_detector import load_detector, normalize_image  # noqa: E402
from photo_trust.config import AI_DETECTORS, AI_DIR, EVAL_DIR, RESULTS, SEED, SHOPEE_IMAGES  # noqa: E402
from photo_trust.data import open_image  # noqa: E402

TARGET_FPR = 0.05


def beautify(img, rng):
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=rng.randint(80, 150)))
    img = ImageEnhance.Color(img).enhance(rng.uniform(1.15, 1.4))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(1.05, 1.2))
    w, h = img.size
    return img.resize((int(w * 0.8), int(h * 0.8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=160)
    ap.add_argument("--detectors", nargs="+", default=["ufd"],
                    help="ids from config.AI_DETECTORS values; HF ones need a working huggingface.co")
    args = ap.parse_args()
    rng = random.Random(SEED)

    gal = pd.read_csv(EVAL_DIR / "gallery.csv")
    real_files = rng.sample(list(gal.image), args.n_real)
    real = [normalize_image(open_image(SHOPEE_IMAGES / f)) for f in real_files]
    edited = [normalize_image(beautify(open_image(SHOPEE_IMAGES / f), rng)) for f in real_files]
    ai_files = sorted(AI_DIR.rglob("*.png"))
    ai = [normalize_image(open_image(p)) for p in ai_files]
    print(f"real={len(real)} edited={len(edited)} ai={len(ai)}", flush=True)

    y = np.r_[np.zeros(len(real)), np.ones(len(ai))]
    summary, rocs = {}, {}
    for label, model_id in [(k, v) for k, v in AI_DETECTORS.items() if v in args.detectors]:
        try:
            det = load_detector(model_id)
        except Exception as e:  # model not downloadable / unknown labels
            print("skip", label, e)
            continue
        s_real, s_edit, s_ai = det.score(real), det.score(edited), det.score(ai)
        scores = np.r_[s_real, s_ai]
        thr = float(np.quantile(s_real, 1 - TARGET_FPR))
        summary[label] = {
            "model_id": model_id,
            "roc_auc": float(roc_auc_score(y, scores)),
            "threshold_at_fpr5_real": thr,
            "tpr_ai_at_fpr5": float(np.mean(s_ai > thr)),
            "fpr_edited_real_at_thr": float(np.mean(s_edit > thr)),
            "default_0.5": {"tpr_ai": float(np.mean(s_ai > 0.5)), "fpr_real": float(np.mean(s_real > 0.5)),
                            "fpr_edited": float(np.mean(s_edit > 0.5))},
            "n_real": len(real), "n_ai": len(ai),
        }
        fpr, tpr, _ = roc_curve(y, scores)
        rocs[label] = (fpr, tpr)
        print(label, json.dumps(summary[label], indent=1), flush=True)

    (RESULTS / "ai_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    plt.figure(figsize=(6.5, 5))
    for label, (f, t) in rocs.items():
        plt.plot(f, t, lw=2, label=f"{label}: AUC={summary[label]['roc_auc']:.2f}")
    plt.plot([0, 1], [0, 1], color="#9aa4ae", ls="--", lw=1)
    plt.axvline(TARGET_FPR, color="#c62828", ls=":", lw=1)
    plt.xlabel("FPR: реальные фото продавцов, ошибочно помеченные как ИИ")
    plt.ylabel("TPR: найденные ИИ-фото")
    plt.title("Сигнал «ИИ»: готовые детекторы на генераторах 2026 года")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "ai_roc.png", dpi=150)


if __name__ == "__main__":
    main()
