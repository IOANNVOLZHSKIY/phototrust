"""Compare pHash (baseline) vs SSCD vs DINOv2 on the 'copy of another listing's photo' task.

Protocol
  * each query is matched against the gallery; score = similarity of the top-1 match
  * honest photos were manually reviewed (data/eval/review/labels.csv): many Shopee listings of the
    same product re-use the SAME picture with their own logo/frame. Those become 'real_reuse'
    (real-world copies); ambiguous pairs are dropped; the rest stay honest (hard_neg / easy_neg)
  * threshold per method is set so that FPR on honest same-product photos (hard_neg) = 1%
  * synthetic positive is 'found' if top-1 match is its true source AND score > threshold
  * real_reuse is 'found' if score > threshold (the base can hold several copies of that picture)
Outputs results/copy_metrics.json, results/copy_per_attack.csv, plots, flagged negatives.

Usage: python scripts/eval_copy.py --methods pHash SSCD DINOv2
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from photo_trust import embedders  # noqa: E402
from photo_trust.attacks import ATTACK_LABELS_RU, ATTACKS, HARD_ATTACKS  # noqa: E402
from photo_trust.config import ANNOTATIONS, EVAL_DIR, RESULTS, SHOPEE_IMAGES  # noqa: E402
from photo_trust.data import cached_embed, paths_of  # noqa: E402

TARGET_FPR = 0.01
OPERATING_FPRS = [0.01, 0.02, 0.05, 0.10]
PRODUCT_FPR = 0.05   # queue threshold used by the demo: <= 5% honest photos go to a moderator
COLORS = {"pHash": "#9aa4ae", "DINOv2": "#e8a33d", "SSCD": "#7fa7d9", "SSCD-288": "#1565c0"}


def top1(method, q, g, chunk=200):
    idx, score = [], []
    for i in range(0, len(q), chunk):
        s = method.similarity(q[i:i + chunk], g)
        idx.append(s.argmax(1))
        score.append(s.max(1))
    return np.concatenate(idx), np.concatenate(score)


def recall_at_fpr(pos_ok, pos_score, neg_score, fpr):
    thr = float(np.quantile(neg_score, 1 - fpr, method="higher"))
    # strictly above the quantile keeps FPR <= target even with ties (pHash has discrete scores)
    return thr, float(np.mean(pos_ok & (pos_score > thr))), float(np.mean(neg_score > thr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["pHash", "SSCD", "DINOv2"])
    args = ap.parse_args()

    gal = pd.read_csv(EVAL_DIR / "gallery.csv")
    qs = pd.read_csv(EVAL_DIR / "queries.csv").fillna("")
    labels_f = ANNOTATIONS / "manual_review_labels.csv"
    if labels_f.exists():
        lab = dict(pd.read_csv(labels_f)[["query", "label"]].itertuples(index=False))
        relabel = {"S": "real_reuse", "ambiguous": "ambiguous"}
        qs["type"] = [relabel.get(lab.get(q), t) if t in ("hard_neg", "easy_neg") else t
                      for q, t in zip(qs["query"], qs.type)]
    gal_paths = [SHOPEE_IMAGES / i for i in gal.image]
    q_paths = paths_of(qs["query"])
    src_index = {img: k for k, img in enumerate(gal.image)}
    pos = (qs.type == "positive").values
    hard = (qs.type == "hard_neg").values
    easy = (qs.type == "easy_neg").values
    reuse = (qs.type == "reuse_same_photo").values
    real = (qs.type == "real_reuse").values
    true_src = np.array([src_index.get(s, -1) for s in qs.source])

    summary, per_attack, curves, curves_real = {}, [], {}, {}
    op_rows, attack_rows = [], []
    for name in args.methods:
        m = embedders.load(name)
        g = cached_embed(m, gal_paths, "gallery")
        q = cached_embed(m, q_paths, "queries")
        idx, score = top1(m, q, g)
        ok = idx == true_src
        thr, rec, fpr_hard = recall_at_fpr(ok[pos], score[pos], score[hard], TARGET_FPR)
        hard_mask = pos & qs.attack.isin(HARD_ATTACKS).values
        summary[name] = {
            "threshold": thr,
            "threshold_fpr_0.1pct": float(np.quantile(score[hard], 0.999, method="higher")),
            "recall_all_attacks": rec,
            "recall_hard_attacks": float(np.mean(ok[hard_mask] & (score[hard_mask] > thr))),
            "top1_retrieval_accuracy": float(ok[pos].mean()),
            "fpr_hard_neg_same_product": fpr_hard,
            "fpr_easy_neg_other_product": float(np.mean(score[easy] > thr)),
            "flagged_reuse_same_photo": float(np.mean(score[reuse] > thr)),
            "recall_real_reuse_manual": float(np.mean(score[real] > thr)),
            "n_positive": int(pos.sum()), "n_hard_neg": int(hard.sum()), "n_easy_neg": int(easy.sum()),
            "n_reuse_same_photo": int(reuse.sum()), "n_real_reuse": int(real.sum()),
        }
        for a in ATTACKS:
            msk = pos & (qs.attack == a).values
            per_attack.append({"method": name, "attack": a, "attack_ru": ATTACK_LABELS_RU[a],
                               "recall": float(np.mean(ok[msk] & (score[msk] > thr)))})
        for f in OPERATING_FPRS:
            t = float(np.quantile(score[hard], 1 - f, method="higher"))
            op_rows.append({"method": name, "fpr_target": f, "thr": round(t, 4),
                            "fpr": round(float(np.mean(score[hard] > t)), 4),
                            "recall_synth": round(float(np.mean(ok[pos] & (score[pos] > t))), 4),
                            "recall_real": round(float(np.mean(score[real] > t)), 4)})
            if f == PRODUCT_FPR:
                for a in ATTACKS:
                    msk = pos & (qs.attack == a).values
                    attack_rows.append({"method": name, "attack": a,
                                        "recall_fpr5": round(float(np.mean(ok[msk] & (score[msk] > t))), 4)})
        # recall vs FPR curve
        neg = np.sort(score[hard])[::-1]
        fprs = np.linspace(0.001, 0.2, 80)
        curves[name] = [(f, float(np.mean(ok[pos] & (score[pos] > np.quantile(neg, 1 - f))))) for f in fprs]
        curves_real[name] = [(f, float(np.mean(score[real] > np.quantile(neg, 1 - f)))) for f in fprs]
        # honest photos that would be flagged -> for manual review
        flagged = qs[hard & (score > thr)].assign(score=score[hard & (score > thr)],
                                                  matched=[gal.image[i] for i in idx[hard & (score > thr)]])
        flagged.to_csv(RESULTS / f"flagged_hard_neg_{name}.csv", index=False)
        print(name, json.dumps(summary[name], indent=1), flush=True)

    (RESULTS / "copy_metrics.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(op_rows).to_csv(RESULTS / "copy_operating_points.csv", index=False)
    pd.DataFrame(attack_rows).pivot(index="attack", columns="method", values="recall_fpr5").to_csv(
        RESULTS / "copy_per_attack_fpr5.csv")
    pa = pd.DataFrame(per_attack)
    pa.to_csv(RESULTS / "copy_per_attack.csv", index=False)

    # Plot 1: recall per attack at FPR=1%
    piv = pa.pivot(index="attack_ru", columns="method", values="recall")
    piv = piv.loc[[ATTACK_LABELS_RU[a] for a in ATTACKS], [m for m in args.methods if m in piv]]
    ax = piv.plot.barh(figsize=(9, 6), color=[COLORS.get(c, None) for c in piv.columns], width=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Доля найденных копий (recall) при FPR = 1% на честных фото того же товара")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(title="Метод", loc="lower right")
    plt.title("Сигнал «копия»: recall по типам изменений фото")
    plt.tight_layout()
    plt.savefig(RESULTS / "copy_recall_per_attack.png", dpi=150)
    plt.close()

    # Plot 2: recall vs FPR
    plt.figure(figsize=(7, 4.5))
    for name, pts in curves.items():
        f, r = zip(*pts)
        plt.plot(np.array(f) * 100, r, label=f"{name}: синтетические атаки", color=COLORS.get(name), lw=2)
        f, r = zip(*curves_real[name])
        plt.plot(np.array(f) * 100, r, label=f"{name}: реальные перезаливы", color=COLORS.get(name), lw=1.5, ls="--")
    plt.axvline(1, color="#c62828", ls="--", lw=1)
    plt.text(1.2, 0.05, "целевой FPR 1%", color="#c62828")
    plt.xlabel("FPR на честных фото того же товара, %")
    plt.ylabel("Recall копий")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.title("Компромисс «найти копию ↔ не задеть честного»")
    plt.tight_layout()
    plt.savefig(RESULTS / "copy_recall_vs_fpr.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
