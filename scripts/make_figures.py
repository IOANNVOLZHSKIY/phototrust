"""Figures for the Gate 3 report from results/*.csv (run after eval_copy.py and the
operating-point export).  Usage: python scripts/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from photo_trust.attacks import ATTACK_LABELS_RU, ATTACKS  # noqa: E402
from photo_trust.config import RESULTS  # noqa: E402

METHODS = ["pHash", "DINOv2", "SSCD-288"]
COLORS = {"pHash": "#9aa4ae", "DINOv2": "#e8a33d", "SSCD-288": "#1565c0"}


def per_attack():
    pa = pd.read_csv(RESULTS / "copy_per_attack_fpr5.csv").set_index("attack")
    pa = pa.loc[list(ATTACKS), METHODS]
    pa.index = [ATTACK_LABELS_RU[a] for a in pa.index]
    ax = pa.plot.barh(figsize=(9, 6.2), color=[COLORS[m] for m in METHODS], width=0.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall копий при FPR 5% на честных фото того же товара")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(title="Метод", loc="lower right")
    plt.title("Сигнал «копия»: какие изменения фото ловит каждый метод")
    plt.tight_layout()
    plt.savefig(RESULTS / "fig_copy_per_attack_fpr5.png", dpi=150)
    plt.close()


def operating_points():
    op = pd.read_csv(RESULTS / "copy_operating_points.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col, title in [(axes[0], "recall_synth", "Синтетические атаки (2 200 копий)"),
                           (axes[1], "recall_real", "Реальные перезаливы в Shopee (186, ручная разметка)")]:
        for m in METHODS:
            d = op[op.method == m]
            ax.plot(d.fpr * 100, d[col], marker="o", lw=2, color=COLORS[m], label=m)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("FPR на честных фото того же товара, %")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Recall")
    axes[0].legend()
    plt.suptitle("Компромисс «найти копию ↔ не задеть честного продавца»")
    plt.tight_layout()
    plt.savefig(RESULTS / "fig_copy_operating_points.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    per_attack()
    operating_points()
    print("ok")
