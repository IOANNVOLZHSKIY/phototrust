"""Pick demo examples for app.py from the evaluated sets: typical successes AND typical failures
(the Gate 3 demo must show where the prototype is wrong). Writes results/demo_examples.json.

Usage: python scripts/pick_demo_examples.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from photo_trust.attacks import ATTACK_LABELS_RU  # noqa: E402
from photo_trust.config import AI_DIR, ANNOTATIONS, CACHE, EVAL_DIR, RESULTS  # noqa: E402

MODEL, FPR = "SSCD-288", 0.05


def main():
    gal = pd.read_csv(EVAL_DIR / "gallery.csv")
    qs = pd.read_csv(EVAL_DIR / "queries.csv").fillna("")
    lab = dict(pd.read_csv(ANNOTATIONS / "manual_review_labels.csv")[["query", "label"]].itertuples(index=False))
    ops = pd.read_csv(RESULTS / "copy_operating_points.csv")
    thr = float(ops[(ops.method == MODEL) & (ops.fpr_target == FPR)].thr.iloc[0])
    S = np.load(CACHE / f"{MODEL}_queries.npy") @ np.load(CACHE / f"{MODEL}_gallery.npy").T
    top, score = S.argmax(1), S.max(1)
    src = {img: k for k, img in enumerate(gal.image)}
    found = np.array([t == src.get(s, -2) for t, s in zip(top, qs.source)]) & (score > thr)
    rev = qs["query"].map(lab).fillna("")

    ex = []
    for attack, want in [("hflip", True), ("screenshot", True), ("border", True), ("crop_80", True),
                         ("crop_55", False), ("combo", False)]:
        k = np.where((qs.attack == attack).values & (found == want))[0][2]
        tag = "" if want else " — ПРОМАХ"
        ex.append((f"Мошенник: копия чужого фото ({ATTACK_LABELS_RU[attack]}){tag}", qs["query"][k]))
    real = (rev == "S").values
    ex.append(("Реальный перезалив чужого фото (Shopee)", qs["query"][np.where(real & (score > thr))[0][5]]))
    ex.append(("Реальный перезалив чужого фото (Shopee) — ПРОМАХ", qs["query"][np.where(real & (score <= thr))[0][3]]))
    honest = (qs.type == "hard_neg").values & (rev != "S").values & (rev != "ambiguous").values
    for i, k in enumerate(np.where(honest & (score <= thr))[0][[0, 5]]):
        ex.append((f"Честный продавец: тот же товар, своё фото #{i + 1}", qs["query"][k]))
    ex.append(("Честный продавец — ЛОЖНОЕ СРАБАТЫВАНИЕ", qs["query"][np.where(honest & (score > thr))[0][0]]))
    for i, p in enumerate(sorted(AI_DIR.rglob("*.png"))[:3]):
        ex.append((f"ИИ-сгенерированное фото #{i + 1}", f"ai/{p.relative_to(AI_DIR).as_posix()}"))
    (RESULTS / "demo_examples.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(ex, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
