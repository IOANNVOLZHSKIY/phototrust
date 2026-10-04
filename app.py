"""PhotoTrust demo: upload a listing photo -> copy search + AI check -> moderation decision.

Run after the eval scripts (they build the gallery embeddings and thresholds):
    streamlit run app.py
"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from photo_trust import embedders  # noqa: E402
from photo_trust.ai_detector import load_detector, normalize_image  # noqa: E402
from photo_trust.config import (AI_DETECTORS, AI_DIR, ANNOTATIONS, CACHE, DEVICE, EVAL_DIR, RESULTS,  # noqa: E402
                                SHOPEE_CSV, SHOPEE_IMAGES)
from photo_trust.data import resolve  # noqa: E402
from photo_trust.decision import ACTION_RU, Match, Thresholds, decide  # noqa: E402

st.set_page_config(page_title="PhotoTrust — проверка фото объявления", layout="wide")
COPY_MODEL = "SSCD-288"
REVIEW_FPR = 0.05   # queue threshold: 5% of honest same-product photos go to a moderator
ACTION_COLOR = {"PUBLISH": "green", "SUGGEST_AI_LABEL": "orange", "REVIEW": "orange", "HIDE_AND_REVIEW": "red"}


def fake_seller(group):
    """Shopee has no seller ids; derive a stable synthetic seller + price per gallery listing."""
    h = zlib.crc32(str(group).encode())
    return f"seller_{h % 9973:04d}", 5_000 + (h % 40) * 1_000


@st.cache_resource
def load_state():
    gal = pd.read_csv(EVAL_DIR / "gallery.csv")
    reuse = pd.read_csv(SHOPEE_CSV).groupby("image").posting_id.size()
    gal["n_listings"] = gal.image.map(reuse).fillna(1).astype(int)
    copy_m = json.loads((RESULTS / "copy_metrics.json").read_text())[COPY_MODEL]
    ops = pd.read_csv(RESULTS / "copy_operating_points.csv")
    review_thr = float(ops[(ops.method == COPY_MODEL) & (ops.fpr_target == REVIEW_FPR)].thr.iloc[0])
    ai_res = json.loads((RESULTS / "ai_metrics.json").read_text()) if (RESULTS / "ai_metrics.json").exists() else {}
    best_ai = max(ai_res, key=lambda k: ai_res[k]["roc_auc"]) if ai_res else list(AI_DETECTORS)[0]
    th = Thresholds(copy_review=review_thr, copy_hide=copy_m["threshold_fpr_0.1pct"],
                    ai_label=ai_res.get(best_ai, {}).get("threshold_at_fpr5_real", 0.9))
    return {
        "gal": gal, "th": th, "best_ai": best_ai,
        "sscd": embedders.load(COPY_MODEL), "phash": embedders.PHash(),
        "g_sscd": np.load(CACHE / f"{COPY_MODEL}_gallery.npy"), "g_phash": np.load(CACHE / "pHash_gallery.npy"),
        "ai": load_detector(AI_DETECTORS[best_ai]),
    }


def examples():
    """Curated by scripts/pick_demo_examples.py: successes and typical failures."""
    pairs = json.loads((RESULTS / "demo_examples.json").read_text(encoding="utf-8"))
    return {name: resolve(ref) for name, ref in pairs}


S = load_state()
st.title("PhotoTrust — проверка фото до публикации")
st.caption("Прототип Gate 3 · категория «Электроника» · база: "
           f"{len(S['gal'])} опубликованных фото (Shopee) · сигналы: копия ({COPY_MODEL}) + ИИ ({S['best_ai']})")

left, right = st.columns([1, 2])
with left:
    mode = st.radio("Фото объявления", ["Пример из тестового набора", "Загрузить своё"])
    if mode == "Загрузить своё":
        up = st.file_uploader("JPG/PNG", type=["jpg", "jpeg", "png", "webp"])
        img = Image.open(up).convert("RGB") if up else None
    else:
        ex = examples()
        try:  # ?ex=N preselects a scenario (deep links / screenshots)
            start = min(int(st.query_params.get("ex", 0)), len(ex) - 1)
        except ValueError:
            start = 0
        choice = st.selectbox("Сценарий", list(ex), index=start)
        img = Image.open(ex[choice]).convert("RGB")
    seller = st.text_input("ID продавца", "seller_new_0001")
    price = st.number_input("Цена, ₽", value=15_000, step=500)
    if img is not None:
        st.image(img, caption="Новое фото", use_container_width=True)

if img is None:
    st.stop()

t0 = time.perf_counter()
q = S["sscd"].embed([img])
sims = (q @ S["g_sscd"].T)[0]
top = np.argsort(-sims)[:3]
t_copy = (time.perf_counter() - t0) * 1000
ph = S["phash"].similarity(S["phash"].embed([img]), S["g_phash"])[0]
t1 = time.perf_counter()
ai_score = float(S["ai"].score([normalize_image(img)])[0])
t_ai = (time.perf_counter() - t1) * 1000

best = S["gal"].iloc[top[0]]
b_seller, b_price = fake_seller(best.group)
match = Match(listing_id=f"#{best.group}", seller_id=b_seller, similarity=float(sims[top[0]]),
              n_listings_with_photo=int(best.n_listings))
dec = decide(seller, match, ai_score, S["th"], price_ratio=price / b_price)

with right:
    st.subheader(f":{ACTION_COLOR[dec.action]}[{ACTION_RU[dec.action]}]")
    for r in dec.reasons:
        st.write("•", r)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Сходство SSCD (топ-1)", f"{sims[top[0]]:.2f}", help=f"порог очереди {S['th'].copy_review:.2f}")
    c2.metric("Сходство pHash (бейзлайн)", f"{ph[top[0]]:.2f}", help="1 − расстояние Хэмминга / 64")
    c3.metric("P(ИИ-генерация)", f"{ai_score:.2f}", help=f"порог метки {S['th'].ai_label:.2f}")
    c4.metric(f"Время проверки ({DEVICE.upper()})", f"{t_copy + t_ai:.0f} мс")
    st.markdown("**Ближайшие фото в базе площадки**")
    cols = st.columns(3)
    for c, i in zip(cols, top):
        row = S["gal"].iloc[i]
        s_id, s_price = fake_seller(row.group)
        c.image(str(SHOPEE_IMAGES / row.image), use_container_width=True)
        c.caption(f"SSCD {sims[i]:.2f} · pHash {ph[i]:.2f} · {s_id} · {s_price:,} ₽ · "
                  f"фото в {row.n_listings} объявл.".replace(",", " "))

with st.expander("Как читать результат"):
    st.markdown(
        "- **Копия**: SSCD сравнивает фото со всей базой; порог выбран так, чтобы ≤ 1% честных фото того же "
        "товара уходили в очередь.\n"
        "- **pHash** показан для сравнения: на кропе, отражении и скриншоте его сходство падает.\n"
        "- **Стоковое фото**: если фото встречается в ≥ 3 объявлениях, это скорее фото производителя / коробки — "
        "не нарушение.\n"
        "- **ИИ**: детектор только предлагает метку «Иллюстрация (ИИ)»; заблокировать продавца он не может.\n"
        "- Продавцы и цены в базе синтетические (в Shopee их нет).")
