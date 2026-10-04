"""Synthetic 'fraudster' edits applied to a stolen photo before re-posting it.

Each attack takes and returns a PIL RGB image. `rng` makes them reproducible.
HARD_ATTACKS are the ones a perceptual hash is expected to miss.
"""
import io
import random

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

_FONT = "C:/Windows/Fonts/arialbd.ttf"


def _font(size):
    try:
        return ImageFont.truetype(_FONT, size)
    except OSError:
        return ImageFont.load_default()


def crop(img, rng, keep=0.8):
    w, h = img.size
    cw, ch = int(w * keep), int(h * keep)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return img.crop((x, y, x + cw, y + ch))


def crop_80(img, rng):
    return crop(img, rng, 0.8)


def crop_55(img, rng):
    return crop(img, rng, 0.55)


def hflip(img, rng):
    return ImageOps.mirror(img)


def rotate(img, rng):
    angle = rng.choice([-1, 1]) * rng.uniform(8, 15)
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255))


def color_filter(img, rng):
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.3))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.3))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.4, 1.6))
    return img


def jpeg_low(img, rng):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(12, 25))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def downscale(img, rng):
    w, h = img.size
    s = rng.uniform(0.25, 0.4)
    return img.resize((max(32, int(w * s)), max(32, int(h * s))), Image.BILINEAR)


def watermark(img, rng):
    img = img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    text = rng.choice(["ORIGINAL", "ЛУЧШАЯ ЦЕНА", "SALE -50%", "@best_shop", "В НАЛИЧИИ"])
    f = _font(max(14, w // 9))
    for _ in range(2):
        x, y = rng.randint(0, max(1, w // 3)), rng.randint(0, max(1, h - h // 5))
        d.text((x, y), text, font=f, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return img


def border(img, rng):
    w, h = img.size
    pad = int(min(w, h) * rng.uniform(0.06, 0.12))
    color = rng.choice([(255, 255, 255), (230, 30, 30), (20, 20, 20), (255, 200, 0)])
    return ImageOps.expand(img, border=pad, fill=color)


def screenshot(img, rng):
    """Photo embedded into a fake mobile-app listing screen (status bar, title, price)."""
    w, h = img.size
    W = int(w * 1.0)
    top, bottom = int(W * 0.18), int(W * 0.45)
    canvas = Image.new("RGB", (W, top + h + bottom), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, W, int(top * 0.35)], fill=(30, 30, 30))
    d.text((10, int(top * 0.45)), "< Объявление", font=_font(max(12, W // 20)), fill=(0, 0, 0))
    canvas.paste(img, (0, top))
    d.text((10, top + h + 10), f"{rng.randint(5, 90)} 990 ₽", font=_font(max(14, W // 12)), fill=(0, 0, 0))
    d.rectangle([10, top + h + int(bottom * 0.55), W - 10, top + h + int(bottom * 0.85)], fill=(0, 170, 80))
    return canvas


def combo(img, rng):
    """Realistic fraudster pipeline: crop + mirror + color + watermark + recompress."""
    img = crop(img, rng, 0.75)
    img = hflip(img, rng)
    img = color_filter(img, rng)
    img = watermark(img, rng)
    return jpeg_low(img, rng)


ATTACKS = {
    "crop_80": crop_80,
    "crop_55": crop_55,
    "hflip": hflip,
    "rotate": rotate,
    "color_filter": color_filter,
    "jpeg_low": jpeg_low,
    "downscale": downscale,
    "watermark": watermark,
    "border": border,
    "screenshot": screenshot,
    "combo": combo,
}

ATTACK_LABELS_RU = {
    "crop_80": "Кроп 80%",
    "crop_55": "Кроп 55%",
    "hflip": "Зеркальное отражение",
    "rotate": "Поворот 8–15°",
    "color_filter": "Цветовой фильтр",
    "jpeg_low": "Сильное JPEG-сжатие",
    "downscale": "Уменьшение размера",
    "watermark": "Текст / водяной знак",
    "border": "Рамка",
    "screenshot": "Скриншот объявления",
    "combo": "Комбо (кроп+зеркало+фильтр+текст)",
}

HARD_ATTACKS = ["crop_55", "hflip", "rotate", "screenshot", "combo"]


def apply(name, img, seed):
    return ATTACKS[name](img.convert("RGB"), random.Random(seed))
