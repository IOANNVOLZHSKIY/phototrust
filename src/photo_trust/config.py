"""Paths and constants shared by scripts and the demo app."""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SHOPEE_DIR = DATA / "shopee"
SHOPEE_IMAGES = SHOPEE_DIR / "train_images"
SHOPEE_CSV = SHOPEE_DIR / "train.csv"
AI_DIR = DATA / "ai_generated"
EVAL_DIR = DATA / "eval"          # generated eval sets (attacked copies etc.)
CACHE = ROOT / "cache"            # embeddings, indexes
MODELS = ROOT / "models"          # downloaded weights
RESULTS = ROOT / "results"        # metrics + plots (committed to git)
ANNOTATIONS = ROOT / "annotations"  # manual labels (committed to git)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Shopee titles are mostly Indonesian/English. Keywords select the "Electronics" category
# (phones, accessories, audio, computer peripherals, small gadgets).
ELECTRONICS_KEYWORDS = [
    "handphone", " hp ", "iphone", "samsung", "xiaomi", "redmi", "oppo", "vivo", "realme",
    "infinix", "case ", "casing", "softcase", "hardcase", "tempered glass", "anti gores",
    "charger", "kabel data", "cable", "usb", "type c", "lightning", "adaptor", "powerbank",
    "power bank", "earphone", "headset", "headphone", "tws", "earbuds", "bluetooth", "speaker",
    "mouse", "keyboard", "laptop", "notebook", "tablet", "smartwatch", "smart watch",
    "flashdisk", "flash disk", "memory card", "micro sd", "microsd", "hdmi", "webcam",
    "router", "wifi", "lampu led", "holder hp", "tripod", "ring light", "baterai", "battery",
]

SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
CLIP_L14_URL = ("https://openaipublic.azureedge.net/clip/models/"
                "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt")
UFD_FC_URL = "https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/main/pretrained_weights/fc_weights.pth"

# Off-the-shelf AI-image detectors compared in the prototype.
# "ufd" = UniversalFakeDetect (CLIP-based); the rest are Hugging Face model ids.
AI_DETECTORS = {
    "UniversalFakeDetect (CLIP, CVPR 2023)": "ufd",
    "sdxl-detector (Organika, 2024)": "Organika/sdxl-detector",
    "ai-vs-human SigLIP (Ateeqq, 2025)": "Ateeqq/ai-vs-human-image-detector",
    "ai-image-detector (haywoodsloan, 2026)": "haywoodsloan/ai-image-detector-deploy",
}

for _p in (DATA, EVAL_DIR, CACHE, MODELS, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)
