"""Wrapper around off-the-shelf Hugging Face AI-image classifiers.

Every model has its own label names; we map them to a single P(AI-generated) score.
"""
import io
import urllib.request

import numpy as np
import torch
from PIL import Image

from .config import DEVICE, MODELS

_AI_WORDS = ("ai", "artificial", "fake", "generated", "sdxl", "synthetic")
_REAL_WORDS = ("real", "human", "hum", "natural", "authentic")


def normalize_image(img, max_side=512, quality=90):
    """Same format for every image (resize + JPEG q90) so detectors cannot cheat on
    PNG-vs-JPEG or resolution differences between sources."""
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


class AIDetector:
    def __init__(self, model_id):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        subfolder = _checkpoint_subfolder(model_id)
        kw = {"cache_dir": MODELS}
        if subfolder:
            kw["subfolder"] = subfolder
        self.model = AutoModelForImageClassification.from_pretrained(model_id, **kw).eval().to(DEVICE)
        try:
            self.proc = AutoImageProcessor.from_pretrained(model_id, **kw)
        except OSError:  # some repos keep the processor only at the root
            self.proc = AutoImageProcessor.from_pretrained(model_id, cache_dir=MODELS)
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        ai = [i for i, l in labels.items() if any(w in l for w in _AI_WORDS)]
        real = [i for i, l in labels.items() if any(w in l for w in _REAL_WORDS)]
        if ai:
            self.ai_idx = ai[0]
        elif real and len(labels) == 2:
            self.ai_idx = 1 - real[0]
        else:
            raise ValueError(f"Cannot map labels of {model_id}: {labels}")
        self.labels = labels

    @torch.no_grad()
    def score(self, images, batch=16):
        out = []
        for i in range(0, len(images), batch):
            x = self.proc(images=[im.convert("RGB") for im in images[i:i + batch]], return_tensors="pt").to(DEVICE)
            p = self.model(**x).logits.softmax(-1)[:, self.ai_idx]
            out.append(p.cpu().numpy())
        return np.concatenate(out)


class UniversalFakeDetect:
    """Ojha et al., CVPR 2023: frozen CLIP ViT-L/14 features + linear probe trained on ProGAN.
    Known to generalise better to unseen generators than CNN detectors.
    Weights come from the OpenAI CDN and GitHub (no Hugging Face needed)."""

    def __init__(self):
        import open_clip
        from torchvision import transforms

        from .config import CLIP_L14_URL, UFD_FC_URL

        clip_path, fc_path = MODELS / "ViT-L-14.pt", MODELS / "ufd_fc_weights.pth"
        for url, p in [(CLIP_L14_URL, clip_path), (UFD_FC_URL, fc_path)]:
            if not p.exists():
                urllib.request.urlretrieve(url, p)
        with open(clip_path, "rb") as f:  # file object: torch.jit.load fails on non-ASCII paths
            sd = torch.jit.load(f, map_location="cpu").state_dict()
        self.clip = open_clip.model.build_model_from_openai_state_dict(sd).float().eval().to(DEVICE)
        self.fc = torch.nn.Linear(768, 1)
        self.fc.load_state_dict(torch.load(fc_path, map_location="cpu"))
        self.fc.to(DEVICE)
        self.tf = transforms.Compose([
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])])

    @torch.no_grad()
    def score(self, images, batch=16):
        out = []
        for i in range(0, len(images), batch):
            x = torch.stack([self.tf(im.convert("RGB")) for im in images[i:i + batch]]).to(DEVICE)
            out.append(torch.sigmoid(self.fc(self.clip.encode_image(x).float()))[:, 0].cpu().numpy())
        return np.concatenate(out)


def load_detector(model_id):
    return UniversalFakeDetect() if model_id == "ufd" else AIDetector(model_id)


def _checkpoint_subfolder(model_id):
    """Some repos store weights only under checkpoint-*/ ; pick the last one."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(model_id)
    if "config.json" in files:
        return None
    ckpts = sorted({f.split("/")[0] for f in files if f.startswith("checkpoint-") and f.endswith("config.json")},
                   key=lambda s: int(s.split("-")[1]))
    return ckpts[-1] if ckpts else None
