"""Image -> vector models: pHash baseline and two neural descriptors (SSCD, DINOv2).

All embedders expose `name`, `embed(list[PIL.Image]) -> np.ndarray` and a `similarity`
convention: higher = more similar. Neural vectors are L2-normalised, so inner product
equals cosine similarity.
"""
import urllib.request

import imagehash
import numpy as np
import torch
from torchvision import transforms

from .config import DEVICE, DINOV2_URL, MODELS, SSCD_URL

torch.set_num_threads(max(1, torch.get_num_threads()))
_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class PHash:
    """Perceptual hash (64 bit). Similarity = 1 - hamming / 64."""
    name = "pHash"
    dim = 64

    def embed(self, images):
        bits = [imagehash.phash(im.convert("RGB")).hash.flatten() for im in images]
        return np.asarray(bits, dtype=np.uint8)

    @staticmethod
    def similarity(q_bits, g_bits):
        # q: (n,64), g: (m,64) -> (n,m) similarity in [0,1]
        q = q_bits.astype(np.int16)
        g = g_bits.astype(np.int16)
        ham = (q[:, None, :] != g[None, :, :]).sum(-1)
        return 1.0 - ham / 64.0


class _Torch:
    batch = 16

    @torch.no_grad()
    def embed(self, images):
        out = []
        for i in range(0, len(images), self.batch):
            x = torch.stack([self.tf(im.convert("RGB")) for im in images[i:i + self.batch]]).to(DEVICE)
            v = self.forward(x)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out).astype("float32")

    @staticmethod
    def similarity(q, g):
        return q @ g.T


class SSCD(_Torch):
    """Meta AI SSCD copy-detection descriptor (ResNet-50, 512-d, CVPR 2022)."""
    name = "SSCD"
    dim = 512

    def __init__(self):
        path = MODELS / "sscd_disc_mixup.torchscript.pt"
        if not path.exists():
            urllib.request.urlretrieve(SSCD_URL, path)
        with open(path, "rb") as f:  # file object: torch.jit.load fails on non-ASCII paths
            self.model = torch.jit.load(f, map_location=DEVICE).eval()
        self.tf = transforms.Compose([transforms.Resize([320, 320]), transforms.ToTensor(), _NORM])

    def forward(self, x):
        return self.model(x)


class SSCDAspect(SSCD):
    """SSCD with the authors' recommended preprocessing: short side -> 288 px, aspect ratio kept
    (no squashing), so crops, frames and screenshots keep their geometry. Variable image sizes ->
    batch of 1."""
    name = "SSCD-288"
    batch = 1

    def __init__(self):
        super().__init__()
        self.tf = transforms.Compose([transforms.Resize(288, max_size=512), transforms.ToTensor(), _NORM])


class DINOv2(_Torch):
    """Self-supervised ViT-S/14 (CLS token, 384-d)."""
    name = "DINOv2"
    dim = 384

    def __init__(self):
        # Original Meta weights from dl.fbaipublicfiles.com (Hugging Face is slow from RU),
        # loaded into timm's ViT-S/14 with position embeddings resized to 224 px.
        import timm
        from timm.models.vision_transformer import checkpoint_filter_fn

        path = MODELS / "dinov2_vits14_pretrain.pth"
        if not path.exists():
            urllib.request.urlretrieve(DINOV2_URL, path)
        self.model = timm.create_model("vit_small_patch14_dinov2", pretrained=False, img_size=224,
                                       num_classes=0).eval()
        sd = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint_filter_fn(sd, self.model))
        self.model.to(DEVICE)
        self.tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                      transforms.ToTensor(), _NORM])

    def forward(self, x):
        return self.model(x)  # pooled CLS token


def load(name):
    return {"pHash": PHash, "SSCD": SSCD, "SSCD-288": SSCDAspect, "DINOv2": DINOv2}[name]()
