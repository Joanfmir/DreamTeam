"""Embeddings d'imatge (CLIP) i de text (sentence-transformers) amb caché a disc.

Requereix les dependències opcionals: pip install -r requirements-dl.txt

    from dreamteam import embeddings
    E = embeddings.image_embeddings(paths)                 # (n, 512), normalitzats
    T = embeddings.text_embeddings(textos)                 # (n, 384)
    s = embeddings.zero_shot(E, ["slim dress", "oversize dress"])   # puntuació imatge ↔ text

Calcular embeddings és el pas lent: el resultat es desa a `cache/` i la segona vegada és instantani.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CACHE = Path("cache/embeddings")
CLIP_MODEL = "openai/clip-vit-base-patch32"
TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # multilingüe (ca/es/en)


def _cache_path(kind: str, model: str, items: list[str]) -> Path:
    h = hashlib.md5(("\n".join(items) + model).encode()).hexdigest()[:16]
    return CACHE / f"{kind}_{model.split('/')[-1]}_{len(items)}_{h}.npy"


def _features(out):
    """transformers 4 retorna un tensor; transformers 5, un objecte amb la projecció a `pooler_output`."""
    return out if hasattr(out, "norm") else out.pooler_output


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # GPU dels Mac amb Apple Silicon
    return "cpu"


def image_embeddings(paths, model_name: str = CLIP_MODEL, batch_size: int = 32,
                     use_cache: bool = True) -> np.ndarray:
    """Embedding CLIP normalitzat de cada imatge. Les imatges que no es poden obrir donen un vector de zeros."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    paths = [str(p) for p in paths]
    cache = _cache_path("img", model_name, paths)
    if use_cache and cache.exists():
        return np.load(cache)

    device = _device()
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    dim = model.config.projection_dim
    out = np.zeros((len(paths), dim), dtype=np.float32)

    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        imgs, ok = [], []
        for i, p in enumerate(batch):
            try:
                imgs.append(Image.open(p).convert("RGB"))
                ok.append(start + i)
            except Exception:  # imatge trencada o inexistent → es queda a zero
                pass
        if not imgs:
            continue
        with torch.no_grad():
            inputs = proc(images=imgs, return_tensors="pt").to(device)
            feats = _features(model.get_image_features(**inputs))
            feats = torch.nn.functional.normalize(feats, dim=-1)
        out[ok] = feats.cpu().numpy()

    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out


def clip_text_embeddings(texts, model_name: str = CLIP_MODEL) -> np.ndarray:
    """Embeddings CLIP de text (mateix espai que les imatges) per fer zero-shot."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = _device()
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    with torch.no_grad():
        inputs = proc(text=list(texts), return_tensors="pt", padding=True).to(device)
        feats = torch.nn.functional.normalize(_features(model.get_text_features(**inputs)), dim=-1)
    return feats.cpu().numpy()


def zero_shot(image_emb: np.ndarray, labels: list[str], template: str = "a photo of a {}") -> np.ndarray:
    """Probabilitat de cada etiqueta per a cada imatge sense entrenar res (CLIP zero-shot).

    Útil per a atributs visuals (Mango 2024): labels = ["slim", "oversize", "regular"].
    """
    T = clip_text_embeddings([template.format(lab) for lab in labels])
    logits = 100 * image_emb @ T.T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def text_embeddings(texts, model_name: str = TEXT_MODEL, batch_size: int = 64,
                    use_cache: bool = True) -> np.ndarray:
    """Embeddings de frases (ressenyes, objectius de participants, descripcions de producte)."""
    from sentence_transformers import SentenceTransformer

    texts = [str(t) for t in texts]
    cache = _cache_path("txt", model_name, texts)
    if use_cache and cache.exists():
        return np.load(cache)
    model = SentenceTransformer(model_name, device=_device())
    out = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=len(texts) > 1000).astype(np.float32)
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out


def prefetch_models():
    """Baixa els models a la caché local. Executeu-ho ABANS del dia D (la wifi pot fallar)."""
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor

    CLIPModel.from_pretrained(CLIP_MODEL)
    CLIPProcessor.from_pretrained(CLIP_MODEL)
    SentenceTransformer(TEXT_MODEL)
    print(f"✓ Models a la caché: {CLIP_MODEL}, {TEXT_MODEL}")


if __name__ == "__main__":
    prefetch_models()
