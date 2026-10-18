"""Face models: reusable identities saved as small .safetensors files.

A Visage face model stores one averaged ArcFace embedding plus JSON metadata.
ReActor face models (``models/reactor/faces/*.safetensors``) can be loaded too,
since they also carry an ArcFace ``embedding`` tensor.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

FORMAT = "visage/1"


@dataclass
class Identity:
    embedding: np.ndarray  # raw or normalised ArcFace vector (512,)
    name: str = ""
    sources: int = 1
    meta: dict = field(default_factory=dict)

    @property
    def normed(self) -> np.ndarray:
        return self.embedding / (np.linalg.norm(self.embedding) + 1e-12)


def average(identities: list[Identity], weights: list[float] | None = None, name: str = "") -> Identity:
    if not identities:
        raise ValueError("no identities to average")
    w = np.ones(len(identities), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    stack = np.stack([i.normed for i in identities]) * w[:, None]
    mean = stack.sum(axis=0) / max(float(w.sum()), 1e-6)
    return Identity(mean / (np.linalg.norm(mean) + 1e-12), name=name, sources=sum(i.sources for i in identities))


def safe_name(name: str) -> str:
    name = re.sub(r"[^\w\- .]", "_", name.strip()).strip(" .")
    return name or time.strftime("face_%Y%m%d_%H%M%S")


def save(identity: Identity, folder: str, name: str | None = None, overwrite: bool = False) -> str:
    from safetensors.numpy import save_file

    os.makedirs(folder, exist_ok=True)
    name = safe_name(name or identity.name)
    path = os.path.join(folder, name + ".safetensors")
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Face model '{name}' already exists")
    meta = {"format": FORMAT, "name": name, "sources": str(identity.sources), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    meta.update({k: str(v) for k, v in identity.meta.items()})
    save_file({"embedding": identity.normed.astype(np.float32)}, path, metadata=meta)
    return path


def load(path: str) -> Identity:
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as f:
        meta = f.metadata() or {}
        keys = list(f.keys())
        if "embedding" not in keys:
            raise ValueError(f"{os.path.basename(path)} has no 'embedding' tensor (keys: {keys[:8]})")
        emb = np.asarray(f.get_tensor("embedding"), dtype=np.float32).reshape(-1)
    if emb.shape[0] != 512:
        raise ValueError(f"{os.path.basename(path)}: unexpected embedding size {emb.shape[0]}")
    name = meta.get("name") or os.path.splitext(os.path.basename(path))[0]
    sources = int(meta.get("sources", "1") or 1) if str(meta.get("sources", "1")).isdigit() else 1
    return Identity(emb, name=name, sources=sources, meta={"path": path, "format": meta.get("format", "reactor")})


def list_models(folders: list[str]) -> dict[str, str]:
    """display name -> path; later folders never shadow earlier ones."""
    found: dict[str, str] = {}
    for folder in folders:
        if not folder or not os.path.isdir(folder):
            continue
        tag = "" if folder == folders[0] else " (ReActor)"
        for f in sorted(os.listdir(folder), key=str.lower):
            if f.lower().endswith(".safetensors"):
                label = os.path.splitext(f)[0] + tag
                found.setdefault(label, os.path.join(folder, f))
    return found


def delete(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)


def describe(identity: Identity) -> str:
    return json.dumps({"name": identity.name, "sources": identity.sources, **identity.meta}, ensure_ascii=False)
