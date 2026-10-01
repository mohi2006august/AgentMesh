"""Text embeddings.

`HashingEmbedder` is the zero-dependency default: words, word bigrams and character
trigrams hashed into a fixed-size signed vector, log-scaled and L2-normalised. It needs
no model download and is deterministic, so local dev and tests behave the same
everywhere. It captures lexical and morphological overlap rather than deep semantics;
plug in a hosted embedding model through the `Embedder` protocol for production quality.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Protocol

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]
_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> Vector:
        """Return a (len(texts), dim) float32 matrix with L2-normalised rows."""
        ...


def _features(text: str) -> Counter[str]:
    words = _WORD.findall(text.lower())
    features: Counter[str] = Counter(f"w:{word}" for word in words)
    features.update(f"b:{a}_{b}" for a, b in zip(words, words[1:], strict=False))
    for word in words:
        padded = f"#{word}#"
        features.update(f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2))
    return features


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vector(self, text: str) -> Vector:
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature, count in _features(text).items():
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            weight = 1.0 + np.log(count)
            # Words carry more meaning than sub-word trigrams.
            if feature.startswith("c:"):
                weight *= 0.5
            vector[index] += sign * weight
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed(self, texts: list[str]) -> Vector:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(text) for text in texts])


def cosine(query: Vector, matrix: Vector) -> npt.NDArray[np.float32]:
    """Cosine similarity of one normalised query against normalised rows."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    result: npt.NDArray[np.float32] = matrix @ query
    return result
