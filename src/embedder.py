"""
src/embedder.py
===============
Sentence-Transformer embedding for semantic candidate-JD similarity.

Uses all-MiniLM-L6-v2 (~80MB, CPU-friendly, 384-dim).
Embeddings are pre-computed offline and cached to disk.
rank.py loads from cache — no re-computation at submission time.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ── Embedding model ────────────────────────────────────────────────────────────

_MODEL_CACHE: Optional[object] = None


def _get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy-load the sentence-transformer model (singleton)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading embedding model: {model_name}")
        _MODEL_CACHE = SentenceTransformer(model_name)
        print("  Model loaded.")
    return _MODEL_CACHE


# ── Embedding functions ────────────────────────────────────────────────────────

def embed_texts(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts into dense vectors.

    Args:
        texts: List of strings to embed.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for CPU-efficient inference.
        show_progress: Show tqdm progress bar.

    Returns:
        np.ndarray of shape (N, D) — float32, L2-normalized.
    """
    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # cosine sim = dot product on normalized vectors
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def cosine_similarity_to_jd(
    candidate_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between N candidate embeddings and the JD embedding.
    Since both are L2-normalized, cosine sim = dot product.

    Returns:
        np.ndarray of shape (N,) — float32 in [-1, 1], typically [0.2, 0.8].
    """
    # jd_embedding shape: (D,) → reshape to (D, 1) for matmul
    sims = candidate_embeddings @ jd_embedding.reshape(-1, 1)
    return sims.flatten().astype(np.float32)


def normalize_similarities(sims: np.ndarray) -> np.ndarray:
    """Min-max normalize similarities to [0, 1] for score composition."""
    lo, hi = sims.min(), sims.max()
    if hi - lo < 1e-8:
        return np.ones_like(sims) * 0.5
    return ((sims - lo) / (hi - lo)).astype(np.float32)


# ── Cache I/O ──────────────────────────────────────────────────────────────────

def save_embeddings(
    cache_dir: str,
    candidate_ids: List[str],
    embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    similarities: np.ndarray,
) -> None:
    """Save embeddings and similarities to a cache directory."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    np.save(str(cache_path / "jd_embedding.npy"), jd_embedding)
    np.savez_compressed(
        str(cache_path / "embeddings.npz"),
        embeddings=embeddings,
        similarities=similarities,
    )

    # Save candidate IDs mapping
    ids_path = cache_path / "candidate_ids.txt"
    ids_path.write_text("\n".join(candidate_ids), encoding="utf-8")

    print(f"  Saved embeddings cache to {cache_path}")
    print(f"  Embeddings shape: {embeddings.shape} | Similarities shape: {similarities.shape}")


def load_embeddings(
    cache_dir: str,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cached embeddings.

    Returns:
        (candidate_ids, embeddings, jd_embedding, similarities)
    """
    cache_path = Path(cache_dir)

    jd_embedding = np.load(str(cache_path / "jd_embedding.npy"))
    data = np.load(str(cache_path / "embeddings.npz"))
    embeddings = data["embeddings"]
    similarities = data["similarities"]

    ids_path = cache_path / "candidate_ids.txt"
    candidate_ids = ids_path.read_text(encoding="utf-8").strip().split("\n")

    print(f"  Loaded cached embeddings: {embeddings.shape}")
    return candidate_ids, embeddings, jd_embedding, similarities


def embeddings_cache_exists(cache_dir: str) -> bool:
    """Check if embedding cache files exist."""
    cache_path = Path(cache_dir)
    return (
        (cache_path / "jd_embedding.npy").exists() and
        (cache_path / "embeddings.npz").exists() and
        (cache_path / "candidate_ids.txt").exists()
    )


# ── High-level interface ───────────────────────────────────────────────────────

def compute_and_cache_embeddings(
    candidate_ids: List[str],
    candidate_texts: List[str],
    jd_text: str,
    cache_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute embeddings for all candidates and JD, then save to cache.

    Returns:
        (embeddings, jd_embedding, similarities)
    """
    start = time.time()
    print(f"\n[Embedder] Embedding {len(candidate_texts)} candidates...")

    # Embed JD
    print("  Embedding JD...")
    jd_embedding = embed_texts([jd_text], model_name=model_name, batch_size=1, show_progress=False)[0]

    # Embed all candidates in batches
    embeddings = embed_texts(
        candidate_texts,
        model_name=model_name,
        batch_size=batch_size,
        show_progress=True,
    )

    # Compute cosine similarities
    print("  Computing cosine similarities...")
    similarities_raw = cosine_similarity_to_jd(embeddings, jd_embedding)
    similarities = normalize_similarities(similarities_raw)

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s | Sim range: [{similarities.min():.3f}, {similarities.max():.3f}]")

    # Save to cache
    save_embeddings(cache_dir, candidate_ids, embeddings, jd_embedding, similarities)

    return embeddings, jd_embedding, similarities
