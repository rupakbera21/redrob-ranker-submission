#!/usr/bin/env python3
"""
precompute.py
=============
Offline pre-computation script for the Redrob Hackathon ranking system.

Run this ONCE before running rank.py. It:
  1. Streams 100K candidates from candidates.jsonl
  2. Extracts features for all candidates (saves feature matrix)
  3. Computes sentence-transformer embeddings (all-MiniLM-L6-v2) for all candidates
  4. Computes cosine similarity between each candidate and the JD
  5. Takes the top-N candidates by initial score → sends to Groq for reasoning
  6. Saves everything to ./cache/ for rank.py to load

Requirements:
  - pip install -r requirements.txt
  - GROQ_API_KEY environment variable (or --groq-key argument)
  - ~4GB RAM peak, ~15-20 minutes CPU time, ~500MB disk

Usage:
  python precompute.py --candidates ./path/to/candidates.jsonl
  python precompute.py --candidates ./path/to/candidates.jsonl --groq-key YOUR_KEY
  python precompute.py --candidates ./path/to/candidates.jsonl --skip-groq  # no LLM reasoning
"""

import argparse
import json
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
import pickle
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.features import extract_features, extract_candidate_text, features_to_subscores
from src.embedder import compute_and_cache_embeddings, embeddings_cache_exists
from src.scorer import score_all_candidates
from src.reasoner import (
    generate_reasoning_for_all,
    save_reasoning_cache,
    load_reasoning_cache,
)


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CANDIDATES = r"c:\Users\RUPAK\Desktop\INDIA RUNS\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── JSONL Loader ──────────────────────────────────────────────────────────────

def load_candidates_jsonl(path: str, max_count: int = None) -> list:
    """Stream candidates from JSONL or JSON array file. Returns list of dicts."""
    candidates = []
    print(f"\n[Loader] Reading candidates from: {path}")
    start = time.time()

    # Detect format: JSON array vs JSONL
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1).strip()

    if first_char == "[":
        # Plain JSON array (e.g. sample_candidates.json)
        with open(path, "r", encoding="utf-8") as f:
            all_cands = json.load(f)
        candidates = all_cands[:max_count] if max_count else all_cands
        print(f"  Loaded {len(candidates):,} candidates (JSON array) in {time.time()-start:.1f}s")
        return candidates

    # JSONL (main candidates.jsonl)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="  Loading", unit=" candidates")):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  Warning: skipping malformed line {i+1}")
            if max_count and len(candidates) >= max_count:
                break

    elapsed = time.time() - start
    print(f"  Loaded {len(candidates):,} candidates in {elapsed:.1f}s")
    return candidates


# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_all_features(candidates: list, config: dict) -> tuple:
    """
    Extract features for all candidates.

    Returns:
        (candidate_ids, candidate_texts, feature_list)
    """
    print(f"\n[Features] Extracting features for {len(candidates):,} candidates...")
    start = time.time()

    candidate_ids = []
    candidate_texts = []
    feature_list = []

    for cand in tqdm(candidates, desc="  Extracting", unit=" candidates"):
        cid = cand.get("candidate_id", "")
        candidate_ids.append(cid)
        candidate_texts.append(extract_candidate_text(cand))
        feature_list.append(extract_features(cand, config))

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")
    return candidate_ids, candidate_texts, feature_list


# ── Feature Cache ─────────────────────────────────────────────────────────────

def save_feature_cache(
    cache_dir: str,
    candidate_ids: list,
    feature_list: list,
) -> None:
    """Save extracted features to cache."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(cache_dir, "features.pkl")
    with open(path, "wb") as f:
        pickle.dump({"candidate_ids": candidate_ids, "features": feature_list}, f)
    print(f"  Saved feature cache: {path} ({len(feature_list):,} entries)")


def load_feature_cache(cache_dir: str) -> tuple:
    """Load feature cache. Returns (candidate_ids, feature_list)."""
    path = os.path.join(cache_dir, "features.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["candidate_ids"], data["features"]


def feature_cache_exists(cache_dir: str) -> bool:
    return os.path.exists(os.path.join(cache_dir, "features.pkl"))


# ── Initial Scoring (for Groq candidate selection) ────────────────────────────

def compute_initial_scores(
    feature_list: list,
    similarities: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Compute composite scores without LLM. Used to select top-N for Groq."""
    weights = config.get("weights", {})
    n = len(feature_list)

    skill_scores = np.zeros(n, dtype=np.float32)
    career_scores = np.zeros(n, dtype=np.float32)
    behavioral_scores = np.zeros(n, dtype=np.float32)

    for i, feats in enumerate(feature_list):
        sub = features_to_subscores(feats)
        skill_scores[i] = sub["skill_score"]
        career_scores[i] = sub["career_score"]
        behavioral_scores[i] = sub["behavioral_score"]

    composite = (
        weights.get("semantic_fit", 0.35) * similarities +
        weights.get("skill_match", 0.30) * skill_scores +
        weights.get("career_quality", 0.20) * career_scores +
        weights.get("behavioral", 0.15) * behavioral_scores
    )
    return np.clip(composite, 0.0, 1.0)


# ── Main Precompute Pipeline ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Hackathon — Pre-computation Script")
    parser.add_argument(
        "--candidates",
        default=DEFAULT_CANDIDATES,
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--groq-key",
        default=os.environ.get("GROQ_API_KEY", ""),
        help="Groq API key (or set GROQ_API_KEY env var)",
    )
    parser.add_argument(
        "--skip-groq",
        action="store_true",
        help="Skip Groq reasoning generation (use rule-based fallback)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding computation if cache exists",
    )
    parser.add_argument(
        "--cache-dir",
        default="./cache",
        help="Directory to save computed cache files",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Limit candidate count for testing (default: all)",
    )
    parser.add_argument(
        "--groq-top-n",
        type=int,
        default=None,
        help="Override: number of top candidates to send to Groq",
    )
    args = parser.parse_args()

    total_start = time.time()
    print("=" * 60)
    print(" Redrob Hackathon — Pre-computation Pipeline")
    print("=" * 60)

    # ── Load config ──────────────────────────────────────────────
    config = load_config()
    cache_dir = args.cache_dir
    precompute_cfg = config.get("precompute", {})
    embed_model = precompute_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    embed_batch = precompute_cfg.get("embedding_batch_size", 512)
    groq_top_n = args.groq_top_n or precompute_cfg.get("top_n_for_groq", 300)
    groq_model = precompute_cfg.get("groq_model", "llama-3.3-70b-versatile")
    groq_batch = precompute_cfg.get("groq_batch_size", 5)
    groq_retries = precompute_cfg.get("groq_max_retries", 3)
    jd_text = config.get("jd_embedding_text", "Senior AI Engineer ML NLP retrieval ranking Python")

    print(f"\nConfig loaded:")
    print(f"  Candidates: {args.candidates}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Embedding model: {embed_model}")
    print(f"  Groq top-N: {groq_top_n}")
    print(f"  Skip Groq: {args.skip_groq}")

    # ── Step 1: Load candidates ──────────────────────────────────
    candidates = load_candidates_jsonl(args.candidates, max_count=args.max_candidates)
    n_total = len(candidates)

    # ── Step 2: Extract features ─────────────────────────────────
    if feature_cache_exists(cache_dir) and args.skip_embeddings:
        print(f"\n[Features] Loading from cache...")
        candidate_ids, feature_list = load_feature_cache(cache_dir)
        candidate_texts = [extract_candidate_text(c) for c in tqdm(candidates, desc="  Rebuilding texts")]
    else:
        candidate_ids, candidate_texts, feature_list = extract_all_features(candidates, config)
        save_feature_cache(cache_dir, candidate_ids, feature_list)

    # ── Step 3: Compute embeddings ───────────────────────────────
    if embeddings_cache_exists(cache_dir) and args.skip_embeddings:
        print(f"\n[Embedder] Loading from cache (--skip-embeddings)...")
        from src.embedder import load_embeddings
        _, _, jd_embedding, similarities = load_embeddings(cache_dir)
    else:
        embeddings, jd_embedding, similarities = compute_and_cache_embeddings(
            candidate_ids=candidate_ids,
            candidate_texts=candidate_texts,
            jd_text=jd_text,
            cache_dir=cache_dir,
            model_name=embed_model,
            batch_size=embed_batch,
        )

    # ── Step 4: Initial scoring to select top-N for Groq ─────────
    print(f"\n[Scorer] Computing initial composite scores...")
    initial_scores = compute_initial_scores(feature_list, similarities, config)
    top_n_indices = np.argsort(initial_scores)[::-1][:groq_top_n]
    top_n_scores = initial_scores[top_n_indices]

    print(f"  Initial score stats:")
    print(f"    Min: {initial_scores.min():.4f} | Max: {initial_scores.max():.4f} | Mean: {initial_scores.mean():.4f}")
    print(f"  Top-{groq_top_n} score range: [{top_n_scores.min():.4f}, {top_n_scores.max():.4f}]")

    # Show top-10 preview
    print(f"\n  Top-10 initial ranking:")
    for rank_pos, idx in enumerate(top_n_indices[:10]):
        cid = candidate_ids[idx]
        cand = candidates[idx]
        title = cand.get("profile", {}).get("current_title", "?")
        print(f"    #{rank_pos+1} {cid}: {title} | score={initial_scores[idx]:.4f}")

    # ── Step 5: Groq reasoning for top-N ─────────────────────────
    reasoning_cache = load_reasoning_cache(cache_dir)

    if not args.skip_groq and args.groq_key:
        # Find candidates not yet in cache
        top_n_candidates = [candidates[idx] for idx in top_n_indices]
        missing = [c for c in top_n_candidates if c.get("candidate_id") not in reasoning_cache]

        if missing:
            print(f"\n[Groq] Generating reasoning for {len(missing)} candidates "
                  f"({len(reasoning_cache)} already cached)...")
            new_reasoning = generate_reasoning_for_all(
                top_candidates=missing,
                api_key=args.groq_key,
                model=groq_model,
                batch_size=groq_batch,
                max_retries=groq_retries,
            )
            reasoning_cache.update(new_reasoning)
            save_reasoning_cache(reasoning_cache, cache_dir)
        else:
            print(f"\n[Groq] All {len(reasoning_cache)} candidates already have reasoning (cached).")

    elif args.skip_groq:
        print(f"\n[Groq] Skipped (--skip-groq flag). Using fallback reasoning in rank.py.")
    else:
        print(f"\n[Groq] No API key provided. Using fallback reasoning in rank.py.")
        print(f"       Set GROQ_API_KEY or pass --groq-key to enable LLM reasoning.")

    # ── Save final id-to-index mapping ────────────────────────────
    id_to_index_path = os.path.join(cache_dir, "id_to_index.json")
    id_to_index = {cid: i for i, cid in enumerate(candidate_ids)}
    with open(id_to_index_path, "w") as f:
        json.dump(id_to_index, f)
    print(f"\n[Cache] Saved id→index mapping: {id_to_index_path}")

    # ── Summary ────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f" Pre-computation complete in {total_elapsed/60:.1f} minutes")
    print(f"  Candidates processed: {n_total:,}")
    print(f"  Reasoning cache entries: {len(reasoning_cache)}")
    print(f"  Cache dir: {cache_dir}/")
    print(f"\n Next step: python rank.py --candidates {args.candidates} --out ./output/rupak_bera.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
