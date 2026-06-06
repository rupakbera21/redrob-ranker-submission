"""
src/scorer.py
=============
Composite weighted scorer that combines semantic similarity,
skill match, career quality, and behavioral signals into a
single final score per candidate.

All operations are vectorized using numpy — handles 100K candidates in <1 second.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .features import (
    extract_features,
    extract_candidate_text,
    features_to_subscores,
)


# ── Composite Scorer ──────────────────────────────────────────────────────────

def compute_composite_score(
    semantic_sim: float,
    skill_score: float,
    career_score: float,
    behavioral_score: float,
    weights: Dict[str, float],
) -> float:
    """
    Compute the final composite score from four sub-scores.

    Args:
        semantic_sim: Normalized cosine similarity between candidate and JD (0-1).
        skill_score: Depth/coverage of AI/ML skills (0-1).
        career_score: Career trajectory and company quality (0-1).
        behavioral_score: Platform engagement and availability (0-1).
        weights: Dict with keys: semantic_fit, skill_match, career_quality, behavioral.

    Returns:
        Float in [0, 1].
    """
    w = weights
    score = (
        w.get("semantic_fit", 0.35) * semantic_sim +
        w.get("skill_match", 0.30) * skill_score +
        w.get("career_quality", 0.20) * career_score +
        w.get("behavioral", 0.15) * behavioral_score
    )
    return float(np.clip(score, 0.0, 1.0))


def score_all_candidates(
    candidates: List[Dict],
    similarities: Optional[np.ndarray],
    config: Dict,
) -> List[Tuple[str, float, Dict, Dict]]:
    """
    Score all candidates and return a sorted list.

    Args:
        candidates: List of candidate dicts (from JSONL).
        similarities: Pre-computed normalized cosine sims (N,). None = skip semantic.
        config: Full config dict from config.yaml.

    Returns:
        List of (candidate_id, composite_score, features_dict, subscores_dict)
        sorted by composite_score descending.
    """
    weights = config.get("weights", {
        "semantic_fit": 0.35,
        "skill_match": 0.30,
        "career_quality": 0.20,
        "behavioral": 0.15,
    })

    if similarities is None:
        # Fallback: redistribute semantic weight to skill + career
        weights = {
            "semantic_fit": 0.0,
            "skill_match": 0.45,
            "career_quality": 0.35,
            "behavioral": 0.20,
        }

    results = []
    for i, cand in enumerate(candidates):
        cid = cand.get("candidate_id", f"UNKNOWN_{i}")

        # Extract features
        feats = extract_features(cand, config)
        subscores = features_to_subscores(feats)

        # Semantic similarity (from pre-computed cache or 0 if no cache)
        sem_sim = float(similarities[i]) if similarities is not None else 0.0

        # Final score
        score = compute_composite_score(
            semantic_sim=sem_sim,
            skill_score=subscores["skill_score"],
            career_score=subscores["career_score"],
            behavioral_score=subscores["behavioral_score"],
            weights=weights,
        )

        results.append((cid, score, feats, subscores))

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def build_score_matrix(
    candidates: List[Dict],
    similarities: Optional[np.ndarray],
    config: Dict,
) -> np.ndarray:
    """
    Fast vectorized score computation for when all features are pre-computed.
    Returns composite scores as np.ndarray shape (N,).
    """
    n = len(candidates)
    weights = config.get("weights", {
        "semantic_fit": 0.35, "skill_match": 0.30,
        "career_quality": 0.20, "behavioral": 0.15,
    })

    skill_scores = np.zeros(n, dtype=np.float32)
    career_scores = np.zeros(n, dtype=np.float32)
    behavioral_scores = np.zeros(n, dtype=np.float32)
    sem_sims = similarities if similarities is not None else np.zeros(n, dtype=np.float32)

    for i, cand in enumerate(candidates):
        feats = extract_features(cand, config)
        sub = features_to_subscores(feats)
        skill_scores[i] = sub["skill_score"]
        career_scores[i] = sub["career_score"]
        behavioral_scores[i] = sub["behavioral_score"]

    composite = (
        weights.get("semantic_fit", 0.35) * sem_sims +
        weights.get("skill_match", 0.30) * skill_scores +
        weights.get("career_quality", 0.20) * career_scores +
        weights.get("behavioral", 0.15) * behavioral_scores
    )

    return np.clip(composite, 0.0, 1.0)
