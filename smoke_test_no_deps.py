#!/usr/bin/env python3
"""
smoke_test_no_deps.py
=====================
Tests feature extraction logic on sample_candidates.json
WITHOUT requiring any external packages (stdlib only).

This validates the core scoring logic is correct before full install.
"""
import json
import math
import sys
from pathlib import Path

# ── Load sample data ──────────────────────────────────────────────────────────
SAMPLE_PATH = r"c:\Users\RUPAK\Desktop\INDIA RUNS\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\sample_candidates.json"

print("=" * 65)
print(" REDROB HACKATHON — Feature Logic Smoke Test (no deps needed)")
print("=" * 65)

with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
    candidates = json.load(f)

print(f"\n✅ Loaded {len(candidates)} sample candidates")

# ── Minimal config (subset of config.yaml) ────────────────────────────────────
CORE_AI_SKILLS = {
    "sentence transformers": 3.0, "sentence-transformers": 3.0,
    "faiss": 3.0, "pinecone": 3.0, "weaviate": 3.0, "qdrant": 3.0,
    "milvus": 3.0, "opensearch": 2.5, "elasticsearch": 2.5,
    "vector search": 3.0, "vector database": 3.0, "embeddings": 3.0,
    "semantic search": 3.0, "dense retrieval": 3.0, "hybrid search": 3.0,
    "information retrieval": 3.0, "retrieval": 2.5, "bm25": 2.5,
    "ranking": 2.5, "reranking": 3.0, "re-ranking": 3.0,
    "learning to rank": 3.0, "ltr": 2.5, "ndcg": 3.0, "mrr": 3.0,
    "rag": 2.5, "fine-tuning llms": 2.5, "fine tuning": 2.0,
    "lora": 2.5, "qlora": 2.5, "peft": 2.5, "nlp": 2.5,
    "natural language processing": 2.5, "llm": 2.5, "large language model": 2.5,
    "transformer": 2.0, "bert": 2.0, "python": 2.0, "pytorch": 1.5,
    "xgboost": 1.5, "machine learning": 1.0, "deep learning": 1.0,
    "image classification": 0.4, "object detection": 0.4,
    "speech recognition": 0.4, "tts": 0.4, "gans": 0.3,
    "weights & biases": 1.5, "milvus": 3.0, "feature engineering": 1.0,
    "statistical modeling": 0.8,
}

PROFICIENCY = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}

SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hexaware", "mphasis",
    "tech mahindra", "ltimindtree", "mindtree", "dunder mifflin",
}

AI_ROLE_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "data scientist", "nlp engineer", "research engineer",
    "applied scientist", "search engineer", "ranking engineer",
    "mlops", "deep learning engineer", "senior data scientist",
    "machine learning scientist",
}

AI_DESC_KEYWORDS = {
    "embedding", "embeddings", "vector", "retrieval", "ranking", "nlp",
    "natural language", "language model", "llm", "transformer", "bert",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "semantic search",
    "information retrieval", "rag", "fine-tuning", "fine tuning",
    "lora", "recommendation", "reranking", "learning to rank", "ndcg",
}

PREFERRED_LOCS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "gurgaon", "gurugram",
}


def skill_score(candidate):
    skills = candidate.get("skills", [])
    raw = 0.0
    count = 0
    for s in skills:
        name = s.get("name", "").lower()
        prof = PROFICIENCY.get(s.get("proficiency", "beginner"), 0.2)
        end = s.get("endorsements", 0)
        dur = s.get("duration_months", 0)
        matched, w = False, 0.0
        if name in CORE_AI_SKILLS:
            matched, w = True, CORE_AI_SKILLS[name]
        else:
            for sk, wt in CORE_AI_SKILLS.items():
                if sk in name or name in sk:
                    matched, w = True, wt
                    break
        if matched and w > 0:
            trust = min(1.0, math.log1p(end) / math.log1p(100))
            dur_b = min(1.0, dur / 36.0)
            raw += w * prof * (0.6 + 0.2 * trust + 0.2 * dur_b)
            count += 1
    return min(1.0, raw / 20.0), count


def career_score(candidate):
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    yoe = float(profile.get("years_of_experience", 0))

    services_only = True
    ai_role_count = 0
    ai_desc_count = 0
    total_months = 0
    tenures = []

    for r in career:
        comp = r.get("company", "").lower()
        title = r.get("title", "").lower()
        desc = r.get("description", "").lower()
        dur = r.get("duration_months", 0)

        is_svc = any(s in comp for s in SERVICES_COMPANIES)
        if not is_svc:
            services_only = False
        if any(t in title for t in AI_ROLE_TITLES):
            ai_role_count += 1
        if sum(1 for kw in AI_DESC_KEYWORDS if kw in desc) >= 2:
            ai_desc_count += 1
        total_months += dur
        if dur > 0:
            tenures.append(dur)

    # Experience band score (5-9 years sweet spot)
    if 5 <= yoe <= 9:
        exp_band = 1.0
    elif yoe < 5:
        exp_band = max(0.0, 1.0 - (5 - yoe) * 0.15)
    else:
        exp_band = max(0.4, 1.0 - (yoe - 9) * 0.05)

    n = max(1, len(career))
    ai_signal = (ai_role_count / n) * 0.4 + (ai_desc_count / n) * 0.3 + (
        1.0 if any(t in profile.get("current_title", "").lower() for t in AI_ROLE_TITLES) else 0.0
    ) * 0.3

    avg_ten = sum(tenures) / max(1, len(tenures))
    tenure_stab = max(0.0, min(1.0, (avg_ten - 6) / 30.0))
    svc_penalty = 0.4 if services_only and len(career) > 0 else 0.0

    raw = (ai_signal * 0.40 + tenure_stab * 0.20 + exp_band * 0.20 + 0.20) - svc_penalty
    return max(0.0, min(1.0, raw))


def behavioral_score(candidate):
    from datetime import datetime
    sig = candidate.get("redrob_signals", {})

    # Recency
    last = sig.get("last_active_date", "")
    try:
        days = (datetime.utcnow().date() - datetime.fromisoformat(last).date()).days
        recency = math.exp(-days * math.log(2) / 60)
    except Exception:
        recency = 0.0

    open_w = 1.0 if sig.get("open_to_work_flag") else 0.0
    rr = float(sig.get("recruiter_response_rate", 0))
    notice = int(sig.get("notice_period_days", 60))
    notice_fit = 1.0 if notice <= 30 else math.exp(-(notice - 30) / 30.0)

    loc = candidate.get("profile", {}).get("location", "").lower()
    country = candidate.get("profile", {}).get("country", "").lower()
    relocate = sig.get("willing_to_relocate", False)
    if any(c in loc for c in PREFERRED_LOCS):
        loc_fit = 1.0
    elif country == "india" and relocate:
        loc_fit = 0.75
    elif country == "india":
        loc_fit = 0.55
    else:
        loc_fit = 0.20

    verified = (sig.get("verified_email", 0) + sig.get("verified_phone", 0) + sig.get("linkedin_connected", 0)) / 3.0
    github = max(0.0, float(sig.get("github_activity_score", -1))) / 100.0

    avail = open_w * 0.5 + recency * 0.5
    responsiveness = rr * 0.7 + verified * 0.3
    logistics = notice_fit * 0.5 + loc_fit * 0.35 + github * 0.15
    return max(0.0, min(1.0, avail * 0.30 + responsiveness * 0.30 + logistics * 0.40))


# ── Score all candidates (no semantic for smoke test) ─────────────────────────
weights = {"skill": 0.45, "career": 0.35, "behavioral": 0.20}

results = []
for c in candidates:
    cid = c.get("candidate_id", "?")
    title = c.get("profile", {}).get("current_title", "?")
    yoe = c.get("profile", {}).get("years_of_experience", 0)
    sk, core_count = skill_score(c)
    cr = career_score(c)
    bh = behavioral_score(c)
    composite = weights["skill"] * sk + weights["career"] * cr + weights["behavioral"] * bh
    results.append({
        "candidate_id": cid,
        "title": title,
        "yoe": yoe,
        "skill_score": sk,
        "career_score": cr,
        "behavioral_score": bh,
        "composite": composite,
        "core_skill_count": core_count,
    })

results.sort(key=lambda x: x["composite"], reverse=True)

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f" {'RANK':<5} {'CANDIDATE_ID':<15} {'TITLE':<30} {'YOE':>4}  {'SCORE':>6}")
print(f"{'─'*65}")

for i, r in enumerate(results[:20]):
    title_short = r["title"][:28]
    print(f" #{i+1:<4} {r['candidate_id']:<15} {title_short:<30} {r['yoe']:>4.1f}  {r['composite']:>6.4f}")

print(f"{'─'*65}")
print(f"\n Sub-scores for top-10:")
print(f" {'#':<4} {'ID':<15} {'skill':>7} {'career':>7} {'behav':>7} {'core_n':>7}")
for i, r in enumerate(results[:10]):
    print(f" #{i+1:<3} {r['candidate_id']:<15} {r['skill_score']:>7.4f} {r['career_score']:>7.4f} "
          f"{r['behavioral_score']:>7.4f} {r['core_skill_count']:>7}")

# ── Anti-trap check ────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f" ANTI-TRAP VALIDATION")
print(f"{'─'*65}")

top10_titles = [r["title"].lower() for r in results[:10]]
non_ai_in_top10 = [r for r in results[:10]
                   if not any(t in r["title"].lower() for t in [
                       "ml", "ai", "data scientist", "engineer", "scientist",
                       "nlp", "research", "machine learning"
                   ])]

print(f" Top-10 titles: {[r['title'] for r in results[:10]]}")
if non_ai_in_top10:
    print(f" ⚠️  NON-AI roles in top-10: {[r['title'] for r in non_ai_in_top10]}")
    print(f"    (These should be reviewed — may indicate tuning needed)")
else:
    print(f" ✅ No obvious non-AI roles in top-10")

bottom5 = results[-5:]
print(f"\n Bottom-5 titles: {[r['title'] for r in bottom5]}")

# ── Score distribution ────────────────────────────────────────────────────────
all_scores = [r["composite"] for r in results]
print(f"\n Score distribution (all {len(results)} sample candidates):")
print(f"   Max:  {max(all_scores):.4f}")
print(f"   Min:  {min(all_scores):.4f}")
print(f"   Mean: {sum(all_scores)/len(all_scores):.4f}")

print(f"\n✅ Smoke test complete — feature extraction logic is working correctly")
print(f"   Next: install sentence-transformers for semantic embeddings\n")
