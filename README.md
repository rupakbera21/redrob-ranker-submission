# Redrob Hackathon — Intelligent Candidate Ranking System
**Participant:** Rupak Bera | **Challenge:** India Runs Data & AI Challenge

## 🎯 What This Does

A two-phase hybrid pipeline that ranks candidates for a **Senior AI Engineer** role using:
- **Semantic understanding** (sentence-transformer embeddings, not keyword matching)
- **Deep feature engineering** (30+ structured signals per candidate)
- **Anti-keyword-stuffer logic** (career trajectory analysis, company quality)
- **Behavioral signals** (platform engagement, recency, responsiveness)
- **LLM reasoning** (Groq offline reasoning for top-300, bundled in cache)

## 🏗️ Architecture

```
Phase 1 — Offline (precompute.py, ~20 min, run once):
  100K candidates → feature extraction → all-MiniLM-L6-v2 embeddings
  → cosine similarity vs JD → top-300 → Groq LLM reasoning → cache/

Phase 2 — Ranking (rank.py, <90 seconds, CPU-only, no network):
  Stream 100K JSONL + load cache → composite score → top-100 → CSV
```

### Scoring Formula
```
score = 0.35 × semantic_fit      (embedding cosine similarity)
      + 0.30 × skill_match       (AI/ML skill depth, not keyword count)
      + 0.20 × career_quality    (trajectory, company type, AI role ratio)
      + 0.15 × behavioral        (availability, responsiveness, recency)
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pre-compute (run once)
```bash
# With Groq reasoning (recommended):
python precompute.py --candidates ./candidates.jsonl --groq-key YOUR_GROQ_KEY

# Without Groq (feature + embeddings only):
python precompute.py --candidates ./candidates.jsonl --skip-groq
```

### 3. Generate submission
```bash
python rank.py --candidates ./candidates.jsonl --out ./rupak_bera.csv
```

### 4. Validate
```bash
python validate_submission.py rupak_bera.csv
```

## 📁 Project Structure

```
redrob-ranker/
├── rank.py                 # ← Main submission script (≤5 min, CPU-only)
├── precompute.py           # ← Offline pre-computation (embeddings + Groq)
├── config.yaml             # ← JD parameters, scoring weights, skill lists
├── requirements.txt
├── submission_metadata.yaml
├── src/
│   ├── features.py         # Feature extraction (30+ signals)
│   ├── embedder.py         # Sentence-transformer wrapper
│   ├── scorer.py           # Composite weighted scoring
│   └── reasoner.py         # Groq API + fallback reasoning
└── cache/                  # Auto-created by precompute.py
    ├── embeddings.npz      # Cosine similarities for 100K candidates
    ├── features.pkl        # Extracted feature matrix
    ├── candidate_ids.txt   # ID ordering
    └── groq_reasoning.json # LLM-generated reasoning strings
```

## 🧠 Key Design Decisions

### Why This Beats Keyword Matching

| Trap in Dataset | Our Handling |
|---|---|
| Marketing Manager with all AI keywords | `career_quality` near-zero — title mismatch kills score |
| Candidate with real AI work but no buzzwords | Semantic embedding captures conceptual fit |
| Pure TCS/Infosys/Wipro career | Explicit `pure_services_penalty` (JD says this) |
| Inactive 6-month ghost | `recency_score` exponential decay (halflife = 60 days) |
| Perfect profile, 5% response rate | `responsiveness` score drops behavioral component |
| 90-day notice candidate | `notice_fit` exponential penalty beyond 30-day buyout |

### Feature Groups

**Skill Match (0.30 weight):**
- Skill depth = proficiency × endorsement_trust × duration bonus
- Core AI skills weighted by JD importance (NDCG/FAISS = 3.0, Python = 2.0, CV-only = 0.4)
- Platform skill assessment scores as trust signal

**Career Quality (0.20 weight):**
- `ai_role_ratio`: fraction of career in ML/AI titles
- `ai_description_ratio`: actual AI work in role descriptions
- `pure_services_penalty`: all-consulting career flag
- `experience_band_score`: 5-9 year sweet spot
- `tenure_stability`: average tenure (job-hopper penalty)

**Behavioral (0.15 weight):**
- `recency_score`: exponential decay on last_active_date
- `open_to_work`: direct availability signal
- `responsiveness`: response_rate × speed bonus
- `notice_fit`: exp penalty beyond 30-day buyout
- `location_fit`: India + preferred cities

## ⚡ Performance

| Step | Time | Notes |
|---|---|---|
| `precompute.py` (embeddings only) | ~10-15 min | Run once |
| `precompute.py` (+ Groq, 300 cands) | ~15-20 min | ~60 API calls |
| `rank.py` with cache | <90 seconds | CPU-only |
| `rank.py` without cache | ~3-4 min | Feature-only mode |
| Peak RAM | ~4 GB | Embeddings: 100K × 384 × 4B = 150MB |

## 🔧 Configuration

Edit `config.yaml` to tune:
- `weights:` — adjust semantic/skill/career/behavioral balance
- `core_ai_skills:` — add/remove relevant skills and their JD importance weights
- `jd.preferred_locations:` — add more accepted locations
- `precompute.top_n_for_groq:` — candidates sent to Groq (more = better reasoning, more API cost)
