"""
src/reasoner.py
===============
Groq API integration for generating high-quality, specific reasoning strings.

IMPORTANT: This module is ONLY used during precompute.py (offline phase).
rank.py loads pre-generated reasoning from cache — NO API calls at runtime.

The Groq LLM serves two purposes:
  1. Re-rank the top-300 candidates with deeper semantic reasoning
  2. Generate Stage-4-quality reasoning strings (specific, honest, non-hallucinated)
"""
from __future__ import annotations

import json
import os
import time
import re
from typing import Any, Dict, List, Optional, Tuple

# ── JD Context (injected into every Groq call) ────────────────────────────────

JD_SUMMARY = """
**Role:** Senior AI Engineer (Founding Team) at Redrob AI — Series A, Pune/Noida, Hybrid

**What they ACTUALLY need (beyond keywords):**
- 5-9 years of APPLIED ML at product companies (NOT pure IT services like TCS/Infosys/Wipro/Accenture entire career)
- PRODUCTION deployment of embedding/retrieval/ranking systems to real users at scale
- Hands-on with vector databases (FAISS, Pinecone, Weaviate, Qdrant, Milvus, OpenSearch)
- Strong NLP/IR background (not just computer vision/speech without NLP)
- Evaluation frameworks: NDCG, MRR, MAP, A/B testing — must have used these
- Python expert who writes production code (not an "architect")
- LLM fine-tuning (LoRA, QLoRA, PEFT) is nice-to-have, not required
- Startup mentality: ship first, optimize later

**Hard disqualifiers:**
- Pure academic/research career with no production deployment
- Entire career in consulting/IT services firms with no product-company experience
- "AI experience" is only LangChain wrappers over OpenAI API (< 12 months, no prior ML)
- Primary expertise is computer vision / speech / robotics WITHOUT NLP/IR exposure
- Has not written production code in 18+ months (pure "architect")
- Title-hoppers (switching every 1.5 years for title bumps)

**Behavioral requirements:**
- Recently active on platform (logged in within last 60 days = strong signal)
- Notice period ≤ 30 days preferred (will buy out up to 30 days)
- Responds to recruiters (response rate > 30% = good)
- India-based preferred (Pune, Noida, Hyderabad, Mumbai, Delhi NCR, Bangalore)
- Open to work flag set = bonus

**Scoring scale for this role:**
- 9.0-10.0: Near-perfect. Right title, product company, production AI, active, India-based
- 7.0-8.9: Strong fit. Good AI background, maybe 1-2 concerns (notice, location, or minor skill gap)
- 5.0-6.9: Moderate fit. Adjacent skills, willingness to learn, or strong behavioral signals offset skill gap
- 3.0-4.9: Weak fit. More domain-adjacent than technical AI fit
- 1.0-2.9: Poor fit. Keyword match only, no real AI depth
- 0.0-1.0: Disqualify. Pure keyword stuffer or disqualifier applies
"""

RANKING_PROMPT_TEMPLATE = """You are an expert technical recruiter evaluating candidates for a specific role. Evaluate each candidate against the JD context below and return ONLY a JSON array — no markdown, no commentary, no code blocks.

{jd_context}

---

For each candidate below, provide:
- "candidate_id": the exact candidate_id string
- "llm_score": float 0.0-10.0 (your assessment of fit for THIS specific role)
- "reasoning": 1-2 sentences, specific facts from the profile, honest about concerns, NO hallucination. Reference actual titles, companies, skills. Tone must match the score.
- "fit_tier": one of "strong" | "moderate" | "weak" | "disqualify"

Return ONLY this exact JSON format:
[
  {{"candidate_id": "CAND_XXXXXXX", "llm_score": X.X, "reasoning": "...", "fit_tier": "..."}},
  ...
]

**Candidates to evaluate:**
{candidates_json}"""


# ── Groq API Client ───────────────────────────────────────────────────────────

def _build_candidate_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact candidate summary for the LLM prompt (to save tokens)."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    education = candidate.get("education", [])

    # Top skills by proficiency
    sorted_skills = sorted(
        skills,
        key=lambda s: ({"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(
            s.get("proficiency", "beginner"), 1), s.get("endorsements", 0)),
        reverse=True
    )
    top_skills = [f"{s['name']} ({s.get('proficiency','')})" for s in sorted_skills[:12]]

    # Last 3 roles
    recent_roles = []
    for r in career[:3]:
        desc_preview = r.get("description", "")[:200]
        recent_roles.append({
            "title": r.get("title", ""),
            "company": r.get("company", ""),
            "duration_months": r.get("duration_months", 0),
            "is_current": r.get("is_current", False),
            "description_preview": desc_preview,
        })

    # Education (top 1)
    edu_summary = []
    for e in education[:1]:
        edu_summary.append(f"{e.get('degree','')} {e.get('field_of_study','')} from {e.get('institution','')} ({e.get('tier','?')})")

    # Key behavioral signals
    last_active = signals.get("last_active_date", "unknown")
    open_to_work = signals.get("open_to_work_flag", False)
    response_rate = signals.get("recruiter_response_rate", 0)
    notice_days = signals.get("notice_period_days", 0)
    github_score = signals.get("github_activity_score", -1)
    location = profile.get("location", "")
    country = profile.get("country", "")

    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "current_title": profile.get("current_title", ""),
        "headline": profile.get("headline", ""),
        "years_of_experience": profile.get("years_of_experience", 0),
        "location": f"{location}, {country}",
        "summary_excerpt": profile.get("summary", "")[:300],
        "top_skills": top_skills,
        "recent_roles": recent_roles,
        "education": edu_summary,
        "behavioral_signals": {
            "last_active": last_active,
            "open_to_work": open_to_work,
            "recruiter_response_rate": round(response_rate, 2),
            "notice_period_days": notice_days,
            "github_activity_score": github_score,
        }
    }


def call_groq_batch(
    candidates: List[Dict[str, Any]],
    groq_client: Any,
    model: str = "llama-3.3-70b-versatile",
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Call Groq API for a batch of candidates. Returns list of result dicts.
    Each result: {candidate_id, llm_score, reasoning, fit_tier}
    """
    summaries = [_build_candidate_summary(c) for c in candidates]
    candidates_json = json.dumps(summaries, indent=2, ensure_ascii=False)

    prompt = RANKING_PROMPT_TEMPLATE.format(
        jd_context=JD_SUMMARY,
        candidates_json=candidates_json,
    )

    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise technical recruiter. Return ONLY valid JSON arrays. No markdown, no code blocks, no commentary."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000,
            )

            content = response.choices[0].message.content.strip()

            # Strip any accidental markdown code blocks
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
            content = re.sub(r"\s*```\s*$", "", content, flags=re.MULTILINE)
            content = content.strip()

            results = json.loads(content)

            # Validate structure
            validated = []
            for item in results:
                if "candidate_id" in item and "llm_score" in item and "reasoning" in item:
                    validated.append({
                        "candidate_id": str(item["candidate_id"]),
                        "llm_score": float(item.get("llm_score", 5.0)),
                        "reasoning": str(item.get("reasoning", "")),
                        "fit_tier": str(item.get("fit_tier", "moderate")),
                    })
            return validated

        except json.JSONDecodeError as e:
            print(f"    [Groq] JSON parse error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str.lower() or "429" in err_str:
                print(f"    [Groq] Rate limit hit. Aborting Groq calls to use fallback.")
                raise Exception("RATE_LIMIT")
            else:
                print(f"    [Groq] Error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

    # Return empty results on full failure
    return []


def generate_reasoning_for_all(
    top_candidates: List[Dict[str, Any]],
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
    batch_size: int = 5,
    max_retries: int = 3,
    delay_between_batches: float = 1.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate reasoning + LLM scores for the top-N candidates using Groq.

    Args:
        top_candidates: List of candidate dicts (pre-filtered top-N).
        api_key: Groq API key.
        model: Groq model identifier.
        batch_size: Candidates per API call.
        max_retries: Retry attempts per batch.
        delay_between_batches: Seconds between API calls (rate limit safety).

    Returns:
        Dict mapping candidate_id → {llm_score, reasoning, fit_tier}.
    """
    from groq import Groq

    client = Groq(api_key=api_key)
    results: Dict[str, Dict[str, Any]] = {}

    # Split into batches
    batches = [
        top_candidates[i:i + batch_size]
        for i in range(0, len(top_candidates), batch_size)
    ]

    print(f"\n[Reasoner] Generating reasoning for {len(top_candidates)} candidates "
          f"in {len(batches)} batches of {batch_size}...")

    for batch_idx, batch in enumerate(batches):
        batch_ids = [c.get("candidate_id", "?") for c in batch]
        print(f"  Batch {batch_idx + 1}/{len(batches)}: {batch_ids[:3]}{'...' if len(batch_ids) > 3 else ''}", end=" ")

        try:
            batch_results = call_groq_batch(
                candidates=batch,
                groq_client=client,
                model=model,
                max_retries=max_retries,
            )
        except Exception as e:
            if str(e) == "RATE_LIMIT":
                break
            batch_results = []

        for item in batch_results:
            cid = item["candidate_id"]
            results[cid] = {
                "llm_score": item["llm_score"],
                "reasoning": item["reasoning"],
                "fit_tier": item["fit_tier"],
            }

        print(f"OK ({len(batch_results)}/{len(batch)} ok)")

        # Rate limit safety
        if batch_idx < len(batches) - 1:
            time.sleep(delay_between_batches)

    success_rate = len(results) / max(1, len(top_candidates))
    print(f"[Reasoner] Done. Coverage: {len(results)}/{len(top_candidates)} ({success_rate:.1%})")

    return results


def save_reasoning_cache(reasoning: Dict[str, Dict], cache_dir: str) -> None:
    """Save Groq reasoning results to cache."""
    import os
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "groq_reasoning.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reasoning, f, indent=2, ensure_ascii=False)
    print(f"  Saved reasoning cache: {path} ({len(reasoning)} entries)")


def load_reasoning_cache(cache_dir: str) -> Dict[str, Dict]:
    """Load Groq reasoning from cache. Returns empty dict if not found."""
    path = os.path.join(cache_dir, "groq_reasoning.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_fallback_reasoning(
    candidate: Dict[str, Any],
    subscores: Dict[str, float],
    composite_score: float,
    rank: int,
) -> str:
    """
    Generate a rule-based reasoning string when Groq is not available.
    Uses only actual facts from the candidate profile — no hallucination.
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])

    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    company = profile.get("current_company", "")

    # Top skills by proficiency
    sorted_skills = sorted(
        skills,
        key=lambda s: ({"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(
            s.get("proficiency", "beginner"), 1), s.get("endorsements", 0)),
        reverse=True
    )
    top_skill_names = [s["name"] for s in sorted_skills[:3]]
    skills_str = ", ".join(top_skill_names) if top_skill_names else "N/A"

    response_rate = float(signals.get("recruiter_response_rate", 0))
    notice = int(signals.get("notice_period_days", 0))
    open_work = signals.get("open_to_work_flag", False)

    # Company diversity check
    all_companies = [r.get("company", "") for r in career]
    is_services_heavy = all(
        any(svc in c.lower() for svc in ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"])
        for c in all_companies if c
    ) and len(all_companies) > 0

    parts = [f"{title} with {yoe:.1f} yrs exp at {company}; top skills: {skills_str}"]

    if open_work:
        parts.append("marked open to work")
    if response_rate > 0.5:
        parts.append(f"strong recruiter response rate ({response_rate:.0%})")
    elif response_rate < 0.2:
        parts.append(f"low recruiter response rate ({response_rate:.0%})")

    if notice <= 30:
        parts.append(f"notice period {notice}d (within buyout window)")
    elif notice > 60:
        parts.append(f"notice period {notice}d (concern: above buyout threshold)")

    if is_services_heavy:
        parts.append("career predominantly in IT services (JD prefers product-company experience)")

    if composite_score < 0.4:
        parts.append("limited AI/ML technical depth relative to JD requirements")

    return "; ".join(parts) + "."
