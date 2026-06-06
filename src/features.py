"""
src/features.py
===============
Feature extraction for candidate ranking.

Extracts ~30 structured numeric features per candidate, designed specifically
for the Senior AI Engineer (Founding Team) JD at Redrob AI.

Anti-keyword-stuffing logic is built in:
  - Skill depth (proficiency + endorsements), not just presence
  - Career trajectory and company quality check
  - Behavioral engagement signals as multiplier
  - Pure-services-company penalty when applicable
"""
from __future__ import annotations

import re
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────

SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hexaware", "mphasis",
    "tech mahindra", "techmahindra", "ltimindtree", "l&t infotech",
    "mindtree", "niit technologies", "mastech", "kpit", "birlasoft",
    "dxc technology", "unisys", "zensar", "cyient", "igate", "patni",
    "dunder mifflin",   # fictional but in dataset
}

AI_ROLE_KEYWORDS = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "artificial intelligence engineer", "data scientist", "nlp engineer",
    "research engineer", "applied scientist", "search engineer",
    "ranking engineer", "recommendation engineer", "applied researcher",
    "research scientist", "mlops", "ml platform", "senior ml",
    "lead ml", "principal ml", "staff ml", "deep learning engineer",
    "senior data scientist", "lead data scientist", "principal data scientist",
    "machine learning scientist",
}

AI_DESCRIPTION_KEYWORDS = {
    "embedding", "embeddings", "vector", "retrieval", "ranking", "nlp",
    "natural language", "language model", "llm", "transformer", "bert",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "semantic search", "information retrieval", "rag",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "recommendation", "search", "reranking", "re-ranking",
    "sentence-transformer", "sentence transformer", "dense retrieval",
    "hybrid search", "learning to rank", "ndcg", "mrr",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "gurgaon", "gurugram", "ncr",
}

PROFICIENCY_SCORE = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.2,
}

COMPANY_SIZE_SCORE = {
    "1-10": 0.6,
    "11-50": 0.7,
    "51-200": 0.8,
    "201-500": 0.85,
    "501-1000": 0.9,
    "1001-5000": 0.95,
    "5001-10000": 1.0,
    "10001+": 1.0,
}

EDUCATION_TIER_SCORE = {
    "tier_1": 1.0,
    "tier_2": 0.8,
    "tier_3": 0.6,
    "tier_4": 0.4,
    "unknown": 0.5,
}

DEGREE_RELEVANCE = {
    "b.tech": 0.9, "b.e.": 0.9, "m.tech": 1.0, "m.e.": 1.0,
    "m.sc": 0.85, "b.sc": 0.7, "mba": 0.3, "b.com": 0.2,
    "b.a": 0.2, "m.a": 0.2, "ph.d": 1.0, "phd": 1.0,
    "m.s.": 1.0, "ms": 1.0, "bs": 0.8,
}

FIELD_RELEVANCE = {
    "computer science": 1.0, "cs": 1.0, "information technology": 0.9,
    "software engineering": 0.95, "machine learning": 1.0,
    "artificial intelligence": 1.0, "data science": 1.0,
    "electronics": 0.7, "electrical": 0.6, "mathematics": 0.85,
    "statistics": 0.9, "physics": 0.7, "mechanical": 0.3,
    "civil": 0.2, "chemical": 0.2, "management": 0.3,
    "information systems": 0.85, "computer engineering": 1.0,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _today() -> date:
    return datetime.utcnow().date()


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        return None


def _days_since(d: Optional[date]) -> int:
    """Days since a date. Returns 9999 if None."""
    if d is None:
        return 9999
    return max(0, (_today() - d).days)


def _normalize(value: float, lo: float, hi: float) -> float:
    """Clamp then normalize to [0, 1]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _is_services_company(company_name: str) -> bool:
    name_lower = company_name.lower().strip()
    for svc in SERVICES_COMPANIES:
        if svc in name_lower:
            return True
    return False


def _title_is_ai_role(title: str) -> bool:
    title_lower = title.lower()
    return any(kw in title_lower for kw in AI_ROLE_KEYWORDS)


def _description_has_ai_work(description: str) -> bool:
    desc_lower = description.lower()
    matches = sum(1 for kw in AI_DESCRIPTION_KEYWORDS if kw in desc_lower)
    return matches >= 2


def _skill_name_matches_core(name: str, core_skills: Dict[str, float]) -> Tuple[bool, float]:
    """Check if a skill name matches any core AI skill. Returns (matched, weight)."""
    name_lower = name.lower().strip()
    if name_lower in core_skills:
        return True, core_skills[name_lower]
    # Partial match (e.g., "Fine-tuning LLMs" should match "fine-tuning")
    for skill_key, weight in core_skills.items():
        if skill_key in name_lower or name_lower in skill_key:
            return True, weight
    return False, 0.0


def _endorsement_trust(endorsements: int) -> float:
    """Logarithmic trust bonus from endorsements (0 → 0, 100+ → ~1.0)."""
    return min(1.0, math.log1p(endorsements) / math.log1p(100))


def _recency_score(last_active: Optional[date], halflife_days: int = 60) -> float:
    """Exponential decay based on days since last active."""
    days = _days_since(last_active)
    return math.exp(-days * math.log(2) / halflife_days)


# ── Main Feature Extractor ─────────────────────────────────────────────────────

def extract_features(candidate: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract ~30 numeric features from a candidate profile.
    All features are in [0, 1] or small positive floats.
    Returns a flat dict of feature_name → float.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    certs = candidate.get("certifications", [])
    signals = candidate.get("redrob_signals", {})
    core_skills = config.get("core_ai_skills", {})
    jd = config.get("jd", {})

    features: Dict[str, float] = {}

    # ─────────────────────────────────────────
    # 1. SKILL FEATURES
    # ─────────────────────────────────────────

    raw_skill_score = 0.0
    core_count = 0
    assessment_sum = 0.0
    assessment_count = 0

    for skill in skills:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "beginner")
        endorsements = int(skill.get("endorsements", 0))
        duration_months = int(skill.get("duration_months", 0))

        matched, weight = _skill_name_matches_core(name, core_skills)
        if matched and weight > 0:
            prof_score = PROFICIENCY_SCORE.get(proficiency, 0.2)
            trust = _endorsement_trust(endorsements)
            duration_bonus = min(1.0, duration_months / 36.0)
            # Composite per-skill score
            raw_skill_score += weight * prof_score * (0.6 + 0.2 * trust + 0.2 * duration_bonus)
            core_count += 1

    # Normalize skill score (cap at ~10 skills contributing fully at weight 3)
    features["skill_raw_score"] = min(1.0, raw_skill_score / 20.0)
    features["core_skill_count"] = min(1.0, core_count / 10.0)

    # Skill assessment scores (on-platform validated scores)
    skill_assessment_scores = signals.get("skill_assessment_scores", {})
    for skill_name, score in skill_assessment_scores.items():
        matched, weight = _skill_name_matches_core(skill_name, core_skills)
        if matched and weight > 0:
            assessment_sum += (score / 100.0) * weight
            assessment_count += 1

    if assessment_count > 0:
        features["assessment_score"] = min(1.0, assessment_sum / (assessment_count * 2.0))
    else:
        features["assessment_score"] = 0.0  # No assessment penalty — many good candidates skip it

    # GitHub activity (strong signal for active technical contributors)
    github_raw = float(signals.get("github_activity_score", -1))
    features["github_score"] = max(0.0, github_raw / 100.0) if github_raw >= 0 else 0.0

    # ─────────────────────────────────────────
    # 2. CAREER QUALITY FEATURES
    # ─────────────────────────────────────────

    yoe = float(profile.get("years_of_experience", 0))
    exp_min = jd.get("experience_min", 5)
    exp_max = jd.get("experience_max", 9)

    # Experience band score — JD says 5-9 years, penalize extremes
    if exp_min <= yoe <= exp_max:
        features["experience_band_score"] = 1.0
    elif yoe < exp_min:
        features["experience_band_score"] = max(0.0, 1.0 - (exp_min - yoe) * 0.15)
    else:  # over-experienced
        features["experience_band_score"] = max(0.4, 1.0 - (yoe - exp_max) * 0.05)

    # Current title: does it signal AI/ML seniority?
    current_title = profile.get("current_title", "").lower()
    features["is_ai_title"] = 1.0 if _title_is_ai_role(current_title) else 0.0

    # Career trajectory: AI roles in history + quality signal
    ai_role_count = 0
    services_only = True
    product_company_months = 0
    total_career_months = 0
    tenure_list = []
    ai_description_match_count = 0

    for role in career:
        company = role.get("company", "")
        title = role.get("title", "")
        description = role.get("description", "")
        duration = int(role.get("duration_months", 0))
        is_services = _is_services_company(company)

        if not is_services:
            services_only = False
            product_company_months += duration

        total_career_months += duration
        if duration > 0:
            tenure_list.append(duration)

        if _title_is_ai_role(title):
            ai_role_count += 1
        if _description_has_ai_work(description):
            ai_description_match_count += 1

    # Pure-services penalty (JD explicitly penalizes entire career in consulting)
    features["pure_services_penalty"] = 1.0 if services_only and len(career) > 0 else 0.0
    features["product_company_ratio"] = (product_company_months / max(1, total_career_months))

    # AI role ratio (how much of career in AI/ML roles)
    features["ai_role_ratio"] = min(1.0, ai_role_count / max(1, len(career)))
    features["ai_description_ratio"] = min(1.0, ai_description_match_count / max(1, len(career)))

    # Job hopper penalty (avg tenure < 18 months is a concern)
    if tenure_list:
        avg_tenure = sum(tenure_list) / len(tenure_list)
        features["tenure_stability"] = _normalize(avg_tenure, 6, 36)
    else:
        features["tenure_stability"] = 0.5

    # Company size diversity (product companies tend to be varied sizes)
    if career:
        latest_company_size = career[0].get("company_size", "1001-5000")
        features["company_prestige"] = COMPANY_SIZE_SCORE.get(latest_company_size, 0.8)
    else:
        features["company_prestige"] = 0.5

    # ─────────────────────────────────────────
    # 3. EDUCATION FEATURES
    # ─────────────────────────────────────────

    if education:
        best_edu = max(education, key=lambda e: EDUCATION_TIER_SCORE.get(e.get("tier", "unknown"), 0.5))
        tier = best_edu.get("tier", "unknown")
        degree = best_edu.get("degree", "").lower().strip()
        field = best_edu.get("field_of_study", "").lower()

        features["edu_tier"] = EDUCATION_TIER_SCORE.get(tier, 0.5)

        deg_score = 0.5
        for key, val in DEGREE_RELEVANCE.items():
            if key in degree:
                deg_score = max(deg_score, val)
        features["edu_degree_relevance"] = deg_score

        field_score = 0.5
        for key, val in FIELD_RELEVANCE.items():
            if key in field:
                field_score = max(field_score, val)
        features["edu_field_relevance"] = field_score
    else:
        features["edu_tier"] = 0.4
        features["edu_degree_relevance"] = 0.5
        features["edu_field_relevance"] = 0.5

    # Certifications (AI/ML relevant)
    ai_cert_keywords = {"aws", "gcp", "azure", "tensorflow", "pytorch", "ml", "ai",
                        "deep learning", "databricks", "spark", "kubernetes", "docker",
                        "machine learning", "data science", "cloud"}
    ai_cert_count = 0
    for cert in certs:
        cert_name = cert.get("name", "").lower()
        if any(kw in cert_name for kw in ai_cert_keywords):
            ai_cert_count += 1
    features["ai_cert_count"] = min(1.0, ai_cert_count / 3.0)

    # ─────────────────────────────────────────
    # 4. BEHAVIORAL / PLATFORM SIGNALS
    # ─────────────────────────────────────────

    # Recency of activity (exponential decay, halflife=60 days)
    last_active = _parse_date(signals.get("last_active_date"))
    features["recency_score"] = _recency_score(last_active, halflife_days=60)

    # Open-to-work flag (direct availability signal)
    features["open_to_work"] = 1.0 if signals.get("open_to_work_flag", False) else 0.0

    # Recruiter responsiveness
    response_rate = float(signals.get("recruiter_response_rate", 0.0))
    avg_response_hrs = float(signals.get("avg_response_time_hours", 168.0))
    # Fast responders (< 24 hrs) score higher
    response_speed = _normalize(-math.log1p(avg_response_hrs), -math.log1p(720), -math.log1p(0.5))
    features["responsiveness"] = response_rate * 0.6 + response_speed * 0.4

    # Platform trust signals
    verified_email = float(signals.get("verified_email", False))
    verified_phone = float(signals.get("verified_phone", False))
    linkedin = float(signals.get("linkedin_connected", False))
    features["platform_trust"] = (verified_email + verified_phone + linkedin) / 3.0

    # Profile completeness
    features["profile_completeness"] = float(signals.get("profile_completeness_score", 50.0)) / 100.0

    # Recruiter engagement (interest from market)
    profile_views = int(signals.get("profile_views_received_30d", 0))
    saved_by = int(signals.get("saved_by_recruiters_30d", 0))
    features["recruiter_interest"] = _normalize(profile_views + saved_by * 3, 0, 60)

    # Interview reliability
    features["interview_reliability"] = float(signals.get("interview_completion_rate", 0.5))

    # Offer acceptance rate (-1 = no history, treat as neutral)
    offer_rate = float(signals.get("offer_acceptance_rate", -1))
    features["offer_reliability"] = max(0.0, offer_rate) if offer_rate >= 0 else 0.5

    # Notice period: JD wants ≤30 days (buys out up to 30 days)
    notice_days = int(signals.get("notice_period_days", 60))
    buyout = jd.get("notice_buyout_days", 30)
    if notice_days <= buyout:
        features["notice_fit"] = 1.0
    else:
        features["notice_fit"] = math.exp(-(notice_days - buyout) / buyout)

    # ─────────────────────────────────────────
    # 5. LOGISTICAL FIT
    # ─────────────────────────────────────────

    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_to_relocate = bool(signals.get("willing_to_relocate", False))
    preferred_modes = {m.lower() for m in jd.get("preferred_work_modes", [])}
    work_mode = signals.get("preferred_work_mode", "").lower()

    # Location fit: India-based in preferred cities = 1.0, willing to relocate = 0.7, outside India = 0.3
    if any(city in location for city in PREFERRED_LOCATIONS):
        features["location_fit"] = 1.0
    elif country == "india" and willing_to_relocate:
        features["location_fit"] = 0.75
    elif country == "india":
        features["location_fit"] = 0.55
    elif willing_to_relocate:
        features["location_fit"] = 0.35
    else:
        features["location_fit"] = 0.20

    # Work mode fit
    features["work_mode_fit"] = 1.0 if work_mode in preferred_modes else 0.4

    # Applications activity (signal of active job search)
    apps = int(signals.get("applications_submitted_30d", 0))
    features["job_search_activity"] = min(1.0, apps / 5.0)

    return features


def extract_candidate_text(candidate: Dict[str, Any], max_tokens: int = 512) -> str:
    """
    Create a rich text representation of the candidate for embedding.
    Designed to capture semantic meaning beyond keyword matching.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    certs = candidate.get("certifications", [])

    parts = []

    # Professional identity
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    headline = profile.get("headline", "")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "")

    parts.append(f"{title} at {company} | {headline} | {yoe} years experience | {location}")

    # Summary (most semantically rich part)
    summary = profile.get("summary", "")
    if summary:
        parts.append(summary[:400])

    # Career history — last 3 roles (title + description)
    for role in career[:3]:
        role_title = role.get("title", "")
        role_company = role.get("company", "")
        desc = role.get("description", "")
        duration = role.get("duration_months", 0)
        parts.append(f"{role_title} at {role_company} ({duration}mo): {desc[:250]}")

    # Skills — prioritize by proficiency and endorsements
    sorted_skills = sorted(
        skills,
        key=lambda s: (PROFICIENCY_SCORE.get(s.get("proficiency", "beginner"), 0.2),
                       s.get("endorsements", 0)),
        reverse=True
    )
    skill_strs = [
        f"{s['name']} ({s.get('proficiency','?')})"
        for s in sorted_skills[:20]
    ]
    if skill_strs:
        parts.append("Skills: " + ", ".join(skill_strs))

    # Education
    for edu in education[:2]:
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        institution = edu.get("institution", "")
        parts.append(f"Education: {degree} {field} from {institution}")

    # Certifications
    if certs:
        cert_names = [c.get("name", "") for c in certs[:4]]
        parts.append("Certifications: " + ", ".join(cert_names))

    full_text = " | ".join(filter(None, parts))
    # Rough character limit to stay within token budget (~4 chars/token)
    return full_text[:max_tokens * 4]


def compute_career_quality_score(features: Dict[str, float]) -> float:
    """
    Composite career quality score from extracted features.
    Combines role seniority, company type, AI trajectory, and stability.

    KEY: ai_signal dominates (0.55 weight). Non-AI titles get a hard multiplier
    penalty so they can't float high on company/behavioral signals alone.
    """
    # AI role ratio and description match — this is the dominant signal
    ai_signal = (features["ai_role_ratio"] * 0.40 +
                 features["ai_description_ratio"] * 0.35 +
                 features["is_ai_title"] * 0.25)

    # Company quality
    company_quality = (features["product_company_ratio"] * 0.5 +
                       features["company_prestige"] * 0.3 +
                       features["tenure_stability"] * 0.2)

    # Experience fit
    exp_fit = features["experience_band_score"]

    # Education bonus
    edu_signal = (features["edu_tier"] * 0.4 +
                  features["edu_field_relevance"] * 0.4 +
                  features["edu_degree_relevance"] * 0.2)

    # Certification bonus (small)
    cert_bonus = features["ai_cert_count"] * 0.1

    # Pure services penalty (JD explicitly disqualifies all-services careers)
    services_penalty = features["pure_services_penalty"] * 0.4

    raw = (ai_signal * 0.55 +        # RAISED: AI trajectory is the main signal
           company_quality * 0.25 +  # Company type still matters
           exp_fit * 0.12 +
           edu_signal * 0.08 +
           cert_bonus) - services_penalty

    # Hard multiplier: if current title is NOT an AI role and AI description
    # evidence is also thin, cap career score (prevents non-AI candidates
    # from floating high on tenure/company signals alone)
    if features["is_ai_title"] < 0.5 and features["ai_description_ratio"] < 0.3:
        raw *= 0.55

    return max(0.0, min(1.0, raw))


def compute_skill_score(features: Dict[str, float]) -> float:
    """
    Composite skill score from raw features.
    Weighted: raw skill depth > assessment > github.
    """
    raw = (features["skill_raw_score"] * 0.60 +
           features["core_skill_count"] * 0.20 +
           features["assessment_score"] * 0.10 +
           features["github_score"] * 0.10)
    return max(0.0, min(1.0, raw))


def compute_behavioral_score(features: Dict[str, float]) -> float:
    """
    Composite behavioral engagement score from platform signals.
    Availability × Responsiveness × Trust × Reliability.
    """
    # Availability: open-to-work + recent activity
    availability = (features["open_to_work"] * 0.5 +
                    features["recency_score"] * 0.5)

    # Responsiveness: recruiter response + platform trust
    responsiveness = (features["responsiveness"] * 0.6 +
                      features["platform_trust"] * 0.2 +
                      features["profile_completeness"] * 0.2)

    # Practical logistics
    logistics = (features["notice_fit"] * 0.4 +
                 features["location_fit"] * 0.35 +
                 features["work_mode_fit"] * 0.15 +
                 features["job_search_activity"] * 0.10)

    # Reliability track record
    reliability = (features["interview_reliability"] * 0.5 +
                   features["offer_reliability"] * 0.3 +
                   features["recruiter_interest"] * 0.2)

    raw = (availability * 0.30 +
           responsiveness * 0.25 +
           logistics * 0.30 +
           reliability * 0.15)

    return max(0.0, min(1.0, raw))


def features_to_subscores(features: Dict[str, float]) -> Dict[str, float]:
    """Compute the three sub-scores from features. Semantic is added externally."""
    return {
        "skill_score": compute_skill_score(features),
        "career_score": compute_career_quality_score(features),
        "behavioral_score": compute_behavioral_score(features),
    }
