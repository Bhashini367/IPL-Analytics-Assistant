"""
validation.py — ValidationNode for the IPL LangGraph RAG system.

Purpose: detect when two retrieved contexts contain contradicting facts
about the same entity.  This is triggered by Section 11 of the dataset
(conflicting primary vs secondary source values).

Design principle: rule-based check first, LLM only as a fallback.
The LLM check is a last resort — it is intentionally narrow and cheap.
"""

import re
from groq import Groq
from config import GROQ_API_KEY

_client = Groq(api_key=GROQ_API_KEY)

# ── Known conflict pairs from Section 11 ──────────────────────────────────
# Each entry: (entity, field, primary_value, secondary_value, description)
# These are checked with simple string matching — no LLM needed.

KNOWN_CONFLICTS = [
    ("virat kohli",     "career runs",    "7263", "7084",
     "Kohli career runs: primary=7263, secondary=7084"),
    ("yuzvendra chahal","career wickets", "205",  "187",
     "Chahal career wickets: primary=205, secondary=187"),
    ("mi vs csk",       "total matches",  "35",   "33",
     "MI vs CSK total matches: primary=35, secondary=33"),
    ("ms dhoni",        "matches played", "250",  "240",
     "Dhoni matches played: primary=250, secondary=240"),
    ("highest team score","record",       "287",  "263",
     "Highest team score: primary=287/3 (SRH 2024), secondary=263/5 (RCB 2013)"),
    ("best bowling figures","record",     "6/12", "6/14",
     "Best bowling figures: primary=6/12 (Alzarri), secondary=6/14 (Tanvir)"),
]


def _contexts(state: dict) -> str:
    """Concatenate all retrieved context strings for pattern matching."""
    keys = [
        "batting_context", "bowling_context", "h2h_context",
        "venue_context", "form_context", "records_context", "team_profile_context",
    ]
    return " ".join(state.get(k, "") for k in keys).lower()


def _rule_based_check(combined: str) -> tuple[bool, str]:
    """
    Check for known conflicting values appearing together in context.

    Returns (conflict_detected: bool, detail: str).
    """
    for entity, field, primary, secondary, description in KNOWN_CONFLICTS:
        # Only flag if BOTH values appear — one value alone is fine
        if primary.lower() in combined and secondary.lower() in combined:
            return True, (
                f"Conflict detected — {description}. "
                f"Trust the primary source ({primary}). "
                f"Secondary source value ({secondary}) may be outdated or incomplete."
            )
    return False, ""


def _llm_check(state: dict) -> tuple[bool, str]:
    """
    LLM fallback: scan contexts for any numeric inconsistency the rule set missed.

    Intentionally narrow prompt — we only want YES/NO + a one-line reason.
    """
    # Only send a small excerpt to keep token cost low
    batting  = state.get("batting_context", "")[:400]
    bowling  = state.get("bowling_context", "")[:400]
    records  = state.get("records_context", "")[:400]

    if not any([batting, bowling, records]):
        return False, ""

    prompt = f"""You are a fact-checker for an IPL cricket database.
Check if the following context snippets contain contradicting values
for the same player or team statistic (e.g. two different run totals for the same player).

Batting context: {batting}
Bowling context: {bowling}
Records context: {records}

Reply with EXACTLY one of:
  YES_CONFLICT: <one-sentence description of the conflict>
  NO_CONFLICT"""

    try:
        response = _client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("YES_CONFLICT"):
            detail = text.replace("YES_CONFLICT:", "").strip()
            return True, f"LLM detected conflict: {detail}"
    except Exception:
        pass  # Silently skip — do not let validation crash the graph

    return False, ""


def validation_node(state: dict) -> dict:
    """
    ValidationNode — runs rule-based conflict check, then LLM if needed.

    Writes:
        state["conflict_detected"] — bool
        state["conflict_detail"]   — human-readable description (empty if no conflict)

    The synthesis_node reads conflict_detected and incorporates the detail
    into the answer when True.
    """
    combined = _contexts(state)

    # 1. Fast rule-based check — no API call
    conflict, detail = _rule_based_check(combined)

    # 2. LLM fallback only when rule check found nothing
    if not conflict:
        conflict, detail = _llm_check(state)

    state["conflict_detected"] = conflict
    state["conflict_detail"]   = detail

    # If a conflict was detected, append the detail to the answer so the user sees it
    if conflict and state.get("answer"):
        state["answer"] = (
            state["answer"]
            + f"\n\n⚠️ Data conflict detected: {detail}"
        )
    elif conflict:
        state["answer"] = f"⚠️ Data conflict detected: {detail}"

    return state