"""
nodes.py — Specialist retrieval nodes for the IPL LangGraph RAG system.

Each node is responsible for ONE section of the dataset.
Nodes read from state["query"] (or state["rewritten_query"] if available)
and write to their dedicated context key.  They never read each other's keys.
"""

from rag.retriever import get_retriever

# Single shared retriever instance
retriever = get_retriever()

def _query(state: dict) -> str:
    """Use rewritten query when available, fall back to original."""
    return state.get("rewritten_query") or state["query"]


# ── Retrieval nodes ────────────────────────────────────────────────────────

def batting_node(state: dict) -> dict:
    """BattingStatsNode — retrieves runs, average, SR, centuries, role."""
    q = f"batting stats runs average strike rate centuries {_query(state)}"
    docs = retriever.invoke(q)
    state["batting_context"] = "\n".join(d.page_content for d in docs)
    return state


def bowling_node(state: dict) -> dict:
    """BowlingStatsNode — retrieves wickets, economy, bowling average, best figures."""
    q = f"bowling wickets economy average best figures {_query(state)}"
    docs = retriever.invoke(q)
    state["bowling_context"] = "\n".join(d.page_content for d in docs)
    return state


def h2h_node(state: dict) -> dict:
    """H2HNode — retrieves head-to-head records between two teams."""
    base = _query(state)
    docs = retriever.invoke(f"head to head record matches wins {base}")
    state["h2h_context"] = "\n".join(d.page_content for d in docs)
    return state


def venue_node(state: dict) -> dict:
    """VenueNode — retrieves pitch report and derives pitch_type for conditional routing."""
    base = _query(state)
    docs = retriever.invoke(f"venue pitch report batting bowling strategy {base}")
    context = "\n".join(d.page_content for d in docs)
    state["venue_context"] = context

    # Score spin vs pace keywords in retrieved context to set pitch_type.
    # Counting occurrences is more robust than a single keyword check.
    lower = context.lower()
    spin_score = sum(lower.count(k) for k in
                     ["spin", "turning", "slow", "dry", "off-spin", "leg-spin", "spinner"])
    pace_score = sum(lower.count(k) for k in
                     ["pace", "bounce", "seam", "swing", "yorker", "fast bowl"])
    state["pitch_type"] = "spin" if spin_score >= pace_score else "pace"

    return state


def form_node(state: dict) -> dict:
    """FormNode — retrieves 2024 recent form (last 5 matches).
    
    IMPORTANT: The '2024 season' prefix biases retrieval toward form chunks
    tagged with season=2024 metadata.  Without this, the retriever may surface
    historical averages instead of recent match data.
    """
    base = _query(state)
    q = f"2024 season recent form last 5 matches {base}"
    docs = retriever.invoke(q)
    state["form_context"] = "\n".join(d.page_content for d in docs)
    return state


def records_node(state: dict) -> dict:
    """RecordsNode — retrieves exact IPL records and milestones.
    
    No LLM reasoning here — exact fact retrieval only.
    The synthesis_node will format the answer.
    """
    q = f"IPL record milestone highest most {_query(state)}"
    docs = retriever.invoke(q)
    state["records_context"] = "\n".join(d.page_content for d in docs)
    return state


def team_profile_node(state: dict) -> dict:
    """TeamProfileNode — retrieves team identity, captain, coach, home ground."""
    q = f"IPL team profile captain coach home ground {_query(state)}"
    docs = retriever.invoke(q)
    state["team_profile_context"] = "\n".join(d.page_content for d in docs)
    return state


# ── Synthesis node ─────────────────────────────────────────────────────────

from groq import Groq
from config import GROQ_API_KEY

_client = Groq(api_key=GROQ_API_KEY)

# Context keys in priority order — higher-specificity keys first
_CONTEXT_KEYS = [
    ("team_profile_context", "TEAM PROFILES"),
    ("batting_context",      "BATTING STATS"),
    ("bowling_context",      "BOWLING STATS"),
    ("h2h_context",          "HEAD-TO-HEAD RECORDS"),
    ("venue_context",        "VENUE & PITCH REPORT"),
    ("form_context",         "RECENT FORM (2024)"),
    ("records_context",      "IPL RECORDS"),
]


def synthesis_node(state: dict) -> dict:
    """SynthesisNode — combines all retrieved contexts and calls the LLM once."""

    # Build context block from whichever keys are populated
    context_parts = []
    for key, label in _CONTEXT_KEYS:
        value = state.get(key, "").strip()
        if value:
            context_parts.append(f"[{label}]\n{value}")

    if context_parts:
        context_block = "\n\n".join(context_parts)
    else:
        # Nothing was retrieved — return fallback immediately without calling LLM
        state["answer"] = (
            "I could not find this information in the dataset. "
            "Please check official IPL or BCCI sources."
        )
        return state

    # Include Dream11 selection if the scoring node ran
    dream11_section = ""
    if state.get("dream11_team"):
        picks = ", ".join(f"{p} ({round(s, 1)})" for p, s in state["dream11_team"])
        dream11_section = f"\n\n[DREAM11 SCORING OUTPUT]\nTop picks: {picks}"

    prompt = f"""You are an IPL analyst answering questions using ONLY the context below.

Rules:
- Answer using ONLY the provided context.
- If the answer is not present, reply: "I could not find this information in the dataset."
- Do NOT invent players, statistics, match results, or Dream11 teams.
- If the query asks for a Dream11 team and dream11 scoring output is present, use it.
- If conflict_detected is True, flag both values and recommend verification.
- Be concise and specific.

CONTEXT:
{context_block}{dream11_section}

QUESTION: {state["query"]}

ANSWER:"""

    response = _client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    state["answer"] = response.choices[0].message.content.strip()
    return state