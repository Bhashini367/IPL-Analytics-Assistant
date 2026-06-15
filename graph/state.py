from typing import TypedDict, Optional


class IPLState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    query: str                   # original user query (never mutated)
    rewritten_query: str         # optional improved query from rewrite_node

    # ── Router outputs ───────────────────────────────────────────────────────
    query_type: str              # batting | bowling | h2h | venue | form |
                                 # records | comparison | dream11 | prediction |
                                 # team_profile | fallback
    entities: list               # extracted player / team names
    active_nodes: list           # nodes the router decided to activate

    # ── Raw retrieval contexts (set by specialist nodes) ──────────────────────
    batting_context: str
    bowling_context: str
    h2h_context: str
    venue_context: str
    form_context: str
    records_context: str
    team_profile_context: str

    # ── Derived / computed (set by scoring_node, dream11 path only) ──────────
    player_scores: dict          # {player_name: float}  — Dream11 path only
    dream11_team: list           # [(player, score), ...]  — Dream11 path only
    pitch_type: str              # "spin" | "pace"  — set by venue_node

    # ── Validation ───────────────────────────────────────────────────────────
    conflict_detected: bool      # set by validation_node
    conflict_detail: str         # human-readable description of conflict

    # ── Output ───────────────────────────────────────────────────────────────
    answer: str