"""
scoring.py — Player scoring and Dream11 selection nodes.

These nodes are ONLY activated on dream11 query paths.
They operate on structured player_stats extracted from batting/form context,
not on raw retrieved text.
"""

import re


# ── Dream11 scoring weights ────────────────────────────────────────────────
# Weights are intentionally not equal — recent form matters more than career
# average for a single match pick.

_BATTING_WEIGHTS = {
    "recent_avg":    0.35,   # average over last 5 matches
    "strike_rate":   0.35,   # career SR (proxy for explosiveness)
    "career_avg":    0.20,   # career average (floor quality signal)
    "centuries":     0.10,   # bonus for match-winners
}

_BOWLING_WEIGHTS = {
    "wickets_per_game": 0.45,  # recent form wickets / games
    "economy":          0.35,  # lower is better — inverted below
    "career_avg":       0.20,
}

# Economy scale: 6.0 is excellent (score 10), 10.0 is poor (score 0)
_ECONOMY_MIN = 6.0
_ECONOMY_MAX = 10.0


def _score_batter(stats: dict) -> float:
    """Compute a Dream11 batting score (0–100 scale)."""
    recent_avg   = float(stats.get("recent_avg", 0))
    strike_rate  = float(stats.get("strike_rate", 100))
    career_avg   = float(stats.get("career_avg", 0))
    centuries    = float(stats.get("centuries", 0))

    # Normalise each component to a 0–100 scale before weighting
    # These ceilings are generous for T20 — adjust if needed
    norm_recent  = min(recent_avg / 80.0, 1.0) * 100
    norm_sr      = min(strike_rate / 200.0, 1.0) * 100
    norm_avg     = min(career_avg / 60.0, 1.0) * 100
    norm_100s    = min(centuries / 7.0, 1.0) * 100  # 7 is Kohli's record

    return (
        norm_recent  * _BATTING_WEIGHTS["recent_avg"] +
        norm_sr      * _BATTING_WEIGHTS["strike_rate"] +
        norm_avg     * _BATTING_WEIGHTS["career_avg"] +
        norm_100s    * _BATTING_WEIGHTS["centuries"]
    )


def _score_bowler(stats: dict) -> float:
    """Compute a Dream11 bowling score (0–100 scale)."""
    wkts_per_game = float(stats.get("wickets_per_game", 0))
    economy       = float(stats.get("economy", 8.0))
    career_avg    = float(stats.get("career_avg", 30.0))

    norm_wkts  = min(wkts_per_game / 3.0, 1.0) * 100  # 3 wkts/game = perfect
    # Economy: lower is better — map [6, 10] → [100, 0]
    norm_econ  = max(0.0, (_ECONOMY_MAX - economy) / (_ECONOMY_MAX - _ECONOMY_MIN)) * 100
    norm_avg   = min(max(0.0, (40.0 - career_avg) / 40.0), 1.0) * 100  # lower avg = better

    return (
        norm_wkts  * _BOWLING_WEIGHTS["wickets_per_game"] +
        norm_econ  * _BOWLING_WEIGHTS["economy"] +
        norm_avg   * _BOWLING_WEIGHTS["career_avg"]
    )


def compute_scores(state: dict) -> dict:
    """
    Scoring node — converts raw player_stats into Dream11 scores.

    Reads:  state["player_stats"]  — populated upstream by a feature extractor
    Writes: state["player_scores"] — {player_name: float}
    """
    player_stats = state.get("player_stats", {})
    scores: dict[str, float] = {}

    for player, stats in player_stats.items():
        role = stats.get("role", "batting").lower()
        if "bowl" in role:
            scores[player] = round(_score_bowler(stats), 2)
        else:
            scores[player] = round(_score_batter(stats), 2)

    state["player_scores"] = scores
    return state


def dream11_selector(state: dict) -> dict:
    """
    Dream11 selection node — picks the top 11 players by score.

    Writes: state["dream11_team"] — [(player_name, score), ...]

    The synthesis_node reads this key and incorporates it into the final answer.
    """
    scores = state.get("player_scores", {})

    if not scores:
        state["dream11_team"] = []
        return state

    top_11 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:11]
    state["dream11_team"] = top_11

    return state