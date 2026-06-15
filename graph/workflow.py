"""
workflow.py — LangGraph graph construction for the IPL Intelligence Assistant.

Graph topology (varies by query_type):

  Simple queries (batting / bowling / records / team_profile / venue / form / h2h):
    router → rewrite → [specialist node] → validate → synthesis → END

  Prediction:
    router → rewrite → h2h → venue → form → validate → synthesis → END

  Dream11:
    router → rewrite → form → batting → bowling → venue → scoring → dream11 → validate → synthesis → END

  Fallback (out-of-corpus / out-of-scope):
    router → synthesis → END   (no retrieval, no validation)
"""

from langgraph.graph import StateGraph, END

from graph.state import IPLState
from graph.router import router_node
from graph.rewrite import rewrite_node
from graph.nodes import (
    batting_node,
    bowling_node,
    h2h_node,
    venue_node,
    form_node,
    records_node,
    team_profile_node,
    synthesis_node,
)
from graph.scoring import compute_scores, dream11_selector
from graph.validation import validation_node


# ── Routing functions ──────────────────────────────────────────────────────

def route_after_router(state: dict) -> str:
    """
    Determines the first specialist node (or fallback) after the router.

    All paths go through rewrite first — this function is the edge decision
    AFTER rewrite.
    """
    qt = state.get("query_type", "batting")

    if qt == "fallback":        return "fallback_synthesis"
    if qt == "team_profile":    return "team_profile"
    if qt == "batting":         return "batting"
    if qt == "bowling":         return "bowling"
    if qt == "records":         return "records"
    if qt == "venue":           return "venue"
    if qt == "form":            return "form"
    if qt == "h2h":             return "h2h"
    if qt == "comparison":      return "batting"   # called once; synthesis handles 2-entity logic
    if qt == "prediction":      return "h2h"       # prediction path: h2h → venue → form
    if qt == "dream11":         return "form"      # dream11 path: form → batting → bowling → venue

    return "batting"  # safe default


def route_after_batting(state: dict) -> str:
    """After batting: dream11 continues to bowling; everything else goes to validate."""
    if state.get("query_type") == "dream11":
        return "bowling"
    return "validate"


def route_after_bowling(state: dict) -> str:
    """After bowling: dream11 continues to venue; everything else goes to validate."""
    if state.get("query_type") == "dream11":
        return "venue"
    return "validate"


def route_after_venue(state: dict) -> str:
    """After venue: dream11 goes to scoring; prediction already has form; others validate."""
    if state.get("query_type") == "dream11":
        return "scoring"
    return "validate"


def route_after_h2h(state: dict) -> str:
    """After h2h: prediction continues to venue; plain h2h goes to validate."""
    if state.get("query_type") == "prediction":
        return "venue_prediction"
    return "validate"


def route_after_venue_prediction(state: dict) -> str:
    """Prediction path: after venue, always go to form."""
    return "form_prediction"


# ── Alias nodes ────────────────────────────────────────────────────────────
# LangGraph requires unique node names per path segment.
# venue_prediction and form_prediction are the same functions,
# just registered under different names so the graph stays acyclic.

def venue_prediction_node(state: dict) -> dict:
    return venue_node(state)


def form_prediction_node(state: dict) -> dict:
    return form_node(state)


def fallback_synthesis_node(state: dict) -> dict:
    """Short-circuit synthesis for out-of-corpus queries — no LLM retrieval."""
    state["answer"] = (
        "I could not find this information in the dataset. "
        "This query may reference data that is out of scope (future predictions, "
        "player salary, awards, or entities not in the IPL 2024 dataset). "
        "Please check official IPL or BCCI sources."
    )
    return state


# ── Graph construction ─────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(IPLState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    graph.add_node("router",             router_node)
    graph.add_node("rewrite",            rewrite_node)

    # Specialist retrieval nodes
    graph.add_node("team_profile",       team_profile_node)
    graph.add_node("batting",            batting_node)
    graph.add_node("bowling",            bowling_node)
    graph.add_node("h2h",                h2h_node)
    graph.add_node("venue",              venue_node)
    graph.add_node("form",               form_node)
    graph.add_node("records",            records_node)

    # Prediction-path aliases (separate nodes to keep graph acyclic)
    graph.add_node("venue_prediction",   venue_prediction_node)
    graph.add_node("form_prediction",    form_prediction_node)

    # Dream11 path
    graph.add_node("scoring",            compute_scores)
    graph.add_node("dream11",            dream11_selector)

    # Terminal nodes
    graph.add_node("validate",           validation_node)
    graph.add_node("synthesis",          synthesis_node)
    graph.add_node("fallback_synthesis", fallback_synthesis_node)

    # ── Edges ──────────────────────────────────────────────────────────────

    # Entry: router → rewrite always
    graph.set_entry_point("router")
    graph.add_edge("router", "rewrite")

    # Rewrite → conditional dispatch to first specialist node
    graph.add_conditional_edges(
        "rewrite",
        route_after_router,
        {
            "team_profile":      "team_profile",
            "batting":           "batting",
            "bowling":           "bowling",
            "h2h":               "h2h",
            "venue":             "venue",
            "form":              "form",
            "records":           "records",
            "fallback_synthesis":"fallback_synthesis",
        },
    )

    # Simple paths: one node → validate → synthesis → END
    graph.add_edge("team_profile",   "validate")
    graph.add_edge("records",        "validate")
    graph.add_edge("form",           "validate")

    # Batting: conditional — dream11 continues, others stop at validate
    graph.add_conditional_edges(
        "batting",
        route_after_batting,
        {"bowling": "bowling", "validate": "validate"},
    )

    # Bowling: conditional — dream11 continues, others stop at validate
    graph.add_conditional_edges(
        "bowling",
        route_after_bowling,
        {"venue": "venue", "validate": "validate"},
    )

    # Venue: conditional — dream11 goes to scoring, others stop at validate
    graph.add_conditional_edges(
        "venue",
        route_after_venue,
        {"scoring": "scoring", "validate": "validate"},
    )

    # H2H: conditional — prediction continues, plain h2h stops at validate
    graph.add_conditional_edges(
        "h2h",
        route_after_h2h,
        {"venue_prediction": "venue_prediction", "validate": "validate"},
    )

    # Prediction path: venue_prediction → form_prediction → validate
    graph.add_edge("venue_prediction", "form_prediction")
    graph.add_edge("form_prediction",  "validate")

    # Dream11 path: scoring → dream11 → validate
    graph.add_edge("scoring",  "dream11")
    graph.add_edge("dream11",  "validate")

    # All paths converge at validate → synthesis → END
    graph.add_edge("validate",          "synthesis")
    graph.add_edge("synthesis",         END)
    graph.add_edge("fallback_synthesis", END)

    return graph


app_graph = build_graph().compile()