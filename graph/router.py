import json
import re
from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

# Nodes the router is allowed to activate.
# Any LLM-hallucinated name that isn't in this set is silently dropped.
VALID_NODES = {
    "batting", "bowling", "h2h", "venue",
    "form", "records", "team_profile",
}

# Query types that must always fall back without touching specialist nodes.
FALLBACK_TYPES = {"fallback", "out_of_scope", "out_of_corpus"}


def router_node(state: dict) -> dict:
    query = state["query"]

    prompt = f"""You are a routing agent for an IPL cricket knowledge base.

The knowledge base contains ONLY:
  team profiles, batting stats, bowling stats, head-to-head records,
  venue/pitch reports, recent form (last 5 matches), IPL records.

Classify the query and return ONLY a JSON object — no markdown, no explanation.

JSON schema:
{{
  "query_type": "batting | bowling | h2h | venue | form | records | team_profile | comparison | prediction | dream11 | fallback",
  "entities": ["list of player or team names mentioned"],
  "active_nodes": ["list of nodes to activate"]
}}

Routing rules (follow exactly):
- team_profile  → ["team_profile"]
- batting       → ["batting"]
- bowling       → ["bowling"]
- h2h           → ["h2h"]
- venue         → ["venue"]
- form          → ["form"]
- records       → ["records"]
- comparison    → ["batting"] (called twice with different entities — do NOT add records)
- prediction    → ["h2h", "venue", "form"]
- dream11       → ["form", "batting", "bowling", "venue"]
- fallback      → []   (use when query is out-of-scope, future prediction, or not in corpus)

Examples of fallback queries:
  "Predict the IPL 2025 champion"
  "What is Sachin Tendulkar's record here?"
  "What is the BCCI's net worth?"
  "Tell me everything about cricket"

Query: {query}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if the model wrapped its output
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

        data = json.loads(raw)

    except (json.JSONDecodeError, KeyError, IndexError):
        # Safe default: treat as a batting query so the graph doesn't crash
        data = {
            "query_type": "batting",
            "entities": [],
            "active_nodes": ["batting"],
        }

    query_type = data.get("query_type", "batting")
    entities = data.get("entities", [])

    # Validate and sanitise the active_nodes list
    if query_type in FALLBACK_TYPES:
        active_nodes = []
    else:
        raw_nodes = data.get("active_nodes", ["batting"])
        active_nodes = [n for n in raw_nodes if n in VALID_NODES]
        if not active_nodes:
            active_nodes = ["batting"]  # last-resort default

    state["query_type"] = query_type
    state["entities"] = entities
    state["active_nodes"] = active_nodes

    return state