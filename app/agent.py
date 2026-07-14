"""Food Rescue Agent — LangGraph pipeline.

Same architecture as the Colab version, but with no input()/print():
state flows in from the API and results flow back out.
"""

import os
from typing import TypedDict, Literal

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------
# State
# ---------------------------------------------------------------

class DonationState(TypedDict):
    restaurant: str
    location: str
    food_description: str   # e.g. "20 boxes of cooked biryani, made 1 hour ago"
    quantity: str           # people can write "20 boxes" or "feeds fifty"
    urgency: str            # critical / same_day / flexible / rejected
    reasoning: str
    assigned_shelter: str


# ---------------------------------------------------------------
# Shelter data (loaded once at startup)
# ---------------------------------------------------------------

SHELTERS_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "shelters.csv")
shelters_df = pd.read_csv(SHELTERS_CSV)


# ---------------------------------------------------------------
# LLM router
# ---------------------------------------------------------------

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

ROUTER_SYSTEM_PROMPT = """You are a surplus food triage router for a food rescue
service. Based on the donation description, classify it into EXACTLY ONE category:

- critical: cooked/perishable food that must be picked up within ~3 hours
  (cooked rice, curries, dairy dishes, cut fruit, anything already prepared)
- same_day: food that lasts until end of day (bread, bakery items, salads,
  sandwiches, milk products still sealed)
- flexible: packaged/sealed/dry food with a long shelf life (canned goods,
  sealed packets, rice bags, biscuits, unopened bottles)
- rejected: food that is UNSAFE to donate (partially eaten, expired, left out
  overnight/many hours, spoiled smell, returned from customer tables)

Respond with ONLY one word: critical, same_day, flexible, or rejected.
"""

# Hard safety-net keywords — these always override the LLM, since donating
# unsafe food is unacceptable. Guardrail before AI.
UNSAFE_KEYWORDS = [
    "half eaten", "partially eaten", "leftover from plates", "expired",
    "left out overnight", "smells bad", "smells off", "spoiled", "mold", "mould",
]
CRITICAL_KEYWORDS = ["cooked", "curry", "rice", "dal", "biryani", "hot food", "dairy"]


def safety_gate_node(state: DonationState) -> DonationState:
    desc_lower = state["food_description"].lower()
    if any(kw in desc_lower for kw in UNSAFE_KEYWORDS):
        return {
            **state,
            "urgency": "rejected",
            "reasoning": "Unsafe keyword detected — donation rejected for food safety.",
        }
    return state


def router_node(state: DonationState) -> DonationState:
    if state.get("urgency") == "rejected":
        return state  # safety gate already decided

    desc_lower = state["food_description"].lower()

    if any(kw in desc_lower for kw in CRITICAL_KEYWORDS):
        return {
            **state,
            "urgency": "critical",
            "reasoning": "Perishable keyword detected — routed as critical for fast pickup.",
        }

    response = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Food: {state['food_description']}\nQuantity: {state['quantity']}"
        ),
    ])
    raw = response.content.strip().lower()

    # Defensive parsing in case the model adds extra words
    if "rejected" in raw:
        urgency = "rejected"
    elif "critical" in raw:
        urgency = "critical"
    elif "same_day" in raw or "same day" in raw:
        urgency = "same_day"
    elif "flexible" in raw:
        urgency = "flexible"
    else:
        urgency = "same_day"  # safe default: don't let food sit around

    return {
        **state,
        "urgency": urgency,
        "reasoning": f"LLM classified based on: '{state['food_description']}'",
    }


# ---------------------------------------------------------------
# Urgency branch nodes
# ---------------------------------------------------------------

def critical_node(state: DonationState) -> DonationState:
    return state

def same_day_node(state: DonationState) -> DonationState:
    return state

def flexible_node(state: DonationState) -> DonationState:
    return state

def rejected_node(state: DonationState) -> DonationState:
    return {**state, "assigned_shelter": "N/A — donation rejected for safety"}


# ---------------------------------------------------------------
# Shelter matching
# ---------------------------------------------------------------

def shelter_match_node(state: DonationState) -> DonationState:
    open_shelters = shelters_df[shelters_df["status"] == "open"]

    # Prefer shelters in the same location first
    local = open_shelters[
        open_shelters["location"].str.lower() == state["location"].lower()
    ]
    candidates = local if not local.empty else open_shelters

    if not candidates.empty:
        shelter = candidates.iloc[0]["shelter_name"]
        assigned = shelter
    else:
        assigned = "No shelter currently open — donation queued"

    return {**state, "assigned_shelter": assigned}


# ---------------------------------------------------------------
# Graph
# ---------------------------------------------------------------

def route_decision(state: DonationState) -> Literal["critical", "same_day", "flexible", "rejected"]:
    return state["urgency"]


builder = StateGraph(DonationState)

builder.add_node("safety_gate", safety_gate_node)
builder.add_node("router", router_node)
builder.add_node("critical", critical_node)
builder.add_node("same_day", same_day_node)
builder.add_node("flexible", flexible_node)
builder.add_node("rejected", rejected_node)
builder.add_node("shelter_match", shelter_match_node)

builder.set_entry_point("safety_gate")
builder.add_edge("safety_gate", "router")

builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "critical": "critical",
        "same_day": "same_day",
        "flexible": "flexible",
        "rejected": "rejected",
    },
)

builder.add_edge("critical", "shelter_match")
builder.add_edge("same_day", "shelter_match")
builder.add_edge("flexible", "shelter_match")
builder.add_edge("rejected", END)

builder.add_edge("shelter_match", END)

graph = builder.compile()
