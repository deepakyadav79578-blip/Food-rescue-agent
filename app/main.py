"""FastAPI backend for the Food Rescue Agent."""

from dotenv import load_dotenv
load_dotenv()  # must run BEFORE importing agent (it needs GROQ_API_KEY)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import graph, shelters_df

app = FastAPI(title="Food Rescue Agent", version="1.0")


class DonationRequest(BaseModel):
    restaurant: str
    location: str
    food_description: str
    quantity: str


class DonationResponse(BaseModel):
    restaurant: str
    urgency: str
    reasoning: str
    assigned_shelter: str


@app.post("/donate", response_model=DonationResponse)
def donate(req: DonationRequest):
    """Submit surplus food — the agent classifies it and matches a shelter."""
    final_state = graph.invoke({
        "restaurant": req.restaurant,
        "location": req.location,
        "food_description": req.food_description,
        "quantity": req.quantity,
        "urgency": "",
        "reasoning": "",
        "assigned_shelter": "",
    })
    return DonationResponse(
        restaurant=final_state["restaurant"],
        urgency=final_state["urgency"],
        reasoning=final_state["reasoning"],
        assigned_shelter=final_state["assigned_shelter"],
    )


@app.get("/shelters")
def list_shelters():
    """List all shelters and their current status."""
    return shelters_df.to_dict(orient="records")


@app.get("/health")
def health():
    return {"status": "ok"}


# Simple demo frontend at /
app.mount("/", StaticFiles(directory="static", html=True), name="static")
