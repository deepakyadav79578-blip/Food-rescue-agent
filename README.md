# 🍲 Food Rescue Agent
## 🌐 Live Demo

https://food-rescue-agent.onrender.com

An agentic AI system (LangGraph + Groq + FastAPI) that connects restaurants with
surplus food to nearby shelters and food banks — before the food becomes unusable.

## Problem

Restaurants often discard surplus food that is safe to eat but cannot be sold.
Meanwhile shelters and food banks struggle to receive timely donations. There is
no efficient system connecting the two before the food spoils.

## How it works

```
donation → safety_gate → router ──► critical  ──┐
                                ├─► same_day  ──┼─► shelter_match → done
                                ├─► flexible  ──┘
                                └─► rejected  ──► end (unsafe food blocked)
```

1. **Safety gate (guardrail before AI):** hard keyword rules reject unsafe food
   (partially eaten, expired, left out overnight) *before* any LLM is called.
2. **Router:** an LLM (Llama 3.3 70B via Groq) classifies the donation by urgency —
   `critical` (cooked food, ~3 hr window), `same_day`, or `flexible` (packaged).
3. **Shelter match:** finds an open shelter, preferring the same locality.

## Run locally

```bash
git clone https://github.com/deepakyadav79578-blip/Food-rescue-agent.git
cd food-rescue-agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then paste your Groq API key into .env

uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the demo form, or http://127.0.0.1:8000/docs
for the interactive API docs (Swagger UI).

## API

| Method | Endpoint    | Description                                  |
|--------|-------------|----------------------------------------------|
| POST   | `/donate`   | Submit surplus food, get urgency + shelter   |
| GET    | `/shelters` | List all shelters and their status           |
| GET    | `/health`   | Health check                                 |

Example:

```bash
curl -X POST http://127.0.0.1:8000/donate \
  -H "Content-Type: application/json" \
  -d '{
    "restaurant": "Spice Villa",
    "location": "Andheri",
    "food_description": "20 boxes of cooked biryani, made 1 hour ago",
    "quantity": "feeds 50 people"
  }'
```

## Tech stack

- **LangGraph** — agent state machine (safety gate → router → matcher)
- **Groq / Llama 3.3 70B** — fast LLM classification
- **FastAPI** — backend REST API
- **Pandas** — shelter data (`data/shelters.csv`)

## Roadmap

- [ ] Real distance-based matching (geocoding)
- [ ] Pickup notifications to shelters (email/SMS)
- [ ] Capacity tracking & retry queue when no shelter is free
- [ ] Database instead of CSV
