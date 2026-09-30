"""TabPFN Titanic survival predictor exposed as an A2A (Agent2Agent) agent.

Endpoints:
  GET  /.well-known/agent-card.json  - A2A agent card (discovery)
  POST /                             - A2A JSON-RPC endpoint (protocol 1.0 and 0.3)
  GET  /health                       - liveness check

Run:  python main.py   (or: uvicorn main:app --host 0.0.0.0 --port 8080)
"""

import asyncio
import json
import logging
import os
import re
import secrets

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from a2a.helpers.proto_helpers import (
    get_data_parts,
    get_text_parts,
    new_message,
    new_data_part,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    APIKeySecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

import tabpfn_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tabpfn-a2a")

HOST = os.getenv("A2A_HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", os.getenv("A2A_PORT", "8080")))
# Public HTTPS URL Joule reaches this agent at (e.g. the Cloud Foundry route)
def default_public_url():
    # On Cloud Foundry, fall back to the app's first route
    uris = json.loads(os.getenv("VCAP_APPLICATION", "{}")).get("application_uris") or []
    return f"https://{uris[0]}" if uris else f"http://localhost:{PORT}"


PUBLIC_URL = (os.getenv("A2A_PUBLIC_URL") or default_public_url()).rstrip("/") + "/"
# Optional shared secret; when set, JSON-RPC calls must send it as the X-API-Key header
API_KEY = os.getenv("A2A_API_KEY")
API_KEY_HEADER = "X-API-Key"

HELP_TEXT = (
    "I predict Titanic passenger survival with TabPFN.\n"
    "Describe a passenger, e.g. \"Would a 30 year old woman in first class survive?\", "
    "or send JSON: {\"passengers\": [{\"Pclass\": 1, \"Sex\": \"female\", \"Age\": 30, "
    "\"SibSp\": 0, \"Parch\": 0, \"Fare\": 80}]}.\n"
    "Say \"evaluate the model\" to run the hold-out accuracy test."
)


# --------------------------------------------------
# Request parsing
# --------------------------------------------------

def passengers_from_data(data):
    if isinstance(data, dict):
        data = data.get("passengers", data.get("passenger", data))
    if isinstance(data, dict):
        data = [data]
    if isinstance(data, list) and all(isinstance(p, dict) for p in data):
        return data
    return None


def passengers_from_json_text(text):
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        return None
    try:
        return passengers_from_data(json.loads(match.group(1)))
    except json.JSONDecodeError:
        return None


def passenger_from_natural_language(text):
    """Best-effort extraction of a single passenger from free text."""
    t = text.lower()
    passenger = {}

    if re.search(r"\b(1st|first)\b|\bclass\s*1\b|\bpclass\s*[:=]?\s*1\b", t):
        passenger["Pclass"] = 1
    elif re.search(r"\b(2nd|second)\b|\bclass\s*2\b|\bpclass\s*[:=]?\s*2\b", t):
        passenger["Pclass"] = 2
    elif re.search(r"\b(3rd|third)\b|\bclass\s*3\b|\bpclass\s*[:=]?\s*3\b", t):
        passenger["Pclass"] = 3

    if re.search(r"\b(female|woman|women|girl|lady|she|her)\b", t):
        passenger["Sex"] = "female"
    elif re.search(r"\b(male|man|men|boy|gentleman|he|his)\b", t):
        passenger["Sex"] = "male"

    age = re.search(r"(\d{1,3})\s*(?:-\s*)?(?:years?|yrs?|y/o|yo)\b|\bage[d]?\s*[:=]?\s*(\d{1,3})\b", t)
    if age:
        passenger["Age"] = float(age.group(1) or age.group(2))

    fare = re.search(r"(?:fare|paid|ticket (?:cost|price))\D{0,10}(\d+(?:\.\d+)?)|[$£](\d+(?:\.\d+)?)", t)
    if fare:
        passenger["Fare"] = float(fare.group(1) or fare.group(2))

    sibsp = re.search(r"(\d+)\s*(?:siblings?|spouses?|sibsp)", t)
    if sibsp:
        passenger["SibSp"] = int(sibsp.group(1))
    elif re.search(r"\b(with (?:her|his) (?:husband|wife|spouse))\b", t):
        passenger["SibSp"] = 1

    parch = re.search(r"(\d+)\s*(?:parents?|children|kids?|parch)", t)
    if parch:
        passenger["Parch"] = int(parch.group(1))

    return passenger if "Pclass" in passenger and "Sex" in passenger else None


def format_predictions(results):
    lines = []
    for i, r in enumerate(results, start=1):
        p = r["passenger"]
        lines.append(
            f"{i}. {p['Sex']}, age {p['Age']:g}, class {p['Pclass']}, fare {p['Fare']:.2f}, "
            f"{p['SibSp']} sibling/spouse, {p['Parch']} parent/child: {r['prediction']} "
            f"({r['survival_probability']:.0%} survival probability)"
        )
    return "TabPFN survival prediction:\n" + "\n".join(lines)


def format_evaluation(result):
    return (
        f"TabPFN hold-out evaluation on {result['testing_rows']} passengers "
        f"(trained in-context on {result['training_rows']} rows): "
        f"accuracy {result['accuracy']:.0%}.\n"
        f"Predicted: {result['predictions']}\n"
        f"Actual:    {result['actual']}"
    )


# --------------------------------------------------
# A2A agent executor
# --------------------------------------------------

class TabPFNAgentExecutor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        parts = list(message.parts) if message else []
        text = "\n".join(get_text_parts(parts)).strip()

        try:
            reply_parts = await asyncio.to_thread(self.handle, text, get_data_parts(parts))
        except ValueError as e:
            reply_parts = [new_text_part(f"{e}\n\n{HELP_TEXT}")]
        except Exception:
            logger.exception("TabPFN request failed")
            reply_parts = [new_text_part("Sorry, the TabPFN prediction service failed. Please try again later.")]

        await event_queue.enqueue_event(new_message(
            parts=reply_parts,
            context_id=context.context_id,
            task_id=context.task_id,
        ))

    def handle(self, text, data_parts):
        passengers = None
        for data in data_parts:
            passengers = passengers_from_data(data)
            if passengers:
                break

        if not passengers and text:
            passengers = passengers_from_json_text(text)

        if not passengers and re.search(r"\b(evaluat\w*|accuracy|test the model|benchmark|demo)\b", text, re.I):
            result = tabpfn_service.evaluate_holdout()
            return [new_text_part(format_evaluation(result)), new_data_part(result)]

        if not passengers and text:
            passenger = passenger_from_natural_language(text)
            passengers = [passenger] if passenger else None

        if not passengers:
            return [new_text_part(HELP_TEXT)]

        results = tabpfn_service.predict_passengers(passengers)
        return [new_text_part(format_predictions(results)), new_data_part({"results": results})]

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancellation is not supported.")


# --------------------------------------------------
# Agent card
# --------------------------------------------------

def build_agent_card():
    card = AgentCard(
        name="TabPFN Titanic Survival Agent",
        description=(
            "Predicts whether Titanic passengers would have survived using a local TabPFN "
            "tabular foundation model, and evaluates the model's accuracy."
        ),
        version="1.0.0",
        # Same endpoint advertised for A2A 1.0 and 0.3 clients
        supported_interfaces=[
            AgentInterface(url=PUBLIC_URL, protocol_binding="JSONRPC", protocol_version="1.0"),
            AgentInterface(url=PUBLIC_URL, protocol_binding="JSONRPC", protocol_version="0.3"),
        ],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
        skills=[
            AgentSkill(
                id="predict_titanic_survival",
                name="Predict Titanic survival",
                description=(
                    "Predicts survival for one or more Titanic passengers. Inputs per passenger: "
                    "Pclass (1-3, required), Sex (male/female, required), Age, SibSp (siblings/spouses "
                    "aboard), Parch (parents/children aboard), Fare. Accepts natural language or JSON "
                    "{\"passengers\": [{...}]}."
                ),
                tags=["tabpfn", "prediction", "classification", "titanic", "machine learning"],
                examples=[
                    "Would a 30 year old woman in first class have survived the Titanic?",
                    "Predict survival for a 25 year old man in third class who paid $8",
                    '{"passengers": [{"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25}]}',
                ],
            ),
            AgentSkill(
                id="evaluate_model",
                name="Evaluate TabPFN model",
                description="Runs a hold-out test on the Titanic dataset and reports TabPFN's accuracy.",
                tags=["tabpfn", "evaluation", "accuracy"],
                examples=["Evaluate the model", "How accurate is the TabPFN model?"],
            ),
        ],
    )

    if API_KEY:
        card.security_schemes["apiKey"].CopyFrom(SecurityScheme(
            api_key_security_scheme=APIKeySecurityScheme(location="header", name=API_KEY_HEADER)
        ))
        card.security_requirements.append(SecurityRequirement(schemes={"apiKey": StringList()}))

    return card


# --------------------------------------------------
# App
# --------------------------------------------------

agent_card = build_agent_card()

request_handler = DefaultRequestHandler(
    agent_executor=TabPFNAgentExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)

app = FastAPI(title="TabPFN A2A Agent")

for route in create_agent_card_routes(agent_card):
    app.router.routes.append(route)

# Older clients (A2A <= 0.2) look for the card at agent.json
for route in create_agent_card_routes(agent_card, card_url="/.well-known/agent.json"):
    app.router.routes.append(route)

for route in create_jsonrpc_routes(request_handler, rpc_url="/", enable_v0_3_compat=True):
    app.router.routes.append(route)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    public_paths = (AGENT_CARD_WELL_KNOWN_PATH, "/.well-known/agent.json", "/health")
    if API_KEY and request.url.path not in public_paths:
        provided = request.headers.get(API_KEY_HEADER, "")
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            provided = provided or auth[7:]
        if not secrets.compare_digest(provided, API_KEY):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    """Predict survival from a regular JSON HTTP request.

    Accepted body: ``{"passengers": [{"Pclass": 1, "Sex": "female", ...}]}``
    A single passenger object is also accepted.
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc

    passengers = passengers_from_data(payload)
    if not passengers:
        raise HTTPException(
            status_code=422,
            detail="Send a passenger object or {\"passengers\": [{...}]}. Pclass and Sex are required.",
        )

    try:
        results = await asyncio.to_thread(tabpfn_service.predict_passengers, passengers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("REST prediction failed")
        raise HTTPException(status_code=503, detail="Prediction service is unavailable.") from exc

    return {"results": results}


if __name__ == "__main__":
    logger.info("A2A agent card: %s.well-known/agent-card.json", PUBLIC_URL)
    uvicorn.run(app, host=HOST, port=PORT)
