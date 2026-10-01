"""A2A/ASGI host for the HANA-to-TabPFN comparison agent.

This follows the structure of the supplied Cloud Foundry sample, updated for
the current A2A Python SDK (v1.x). A2A v1 uses route factories instead of the
removed ``A2AStarletteApplication`` wrapper.
"""

import logging
import os

import httpx
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from dotenv import load_dotenv
from fastapi import FastAPI

from agent_executor import TabPFNComparisonExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cloud Foundry supplies PORT. A2A_PUBLIC_URL must be the externally reachable
# route, never the internal host/port used by uvicorn.
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
base_url = os.getenv("A2A_PUBLIC_URL", "http://localhost:8000").rstrip("/")

capabilities = AgentCapabilities(streaming=True, push_notifications=True)
skill = AgentSkill(
    id="compare-hana-table",
    name="Compare HANA classification models",
    description="Compares TabPFN, Random Forest, SVM, and Logistic Regression on an SAP HANA table.",
    tags=["hana", "tabpfn", "classification", "model-comparison"],
    examples=['{"target_column":"Survived","table_name":"PASSENGERS","test_size":0.2}'],
)
agent_card = AgentCard(
    name="HANA TabPFN Comparison Agent",
    description="Compares a TabPFN deployment with baseline classifiers using an SAP HANA table.",
    version="1.0.0",
    supported_interfaces=[
        AgentInterface(protocol_binding="JSONRPC", protocol_version="1.0", url=f"{base_url}/"),
        # Joule's agent-request action currently uses the A2A 0.3-compatible transport.
        AgentInterface(protocol_binding="JSONRPC", protocol_version="0.3", url=f"{base_url}/"),
    ],
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=capabilities,
    skills=[skill],
)

# These components accept push-notification subscription configuration. This
# synchronous comparison agent still returns its result on the request stream.
httpx_client = httpx.AsyncClient()
push_config_store = InMemoryPushNotificationConfigStore()
push_sender = BasePushNotificationSender(
    httpx_client=httpx_client,
    config_store=push_config_store,
)
request_handler = DefaultRequestHandler(
    agent_executor=TabPFNComparisonExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
    push_config_store=push_config_store,
    push_sender=push_sender,
)

app = FastAPI(title=agent_card.name, description=agent_card.description)
add_a2a_routes_to_fastapi(
    app,
    agent_card_routes=(
        create_agent_card_routes(agent_card)
        + create_agent_card_routes(agent_card, card_url="/.well-known/agent.json")
    ),
    jsonrpc_routes=create_jsonrpc_routes(request_handler, rpc_url="/", enable_v0_3_compat=True),
    rest_routes=create_rest_routes(request_handler, enable_v0_3_compat=True),
)


@app.on_event("shutdown")
async def close_http_client() -> None:
    await httpx_client.aclose()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
