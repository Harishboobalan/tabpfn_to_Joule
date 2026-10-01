"""Send one comparison request to a running local A2A server."""

import asyncio
import json
import os

import httpx
from a2a.client import create_client
from a2a.client import ClientConfig
from a2a.client.errors import AgentCardResolutionError
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import Role, SendMessageConfiguration, SendMessageRequest, TaskState


async def main() -> None:
    agent_url = os.getenv("A2A_AGENT_URL", "http://localhost:8000")
    timeout_seconds = float(os.getenv("A2A_CLIENT_TIMEOUT", "300"))
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=10)) as http_client:
        try:
            client = await create_client(agent_url, client_config=ClientConfig(httpx_client=http_client))
        except AgentCardResolutionError as exc:
            raise SystemExit(
                f"Cannot reach the A2A agent at {agent_url}. Start it with 'python app.py', "
                "or set A2A_AGENT_URL to its actual URL."
            ) from exc
        request = {"target_column": os.getenv("TARGET_COLUMN", "Survived")}
        if table_name := os.getenv("HANA_TABLE"):
            request["table_name"] = table_name
        a2a_request = SendMessageRequest(
            message=new_text_message(json.dumps(request), role=Role.ROLE_USER),
            configuration=SendMessageConfiguration(),
        )
        async for response in client.send_message(a2a_request):
            event_type = response.WhichOneof("payload")
            if event_type == "artifact_update":
                result = json.loads(get_stream_response_text(response))
                print(
                    json.dumps(
                        {
                            "predictions": result["predictions"],
                            "accuracy": result["accuracy"],
                            "comparison": result["comparison"],
                        },
                        indent=2,
                    )
                )
            elif (
                event_type == "status_update"
                and response.status_update.status.state == TaskState.TASK_STATE_FAILED
            ):
                raise RuntimeError(get_stream_response_text(response))


if __name__ == "__main__":
    asyncio.run(main())
