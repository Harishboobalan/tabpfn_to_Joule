"""A2A bridge for the TabPFN comparison business agent."""

import asyncio
import json

from a2a.helpers import new_task_from_user_message, new_text_artifact_update_event, new_text_status_update_event
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import TaskState
from agent import TabPFNComparisonAgent, summarize

class TabPFNComparisonExecutor(AgentExecutor):
    def __init__(self) -> None:
        self._agent = TabPFNComparisonAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task.id,
                task.context_id,
                TaskState.TASK_STATE_WORKING,
                "Loading the HANA table and comparing models.",
            )
        )
        try:
            result = await asyncio.to_thread(self._agent.invoke, context.get_user_input())
            await event_queue.enqueue_event(
                new_text_artifact_update_event(
                    task.id,
                    task.context_id,
                    "comparison-result",
                    json.dumps(result, indent=2),
                    last_chunk=True,
                )
            )
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task.id,
                    task.context_id,
                    TaskState.TASK_STATE_COMPLETED,
                    # Joule shows the final status message, so it carries the readable summary.
                    summarize(result),
                )
            )
        except Exception as exc:
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task.id,
                    task.context_id,
                    TaskState.TASK_STATE_FAILED,
                    str(exc),
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task:
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    context.current_task.id,
                    context.current_task.context_id,
                    TaskState.TASK_STATE_CANCELED,
                    "Comparison cancelled.",
                )
            )
