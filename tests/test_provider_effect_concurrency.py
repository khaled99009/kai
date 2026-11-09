"""Concurrency protection for one durable provider-effect key."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
from sqlalchemy import select

from kai.runtime.effects import ProviderEffect, ProviderEffectStatus
from kai.tools import default_registry
from kai.tools.context import ToolContext
from kai.tools.definition import ToolDefinition
from kai.tools.execution import ToolExecutionOutcome
from kai.tools.registry import ToolRegistry
from kai.tools.result import ToolOutcome
from tool_harness import (
    GoogleBackend,
    close_context,
    insert_account,
    make_context,
    make_gateway,
    make_intent,
    make_task,
    make_tenant_user,
    session_scope,
)

CALENDAR_ARGS = {
    "title": "Concurrent Effect Sync",
    "date": "2026-06-24",
    "start_time": "10:00",
    "end_time": "11:00",
    "timezone": "Europe/Berlin",
}


class BlockingProviderExecutor:
    """Hold the winner at the provider boundary while the loser races."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def execute(
        self,
        *,
        tool: ToolDefinition,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolExecutionOutcome:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return ToolExecutionOutcome(
            ok=True,
            verified=True,
            evidence={"event_id": "concurrent-event-1"},
        )


def registry_with_executor(executor: BlockingProviderExecutor) -> ToolRegistry:
    base_registry = default_registry()
    return ToolRegistry.build(
        replace(definition, executor=executor)
        if definition.name == "calendar.create_event"
        else definition
        for definition in base_registry.definitions().values()
    )


async def execute_in_session(
    gateway: Any,
    *,
    task: Any,
    account_id: Any,
    backend: GoogleBackend,
    intent: Any,
) -> Any:
    async with session_scope() as session:
        context = make_context(
            session=session,
            task=task,
            backend=backend,
            account_id=account_id,
        )
        try:
            return await gateway.execute(intent, context)
        finally:
            await close_context(context)


@pytest.mark.asyncio
async def test_concurrent_workers_share_one_provider_effect_reservation() -> None:
    """Only one concurrent worker may pass the durable reservation boundary."""
    tenant, user = await make_tenant_user("concurrent-effects@example.com")
    account = await insert_account(
        tenant_id=tenant.id,
        user_id=user.id,
        email="alice@example.com",
        sub="concurrent-effects-sub",
    )
    task = await make_task(
        tenant_id=tenant.id,
        user_id=user.id,
        allowed_tools=["calendar.create_event"],
    )
    backend = GoogleBackend()
    executor = BlockingProviderExecutor()
    registry = registry_with_executor(executor)
    first_gateway, _ = make_gateway(registry=registry)
    second_gateway, _ = make_gateway(registry=registry)
    intent = make_intent(
        "calendar.create_event",
        task=task,
        arguments=CALENDAR_ARGS,
        request_id="concurrent-provider-effect-1",
    )

    winner = asyncio.create_task(
        execute_in_session(
            first_gateway,
            task=task,
            account_id=account.id,
            backend=backend,
            intent=intent,
        )
    )
    await asyncio.wait_for(executor.started.wait(), timeout=10)

    loser = asyncio.create_task(
        execute_in_session(
            second_gateway,
            task=task,
            account_id=account.id,
            backend=backend,
            intent=intent,
        )
    )
    loser_result = await asyncio.wait_for(loser, timeout=10)

    executor.release.set()
    winner_result = await asyncio.wait_for(winner, timeout=10)

    assert winner_result.outcome is ToolOutcome.EXECUTION_SUCCEEDED
    assert winner_result.ok is True
    assert loser_result.outcome is ToolOutcome.REQUIRES_ATTENTION
    assert loser_result.ok is False
    assert loser_result.error_code == "PROVIDER_EFFECT_ALREADY_RESERVED_OR_SETTLED"
    assert executor.calls == 1

    async with session_scope() as session:
        effects = (
            await session.execute(
                select(ProviderEffect).where(
                    ProviderEffect.tenant_id == tenant.id,
                    ProviderEffect.task_id == task.id,
                    ProviderEffect.request_id == intent.request_id,
                )
            )
        ).scalars().all()
        assert len(effects) == 1
        assert effects[0].status == ProviderEffectStatus.APPLIED.value
        assert effects[0].provider_reference == "concurrent-event-1"
