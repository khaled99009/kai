"""Crash recovery for the durable provider-effect reservation boundary."""

from __future__ import annotations

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
    "title": "Crash Recovery Sync",
    "date": "2026-06-24",
    "start_time": "10:00",
    "end_time": "11:00",
    "timezone": "Europe/Berlin",
}


class WorkerCrash(BaseException):
    """Process-level interruption; normal gateway exception handling must not run."""


class CrashBeforeProvider:
    """Executor that simulates a worker dying before it reaches the provider."""

    calls = 0

    async def execute(
        self,
        *,
        tool: ToolDefinition,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolExecutionOutcome:
        self.calls += 1
        raise WorkerCrash("simulated worker termination")


@pytest.mark.asyncio
async def test_worker_crash_after_reservation_blocks_provider_redispatch() -> None:
    """A committed RESERVED row protects against a duplicate after a crash."""
    tenant, user = await make_tenant_user("crash-recovery@example.com")
    account = await insert_account(
        tenant_id=tenant.id,
        user_id=user.id,
        email="alice@example.com",
        sub="crash-recovery-sub",
    )
    task = await make_task(
        tenant_id=tenant.id,
        user_id=user.id,
        allowed_tools=["calendar.create_event"],
    )
    backend = GoogleBackend()
    crashing_executor = CrashBeforeProvider()
    base_registry = default_registry()
    production_registry = ToolRegistry.build(
        replace(
            definition,
            executor=crashing_executor,
        )
        if definition.name == "calendar.create_event"
        else definition
        for definition in base_registry.definitions().values()
    )
    first_gateway, _ = make_gateway(registry=production_registry)
    intent = make_intent(
        "calendar.create_event",
        task=task,
        arguments=CALENDAR_ARGS,
        request_id="crash-before-provider-1",
    )

    async with session_scope() as session:
        first_context = make_context(
            session=session,
            task=task,
            backend=backend,
            account_id=account.id,
        )
        with pytest.raises(WorkerCrash):
            await first_gateway.execute(intent, first_context)
        await close_context(first_context)

    async with session_scope() as session:
        effect = (
            await session.execute(
                select(ProviderEffect).where(
                    ProviderEffect.tenant_id == tenant.id,
                    ProviderEffect.task_id == task.id,
                    ProviderEffect.request_id == intent.request_id,
                )
            )
        ).scalar_one()
        assert effect.status == ProviderEffectStatus.RESERVED.value
        assert effect.settled_at is None

    second_gateway, _ = make_gateway()
    async with session_scope() as session:
        second_context = make_context(
            session=session,
            task=task,
            backend=backend,
            account_id=account.id,
        )
        result = await second_gateway.execute(intent, second_context)
        await close_context(second_context)

    assert result.ok is False
    assert result.error_code == "PROVIDER_EFFECT_ALREADY_RESERVED_OR_SETTLED"
    assert crashing_executor.calls == 1
    assert not [call for call in backend.calendar_calls if call.method == "POST"]

    async with session_scope() as session:
        effects = (
            await session.execute(
                select(ProviderEffect).where(
                    ProviderEffect.tenant_id == tenant.id,
                    ProviderEffect.task_id == task.id,
                )
            )
        ).scalars().all()
        assert len(effects) == 1
        assert effects[0].status == ProviderEffectStatus.RESERVED.value
