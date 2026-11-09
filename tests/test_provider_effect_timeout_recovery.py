"""Timeout recovery for durable provider effects."""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy import select

from kai.runtime.effects import ProviderEffect, ProviderEffectStatus
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
    "title": "Timeout Recovery Sync",
    "date": "2026-06-24",
    "start_time": "10:00",
    "end_time": "11:00",
    "timezone": "Europe/Berlin",
}


class TimeoutBackend(GoogleBackend):
    """Provider transport that receives the write and loses the response."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/events"):
            self.calendar_calls.append(request)
            raise httpx.ReadTimeout("provider response timed out", request=request)
        return super().handler(request)


@pytest.mark.asyncio
async def test_provider_timeout_settles_ambiguous_and_blocks_retry() -> None:
    """A timeout after dispatch must never trigger an automatic redispatch."""
    tenant, user = await make_tenant_user("timeout-recovery@example.com")
    account = await insert_account(
        tenant_id=tenant.id,
        user_id=user.id,
        email="alice@example.com",
        sub="timeout-recovery-sub",
    )
    task = await make_task(
        tenant_id=tenant.id,
        user_id=user.id,
        allowed_tools=["calendar.create_event"],
    )
    backend = TimeoutBackend()
    intent = make_intent(
        "calendar.create_event",
        task=task,
        arguments=CALENDAR_ARGS,
        request_id="timeout-after-provider-dispatch-1",
    )

    first_gateway, _ = make_gateway()
    async with session_scope() as session:
        first_context = make_context(
            session=session,
            task=task,
            backend=backend,
            account_id=account.id,
        )
        first_result = await first_gateway.execute(intent, first_context)
        await close_context(first_context)

    assert first_result.outcome is ToolOutcome.EXECUTION_FAILED
    assert first_result.ok is False
    assert first_result.error_code == "PROVIDER_UNREACHABLE"
    assert len(backend.calendar_calls) == 1

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
        assert effect.status == ProviderEffectStatus.AMBIGUOUS.value
        assert effect.error_code == "PROVIDER_RESULT_AMBIGUOUS"
        assert effect.settled_at is not None

    second_gateway, _ = make_gateway()
    async with session_scope() as session:
        second_context = make_context(
            session=session,
            task=task,
            backend=backend,
            account_id=account.id,
        )
        second_result = await second_gateway.execute(intent, second_context)
        await close_context(second_context)

    assert second_result.outcome is ToolOutcome.REQUIRES_ATTENTION
    assert second_result.ok is False
    assert second_result.error_code == "PROVIDER_EFFECT_ALREADY_RESERVED_OR_SETTLED"
    assert len(backend.calendar_calls) == 1
