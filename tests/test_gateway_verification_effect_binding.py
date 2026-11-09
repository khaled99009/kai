"""Gateway-level binding between verification failure and provider effects."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from kai.runtime.effects import ProviderEffect, ProviderEffectStatus
from kai.tools.result import ToolOutcome
from tool_harness import (
    GoogleBackend,
    close_context,
    event_payload,
    insert_account,
    make_context,
    make_gateway,
    make_intent,
    make_task,
    make_tenant_user,
    session_scope,
)

CALENDAR_ARGS = {
    "title": "Expected Team Sync",
    "date": "2026-06-24",
    "start_time": "10:00",
    "end_time": "11:00",
    "timezone": "Europe/Berlin",
}


@pytest.mark.asyncio
async def test_verification_mismatch_is_attention_after_applied_effect() -> None:
    """A landed write with bad read-back is not success and is not redispatched."""
    tenant, user = await make_tenant_user("gateway-verification@example.com")
    account = await insert_account(
        tenant_id=tenant.id,
        user_id=user.id,
        email="alice@example.com",
        sub="gateway-verification-sub",
    )
    task = await make_task(
        tenant_id=tenant.id,
        user_id=user.id,
        allowed_tools=["calendar.create_event"],
    )
    backend = GoogleBackend()
    backend.read_event_payload = event_payload(
        event_id="created-event-1",
        summary="Provider Stored A Different Title",
        start="2026-06-24T10:00:00+02:00",
        end="2026-06-24T11:00:00+02:00",
        timezone="Europe/Berlin",
    )
    intent = make_intent(
        "calendar.create_event",
        task=task,
        arguments=CALENDAR_ARGS,
        request_id="gateway-verification-mismatch-1",
    )

    gateway, _ = make_gateway()
    async with session_scope() as session:
        context = make_context(
            session=session,
            task=task,
            backend=backend,
            account_id=account.id,
        )
        first_result = await gateway.execute(intent, context)
        await close_context(context)

    assert first_result.outcome is ToolOutcome.REQUIRES_ATTENTION
    assert first_result.ok is False
    assert first_result.verified is False
    assert first_result.error_code == "VERIFICATION_FAILED"
    assert any(mismatch.field == "title" for mismatch in first_result.mismatches)
    assert len([call for call in backend.calendar_calls if call.method == "POST"]) == 1

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
        assert effect.status == ProviderEffectStatus.APPLIED.value
        assert effect.provider_reference == "created-event-1"
        assert effect.evidence["event_id"] == "created-event-1"
        assert effect.evidence["mismatch_count"] == 1

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
    assert second_result.error_code == "PROVIDER_EFFECT_ALREADY_RESERVED_OR_SETTLED"
    assert len([call for call in backend.calendar_calls if call.method == "POST"]) == 1
