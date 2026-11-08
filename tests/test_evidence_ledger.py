"""Deterministic and PostgreSQL contracts for the immutable evidence ledger."""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from kai.db.engine import async_session_factory
from kai.runtime.ledger import GENESIS_HASH, append_event, event_hash
from kai.runtime.models import EvidenceLedger, RuntimeExecution
from tool_harness import make_task, make_tenant_user

EVENT_ID = UUID("00000000-0000-0000-0000-000000000001")
EXECUTION_ID = UUID("00000000-0000-0000-0000-000000000002")


@pytest.mark.no_db_unit
def test_genesis_hash_is_fixed_length_zero_digest() -> None:
    assert GENESIS_HASH == "0" * 64


@pytest.mark.no_db_unit
def test_event_hash_is_stable_for_same_canonical_event() -> None:
    first = event_hash(
        event_id=EVENT_ID,
        execution_id=EXECUTION_ID,
        sequence=1,
        event_type="tool_observation",
        payload={"b": 2, "a": {"nested": True}},
        previous_hash=GENESIS_HASH,
    )
    second = event_hash(
        event_id=EVENT_ID,
        execution_id=EXECUTION_ID,
        sequence=1,
        event_type="tool_observation",
        payload={"a": {"nested": True}, "b": 2},
        previous_hash=GENESIS_HASH,
    )
    assert first == second
    assert len(first) == 64


@pytest.mark.no_db_unit
def test_any_chain_input_change_changes_event_hash() -> None:
    base = event_hash(
        event_id=EVENT_ID,
        execution_id=EXECUTION_ID,
        sequence=1,
        event_type="tool_observation",
        payload={"ok": True},
        previous_hash=GENESIS_HASH,
    )
    changed = event_hash(
        event_id=EVENT_ID,
        execution_id=EXECUTION_ID,
        sequence=2,
        event_type="tool_observation",
        payload={"ok": True},
        previous_hash=base,
    )
    assert changed != base


@pytest.mark.no_db_unit
def test_ledger_model_declares_unique_execution_sequence() -> None:
    constraints = EvidenceLedger.__table__.constraints
    assert any(
        constraint.name == "uq_evidence_ledger_execution_sequence"
        for constraint in constraints
    )
    assert EvidenceLedger.__table__.c.payload.nullable is False


@pytest.mark.asyncio
async def test_postgres_chain_is_append_only() -> None:
    tenant, user = await make_tenant_user("ledger-owner@example.com")
    task = await make_task(tenant_id=tenant.id, user_id=user.id, state="EXECUTING")
    execution = RuntimeExecution(
        task_id=task.id,
        tenant_id=tenant.id,
        user_id=user.id,
        timezone="UTC",
        scenario=None,
        identities=[],
        evidence=[],
        cursor=0,
        steps_used=0,
    )
    async with async_session_factory() as session:
        session.add(execution)
        await session.commit()
        await session.refresh(execution)
        execution_id = execution.id
        first = await append_event(
            session,
            tenant_id=tenant.id,
            task_id=task.id,
            execution_id=execution_id,
            event_type="tool_observation",
            payload={"step": 1, "ok": True},
        )
        second = await append_event(
            session,
            tenant_id=tenant.id,
            task_id=task.id,
            execution_id=execution_id,
            event_type="tool_observation",
            payload={"step": 2, "ok": False},
        )
        await session.commit()
        await session.refresh(first)
        await session.refresh(second)
        assert first.sequence == 1
        assert first.previous_hash == GENESIS_HASH
        assert second.sequence == 2
        assert second.previous_hash == first.event_hash

        with pytest.raises(DBAPIError, match="append-only"):
            await session.execute(
                text("UPDATE evidence_ledger SET event_type = 'tampered'")
            )
        await session.rollback()

        with pytest.raises(DBAPIError, match="append-only"):
            await session.execute(text("DELETE FROM evidence_ledger"))
        await session.rollback()

        rows = list(
            (
                await session.execute(
                    text(
                        "SELECT sequence, event_type FROM evidence_ledger "
                        "WHERE execution_id = :execution_id ORDER BY sequence"
                    ),
                    {"execution_id": execution_id},
                )
            ).all()
        )
        assert rows == [(1, "tool_observation"), (2, "tool_observation")]
