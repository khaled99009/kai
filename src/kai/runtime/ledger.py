"""Append-only evidence ledger primitives.

The runtime JSONB evidence list remains the compatibility projection.  This module
records the same observation as a separately chained database event so readers can
verify provenance without trusting a mutable aggregate JSON document.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kai.runtime.models import EvidenceLedger

GENESIS_HASH = "0" * 64


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def event_hash(
    *,
    event_id: UUID,
    execution_id: UUID,
    sequence: int,
    event_type: str,
    payload: Mapping[str, Any],
    previous_hash: str,
) -> str:
    """Return the deterministic SHA-256 hash committed for one ledger event."""
    material = {
        "event_id": str(event_id),
        "execution_id": str(execution_id),
        "sequence": sequence,
        "event_type": event_type,
        "payload": payload,
        "previous_hash": previous_hash,
    }
    return hashlib.sha256(_canonical_json(material)).hexdigest()


async def append_event(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    task_id: UUID,
    execution_id: UUID,
    event_type: str,
    payload: Mapping[str, Any],
) -> EvidenceLedger:
    """Append one chained event in the caller's transaction.

    The execution claim serializes runtime writers.  The unique ``(execution_id,
    sequence)`` constraint remains the database backstop if a caller violates that
    contract.
    """
    previous = await session.scalar(
        select(EvidenceLedger)
        .where(EvidenceLedger.execution_id == execution_id)
        .order_by(EvidenceLedger.sequence.desc())
        .limit(1)
    )
    previous_hash = previous.event_hash if previous is not None else GENESIS_HASH
    sequence = previous.sequence + 1 if previous is not None else 1
    event_id = uuid4()
    safe_payload = dict(payload)
    row = EvidenceLedger(
        id=event_id,
        tenant_id=tenant_id,
        task_id=task_id,
        execution_id=execution_id,
        sequence=sequence,
        event_type=event_type,
        payload=safe_payload,
        previous_hash=previous_hash,
        event_hash=event_hash(
            event_id=event_id,
            execution_id=execution_id,
            sequence=sequence,
            event_type=event_type,
            payload=safe_payload,
            previous_hash=previous_hash,
        ),
    )
    session.add(row)
    await session.flush()
    return row


__all__ = ["GENESIS_HASH", "append_event", "event_hash"]
