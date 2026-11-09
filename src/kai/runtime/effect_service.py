"""Transactional helpers for durable provider-effect idempotency."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from kai.runtime.effects import ProviderEffect, ProviderEffectStatus


async def reserve_provider_effect(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    task_id: UUID,
    request_id: str,
    tool_name: str,
    idempotency_key: str,
) -> ProviderEffect:
    """Atomically create or lock the durable reservation for one effect."""
    statement = (
        insert(ProviderEffect)
        .values(
            tenant_id=tenant_id,
            task_id=task_id,
            request_id=request_id,
            tool_name=tool_name,
            idempotency_key=idempotency_key,
            status=ProviderEffectStatus.RESERVED.value,
            evidence={},
        )
        .on_conflict_do_nothing(index_elements=[ProviderEffect.idempotency_key])
    )
    await session.execute(statement)
    result = await session.execute(
        select(ProviderEffect)
        .where(
            ProviderEffect.tenant_id == tenant_id,
            ProviderEffect.idempotency_key == idempotency_key,
        )
        .with_for_update()
    )
    effect = result.scalar_one_or_none()
    if effect is None:
        raise RuntimeError("provider effect reservation disappeared")
    return effect


async def settle_provider_effect(
    session: AsyncSession,
    *,
    effect_id: UUID,
    status: ProviderEffectStatus,
    provider_reference: str | None = None,
    error_code: str | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> None:
    """Move a reservation to a terminal safe state exactly once."""
    if status not in (ProviderEffectStatus.APPLIED, ProviderEffectStatus.AMBIGUOUS):
        raise ValueError("provider effects can only settle to APPLIED or AMBIGUOUS")
    await session.execute(
        update(ProviderEffect)
        .where(
            ProviderEffect.id == effect_id,
            ProviderEffect.status == ProviderEffectStatus.RESERVED.value,
        )
        .values(
            status=status.value,
            provider_reference=provider_reference,
            error_code=error_code,
            evidence=dict(evidence or {}),
            settled_at=func.now(),
        )
    )


__all__ = ["reserve_provider_effect", "settle_provider_effect"]
