"""Durable provider-effect idempotency state machine."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from kai.db.base import Base, PKMixin, TenantMixin


class ProviderEffectStatus(StrEnum):
    """Safe states for one externally visible provider effect."""

    RESERVED = "RESERVED"
    APPLIED = "APPLIED"
    AMBIGUOUS = "AMBIGUOUS"


class ProviderEffect(PKMixin, TenantMixin, Base):
    """One durable reservation for one provider-side effect.

    ``RESERVED`` is committed before the external call. Any retry that finds it
    must not send again because the previous worker may have reached the
    provider. ``APPLIED`` means the provider returned a usable result. ``AMBIGUOUS``
    means the outcome cannot safely be classified and requires reconciliation.
    No credentials or provider response bodies are stored here.
    """

    __tablename__ = "provider_effects"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_provider_effects_key"),
        Index(
            "ix_provider_effects_tenant_task",
            "tenant_id",
            "task_id",
        ),
        {"comment": "durable provider-effect idempotency state"},
    )

    task_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, index=True)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    provider_reference: Mapped[str | None] = mapped_column(String(256), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    reserved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    settled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


__all__ = ["ProviderEffect", "ProviderEffectStatus"]
