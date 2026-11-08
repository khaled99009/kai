"""The :class:`RuntimeExecution` ORM row (one per agent-runtime task).

The row keeps the agent runtime's *resume contract*: the validated scenario,
the resolved attendee identities and their evidence, the list of tool-run
records (evidence), and the progress cursor. It exists so a task can stop at a
human approval checkpoint and resume later without re-running the model or
re-executing completed steps, and so a re-trigger can never create a duplicate
calendar event.

``plan_version``/``plan_hash`` are what the ``cursor`` is bound to. The cursor
counts the steps of exactly one plan, so the row records that plan's version and
a content hash of its steps: a re-plan after execution started can never be
resumed against the old cursor (which would silently skip the new plan's steps
or re-run finished ones). Both stay ``None`` until the first trigger adopts a
plan, and are only re-bound when no step has run yet.

``claim_token``/``claimed_at`` are the runtime's per-task execution claim: the
single-trigger lease that makes a concurrent or duplicate resume a no-op instead
of a second writer racing the owner into a different final state. It is a
committed reservation, never a held transaction or row lock, so no database
lock is ever kept across an external provider call. A claim older than the lease
is treated as abandoned by a crashed worker and may be taken over.

``fulfillment_profile``/``evaluator_version``/``fulfillment_verdict`` record how
fulfillment was decided: which sealed profile the contract named, which version of
that profile's evaluator semantics ran, and what it concluded. They are an audit
trail, and every one of them is nullable on purpose. A run that stops before
verification has no verdict to record, and a row written before these columns
existed cannot be given an identity that was never written down — so it is left
``NULL`` rather than backfilled, because a backfilled ``evaluator_version`` would
be a guess presented as provenance. ``NULL`` therefore means "not recorded", and
never "no evaluator ran": the evaluator, the sealed contract, and the evidence
are what decide, and this row only describes what they decided.

Explicit tenants-and-users FK scope mirrors the Task row; ``task_id`` is
unique so at most one execution exists per task. No secrets are stored here:
no tokens, credentials, or message bodies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from kai.db.base import Base, PKMixin, TenantMixin, TimestampMixin


class RuntimeExecution(PKMixin, TenantMixin, TimestampMixin, Base):
    """Agent-runtime execution state, 1:1 with a ``tasks.id`` row."""

    __tablename__ = "runtime_executions"
    __table_args__ = (
        {"comment": "agent-runtime execution state for one task"},
    )

    task_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)

    scenario: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    identities: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    evidence: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    cursor: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    steps_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: What the cursor above is counting: one plan version plus the hash of its
    #: steps. Nullable until the first trigger binds it.
    #:
    #: The ``comment=`` values below restate what the migrations already applied
    #: to the database (``b3d5f7a9c1e2`` and ``c8e1f3a5b7d9``). They are metadata
    #: only: no type, nullability, default or index changes with them. They are
    #: declared here so this model accurately describes a migrated schema and
    #: ``alembic check`` reports real drift rather than six columns whose
    #: operator-facing descriptions it believes are missing.
    plan_version: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="task plan version this cursor is bound to",
    )
    plan_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="sha256 of the bound plan version steps",
    )

    claim_token: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), nullable=True, index=True
    )
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # The fulfillment audit triple: which sealed profile, which version of its
    # evaluator, and what that evaluator concluded. All three are nullable
    # because a run that never reached verification has nothing to record, and
    # because the rows written before these columns existed cannot honestly
    # claim an identity — see the module docstring.
    #
    # These columns are an audit trail, never a source of authority. The verdict
    # here is re-derived from the evidence above on every settlement, and a
    # re-derivation that disagrees with this row overwrites it. Nothing reads
    # these columns to decide a task state.
    fulfillment_profile: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="sealed contract profile whose evaluator decided fulfillment",
    )
    evaluator_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="semantic version of the fulfillment evaluator that ran",
    )
    fulfillment_verdict: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="verdict the evaluator returned: audit record, not authority",
    )


class EvidenceLedger(PKMixin, TenantMixin, Base):
    """Immutable, hash-chained record of runtime observations.

    The payload is a private durable copy of the gateway observation.  It is not
    used as an authorization source; it proves what the runtime recorded and
    provides a tamper-evident chain independent of ``RuntimeExecution.evidence``.
    PostgreSQL rejects updates and deletes through the migration-installed trigger.
    """

    __tablename__ = "evidence_ledger"
    __table_args__ = (
        UniqueConstraint(
            "execution_id", "sequence", name="uq_evidence_ledger_execution_sequence"
        ),
        Index(
            "ix_evidence_ledger_tenant_task_sequence",
            "tenant_id",
            "task_id",
            "sequence",
        ),
        {"comment": "append-only hash-chained runtime evidence"},
    )

    execution_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), nullable=False, index=True
    )
    task_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    previous_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


__all__ = ["EvidenceLedger", "RuntimeExecution"]
