"""add durable provider-effect idempotency state

Revision ID: d1e7f9a3b5c7
Revises: c9d4e6f8a1b2
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d1e7f9a3b5c7"
down_revision: str | None = "c9d4e6f8a1b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "provider_effects",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("task_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("provider_reference", sa.String(length=256), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "reserved_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key", name="uq_provider_effects_key"),
        comment="durable provider-effect idempotency state",
    )
    op.create_index("ix_provider_effects_tenant_id", "provider_effects", ["tenant_id"])
    op.create_index("ix_provider_effects_task_id", "provider_effects", ["task_id"])
    op.create_index(
        "ix_provider_effects_tenant_task",
        "provider_effects",
        ["tenant_id", "task_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_provider_effects_tenant_task", table_name="provider_effects")
    op.drop_index("ix_provider_effects_task_id", table_name="provider_effects")
    op.drop_index("ix_provider_effects_tenant_id", table_name="provider_effects")
    op.drop_table("provider_effects")
