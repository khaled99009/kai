"""add immutable evidence ledger

Revision ID: c9d4e6f8a1b2
Revises: 7573ac048c34
Create Date: 2026-11-08 17:55:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c9d4e6f8a1b2"
down_revision: str | None = "7573ac048c34"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "evidence_ledger",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("tenant_id", sa.UUID(), nullable=False),
        sa.Column("execution_id", sa.UUID(), nullable=False),
        sa.Column("task_id", sa.UUID(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("previous_hash", sa.String(length=64), nullable=False),
        sa.Column("event_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_hash", name="uq_evidence_ledger_event_hash"),
        sa.UniqueConstraint(
            "execution_id", "sequence", name="uq_evidence_ledger_execution_sequence"
        ),
        comment="append-only hash-chained runtime evidence",
    )
    op.create_index(
        "ix_evidence_ledger_tenant_id", "evidence_ledger", ["tenant_id"], unique=False
    )
    op.create_index(
        "ix_evidence_ledger_execution_id", "evidence_ledger", ["execution_id"], unique=False
    )
    op.create_index(
        "ix_evidence_ledger_task_id", "evidence_ledger", ["task_id"], unique=False
    )
    op.create_index(
        "ix_evidence_ledger_tenant_task_sequence",
        "evidence_ledger",
        ["tenant_id", "task_id", "sequence"],
        unique=False,
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION kai_reject_evidence_ledger_mutation()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION 'evidence_ledger is append-only: % is forbidden', TG_OP
                USING ERRCODE = '55006';
        END;
        $$;
        """
    )
    op.execute(
        """
        CREATE TRIGGER evidence_ledger_append_only
        BEFORE UPDATE OR DELETE ON evidence_ledger
        FOR EACH ROW EXECUTE FUNCTION kai_reject_evidence_ledger_mutation();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS evidence_ledger_append_only ON evidence_ledger")
    op.execute("DROP FUNCTION IF EXISTS kai_reject_evidence_ledger_mutation()")
    op.drop_index("ix_evidence_ledger_tenant_task_sequence", table_name="evidence_ledger")
    op.drop_index("ix_evidence_ledger_task_id", table_name="evidence_ledger")
    op.drop_index("ix_evidence_ledger_execution_id", table_name="evidence_ledger")
    op.drop_index("ix_evidence_ledger_tenant_id", table_name="evidence_ledger")
    op.drop_table("evidence_ledger")
