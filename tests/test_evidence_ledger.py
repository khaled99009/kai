"""Deterministic contracts for the immutable evidence ledger."""

from __future__ import annotations

from uuid import UUID

import pytest

from kai.runtime.ledger import GENESIS_HASH, event_hash
from kai.runtime.models import EvidenceLedger

pytestmark = pytest.mark.no_db_unit

EVENT_ID = UUID("00000000-0000-0000-0000-000000000001")
EXECUTION_ID = UUID("00000000-0000-0000-0000-000000000002")


def test_genesis_hash_is_fixed_length_zero_digest() -> None:
    assert GENESIS_HASH == "0" * 64


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


def test_ledger_model_declares_unique_execution_sequence() -> None:
    constraints = EvidenceLedger.__table__.constraints
    assert any(
        constraint.name == "uq_evidence_ledger_execution_sequence"
        for constraint in constraints
    )
    assert EvidenceLedger.__table__.c.payload.nullable is False
