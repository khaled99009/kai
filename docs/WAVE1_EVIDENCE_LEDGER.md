# Wave 1 — Immutable Evidence Ledger

## Scope

KAI now records each runtime tool observation in two places:

1. `RuntimeExecution.evidence` remains the compatibility projection used by the
   existing resume and fulfillment readers.
2. `evidence_ledger` stores an independent append-only provenance event.

The ledger event is written in the same SQLAlchemy transaction as the runtime
observation. A failed transaction therefore does not leave a ledger event that
claims a result was committed.

## Chain contract

Each event contains:

- `execution_id` and `sequence`, unique together;
- `event_type`;
- JSONB `payload` containing the normalized tool observation;
- `previous_hash`;
- `event_hash`, a SHA-256 digest of the canonical event fields;
- database-generated `created_at`.

The first event in an execution uses 64 zeroes as its `previous_hash`. Payload
objects are serialized with sorted keys and compact separators, so equivalent
JSON object ordering produces the same digest.

## Immutability boundary

The migration installs a PostgreSQL trigger that rejects `UPDATE` and `DELETE`
on `evidence_ledger`. The application has no update or delete service for this
table. The unique sequence constraint is the database backstop against two
writers claiming the same position.

This is tamper-evidence, not a cryptographic trust boundary against a database
superuser or schema owner. Production deployment must keep migration ownership
and application runtime permissions separate and must prevent the runtime role
from altering triggers or table definitions.

## Authority boundary

The ledger is evidence, not authority. It does not grant approval, authorize a
tool, or decide fulfillment. Authority remains in the Policy, Approval, and Tool
Gateway paths. Provider-side idempotency is the next Wave 1 change and must bind
the external effect to the same execution identity before retries are enabled.

## Verification

The CI Quality Gate verifies:

- Alembic upgrade and isolated migration tests;
- normal Python 3.14 tests;
- deterministic hash behavior;
- hash-chain linking;
- PostgreSQL rejection of both `UPDATE` and `DELETE`;
- repository Ruff and Mypy checks.
