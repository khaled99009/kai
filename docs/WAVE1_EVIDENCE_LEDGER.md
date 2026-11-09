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

## Provider idempotency key propagation

The Gateway derives a stable `kai-<sha256>` key after validating and normalizing
tool arguments. The key includes the tenant, task, request/action, tool, account,
and normalized arguments; it never includes access tokens or provider
credentials. Calendar-create and Gmail-send adapters propagate it as
`X-Idempotency-Key` to the injected provider transport.

This propagation is not presented as proof that Google APIs honor the header. A
provider-specific native idempotency mapping and a durable effect state machine
are still required before automatic retries are enabled. Until then, an
ambiguous provider result remains an attention-required outcome.

## Durable provider-effect state

Migration `d1e7f9a3b5c7` adds `provider_effects`. Its unique
`idempotency_key` is the database backstop against two workers reserving the
same external effect. The state machine is intentionally fail-closed:

- `RESERVED`: a worker owns the reservation and may be about to contact the
  provider; another retry must not send;
- `APPLIED`: the provider returned a usable accepted result; another retry must
  not send;
- `AMBIGUOUS`: the request may have reached the provider but cannot be
  classified safely; reconciliation is required and automatic retry is
  forbidden.

Only `RESERVED -> APPLIED` and `RESERVED -> AMBIGUOUS` are legal terminal
transitions. The row stores provider reference and redacted evidence only; it
never stores credentials or raw message bodies.

The `ToolGateway` integrates this state machine for the externally visible
`calendar.create_event` and `gmail.send` operations. It commits a newly created
`RESERVED` row before invoking the provider. A retry that finds any existing row,
including an unfinished `RESERVED` row, is stopped with an
attention-required result and the provider executor is not called. A successful
provider execution is settled as `APPLIED`; an exception, timeout, or failed
execution classification is settled as `AMBIGUOUS` before the failure is
returned to the caller. This deliberately favors duplicate prevention over
automatic retry when the provider outcome cannot be proven.

## Verification

The CI Quality Gate verifies:

- Alembic upgrade and isolated migration tests;
- normal Python 3.14 tests;
- deterministic hash behavior;
- hash-chain linking;
- PostgreSQL rejection of both `UPDATE` and `DELETE`;
- crash recovery after a committed `RESERVED` effect before provider dispatch;
- timeout after provider dispatch settling the effect as `AMBIGUOUS`;
- repository Ruff and Mypy checks.
