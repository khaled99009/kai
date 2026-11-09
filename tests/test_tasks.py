"""Task Manager integration tests — real Dockerized PostgreSQL, no mocks.

Service-level calls and API calls both land in the same shared Postgres that
the app uses at runtime. Cross-tenant and cross-user isolation are asserted at
the row level (list/get/mutation silently resolve to nothing) and at the API
level (404 on foreign ids, 401 without a token).
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest
from pytest import LogCaptureFixture
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select

from alembic_runner import run_alembic
from kai.auth.jwt import TokenClaims, decode_access_token
from kai.db.engine import async_session_factory
from kai.db.models import User
from kai.main import app
from kai.tasks.models import Task
from kai.tasks.schemas import TaskContractInput
from kai.tasks.service import (
    CLIENT_TRANSITION_TARGETS,
    EXTERNAL_ACTOR,
    INTERNAL_ACTOR,
    InvalidActorError,
    InvalidTransition,
    TaskNotFoundError,
    archive_task,
    assign_task,
    count_tasks,
    create_task,
    get_task,
    list_tasks,
    transition_task,
)
from kai.tasks.state import (
    ALL_STATES,
    COMPLETED,
    CREATED,
    ESCALATE,
    EXECUTING,
    FAILED,
    PLANNING,
    POLICY_CHECK,
    REQUIRES_ATTENTION,
    RETRY,
    TRANSITIONS,
    UNDERSTANDING,
    VERIFYING,
    WAITING_APPROVAL,
)
from migration_harness import MigrationGroupState, migration_schema

API_PREFIX = "/api/v1"

#: Everything a client may NOT assert: the full vocabulary minus the three
#: states it is entitled to ask for. Derived, so widening the client allow-list
#: has to be a deliberate edit here too.
_RUNTIME_OWNED_STATES = ALL_STATES - CLIENT_TRANSITION_TARGETS


def _contract(**overrides: object) -> TaskContractInput:
    data: dict[str, object] = {
        "goal": "Book a flight to Berlin",
        "success_criteria": ["booking reference exists", "cost confirmed"],
        "fulfillment_profile": "calendar_meeting",
        "allowed_tools": ["calendar", "gmail"],
        "risk_policy": "LOW",
        "deadline": datetime(2026, 6, 24, 10, 0, tzinfo=UTC),
        "max_cost": 500.0,
        "max_steps": 20,
        "verification_method": "automatic",
        "fallback": "ask the user",
    }
    data.update(overrides)
    return TaskContractInput.model_validate(data)


async def _register(client: httpx.AsyncClient, email: str) -> tuple[str, TokenClaims]:
    resp = await client.post(
        f"{API_PREFIX}/auth/register",
        json={"email": email, "password": "password123"},
    )
    assert resp.status_code == 201, resp.text
    token = resp.json()["access_token"]
    return token, decode_access_token(token)


def _task_json(goal: str = "Book a flight to Berlin") -> dict:
    return {
        "goal": goal,
        "success_criteria": ["booking reference exists"],
        "fulfillment_profile": "calendar_meeting",
        "allowed_tools": ["calendar"],
        "risk_policy": "LOW",
        "deadline": "2026-06-24T10:00:00Z",
        "max_cost": 500.0,
        "max_steps": 20,
        "verification_method": "automatic",
        "fallback": "ask the user",
    }


async def _create_via_api(client: httpx.AsyncClient, token: str) -> dict:
    resp = await client.post(
        f"{API_PREFIX}/tasks",
        json=_task_json(),
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- create / get / list ---------------------------------------------------

async def test_create_get_list_roundtrip(client: httpx.AsyncClient) -> None:
    token, claims = await _register(client, "alice@example.com")
    created = await _create_via_api(client, token)

    assert created["state"] == CREATED
    assert created["tenant_id"] == str(claims.tenant_id)
    assert created["user_id"] == str(claims.user_id)
    assert created["contract"]["goal"] == "Book a flight to Berlin"

    fetched = await client.get(
        f"{API_PREFIX}/tasks/{created['id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert fetched.status_code == 200
    assert fetched.json()["id"] == created["id"]
    assert fetched.json()["contract"]["deadline"] == "2026-06-24T10:00:00Z"

    listing = await client.get(
        f"{API_PREFIX}/tasks", headers={"Authorization": f"Bearer {token}"}
    )
    assert listing.status_code == 200
    assert [t["id"] for t in listing.json()["tasks"]] == [created["id"]]


async def test_contract_roundtrip_preserves_details(
    client: httpx.AsyncClient,
) -> None:
    token, _ = await _register(client, "roundtrip@example.com")
    resp = await client.post(
        f"{API_PREFIX}/tasks",
        json=_task_json(),
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    contract = resp.json()["contract"]
    assert contract["success_criteria"] == ["booking reference exists"]
    assert contract["allowed_tools"] == ["calendar"]
    assert contract["risk_policy"] == "LOW"
    assert contract["max_cost"] == 500.0
    assert contract["max_steps"] == 20
    assert contract["verification_method"] == "automatic"


async def test_create_ignores_caller_supplied_scope_fields(
    client: httpx.AsyncClient,
) -> None:
    token, claims = await _register(client, "scope@example.com")
    resp = await client.post(
        f"{API_PREFIX}/tasks",
        json={**_task_json(), "tenant_id": "other", "user_id": "other"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["tenant_id"] == str(claims.tenant_id)
    assert body["user_id"] == str(claims.user_id)


# --- tenant isolation ------------------------------------------------------

async def test_cross_tenant_read_is_404(client: httpx.AsyncClient) -> None:
    token_a, _ = await _register(client, "acme@corp.example")
    token_b, _ = await _register(client, "globex@corp.example")
    created = await _create_via_api(client, token_a)

    resp = await client.get(
        f"{API_PREFIX}/tasks/{created['id']}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp.status_code == 404


async def test_cross_tenant_transition_is_404_and_does_not_mutate(
    client: httpx.AsyncClient,
) -> None:
    token_a, _ = await _register(client, "acme@corp.example")
    token_b, _ = await _register(client, "globex@corp.example")
    created = await _create_via_api(client, token_a)

    resp = await client.post(
        f"{API_PREFIX}/tasks/{created['id']}/transition",
        json={"target_state": UNDERSTANDING},
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp.status_code == 404

    mine = await client.get(
        f"{API_PREFIX}/tasks/{created['id']}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert mine.status_code == 200
    assert mine.json()["state"] == CREATED


async def test_cross_tenant_archive_is_404(client: httpx.AsyncClient) -> None:
    token_a, _ = await _register(client, "acme@corp.example")
    token_b, _ = await _register(client, "globex@corp.example")
    created = await _create_via_api(client, token_a)
    resp = await client.post(
        f"{API_PREFIX}/tasks/{created['id']}/archive",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp.status_code == 404


async def test_cross_tenant_listing_is_empty(client: httpx.AsyncClient) -> None:
    token_a, _ = await _register(client, "acme@corp.example")
    token_b, _ = await _register(client, "globex@corp.example")
    await _create_via_api(client, token_a)
    await _create_via_api(client, token_a)
    acme_created = await _create_via_api(client, token_a)

    g1 = await _create_via_api(client, token_b)
    g2 = await _create_via_api(client, token_b)

    resp = await client.get(
        f"{API_PREFIX}/tasks", headers={"Authorization": f"Bearer {token_b}"}
    )
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    assert {t["id"] for t in tasks} == {g1["id"], g2["id"]}
    assert acme_created["id"] not in {t["id"] for t in tasks}


async def test_service_scopes_every_lookup_by_tenant() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        _, claims_a = await _register(client, "acme@corp.example")
        _, claims_b = await _register(client, "globex@corp.example")

    async with async_session_factory() as session:
        task = await create_task(
            session,
            tenant_id=claims_a.tenant_id,
            user_id=claims_a.user_id,
            contract=_contract(),
        )
        task_id = task.id

        assert (
            await get_task(
                session, task_id=task_id, tenant_id=claims_a.tenant_id
            )
        ).id == task_id

        with pytest.raises(TaskNotFoundError):
            await get_task(
                session, task_id=task_id, tenant_id=claims_b.tenant_id
            )
        assert await count_tasks(session, tenant_id=claims_b.tenant_id) == 0
        assert await list_tasks(session, tenant_id=claims_b.tenant_id) == []


# --- user isolation within a tenant ---------------------------------------

async def test_same_tenant_other_user_cannot_touch_task() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        _, claims_a = await _register(client, "owner@corp.example")

    other = User(tenant_id=claims_a.tenant_id, email="second@corp.example", password_hash="unused")
    async with async_session_factory() as session:
        session.add(other)
        await session.commit()
        await session.refresh(other)

        task = await create_task(
            session,
            tenant_id=claims_a.tenant_id,
            user_id=claims_a.user_id,
            contract=_contract(),
        )
        task_id = task.id

        with pytest.raises(TaskNotFoundError):
            await get_task(
                session,
                task_id=task_id,
                tenant_id=claims_a.tenant_id,
                user_id=other.id,
            )
        assert await list_tasks(
            session,
            tenant_id=claims_a.tenant_id,
            user_id=other.id,
        ) == []
        with pytest.raises(TaskNotFoundError):
            await transition_task(
                session,
                task_id=task_id,
                tenant_id=claims_a.tenant_id,
                user_id=other.id,
                target=UNDERSTANDING,
            )

        # Assign to the second user *within the same tenant*: the owner loses
        # visibility, the second user gains it.
        await assign_task(
            session,
            task_id=task_id,
            tenant_id=claims_a.tenant_id,
            user_id=other.id,
        )
        moved = await get_task(
            session,
            task_id=task_id,
            tenant_id=claims_a.tenant_id,
            user_id=other.id,
        )
        assert moved.user_id == other.id
        with pytest.raises(TaskNotFoundError):
            await get_task(
                session,
                task_id=task_id,
                tenant_id=claims_a.tenant_id,
                user_id=claims_a.user_id,
            )


# --- state machine through the service ------------------------------------

async def test_every_declared_transition_applies() -> None:
    for current, targets in TRANSITIONS.items():
        for target in targets:
            row = await _persist(state=current)
            async with async_session_factory() as session:
                await transition_task(
                    session,
                    task_id=row.id,
                    tenant_id=row.tenant_id,
                    user_id=row.user_id,
                    target=target,
                )
            async with async_session_factory() as session:
                reloaded = await get_task(
                    session,
                    task_id=row.id,
                    tenant_id=row.tenant_id,
                    user_id=row.user_id,
                )
                assert reloaded.state == target, (current, target)


async def test_client_actor_may_only_assert_its_own_states() -> None:
    """The actor check is the service's, not just the endpoint's."""
    for target in sorted(_RUNTIME_OWNED_STATES):
        row = await _persist(state=CREATED)
        async with async_session_factory() as session:
            with pytest.raises(InvalidActorError) as exc_info:
                await transition_task(
                    session,
                    task_id=row.id,
                    tenant_id=row.tenant_id,
                    user_id=row.user_id,
                    target=target,
                    actor=EXTERNAL_ACTOR,
                )
            assert "asserted by the agent runtime" in str(exc_info.value)
            await session.rollback()

        async with async_session_factory() as session:
            reloaded = await get_task(
                session,
                task_id=row.id,
                tenant_id=row.tenant_id,
                user_id=row.user_id,
            )
            assert reloaded.state == CREATED, target


def test_the_client_allow_list_is_exactly_three_honest_states() -> None:
    """Pin the boundary: FAILED, REQUIRES_ATTENTION, RETRY - and nothing else."""
    assert frozenset({FAILED, REQUIRES_ATTENTION, RETRY}) == CLIENT_TRANSITION_TARGETS
    assert CLIENT_TRANSITION_TARGETS < ALL_STATES
    assert COMPLETED not in CLIENT_TRANSITION_TARGETS
    assert ESCALATE not in CLIENT_TRANSITION_TARGETS


async def test_internal_actor_may_still_assert_lifecycle_states() -> None:
    """The runtime's own writes are untouched by the client restriction."""
    row = await _persist(state=CREATED)
    async with async_session_factory() as session:
        for target in (UNDERSTANDING, PLANNING, POLICY_CHECK, EXECUTING):
            await transition_task(
                session,
                task_id=row.id,
                tenant_id=row.tenant_id,
                user_id=row.user_id,
                target=target,
                actor=INTERNAL_ACTOR,
            )

    async with async_session_factory() as session:
        reloaded = await get_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
        )
        assert reloaded.state == EXECUTING


async def test_illegal_transition_is_rejected_and_persists_nothing() -> None:
    row = await _persist(state=CREATED)
    async with async_session_factory() as session:
        with pytest.raises(InvalidTransition) as exc_info:
            await transition_task(
                session,
                task_id=row.id,
                tenant_id=row.tenant_id,
                user_id=row.user_id,
                target=EXECUTING,
            )
        assert "illegal transition" in str(exc_info.value)
        await session.rollback()
    async with async_session_factory() as session:
        reloaded = await get_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
        )
        assert reloaded.state == CREATED
        assert reloaded.attempts == 0


async def test_completed_is_frozen() -> None:
    row = await _persist(state=COMPLETED)
    async with async_session_factory() as session:
        with pytest.raises(InvalidTransition):
            await transition_task(
                session,
                task_id=row.id,
                tenant_id=row.tenant_id,
                user_id=row.user_id,
                target=EXECUTING,
            )


async def test_retry_path_increments_attempts_and_clears_error() -> None:
    row = await _persist(state=CREATED)
    async with async_session_factory() as session:
        await transition_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            target=FAILED,
            fail_reason="external provider timed out",
        )
        failed = await get_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
        )
        assert failed.state == FAILED
        assert failed.attempts == 0
        assert failed.last_error == "external provider timed out"
        assert failed.failed_at is not None

        await transition_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            target=RETRY,
        )
        retried = await get_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
        )
        assert retried.state == RETRY
        assert retried.attempts == 1
        assert retried.last_error is None
        assert retried.failed_at is None

        await transition_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            target=EXECUTING,
        )
        resumed = await get_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
        )
        assert resumed.state == EXECUTING


async def test_retry_budget_exhaustion_forces_escalation() -> None:
    row = await _persist(state=FAILED, attempts=1, max_retries=1)
    async with async_session_factory() as session:
        with pytest.raises(InvalidTransition) as exc_info:
            await transition_task(
                session,
                task_id=row.id,
                tenant_id=row.tenant_id,
                user_id=row.user_id,
                target=RETRY,
            )
        assert "budget exhausted" in str(exc_info.value)

        await transition_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            target=ESCALATE,
        )
        escalated = await get_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
        )
        assert escalated.state == ESCALATE


async def test_requires_attention_path() -> None:
    row = await _persist(state=CREATED)
    async with async_session_factory() as session:
        await transition_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            target=REQUIRES_ATTENTION,
            fail_reason="needs human input",
        )
        flagged = await get_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
        )
        assert flagged.state == REQUIRES_ATTENTION
        assert flagged.last_error == "needs human input"

        await transition_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            target=RETRY,
        )
        retried = await get_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
        )
        assert retried.state == RETRY
        assert retried.attempts == 1


async def test_failed_can_escalate_directly() -> None:
    row = await _persist(state=FAILED, attempts=0)
    async with async_session_factory() as session:
        await transition_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            target=ESCALATE,
        )
        reloaded = await get_task(
            session,
            task_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
        )
        assert reloaded.state == ESCALATE


async def test_archive_requires_terminal_state() -> None:
    row = await _persist(state=CREATED)
    async with async_session_factory() as session:
        with pytest.raises(InvalidTransition):
            await archive_task(
                session,
                task_id=row.id,
                tenant_id=row.tenant_id,
                user_id=row.user_id,
            )
    terminal = await _persist(state=COMPLETED)
    async with async_session_factory() as session:
        await archive_task(
            session,
            task_id=terminal.id,
            tenant_id=terminal.tenant_id,
            user_id=terminal.user_id,
        )
        assert await list_tasks(session, tenant_id=terminal.tenant_id) == []
        assert len(
            await list_tasks(
                session, tenant_id=terminal.tenant_id, include_archived=True
            )
        ) == 1


async def test_archive_via_api_hides_from_list(client: httpx.AsyncClient) -> None:
    token, claims = await _register(client, "archive@example.com")
    created = await _create_via_api(client, token)
    # archive requires terminal, and a terminal state is not a client's to
    # assert: the machine is walked here the way the runtime walks it, through
    # the service rather than the public endpoint.
    async with async_session_factory() as session:
        for target in (
            UNDERSTANDING,
            PLANNING,
            POLICY_CHECK,
            EXECUTING,
            VERIFYING,
            COMPLETED,
        ):
            await transition_task(
                session,
                task_id=UUID(created["id"]),
                tenant_id=claims.tenant_id,
                user_id=claims.user_id,
                target=target,
            )

    resp = await client.post(
        f"{API_PREFIX}/tasks/{created['id']}/archive",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200

    listing = await client.get(
        f"{API_PREFIX}/tasks", headers={"Authorization": f"Bearer {token}"}
    )
    assert listing.json()["tasks"] == []


# --- API transitions -------------------------------------------------------

async def test_transition_via_api(client: httpx.AsyncClient) -> None:
    """A client can say the honest thing about its own task: that it failed."""
    token, _ = await _register(client, "api@example.com")
    created = await _create_via_api(client, token)

    resp = await client.post(
        f"{API_PREFIX}/tasks/{created['id']}/transition",
        json={"target_state": FAILED},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["state"] == FAILED


async def test_client_can_request_the_three_states_it_owns(
    client: httpx.AsyncClient,
) -> None:
    token, _ = await _register(client, "api@example.com")
    created = await _create_via_api(client, token)
    auth = {"Authorization": f"Bearer {token}"}

    for target in (REQUIRES_ATTENTION, RETRY):
        resp = await client.post(
            f"{API_PREFIX}/tasks/{created['id']}/transition",
            json={"target_state": target},
            headers=auth,
        )
        assert resp.status_code == 200, (target, resp.text)
        assert resp.json()["state"] == target


@pytest.mark.parametrize("target", sorted(_RUNTIME_OWNED_STATES))
async def test_client_cannot_assert_a_runtime_lifecycle_state(
    client: httpx.AsyncClient, target: str
) -> None:
    """No client may write down a state only the runtime can establish.

    Every one of these edges exists in the machine, so nothing but the actor
    check stands between a caller and a task that looks finished and is not.
    """
    token, _ = await _register(client, "api@example.com")
    created = await _create_via_api(client, token)

    resp = await client.post(
        f"{API_PREFIX}/tasks/{created['id']}/transition",
        json={"target_state": target},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403, (target, resp.text)
    assert "asserted by the agent runtime" in resp.json()["detail"]["error"]

    verify = await client.get(
        f"{API_PREFIX}/tasks/{created['id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert verify.json()["state"] == CREATED


async def test_client_is_refused_even_where_the_machine_allows_the_edge(
    client: httpx.AsyncClient,
) -> None:
    """Entitlement is not legality: WAITING_APPROVAL -> EXECUTING is legal here.

    If the refusal depended on the machine, this request would succeed and the
    caller could drive its own task around the approval checkpoint it is
    supposed to be waiting on.
    """
    token, claims = await _register(client, "api@example.com")
    created = await _create_via_api(client, token)
    task_id = UUID(created["id"])

    async with async_session_factory() as session:
        for step in (UNDERSTANDING, PLANNING, POLICY_CHECK, WAITING_APPROVAL):
            await transition_task(
                session,
                task_id=task_id,
                tenant_id=claims.tenant_id,
                user_id=claims.user_id,
                target=step,
            )
        parked = await get_task(
            session,
            task_id=task_id,
            tenant_id=claims.tenant_id,
            user_id=claims.user_id,
        )
        assert parked.state == WAITING_APPROVAL
        assert EXECUTING in TRANSITIONS[WAITING_APPROVAL]

    resp = await client.post(
        f"{API_PREFIX}/tasks/{task_id}/transition",
        json={"target_state": EXECUTING},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403, resp.text

    async with async_session_factory() as session:
        still = await get_task(
            session,
            task_id=task_id,
            tenant_id=claims.tenant_id,
            user_id=claims.user_id,
        )
        assert still.state == "WAITING_APPROVAL"


async def test_illegal_transition_via_api_is_409(client: httpx.AsyncClient) -> None:
    """A client-owned state the machine refuses is still a conflict, not a 403."""
    token, _ = await _register(client, "api@example.com")
    created = await _create_via_api(client, token)

    resp = await client.post(
        f"{API_PREFIX}/tasks/{created['id']}/transition",
        json={"target_state": RETRY},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 409
    assert "illegal transition" in resp.json()["detail"]["error"]

    verify = await client.get(
        f"{API_PREFIX}/tasks/{created['id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert verify.json()["state"] == CREATED


async def test_unknown_target_state_is_422(client: httpx.AsyncClient) -> None:
    token, _ = await _register(client, "api@example.com")
    created = await _create_via_api(client, token)
    resp = await client.post(
        f"{API_PREFIX}/tasks/{created['id']}/transition",
        json={"target_state": "NOT_A_STATE"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


async def test_transition_missing_task_is_404(client: httpx.AsyncClient) -> None:
    token, _ = await _register(client, "api@example.com")
    resp = await client.post(
        f"{API_PREFIX}/tasks/00000000-0000-0000-0000-000000000000/transition",
        json={"target_state": UNDERSTANDING},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


# --- authentication --------------------------------------------------------

async def test_endpoints_require_authentication(client: httpx.AsyncClient) -> None:
    zero = "00000000-0000-0000-0000-000000000000"
    assert (await client.post(f"{API_PREFIX}/tasks", json=_task_json())).status_code == 401
    assert (await client.get(f"{API_PREFIX}/tasks")).status_code == 401
    assert (await client.get(f"{API_PREFIX}/tasks/{zero}")).status_code == 401
    assert (
        await client.post(
            f"{API_PREFIX}/tasks/{zero}/transition",
            json={"target_state": UNDERSTANDING},
        )
    ).status_code == 401
    assert (await client.post(f"{API_PREFIX}/tasks/{zero}/archive")).status_code == 401


async def test_invalid_token_is_401(client: httpx.AsyncClient) -> None:
    headers = {"Authorization": "Bearer not-a-jwt"}
    assert (await client.get(f"{API_PREFIX}/tasks", headers=headers)).status_code == 401


# --- malformed input -------------------------------------------------------

async def test_create_empty_body_is_422(client: httpx.AsyncClient) -> None:
    token, _ = await _register(client, "mal@example.com")
    resp = await client.post(
        f"{API_PREFIX}/tasks", json={}, headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 422


async def test_create_blank_goal_is_422(client: httpx.AsyncClient) -> None:
    token, _ = await _register(client, "mal@example.com")
    resp = await client.post(
        f"{API_PREFIX}/tasks",
        json={**_task_json(), "goal": "   "},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


async def test_create_naive_deadline_is_422(client: httpx.AsyncClient) -> None:
    token, _ = await _register(client, "mal@example.com")
    resp = await client.post(
        f"{API_PREFIX}/tasks",
        json={**_task_json(), "deadline": "2026-06-24T10:00:00"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


async def test_create_bad_max_steps_is_422(client: httpx.AsyncClient) -> None:
    token, _ = await _register(client, "mal@example.com")
    resp = await client.post(
        f"{API_PREFIX}/tasks",
        json={**_task_json(), "max_steps": 1001},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


async def test_create_negative_max_cost_is_422(client: httpx.AsyncClient) -> None:
    token, _ = await _register(client, "mal@example.com")
    resp = await client.post(
        f"{API_PREFIX}/tasks",
        json={**_task_json(), "max_cost": -5},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


async def test_create_high_risk_over_step_cap_is_422(client: httpx.AsyncClient) -> None:
    token, _ = await _register(client, "mal@example.com")
    resp = await client.post(
        f"{API_PREFIX}/tasks",
        json={**_task_json(), "risk_policy": "CRITICAL", "max_steps": 101},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


# --- no sensitive data leakage ---------------------------------------------

async def test_responses_never_expose_internal_metadata(
    client: httpx.AsyncClient, caplog: LogCaptureFixture
) -> None:
    token, _ = await _register(client, "leak@example.com")
    created = await _create_via_api(client, token)
    resp = await client.post(
        f"{API_PREFIX}/tasks/{created['id']}/transition",
        json={"target_state": FAILED},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    listing = await client.get(
        f"{API_PREFIX}/tasks", headers={"Authorization": f"Bearer {token}"}
    )
    for body in (created, resp.json(), listing.json()):
        text = repr(body)
        assert "last_error" not in text
        assert "failed_at" not in text
        assert "attempts" not in text
        assert "is_archived" not in text
        assert "Book a flight to Berlin" in text  # goal is a legit contract field

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        await client.get(
            f"{API_PREFIX}/tasks/{created['id']}",
            headers={"Authorization": f"Bearer {token}"},
        )
    combined = caplog.text
    assert "access_token" not in combined
    assert created["tenant_id"] not in combined
    assert "last_error" not in combined


# --- migration -------------------------------------------------------------

REVISION_PREV = "5f3e1d2c4b6a"
# The head of the *deliverable* migration chain. Pinned explicitly rather than
# read via ``migration_harness.current_head()``: that helper calls
# ``ScriptDirectory.get_current_head()``, which raises when the script directory
# has more than one head -- the state of a working tree that still carries the
# untracked, unapproved ``b2f7c1d9e4a6`` alongside ``a7c9e1f3b5d2``. The
# assertion below only cares that downgrade -> upgrade is clean and that this
# head is applied.
REVISION_HEAD = "d1e7f9a3b5c7"


def _alembic(target_url: str, *args: str) -> subprocess.CompletedProcess[str]:
    return run_alembic(*args, target_url=target_url)


def _table_names_fresh(target_url: str) -> list[str]:
    """Inspect on its own disposable engine + event loop.

    The migration test runs synchronously and must not touch the shared
    ``engine``'s pool, whose connections live on pytest's loop.
    """

    from sqlalchemy.ext.asyncio import create_async_engine

    async def _names() -> list[str]:
        eng = create_async_engine(target_url)
        try:
            async with eng.connect() as conn:
                tables = await conn.run_sync(
                    lambda sync: sa_inspect(sync).get_table_names()
                )
                return list(tables)
        finally:
            await eng.dispose()

    return asyncio.run(_names())


@pytest.mark.db_migration
def test_migration_is_rollbackable_and_forward(
    migration_target: str, migration_group_state: MigrationGroupState
) -> None:
    with migration_schema(migration_group_state, migration_target):
        down = _alembic(migration_target, "downgrade", REVISION_PREV)
        assert down.returncode == 0, down.stderr
        current = _alembic(migration_target, "current").stdout
        assert REVISION_PREV in current

        assert "tasks" not in _table_names_fresh(migration_target)
        assert "task_plans" not in _table_names_fresh(migration_target)

        up = _alembic(migration_target, "upgrade", "head")
        assert up.returncode == 0, up.stderr
        assert "tasks" in _table_names_fresh(migration_target)
        assert "task_plans" in _table_names_fresh(migration_target)
        assert REVISION_HEAD in _alembic(migration_target, "current").stdout


async def _persist(**overrides: object) -> Task:
    from uuid import uuid4

    from kai.db.models import Tenant

    async with async_session_factory() as session:
        tenant = Tenant(name=f"tenant-{uuid4()}")
        session.add(tenant)
        await session.flush()
        user = User(
            tenant_id=tenant.id,
            email=f"user-{uuid4()}@example.com",
            password_hash="unused",
        )
        session.add(user)
        await session.flush()

        params: dict[str, object] = {
            "tenant_id": tenant.id,
            "user_id": user.id,
            "state": CREATED,
            "goal": "persisted goal",
            "success_criteria": ["criterion"],
            "fulfillment_profile": "calendar_meeting",
            "allowed_tools": [],
            "risk_policy": "LOW",
            "max_cost": 0.0,
            "max_steps": 20,
            "verification_method": "automatic",
            "fallback": "ask user",
            "attempts": 0,
            "max_retries": 2,
        }
        params.update(overrides)
        task = Task(**params)  # type: ignore[arg-type]
        session.add(task)
        await session.commit()
        await session.refresh(task)
        return task


# ----------------------------------------------------------------------------------
# Concurrency: two writers, one state machine
# ----------------------------------------------------------------------------------


async def test_a_transition_waits_for_one_already_in_flight() -> None:
    """The loser of a race must block, not decide against a state nobody wrote.

    ``EXECUTING`` is the state that makes the race worth taking: the runtime may
    stop there for a human (``WAITING_APPROVAL``) and a reader that never saw
    that pause may carry on to ``VERIFYING`` - both legal from ``EXECUTING``,
    which is exactly why the machine cannot catch the loser on its own.
    """
    task = await _persist(state=EXECUTING)

    async with async_session_factory() as holder:
        # A transaction that is mid-decide on this row and has not committed.
        locked = (
            await holder.execute(
                select(Task).where(Task.id == task.id).with_for_update()
            )
        ).scalar_one()
        locked.state = WAITING_APPROVAL
        await holder.flush()

        async with async_session_factory() as other:
            second = asyncio.create_task(
                transition_task(
                    other,
                    task_id=task.id,
                    tenant_id=task.tenant_id,
                    user_id=task.user_id,
                    target=VERIFYING,
                )
            )
            await asyncio.sleep(0.3)
            assert not second.done(), (
                "the second transition was decided against EXECUTING while a "
                "writer held the task: it would overwrite the approval pause"
            )

            await holder.commit()

            # Unblocked, it is re-read and judged against what was really
            # written - and WAITING_APPROVAL -> VERIFYING is not a machine edge.
            with pytest.raises(InvalidTransition):
                await asyncio.wait_for(second, timeout=10)

    async with async_session_factory() as session:
        final = await get_task(
            session, task_id=task.id, tenant_id=task.tenant_id
        )
    assert final.state == WAITING_APPROVAL


async def test_two_writers_cannot_both_move_a_task_off_the_same_state() -> None:
    """Exactly one of two racing writers is allowed to believe it moved the task.

    Without the row lock both read ``EXECUTING``, both pass the machine, and both
    return their own state as if they had it: the caller that parked the task for
    approval is told it did, while the row says otherwise.
    """
    task = await _persist(state=EXECUTING)

    async def _move(target: str) -> str:
        async with async_session_factory() as session:
            try:
                moved = await transition_task(
                    session,
                    task_id=task.id,
                    tenant_id=task.tenant_id,
                    user_id=task.user_id,
                    target=target,
                )
            except InvalidTransition:
                return "refused"
            return moved.state

    outcomes = await asyncio.gather(
        _move(WAITING_APPROVAL), _move(VERIFYING)
    )
    winners = [outcome for outcome in outcomes if outcome != "refused"]
    assert len(winners) == 1, f"both writers were told they won: {outcomes}"

    async with async_session_factory() as session:
        final = await get_task(
            session, task_id=task.id, tenant_id=task.tenant_id
        )
    assert final.state == winners[0]


def test_contract_validation_errors_use_ids_not_row_content() -> None:
    # Pydantic's boundary errors mention field names, never stored values.
    from pydantic import ValidationError

    import kai.tasks.schemas as schemas

    with pytest.raises(ValidationError) as exc_info:
        schemas.TaskContractInput.model_validate({})
    text = str(exc_info.value)
    assert "goal" in text
    assert "FAILED" not in text
