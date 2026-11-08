"""Provider idempotency-key contract tests."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest

from kai.calendar.client import GoogleApiError, create_event
from kai.calendar.service_models import CreateEventRequest
from kai.tools.idempotency import provider_idempotency_key

pytestmark = pytest.mark.no_db_unit

TENANT = UUID("00000000-0000-0000-0000-000000000001")
TASK = UUID("00000000-0000-0000-0000-000000000002")
ACCOUNT = UUID("00000000-0000-0000-0000-000000000003")


def test_key_is_stable_for_equivalent_argument_order() -> None:
    first = provider_idempotency_key(
        tenant_id=TENANT,
        task_id=TASK,
        request_id="action-1",
        tool_name="calendar.create_event",
        arguments={"b": 2, "a": {"z": True, "y": 1}},
        account_id=ACCOUNT,
    )
    second = provider_idempotency_key(
        tenant_id=TENANT,
        task_id=TASK,
        request_id="action-1",
        tool_name="calendar.create_event",
        arguments={"a": {"y": 1, "z": True}, "b": 2},
        account_id=ACCOUNT,
    )
    assert first == second
    assert first.startswith("kai-")
    assert len(first) == 68


def test_key_changes_when_governed_action_changes() -> None:
    common = {
        "tenant_id": TENANT,
        "task_id": TASK,
        "request_id": "action-1",
        "tool_name": "gmail.send",
        "account_id": ACCOUNT,
    }
    first = provider_idempotency_key(arguments={"subject": "one"}, **common)
    second = provider_idempotency_key(arguments={"subject": "two"}, **common)
    assert first != second


def test_key_material_does_not_include_access_token() -> None:
    key = provider_idempotency_key(
        tenant_id=TENANT,
        task_id=TASK,
        request_id="action-1",
        tool_name="gmail.send",
        arguments={"to": "user@example.com", "body": "hello"},
        account_id=ACCOUNT,
    )
    assert "access_token" not in key
    assert "secret" not in key


@pytest.mark.asyncio
async def test_calendar_write_sends_idempotency_header() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers["X-Idempotency-Key"]
        return httpx.Response(500, json={})

    request = CreateEventRequest(
        title="idempotent",
        start=datetime(2026, 1, 1, 9, tzinfo=UTC),
        end=datetime(2026, 1, 1, 10, tzinfo=UTC),
        timezone="UTC",
        attendees=[],
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(GoogleApiError):
            await create_event(
                http,
                access_token="token",
                req=request,
                idempotency_key="kai-test-key",
            )
    assert seen == {"key": "kai-test-key"}
