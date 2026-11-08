"""Tenant-scoped Calendar operations.

Data is fetched for exactly one ``(account_id, tenant_id)``; the access token
is decrypted per request and never returned or logged. Verification is
deterministic in application code: after creation the event is read back from
Google and compared field-by-field (title, date, start/end time, timezone,
attendees). A mismatch means the operation must not be reported as success.
"""

from datetime import datetime
from uuid import UUID
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kai.calendar import client
from kai.calendar.client import (
    GoogleApiError,
    GoogleNotFoundError,
    fetch_event,
)
from kai.calendar.models import CalendarEvent
from kai.calendar.service_models import CreateEventOutcome, CreateEventRequest, FieldMismatch
from kai.db.models import GoogleAccount
from kai.google.service import usable_access_token


class AccountNotFoundError(ValueError):
    """Raised when a Google account does not exist for the current tenant."""


async def _load_account(
    session: AsyncSession, *, account_id: UUID, tenant_id: UUID
) -> GoogleAccount:
    result = await session.execute(
        select(GoogleAccount).where(
            GoogleAccount.id == account_id,
            GoogleAccount.tenant_id == tenant_id,
        )
    )
    account = result.scalar_one_or_none()
    if account is None:
        raise AccountNotFoundError("google account not found for tenant")
    return account


async def list_events(
    session: AsyncSession,
    transport: httpx.AsyncClient,
    *,
    account_id: UUID,
    tenant_id: UUID,
    start: datetime,
    end: datetime,
) -> list[CalendarEvent]:
    """Read the primary calendar's timed events within ``[start, end)``."""
    account = await _load_account(session, account_id=account_id, tenant_id=tenant_id)
    return await client.fetch_events(
        transport,
        access_token=await usable_access_token(session, transport, account=account),
        start=start,
        end=end,
    )


async def get_event(
    session: AsyncSession,
    transport: httpx.AsyncClient,
    *,
    account_id: UUID,
    tenant_id: UUID,
    event_id: str,
) -> CalendarEvent:
    """Read one timed event, scoped to ``(account_id, tenant_id)``."""
    account = await _load_account(session, account_id=account_id, tenant_id=tenant_id)
    return await fetch_event(
        transport,
        access_token=await usable_access_token(session, transport, account=account),
        event_id=event_id,
    )


def _zone_matches(requested: str, event_start: datetime, declared: str) -> bool:
    """True when the event's effective zone equals the requested zone.

    Google echoes the IANA name sent in the create payload, and labels that
    resolve to the same offset at the event instant (e.g. ``UTC`` vs
    ``Etc/UTC``) are equivalent. Offset comparison at the exact instant is the
    deterministic tiebreaker.
    """
    if requested.strip().casefold() == declared.strip().casefold():
        return True
    try:
        requested_zone = ZoneInfo(requested)
    except (KeyError, ValueError):
        return False
    try:
        declared_zone = ZoneInfo(declared) if declared.strip() else None
    except (KeyError, ValueError):
        declared_zone = None
    if declared_zone is not None and event_start.astimezone(
        declared_zone
    ).utcoffset() == event_start.astimezone(requested_zone).utcoffset():
        return True
    return (
        event_start.utcoffset() == event_start.astimezone(requested_zone).utcoffset()
        and event_start.tzname() == event_start.astimezone(requested_zone).tzname()
    )


def verify_event(
    event: CalendarEvent, req: CreateEventRequest
) -> tuple[FieldMismatch, ...]:
    """Deterministic field-by-field verification of a read-back event.

    Fields: title, date, start/end time (wall-clock in the requested zone),
    timezone, attendees. Comparisons are exact and do not depend on Google
    echoing our input — anything Google returns is the source of truth.
    """
    mismatches: list[FieldMismatch] = []
    if event.summary != req.title:
        mismatches.append(FieldMismatch("title", req.title, event.summary))
    if event.start != req.start:
        mismatches.append(
            FieldMismatch("start", req.start.isoformat(), event.start.isoformat())
        )
    if event.end != req.end:
        mismatches.append(
            FieldMismatch("end", req.end.isoformat(), event.end.isoformat())
        )
    if not _zone_matches(req.timezone, event.start, event.start_timezone):
        mismatches.append(
            FieldMismatch("timezone", req.timezone, event.start_timezone)
        )
    try:
        requested_zone = ZoneInfo(req.timezone)
    except (KeyError, ValueError):
        return (
            *mismatches,
            FieldMismatch("timezone", req.timezone, "invalid requested timezone"),
        )
    local_start = event.start.astimezone(requested_zone)
    local_end = event.end.astimezone(requested_zone)
    if local_start.date() != req.start.date():
        mismatches.append(
            FieldMismatch(
                "date", req.start.date().isoformat(), local_start.date().isoformat()
            )
        )
    if local_start.time() != req.start.time():
        mismatches.append(
            FieldMismatch(
                "start_time",
                req.start.time().isoformat(),
                local_start.time().isoformat(),
            )
        )
    if local_end.time() != req.end.time():
        mismatches.append(
            FieldMismatch(
                "end_time",
                req.end.time().isoformat(),
                local_end.time().isoformat(),
            )
        )
    expected_emails = sorted(
        {email.strip().casefold() for email in req.attendees if email.strip()}
    )
    actual_emails = sorted(
        {
            attendee.email.strip().casefold()
            for attendee in event.attendees
            if attendee.email.strip()
        }
    )
    if expected_emails != actual_emails:
        mismatches.append(
            FieldMismatch("attendees", ",".join(expected_emails), ",".join(actual_emails))
        )
    return tuple(mismatches)


async def create_and_verify_event(
    session: AsyncSession,
    transport: httpx.AsyncClient,
    *,
    account_id: UUID,
    tenant_id: UUID,
    req: CreateEventRequest,
    idempotency_key: str | None = None,
) -> CreateEventOutcome:
    """Create an event, read it back, and verify it deterministically.

    A failure to *create* propagates: nothing was written, so the caller must
    see the provider's error. A failure to *read the event back* does not. Once
    the provider has accepted the create, a 401/403/429 on the read-back joins
    the 404/malformed/5xx read failures in returning an outcome that is
    explicitly not verified and that still carries the created event id. The
    write may have landed, so reporting it as a tool failure would invite a
    blind retry and a duplicate event.
    """
    account = await _load_account(session, account_id=account_id, tenant_id=tenant_id)
    token = await usable_access_token(session, transport, account=account)
    created = await client.create_event(
        transport,
        access_token=token,
        req=req,
        idempotency_key=idempotency_key,
    )
    try:
        readback = await fetch_event(
            transport, access_token=token, event_id=created.event_id
        )
    except GoogleNotFoundError:
        return CreateEventOutcome(
            event=None,
            created_event_id=created.event_id,
            mismatches=(FieldMismatch("retrieval", "read-after-create", "event not found"),),
        )
    except GoogleApiError as exc:
        return CreateEventOutcome(
            event=None,
            created_event_id=created.event_id,
            mismatches=(FieldMismatch("retrieval", "read-after-create", str(exc)),),
        )
    mismatches = verify_event(readback, req)
    return CreateEventOutcome(event=readback, mismatches=mismatches)
