"""Calendar tool adapter.

Three tools share this adapter; the tool name (already validated by the
registry) selects the code path:

* ``calendar.read_slots`` — list events in the working window, then compute
  free/busy availability in the requested timezone (reuses
  ``calendar.service.list_events`` + ``calendar.slots.compute_availability``).
* ``calendar.get_event`` — read one event (reuses the new
  ``calendar.service.get_event``).
* ``calendar.create_event`` — create then read-after-create verify (reuses
  ``calendar.service.create_and_verify_event``); ``verified`` mirrors the
  service's deterministic verification.

Reads are ``verified=True`` only when the service parsed provider data; event
creation is ``verified`` exactly when the read-back returned no mismatches.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kai.calendar import client as calendar_client
from kai.calendar.models import CalendarEvent
from kai.calendar.service import (
    AccountNotFoundError,
    create_and_verify_event,
    get_event,
    list_events,
)
from kai.calendar.service_models import CreateEventRequest
from kai.calendar.slots import compute_availability, iso_interval
from kai.tools.adapters.base import AdapterBase, provider_failure
from kai.tools.context import ToolContext
from kai.tools.errors import PROVIDER_UNREACHABLE, ToolExecutionError
from kai.tools.execution import ToolExecutionOutcome, VerificationMismatch
from kai.tools.schemas import CalendarCreateEventArgs

_PROVIDER_CODES: dict[type[Exception], str] = {
    calendar_client.GoogleInvalidTokenError: "TOKEN_INVALID",
    calendar_client.GoogleForbiddenError: "PERMISSION_DENIED",
    calendar_client.GoogleRateLimitError: "RATE_LIMITED",
    calendar_client.GoogleUnreadableResultError: PROVIDER_UNREACHABLE,
    calendar_client.GoogleApiError: "PROVIDER_ERROR",
}


def _serialize_event(event: CalendarEvent | None) -> dict[str, Any] | None:
    if event is None:
        return None
    return {
        "event_id": event.event_id,
        "title": event.summary,
        "start": event.start.isoformat(),
        "end": event.end.isoformat(),
        "start_timezone": event.start_timezone,
        "end_timezone": event.end_timezone,
        "attendees": [attendee.email for attendee in event.attendees],
    }


def _mismatches(pair: tuple[Any, ...]) -> tuple[VerificationMismatch, ...]:
    return tuple(
        VerificationMismatch(field=m.field, expected=m.expected, actual=m.actual)
        for m in pair
    )


class CalendarAdapter(AdapterBase):
    """Executes the three calendar tools against the stored Google account."""

    async def execute(
        self,
        *,
        tool: Any,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolExecutionOutcome:
        if tool.name == "calendar.read_slots":
            return await self._read_slots(arguments=arguments, context=context)
        if tool.name == "calendar.get_event":
            return await self._get_event(arguments=arguments, context=context)
        if tool.name == "calendar.create_event":
            return await self._create_event(arguments=arguments, context=context)
        raise self.unimplemented(tool.name)

    async def _read_slots(
        self, *, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecutionOutcome:
        self.record("calendar.read_slots")
        transport = self.require_transport(context)
        args = CalendarReadArgs.from_mapping(arguments)
        try:
            events = await list_events(
                context.session,
                transport,
                account_id=self.require_account(context),
                tenant_id=context.tenant_id,
                start=args.window_start,
                end=args.window_end,
            )
        except AccountNotFoundError:
            raise ToolExecutionError(
                "google account not found for tenant", error_code="ACCOUNT_NOT_FOUND"
            ) from None
        except Exception as exc:
            raise provider_failure(exc, codes=_PROVIDER_CODES) from exc

        availability = compute_availability(
            events,
            day=args.day,
            timezone=arguments["timezone"],
            day_start=args.day_start,
            day_end=args.day_end,
        )
        data = {
            "timezone": availability.timezone,
            "date": args.day.isoformat(),
            "day_start": availability.day_start.isoformat(),
            "day_end": availability.day_end.isoformat(),
            "free": [iso_interval(slot) for slot in availability.free],
            "busy": [iso_interval(slot) for slot in availability.busy],
        }
        return ToolExecutionOutcome(
            ok=True,
            verified=True,
            data=data,
            evidence={
                "free_slots": len(availability.free),
                "busy_slots": len(availability.busy),
            },
        )

    async def _get_event(
        self, *, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecutionOutcome:
        self.record("calendar.get_event")
        transport = self.require_transport(context)
        try:
            event = await get_event(
                context.session,
                transport,
                account_id=self.require_account(context),
                tenant_id=context.tenant_id,
                event_id=arguments["event_id"],
            )
        except AccountNotFoundError:
            raise ToolExecutionError(
                "google account not found for tenant", error_code="ACCOUNT_NOT_FOUND"
            ) from None
        except Exception as exc:
            raise provider_failure(exc, codes=_PROVIDER_CODES) from exc
        return ToolExecutionOutcome(
            ok=True,
            verified=True,
            data=_serialize_event(event),
            evidence={"event_id": event.event_id},
        )

    async def _create_event(
        self, *, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecutionOutcome:
        self.record("calendar.create_event")
        transport = self.require_transport(context)
        args = CalendarCreateEventArgs.model_validate(arguments)
        start, end = args.as_aware_window()
        req = CreateEventRequest(
            title=args.title,
            start=start,
            end=end,
            timezone=args.timezone,
            attendees=tuple(args.attendees),
        )
        try:
            outcome = await create_and_verify_event(
                context.session,
                transport,
                account_id=self.require_account(context),
                tenant_id=context.tenant_id,
                req=req,
                idempotency_key=context.metadata.get("provider_idempotency_key"),
            )
        except AccountNotFoundError:
            raise ToolExecutionError(
                "google account not found for tenant", error_code="ACCOUNT_NOT_FOUND"
            ) from None
        except Exception as exc:
            raise provider_failure(exc, codes=_PROVIDER_CODES) from exc

        mismatches = _mismatches(outcome.mismatches)
        data = _serialize_event(outcome.event)
        evidence: dict[str, Any] = {}
        if outcome.event is not None:
            evidence["event_id"] = outcome.event.event_id
            evidence["mismatch_count"] = len(outcome.mismatches)
        elif outcome.created_event_id is not None:
            # The provider accepted the write but the event could not be read
            # back; keep the id it returned so the accepted creation is still
            # recorded as evidence rather than lost.
            evidence["event_id"] = outcome.created_event_id
            evidence["mismatch_count"] = len(outcome.mismatches)
        return ToolExecutionOutcome(
            ok=True,
            verified=outcome.verified,
            data=data,
            evidence=evidence,
            mismatches=mismatches,
        )


class CalendarReadArgs:
    """Normalized day-window inputs for ``calendar.read_slots``."""

    def __init__(self, *, day: Any, start: Any, end: Any, timezone: str) -> None:
        from datetime import datetime

        from kai.calendar.slots import parse_timezone

        tz = parse_timezone(timezone)
        self.day = day
        self.day_start = start
        self.day_end = end
        self.window_start = datetime.combine(day, start, tzinfo=tz)
        self.window_end = datetime.combine(day, end, tzinfo=tz)

    @classmethod
    def from_mapping(cls, arguments: Mapping[str, Any]) -> CalendarReadArgs:
        return cls(
            day=arguments["date"],
            start=arguments["day_start"],
            end=arguments["day_end"],
            timezone=arguments["timezone"],
        )


__all__ = ["CalendarAdapter"]
