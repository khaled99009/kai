"""Google Calendar API client for the vertical slice.

Network traffic only flows through an injected ``httpx.AsyncClient`` so tests
can isolate Google's network behind ``httpx.MockTransport`` while the
database and credential decryption stay real. The slice reads and writes only
the user's primary calendar — the smallest surface that supports availability
and event creation — under the ``calendar.events`` scope granted at connect.
"""

from datetime import datetime
from urllib.parse import quote

import httpx

from kai.calendar.models import (
    CalendarEvent,
    EventsPage,
    parse_event_payload,
    parse_events_payload,
)
from kai.calendar.service_models import CreateEventRequest
from kai.observability import metrics
from kai.observability.providers import provider_call

EVENTS_COLLECTION_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
MAX_RESULTS = 250
MAX_PAGES = 50
MAX_EVENTS = 10_000


class GoogleApiError(RuntimeError):
    """A Google API request failed. Carries the HTTP status, when there was one.

    ``status_code`` is what lets a caller tell a rejection from a server-side
    failure **without parsing the message**. That distinction decides whether a
    write can be reported as definitely-not-applied, so it has to be data rather
    than prose: a 4xx is the provider saying no, while a 5xx says nothing about
    whether the operation was committed before the error came back. ``None``
    means no response was ever received, which is a third case again and not a
    missing detail.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def is_server_failure(self) -> bool:
        """Did the provider answer, but with an error that does not settle it?

        True only for a status that exists — a 5xx. A 4xx is excluded because
        it is a rejection, and ``None`` is excluded because there was no answer
        to classify.
        """
        return self.status_code is not None and self.status_code >= 500


class _TypedStatusError(GoogleApiError):
    """A named status whose class already says what the status was."""


    """Raised when Google's Calendar API rejects or fails a request."""


class GoogleInvalidTokenError(_TypedStatusError):
    """HTTP 401: the stored access token is invalid or expired."""


class GoogleForbiddenError(_TypedStatusError):
    """HTTP 403: scope is missing or consent was revoked."""


class GoogleRateLimitError(_TypedStatusError):
    """HTTP 429: quota/rate limit exceeded."""


class GoogleNotFoundError(_TypedStatusError):
    """HTTP 404: the requested event does not exist."""


class GoogleUnreadableResultError(GoogleApiError):
    """The request was dispatched and the provider replied with a success status,
    yet the reply cannot be read as the created resource (a malformed body, or a
    missing/unparseable id), so whether the write was applied cannot be
    established from this answer.
    """


def _raise_for_status(resp: httpx.Response) -> None:
    """Map Google's failure modes onto typed errors (never echoes tokens)."""
    if resp.status_code == 401:
        raise GoogleInvalidTokenError(
            "google access token is invalid or expired (consent may be revoked)",
            status_code=resp.status_code,
        )
    if resp.status_code == 403:
        raise GoogleForbiddenError(
            "google denied the request; calendar.events scope may be missing",
            status_code=resp.status_code,
        )
    if resp.status_code == 404:
        raise GoogleNotFoundError(
            "google calendar event was not found",
            status_code=resp.status_code,
        )
    if resp.status_code == 429:
        raise GoogleRateLimitError(
            "google api rate limit exceeded, try again later",
            status_code=resp.status_code,
        )
    if resp.status_code >= 400:
        raise GoogleApiError(
            f"google calendar api request failed ({resp.status_code})",
            status_code=resp.status_code,
        )


async def fetch_events_page(
    transport: httpx.AsyncClient,
    *,
    access_token: str,
    start: datetime,
    end: datetime,
    page_token: str | None = None,
) -> EventsPage:
    """Fetch one page of timed events within ``[start, end)`` on the primary calendar.

    ``singleEvents=true`` expands recurring events and enables
    ``orderBy=startTime``, so busy intervals are resolved, not repeated rules.
    """
    params: dict[str, str] = {
        "timeMin": start.isoformat(),
        "timeMax": end.isoformat(),
        "singleEvents": "true",
        "orderBy": "startTime",
        "maxResults": str(MAX_RESULTS),
    }
    if page_token:
        params["pageToken"] = page_token
    with provider_call(provider=metrics.PROVIDER_GOOGLE, operation="list_events"):
        resp = await transport.get(
            EVENTS_COLLECTION_URL,
            params=params,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        _raise_for_status(resp)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise GoogleApiError(
                "google calendar api returned a malformed response"
            ) from exc
        return parse_events_payload(payload)


async def fetch_events(
    transport: httpx.AsyncClient,
    *,
    access_token: str,
    start: datetime,
    end: datetime,
) -> list[CalendarEvent]:
    """Fetch every timed event in the window, bounded against bad pagination."""
    events: list[CalendarEvent] = []
    page_token: str | None = None
    for _ in range(MAX_PAGES):
        page = await fetch_events_page(
            transport,
            access_token=access_token,
            start=start,
            end=end,
            page_token=page_token,
        )
        events.extend(page.events)
        if not page.has_more or len(events) >= MAX_EVENTS:
            break
        page_token = page.next_page_token
    return events


async def create_event(
    transport: httpx.AsyncClient,
    *,
    access_token: str,
    req: CreateEventRequest,
    idempotency_key: str | None = None,
) -> CalendarEvent:
    """Create a timed event on the primary calendar and parse it back."""
    body: dict = {
        "summary": req.title,
        "start": {"dateTime": req.start.isoformat(), "timeZone": req.timezone},
        "end": {"dateTime": req.end.isoformat(), "timeZone": req.timezone},
        "attendees": [{"email": email} for email in req.attendees],
    }
    with provider_call(provider=metrics.PROVIDER_GOOGLE, operation="create_event"):
        headers = {"Authorization": f"Bearer {access_token}"}
        if idempotency_key is not None:
            headers["X-Idempotency-Key"] = idempotency_key
        resp = await transport.post(
            EVENTS_COLLECTION_URL,
            json=body,
            headers=headers,
        )
        _raise_for_status(resp)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise GoogleUnreadableResultError(
                "google calendar create returned a malformed response",
                status_code=resp.status_code,
            ) from exc
        event = parse_event_payload(payload)
        if event is None:
            raise GoogleUnreadableResultError(
                "google calendar create returned an unparseable event",
                status_code=resp.status_code,
            )
        return event


async def fetch_event(
    transport: httpx.AsyncClient, *, access_token: str, event_id: str
) -> CalendarEvent:
    """Read one event back from the primary calendar (read-after-create)."""
    url = f"{EVENTS_COLLECTION_URL}/{quote(event_id, safe='')}"
    with provider_call(provider=metrics.PROVIDER_GOOGLE, operation="read_event"):
        resp = await transport.get(
            url, headers={"Authorization": f"Bearer {access_token}"}
        )
        _raise_for_status(resp)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise GoogleApiError(
                "google calendar read returned a malformed response"
            ) from exc
        event = parse_event_payload(payload)
        if event is None:
            raise GoogleApiError("google calendar read returned an unparseable event")
        return event
