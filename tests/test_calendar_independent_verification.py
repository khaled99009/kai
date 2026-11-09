"""Independent, field-by-field verification for calendar creation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from kai.calendar.models import Attendee, CalendarEvent
from kai.calendar.service import verify_event
from kai.calendar.service_models import CreateEventRequest

pytestmark = pytest.mark.no_db_unit

REQUEST = CreateEventRequest(
    title="Team Sync",
    start=datetime(2026, 6, 24, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")),
    end=datetime(2026, 6, 24, 11, 0, tzinfo=ZoneInfo("Europe/Berlin")),
    timezone="Europe/Berlin",
    attendees=("bob@example.com", "carol@example.com"),
)

READ_BACK = CalendarEvent(
    event_id="event-1",
    summary="Team Sync",
    start=datetime(2026, 6, 24, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")),
    end=datetime(2026, 6, 24, 11, 0, tzinfo=ZoneInfo("Europe/Berlin")),
    start_timezone="Europe/Berlin",
    end_timezone="Europe/Berlin",
    attendees=(
        Attendee(email="bob@example.com"),
        Attendee(email="carol@example.com"),
    ),
)


@pytest.mark.parametrize(
    ("field", "event"),
    [
        ("title", replace(READ_BACK, summary="Different title")),
        (
            "start",
            replace(
                READ_BACK,
                start=datetime(
                    2026, 6, 24, 10, 30, tzinfo=ZoneInfo("Europe/Berlin")
                ),
            ),
        ),
        (
            "end",
            replace(
                READ_BACK,
                end=datetime(2026, 6, 24, 11, 30, tzinfo=ZoneInfo("Europe/Berlin")),
            ),
        ),
        (
            "timezone",
            replace(
                READ_BACK,
                start_timezone="UTC",
                end_timezone="UTC",
                start=datetime(2026, 6, 24, 8, 0, tzinfo=ZoneInfo("UTC")),
                end=datetime(2026, 6, 24, 9, 0, tzinfo=ZoneInfo("UTC")),
            ),
        ),
        (
            "attendees",
            replace(READ_BACK, attendees=(Attendee(email="mallory@example.com"),)),
        ),
    ],
)
def test_verify_event_rejects_each_independent_field_mismatch(
    field: str, event: CalendarEvent
) -> None:
    mismatches = verify_event(event, REQUEST)

    assert field in {mismatch.field for mismatch in mismatches}
    assert mismatches


def test_verify_event_accepts_exact_readback_without_provider_claims() -> None:
    """Only deterministic comparison, not the provider's 2xx, proves success."""
    assert verify_event(READ_BACK, REQUEST) == ()


def test_verify_event_compares_attendees_as_normalized_sets() -> None:
    reordered = replace(
        READ_BACK,
        attendees=(
            Attendee(email="CAROL@example.com"),
            Attendee(email=" bob@example.com "),
        ),
    )

    assert verify_event(reordered, REQUEST) == ()


def test_verify_event_rejects_invalid_requested_timezone() -> None:
    invalid_request = replace(REQUEST, timezone="Not/AZone")

    mismatches = verify_event(READ_BACK, invalid_request)

    assert any(mismatch.field == "timezone" for mismatch in mismatches)
