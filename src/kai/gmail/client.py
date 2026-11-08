"""Gmail API client for the vertical slice.

Network traffic only flows through an injected ``httpx.AsyncClient`` so tests
can isolate Google's network behind ``httpx.MockTransport`` while the database
and credential decryption stay real. Read operations use the ``gmail.readonly``
grant; send uses the separate ``gmail.send`` grant. Nothing here executes
message content — bodies are parsed as data and returned as data.
"""

from urllib.parse import quote

import httpx

from kai.gmail.models import (
    GmailMessage,
    MessageRef,
    SearchPage,
    parse_message_payload,
    parse_search_payload,
)
from kai.observability import metrics
from kai.observability.providers import provider_call

MESSAGES_URL = "https://www.googleapis.com/gmail/v1/users/me/messages"
SEND_URL = "https://www.googleapis.com/gmail/v1/users/me/messages/send"
MAX_RESULTS = 50
MAX_PAGES = 20
MAX_MESSAGES = 1_000


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


    """Raised when Google's Gmail API rejects or fails a request."""


class GoogleInvalidTokenError(_TypedStatusError):
    """HTTP 401: the stored access token is invalid or expired."""


class GoogleForbiddenError(_TypedStatusError):
    """HTTP 403: scope is missing or consent was revoked."""


class GoogleRateLimitError(_TypedStatusError):
    """HTTP 429: quota/rate limit exceeded."""


class GoogleNotFoundError(_TypedStatusError):
    """HTTP 404: the requested message does not exist."""


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
            "google denied the request; gmail scope may be missing",
            status_code=resp.status_code,
        )
    if resp.status_code == 404:
        raise GoogleNotFoundError(
            "gmail message was not found", status_code=resp.status_code
        )
    if resp.status_code == 429:
        raise GoogleRateLimitError(
            "google api rate limit exceeded, try again later",
            status_code=resp.status_code,
        )
    if resp.status_code >= 400:
        raise GoogleApiError(
            f"gmail api request failed ({resp.status_code})",
            status_code=resp.status_code,
        )


async def search_messages(
    transport: httpx.AsyncClient,
    *,
    access_token: str,
    query: str,
    max_results: int = MAX_RESULTS,
) -> list[MessageRef]:
    """Search the mailbox with a Gmail ``q`` query, following pagination."""
    refs: list[MessageRef] = []
    page_token: str | None = None
    for _ in range(MAX_PAGES):
        page = await _search_page(
            transport,
            access_token=access_token,
            query=query,
            max_results=max_results,
            page_token=page_token,
        )
        refs.extend(page.refs)
        if not page.has_more or len(refs) >= MAX_MESSAGES:
            break
        page_token = page.next_page_token
    return refs[:max_results]


async def _search_page(
    transport: httpx.AsyncClient,
    *,
    access_token: str,
    query: str,
    max_results: int,
    page_token: str | None,
) -> SearchPage:
    params: dict[str, str] = {
        "q": query,
        "maxResults": str(max_results),
        "includeSpamTrash": "false",
    }
    if page_token:
        params["pageToken"] = page_token
    with provider_call(provider=metrics.PROVIDER_GOOGLE, operation="search_messages"):
        resp = await transport.get(
            MESSAGES_URL,
            params=params,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        _raise_for_status(resp)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise GoogleApiError("gmail search returned a malformed response") from exc
        return parse_search_payload(payload)


async def get_message(
    transport: httpx.AsyncClient, *, access_token: str, message_id: str
) -> GmailMessage:
    """Read one message (headers, snippet, text body) from the mailbox."""
    with provider_call(provider=metrics.PROVIDER_GOOGLE, operation="get_message"):
        resp = await transport.get(
            f"{MESSAGES_URL}/{quote(message_id, safe='')}",
            params={"format": "full"},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        _raise_for_status(resp)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise GoogleApiError("gmail read returned a malformed response") from exc
        message = parse_message_payload(payload)
        if message is None:
            raise GoogleApiError("gmail read returned an unparseable message")
        return message


async def send_message(
    transport: httpx.AsyncClient,
    *,
    access_token: str,
    raw: str,
    idempotency_key: str | None = None,
) -> GmailMessage:
    """Send a raw RFC5322 message; return the created Gmail message."""
    with provider_call(provider=metrics.PROVIDER_GOOGLE, operation="send_message"):
        headers = {"Authorization": f"Bearer {access_token}"}
        if idempotency_key is not None:
            headers["X-Idempotency-Key"] = idempotency_key
        resp = await transport.post(
            SEND_URL,
            json={"raw": raw},
            headers=headers,
        )
        _raise_for_status(resp)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise GoogleUnreadableResultError(
                "gmail send returned a malformed response",
                status_code=resp.status_code,
            ) from exc
        message = parse_message_payload(payload)
        if message is None or not message.message_id:
            raise GoogleUnreadableResultError(
                "gmail send returned no accepted message_id",
                status_code=resp.status_code,
            )
        return message
