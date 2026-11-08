"""Tenant-scoped Gmail operations.

Messages are read and sent for exactly one ``(account_id, tenant_id)``; the
access token is decrypted per request and never returned or logged. Sending is
verified deterministically in application code: the provider's accepted
``message_id`` plus a read-after-send of that message are checked for message
id, SENT label, thread, sender, recipient and subject. Gmail exposes no read
receipts, so read-acknowledgement is never claimed.
"""

import base64
from dataclasses import dataclass, field
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kai.db.models import GoogleAccount
from kai.gmail import client
from kai.gmail.client import (
    GoogleApiError,
    GoogleNotFoundError,
)
from kai.gmail.models import GmailMessage, MessageRef
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


async def search_messages(
    session: AsyncSession,
    transport: httpx.AsyncClient,
    *,
    account_id: UUID,
    tenant_id: UUID,
    query: str,
    max_results: int,
) -> list[MessageRef]:
    account = await _load_account(session, account_id=account_id, tenant_id=tenant_id)
    return await client.search_messages(
        transport,
        access_token=await usable_access_token(session, transport, account=account),
        query=query,
        max_results=max_results,
    )


async def get_message(
    session: AsyncSession,
    transport: httpx.AsyncClient,
    *,
    account_id: UUID,
    tenant_id: UUID,
    message_id: str,
) -> GmailMessage:
    account = await _load_account(session, account_id=account_id, tenant_id=tenant_id)
    return await client.get_message(
        transport,
        access_token=await usable_access_token(session, transport, account=account),
        message_id=message_id,
    )


@dataclass(frozen=True)
class FieldMismatch:
    """One deterministic verification mismatch."""

    field: str
    expected: str
    actual: str


@dataclass(frozen=True)
class SendRequest:
    """A validated outbound message. The sender is never client-supplied."""

    to: str
    subject: str
    body: str


@dataclass(frozen=True)
class SendOutcome:
    """Read-after-send result. ``verified`` requires every check to pass."""

    message: GmailMessage | None
    mismatches: tuple[FieldMismatch, ...] = field(default_factory=tuple)

    @property
    def verified(self) -> bool:
        return not self.mismatches


def build_raw_message(
    *, sender_email: str, to: str, subject: str, body: str
) -> str:
    """Build an RFC5322 message from validated, controlled fields only."""
    text = (
        f"From: {sender_email}\r\n"
        f"To: {to}\r\n"
        f"Subject: {subject}\r\n"
        "\r\n"
        f"{body}"
    )
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def verify_send(
    message: GmailMessage,
    *,
    expected_id: str,
    sender_email: str,
    to: str,
    subject: str,
) -> tuple[FieldMismatch, ...]:
    """Deterministic post-send verification with concrete Gmail evidence.

    Fields: provider message_id, SENT label, thread, sender, recipient,
    subject. A failure is a failure — the send must not be reported as success.
    """
    mismatches: list[FieldMismatch] = []
    if message.message_id != expected_id:
        mismatches.append(
            FieldMismatch("message_id", expected_id, message.message_id)
        )
    if not message.thread_id:
        mismatches.append(FieldMismatch("thread_id", "non-empty", "missing"))
    if "SENT" not in message.label_ids:
        mismatches.append(
            FieldMismatch("label_ids", "contains SENT", ",".join(message.label_ids))
        )
    from_emails = [message.from_addr.email.casefold()] if message.from_addr else []
    if sender_email.casefold() not in from_emails:
        mismatches.append(
            FieldMismatch("sender", sender_email, ",".join(from_emails))
        )
    to_emails = [address.email.casefold() for address in message.to]
    if to.casefold() not in to_emails:
        mismatches.append(FieldMismatch("recipient", to, ",".join(to_emails)))
    if message.subject != subject:
        mismatches.append(FieldMismatch("subject", subject, message.subject))
    return tuple(mismatches)


async def send_and_verify(
    session: AsyncSession,
    transport: httpx.AsyncClient,
    *,
    account_id: UUID,
    tenant_id: UUID,
    req: SendRequest,
    idempotency_key: str | None = None,
) -> SendOutcome:
    """Send one message, read it back, and verify it with Gmail evidence.

    A failure to *send* propagates: nothing was delivered, so the caller must
    see the provider's error. A failure to *read the message back* (401/403/429
    as well as 404, malformed payloads, and 5xx) means the evidence cannot be
    established, so the outcome is explicitly not verified rather than a success
    or a tool failure: the message may have been delivered, and reporting a
    failure would invite a duplicate send.
    """
    account = await _load_account(session, account_id=account_id, tenant_id=tenant_id)
    sender_email = account.email
    token = await usable_access_token(session, transport, account=account)
    raw = build_raw_message(
        sender_email=sender_email, to=req.to, subject=req.subject, body=req.body
    )
    sent = await client.send_message(
        transport,
        access_token=token,
        raw=raw,
        idempotency_key=idempotency_key,
    )
    try:
        readback = await client.get_message(
            transport, access_token=token, message_id=sent.message_id
        )
    except GoogleNotFoundError:
        return SendOutcome(
            message=None,
            mismatches=(
                FieldMismatch("read_after_send", "message readable", "not found"),
            ),
        )
    except GoogleApiError as exc:
        return SendOutcome(
            message=None,
            mismatches=(
                FieldMismatch("read_after_send", "message readable", str(exc)),
            ),
        )
    mismatches = verify_send(
        readback,
        expected_id=sent.message_id,
        sender_email=sender_email,
        to=req.to,
        subject=req.subject,
    )
    return SendOutcome(message=readback, mismatches=mismatches)
