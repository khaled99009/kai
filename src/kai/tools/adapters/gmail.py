"""Gmail tool adapter.

Three tools share this adapter:

* ``gmail.search`` — search the mailbox and return message references (reuses
  ``gmail.service.search_messages``).
* ``gmail.read`` — read one message body (reuses
  ``gmail.service.get_message``); the body is returned as *data*, never
  executed or interpreted.
* ``gmail.send`` — send via ``gmail.service.send_and_verify``; ``verified``
  mirrors the service's deterministic read-after-send verification and
  ``mismatches`` surface what the provider actually stored.

Message bodies are T2 untrusted external data: they are payloads, not
commands, and cannot alter policy, the registry, or the verification contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kai.gmail import client as gmail_client
from kai.gmail.models import EmailAddress, GmailMessage, MessageRef
from kai.gmail.service import (
    AccountNotFoundError,
    SendRequest,
    get_message,
    search_messages,
    send_and_verify,
)
from kai.tools.adapters.base import AdapterBase, provider_failure
from kai.tools.context import ToolContext
from kai.tools.errors import PROVIDER_UNREACHABLE, ToolExecutionError
from kai.tools.execution import ToolExecutionOutcome, VerificationMismatch

_PROVIDER_CODES: dict[type[Exception], str] = {
    gmail_client.GoogleInvalidTokenError: "TOKEN_INVALID",
    gmail_client.GoogleForbiddenError: "PERMISSION_DENIED",
    gmail_client.GoogleRateLimitError: "RATE_LIMITED",
    gmail_client.GoogleNotFoundError: "NOT_FOUND",
    gmail_client.GoogleUnreadableResultError: PROVIDER_UNREACHABLE,
    gmail_client.GoogleApiError: "PROVIDER_ERROR",
}


def _serialize_email_address(address: EmailAddress | None) -> dict[str, Any] | None:
    if address is None:
        return None
    return {"name": address.name, "email": address.email}


def _serialize_message(message: GmailMessage) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "thread_id": message.thread_id,
        "subject": message.subject,
        "from": _serialize_email_address(message.from_addr),
        "to": [_serialize_email_address(address) for address in message.to],
        "date": message.date,
        "snippet": message.snippet,
        "body": message.body_text,
        "label_ids": list(message.label_ids),
    }


def _serialize_ref(ref: MessageRef) -> dict[str, Any]:
    return {"message_id": ref.message_id, "thread_id": ref.thread_id}


def _mismatches(pair: tuple[Any, ...]) -> tuple[VerificationMismatch, ...]:
    return tuple(
        VerificationMismatch(field=m.field, expected=m.expected, actual=m.actual)
        for m in pair
    )


class GmailAdapter(AdapterBase):
    """Executes the three gmail tools against the stored Google account."""

    async def execute(
        self,
        *,
        tool: Any,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolExecutionOutcome:
        if tool.name == "gmail.search":
            return await self._search(arguments=arguments, context=context)
        if tool.name == "gmail.read":
            return await self._read(arguments=arguments, context=context)
        if tool.name == "gmail.send":
            return await self._send(arguments=arguments, context=context)
        raise self.unimplemented(tool.name)

    async def _search(
        self, *, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecutionOutcome:
        self.record("gmail.search")
        transport = self.require_transport(context)
        try:
            refs = await search_messages(
                context.session,
                transport,
                account_id=self.require_account(context),
                tenant_id=context.tenant_id,
                query=arguments["query"],
                max_results=arguments["max_results"],
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
            data=[_serialize_ref(ref) for ref in refs],
            evidence={"query": arguments["query"], "total": len(refs)},
        )

    async def _read(
        self, *, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecutionOutcome:
        self.record("gmail.read")
        transport = self.require_transport(context)
        try:
            message = await get_message(
                context.session,
                transport,
                account_id=self.require_account(context),
                tenant_id=context.tenant_id,
                message_id=arguments["message_id"],
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
            data=_serialize_message(message),
            evidence={"message_id": message.message_id},
        )

    async def _send(
        self, *, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecutionOutcome:
        self.record("gmail.send")
        transport = self.require_transport(context)
        req = SendRequest(
            to=str(arguments["to"]),
            subject=arguments["subject"],
            body=arguments["body"],
        )
        try:
            outcome = await send_and_verify(
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

        evidence: dict[str, Any] = {}
        if outcome.message is not None:
            evidence["message_id"] = outcome.message.message_id
            evidence["thread_id"] = outcome.message.thread_id
            evidence["label_ids"] = list(outcome.message.label_ids)
            evidence["mismatch_count"] = len(outcome.mismatches)
        return ToolExecutionOutcome(
            ok=True,
            verified=outcome.verified,
            data=(
                _serialize_message(outcome.message)
                if outcome.message is not None
                else None
            ),
            evidence=evidence,
            mismatches=_mismatches(outcome.mismatches),
        )


__all__ = ["GmailAdapter"]
