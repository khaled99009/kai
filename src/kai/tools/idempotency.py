"""Deterministic provider-effect idempotency keys.

The key is derived only from trusted execution identity, the validated tool name,
and normalized arguments. Raw credentials and provider responses are never part
of it. A retry of the same governed action therefore reuses the same key.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any
from uuid import UUID


def provider_idempotency_key(
    *,
    tenant_id: UUID,
    task_id: UUID,
    request_id: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    account_id: UUID | None,
) -> str:
    material = {
        "tenant_id": str(tenant_id),
        "task_id": str(task_id),
        "request_id": request_id,
        "tool_name": tool_name,
        "arguments": arguments,
        "account_id": str(account_id) if account_id is not None else None,
    }
    encoded = json.dumps(
        material, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return "kai-" + hashlib.sha256(encoded).hexdigest()


__all__ = ["provider_idempotency_key"]
