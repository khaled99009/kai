"""Deterministic contracts for durable provider-effect states."""

from __future__ import annotations

import pytest

from kai.runtime.effect_service import settle_provider_effect
from kai.runtime.effects import ProviderEffectStatus

pytestmark = pytest.mark.no_db_unit


@pytest.mark.asyncio
async def test_only_terminal_states_are_accepted_for_settlement() -> None:
    assert ProviderEffectStatus.APPLIED.value == "APPLIED"
    assert ProviderEffectStatus.AMBIGUOUS.value == "AMBIGUOUS"
    with pytest.raises(ValueError, match="only settle"):
        await settle_provider_effect(
            None,  # type: ignore[arg-type]
            effect_id=None,  # type: ignore[arg-type]
            status=ProviderEffectStatus.RESERVED,
        )


def test_reserved_is_not_a_terminal_success_state() -> None:
    assert ProviderEffectStatus.RESERVED != ProviderEffectStatus.APPLIED
    assert ProviderEffectStatus.RESERVED != ProviderEffectStatus.AMBIGUOUS
