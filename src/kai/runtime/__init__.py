"""The Agent Runtime: one guarded vertical slice over the job system.

The runtime owns fixed orchestration (submit/trigger), the sealed slice
contract, LLM-as-data understanding, deterministic identity and slot logic,
and the resume-safe execution envelope. Execution authority stays with the
Tool Gateway; the runtime only builds intents and reads deterministic results.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AgentRuntime": ("kai.runtime.service", "AgentRuntime"),
    "GoalSubmission": ("kai.runtime.schemas", "GoalSubmission"),
    "RuntimeExecution": ("kai.runtime.models", "RuntimeExecution"),
    "RuntimeFatal": ("kai.runtime.service", "RuntimeFatal"),
    "RuntimeNotFoundError": ("kai.runtime.service", "RuntimeNotFoundError"),
    "RuntimeResumeError": ("kai.runtime.service", "RuntimeResumeError"),
    "RuntimeTaskResponse": ("kai.runtime.schemas", "RuntimeTaskResponse"),
    "SubmitResponse": ("kai.runtime.schemas", "SubmitResponse"),
    "TriggerOutcome": ("kai.runtime.service", "TriggerOutcome"),
    "TriggerResponse": ("kai.runtime.schemas", "TriggerResponse"),
    "UnderstandingFailed": ("kai.runtime.understanding", "UnderstandingFailed"),
    "UnderstandingOutput": ("kai.runtime.schemas", "UnderstandingOutput"),
    "get_runtime_gateway": ("kai.runtime.deps", "get_runtime_gateway"),
    "get_runtime_service": ("kai.runtime.deps", "get_runtime_service"),
    "understand_scenario": ("kai.runtime.understanding", "understand_scenario"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        try:
            module = import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            raise AttributeError(name) from exc
        globals()[name] = module
        return module
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "AgentRuntime",
    "GoalSubmission",
    "RuntimeExecution",
    "RuntimeFatal",
    "RuntimeNotFoundError",
    "RuntimeResumeError",
    "RuntimeTaskResponse",
    "SubmitResponse",
    "TriggerOutcome",
    "TriggerResponse",
    "UnderstandingFailed",
    "UnderstandingOutput",
    "get_runtime_gateway",
    "get_runtime_service",
    "understand_scenario",
]
