"""Application-side guard around one provider attempt.

Whatever the backend does, the caller gets a typed `ModelResult` within the request deadline.
Exceptions and overruns become typed failures, never risk judgments. Cancellation of the caller is
propagated. A cancelled remote request may still be computing server-side; callers must count the
attempt as made.
"""

from __future__ import annotations

import asyncio
import time

from .base import ModelError, ModelProvider, ModelRequest, ModelResult

# Grace on top of the request deadline so a provider can report its own timeout first.
DEADLINE_GRACE_S = 0.25


async def guarded_generate(provider: ModelProvider, request: ModelRequest) -> ModelResult:
    started = time.monotonic()

    def failure(kind: str, message: str) -> ModelResult:
        return ModelResult(
            model_id=provider.model_id,
            elapsed_s=time.monotonic() - started,
            error=ModelError(kind=kind, message=message),  # type: ignore[arg-type]
        )

    try:
        result = await asyncio.wait_for(
            provider.generate(request), timeout=request.timeout_s + DEADLINE_GRACE_S
        )
    except TimeoutError:
        return failure("timeout", "request exceeded its deadline")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Only the exception type is kept; messages can contain request content or credentials.
        return failure("provider_error", f"provider raised {type(exc).__name__}")
    if not isinstance(result, ModelResult):
        return failure("provider_error", "provider returned an invalid result type")
    elapsed = time.monotonic() - started
    if elapsed > request.timeout_s + DEADLINE_GRACE_S:
        # A provider that swallowed cancellation and answered late is still a timeout.
        return failure("timeout", "result arrived after the deadline")
    # The guard's own monotonic measurement is authoritative for latency.
    return result.model_copy(update={"elapsed_s": elapsed})
