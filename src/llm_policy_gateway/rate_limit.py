"""Per-process tenant request and estimated-token buckets."""

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RateState:
    remaining_requests: int
    remaining_tokens: int


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after


class RateLimiter(Protocol):
    def admit(self, tenant_id: str, rpm: int, tpm: int, tokens: int) -> RateState: ...


class InMemoryRateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.lock = threading.Lock()
        self.buckets: dict[str, tuple[float, float, float]] = {}

    def admit(self, tenant_id: str, rpm: int, tpm: int, tokens: int) -> RateState:
        now = self.clock()
        with self.lock:
            request_balance, token_balance, previous = self.buckets.get(
                tenant_id, (float(rpm), float(tpm), now)
            )
            elapsed = max(0.0, now - previous)
            request_balance = min(float(rpm), request_balance + elapsed * rpm / 60)
            token_balance = min(float(tpm), token_balance + elapsed * tpm / 60)
            self.buckets[tenant_id] = (request_balance, token_balance, now)
            if request_balance < 1 or token_balance < tokens:
                request_wait = (1 - request_balance) * 60 / rpm
                token_wait = (
                    (tokens - token_balance) * 60 / tpm if tokens <= tpm else 60.0
                )
                raise RateLimitExceeded(
                    max(1, math.ceil(max(request_wait, token_wait)))
                )
            request_balance -= 1
            token_balance -= tokens
            self.buckets[tenant_id] = (request_balance, token_balance, now)
            return RateState(math.floor(request_balance), math.floor(token_balance))
