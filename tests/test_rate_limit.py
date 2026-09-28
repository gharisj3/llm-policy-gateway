"""Independent request and estimated-token admission limits."""

import pytest

from llm_policy_gateway.rate_limit import InMemoryRateLimiter, RateLimitExceeded


def test_request_bucket_refills_and_returns_retry_after() -> None:
    clock = [0.0]
    limiter = InMemoryRateLimiter(lambda: clock[0])
    assert limiter.admit("one", rpm=1, tpm=100, tokens=1).remaining_requests == 0
    with pytest.raises(RateLimitExceeded) as error:
        limiter.admit("one", rpm=1, tpm=100, tokens=1)
    assert error.value.retry_after == 60
    clock[0] = 60.0
    assert limiter.admit("one", rpm=1, tpm=100, tokens=1).remaining_requests == 0
    assert limiter.admit("other", rpm=1, tpm=100, tokens=1).remaining_requests == 0


def test_token_bucket_limits_estimated_usage() -> None:
    clock = [0.0]
    limiter = InMemoryRateLimiter(lambda: clock[0])
    assert limiter.admit("one", rpm=10, tpm=5, tokens=4).remaining_tokens == 1
    with pytest.raises(RateLimitExceeded) as error:
        limiter.admit("one", rpm=10, tpm=5, tokens=2)
    assert error.value.retry_after == 12
    clock[0] = 12.0
    assert limiter.admit("one", rpm=10, tpm=5, tokens=2).remaining_tokens == 0
