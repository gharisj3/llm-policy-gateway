"""Atomic daily reservations and settlement in Decimal dollars."""

import threading
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import ROUND_UP, Decimal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from llm_policy_gateway.db import Tenant, UsageLedger
from llm_policy_gateway.pricing import PricingDocument

SIX_PLACES = Decimal("0.000001")
THOUSAND = Decimal(1000)
SQLITE_LOCK = threading.RLock()


def utc_now() -> datetime:
    return datetime.now(UTC)


def estimate_tokens(texts: list[str]) -> int:
    return (sum(len(text) for text in texts) + 3) // 4


class BudgetExceeded(Exception):
    pass


class BudgetManager:
    def __init__(
        self,
        factory: sessionmaker,
        pricing: PricingDocument,
        sqlite: bool,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.factory = factory
        self.pricing = pricing
        self.lock = SQLITE_LOCK if sqlite else threading.RLock()
        self.sqlite = sqlite
        self.clock = clock

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
        price = self.pricing.models[model]
        amount = (
            Decimal(prompt_tokens) * price.input_per_1k_tokens
            + Decimal(completion_tokens) * price.output_per_1k_tokens
        ) / THOUSAND
        return amount.quantize(SIX_PLACES, rounding=ROUND_UP)

    def _lock_tenant(self, session: Session, tenant_id: str) -> None:
        if not self.sqlite:
            session.execute(
                select(Tenant.id).where(Tenant.id == tenant_id).with_for_update()
            )

    def _rows(self, session: Session, tenant_id: str, day: date) -> list[UsageLedger]:
        return list(
            session.scalars(
                select(UsageLedger).where(
                    UsageLedger.tenant_id == tenant_id, UsageLedger.day == day
                )
            )
        )

    def reserve(
        self,
        tenant_id: str,
        model: str,
        prompt_tokens: int,
        max_completion_tokens: int,
        daily_limit: Decimal,
    ) -> str:
        estimate = self.cost(model, prompt_tokens, max_completion_tokens)
        now = self.clock()
        day = now.astimezone(UTC).date()
        with self.lock, self.factory() as session, session.begin():
            self._lock_tenant(session, tenant_id)
            used = sum(
                (
                    row.cost_usd
                    for row in self._rows(session, tenant_id, day)
                    if row.status in ("reserved", "settled")
                ),
                Decimal(0),
            )
            if used + estimate > daily_limit:
                raise BudgetExceeded
            request_id = str(uuid4())
            session.add(
                UsageLedger(
                    tenant_id=tenant_id,
                    request_id=request_id,
                    day=day,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=max_completion_tokens,
                    cost_usd=estimate,
                    status="reserved",
                    created_at=now,
                )
            )
        return request_id

    def settle(
        self,
        request_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        daily_limit: Decimal,
    ) -> Decimal:
        with self.lock, self.factory() as session, session.begin():
            row = session.scalar(
                select(UsageLedger).where(UsageLedger.request_id == request_id)
            )
            if row is None or row.status != "reserved":
                raise ValueError("Reservation is not active")
            self._lock_tenant(session, row.tenant_id)
            row.prompt_tokens = prompt_tokens
            row.completion_tokens = completion_tokens
            row.cost_usd = self.cost(row.model, prompt_tokens, completion_tokens)
            row.status = "settled"
            row.settled_at = self.clock()
            tenant_id, day = row.tenant_id, row.day
        return self.remaining(tenant_id, daily_limit, day)

    def release(self, request_id: str) -> None:
        with self.lock, self.factory() as session, session.begin():
            row = session.scalar(
                select(UsageLedger).where(UsageLedger.request_id == request_id)
            )
            if row is None or row.status != "reserved":
                raise ValueError("Reservation is not active")
            self._lock_tenant(session, row.tenant_id)
            row.status = "released"
            row.cost_usd = Decimal(0)
            row.settled_at = self.clock()

    def remaining(
        self, tenant_id: str, daily_limit: Decimal, day: date | None = None
    ) -> Decimal:
        return self.usage(tenant_id, day, daily_limit)["remaining"]

    def usage(
        self,
        tenant_id: str,
        day: date | None = None,
        daily_limit: Decimal | None = None,
    ) -> dict:
        day = day or self.clock().astimezone(UTC).date()
        with self.factory() as session:
            rows = self._rows(session, tenant_id, day)
        settled = sum((r.cost_usd for r in rows if r.status == "settled"), Decimal(0))
        reserved = sum((r.cost_usd for r in rows if r.status == "reserved"), Decimal(0))
        by_model: dict[str, dict] = {}
        for row in rows:
            if row.status != "settled":
                continue
            item = by_model.setdefault(
                row.model,
                {
                    "requests": 0,
                    "cost_usd": Decimal(0),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            )
            item["requests"] += 1
            item["cost_usd"] += row.cost_usd
            item["prompt_tokens"] += row.prompt_tokens
            item["completion_tokens"] += row.completion_tokens
        return {
            "day": day,
            "settled": settled,
            "reserved": reserved,
            "remaining": None
            if daily_limit is None
            else max(Decimal(0), daily_limit - settled - reserved),
            "request_count": sum(item["requests"] for item in by_model.values()),
            "per_model": by_model,
        }
