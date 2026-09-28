"""Daily reservations use exact Decimal amounts and atomic admission."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config

from llm_policy_gateway.budget import BudgetExceeded, BudgetManager, estimate_tokens
from llm_policy_gateway.db import make_engine, make_session_factory
from llm_policy_gateway.keys import create_tenant
from llm_policy_gateway.pricing import load_pricing


@pytest.fixture
def budget(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'budget.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(Config("alembic.ini"), "head")
    engine = make_engine(url)
    factory = make_session_factory(engine)
    with factory() as session:
        tenant = create_tenant(session, "clinical-team")
    clock = [datetime(2026, 1, 1, 23, 59, tzinfo=UTC)]
    manager = BudgetManager(
        factory,
        load_pricing("pricing.example.yaml", {"clinical-local"}),
        sqlite=True,
        clock=lambda: clock[0],
    )
    yield manager, tenant.id, clock
    engine.dispose()


def test_decimal_cost_reserve_settle_and_release(budget) -> None:
    manager, tenant_id, _clock = budget
    assert estimate_tokens(["abc", "de"]) == 2
    assert manager.cost("clinical-local", 1000, 1000) == Decimal("0.060000")
    assert isinstance(manager.cost("clinical-local", 1, 1), Decimal)
    request_id = manager.reserve(
        tenant_id, "clinical-local", 1000, 1000, Decimal("0.06")
    )
    with pytest.raises(BudgetExceeded):
        manager.reserve(tenant_id, "clinical-local", 1000, 1000, Decimal("0.06"))
    remaining = manager.settle(request_id, 100, 10, Decimal("0.06"))
    assert remaining == Decimal("0.057600")
    usage = manager.usage(tenant_id, daily_limit=Decimal("0.06"))
    assert usage["settled"] == Decimal("0.002400")
    assert usage["request_count"] == 1
    assert usage["per_model"]["clinical-local"]["prompt_tokens"] == 100
    second = manager.reserve(tenant_id, "clinical-local", 1000, 0, Decimal("0.06"))
    manager.release(second)
    assert manager.usage(tenant_id)["reserved"] == Decimal(0)
    assert manager.usage(tenant_id)["settled"] == Decimal("0.002400")


def test_atomic_reservations_and_day_rollover(budget) -> None:
    manager, tenant_id, clock = budget

    def attempt() -> bool:
        try:
            manager.reserve(tenant_id, "clinical-local", 1000, 1000, Decimal("0.06"))
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(lambda _index: attempt(), range(2))) == [False, True]
    clock[0] += timedelta(minutes=2)
    assert manager.remaining(tenant_id, Decimal("0.06")) == Decimal("0.06")
    assert attempt()
