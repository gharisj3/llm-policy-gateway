import re
from datetime import timedelta
from typing import Annotated

import pytest
from alembic import command
from alembic.config import Config
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import select

from llm_policy_gateway.app import create_app
from llm_policy_gateway.auth import AuthContext, authenticate_request
from llm_policy_gateway.db import (
    ApiKey,
    Tenant,
    make_engine,
    make_session_factory,
    utc_now,
)
from llm_policy_gateway.keys import create_tenant, issue_key, verify_key


@pytest.fixture
def database(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'keys.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(Config("alembic.ini"), "head")
    engine = make_engine(url)
    factory = make_session_factory(engine)
    yield url, factory
    engine.dispose()


def test_key_is_returned_once_and_only_hash_is_stored(database) -> None:
    _, factory = database
    with factory() as session:
        tenant = create_tenant(session, "clinical-team")
        issued = issue_key(session, tenant.id, "primary")
        row = session.scalar(select(ApiKey).where(ApiKey.id == issued.id))
        assert row is not None
        assert re.fullmatch(r"lpg_[0-9a-f]{8}_[A-Za-z0-9_-]{32,}", issued.key)
        assert row.prefix == issued.prefix
        assert re.fullmatch(r"[0-9a-f]{64}", row.key_hash)
        assert issued.key not in row.key_hash
        assert not hasattr(row, "key")


def test_rejected_keys_all_compute_hmac_and_share_401(database, monkeypatch) -> None:
    from llm_policy_gateway import keys

    url, factory = database
    with factory() as session:
        tenant = create_tenant(session, "clinical-team")
        issued = issue_key(session, tenant.id, "primary")
    calls = []
    real_digest = keys._digest_key

    def spy(full_key: str, pepper: bytes) -> str:
        calls.append(full_key)
        return real_digest(full_key, pepper)

    monkeypatch.setattr(keys, "_digest_key", spy)
    app = create_app(database_url=url)

    @app.get("/private")
    def private(context: Annotated[AuthContext, Depends(authenticate_request)]) -> dict:
        return {"tenant_id": context.tenant_id}

    wrong = issued.key[:-1] + ("A" if issued.key[-1] != "A" else "B")
    unknown = "lpg_deadbeef_" + "a" * 43
    with TestClient(app) as client:
        responses = [
            client.get("/private", headers={"Authorization": f"Bearer {value}"})
            for value in (unknown, wrong)
        ]
        with factory() as session:
            row = session.get(ApiKey, issued.id)
            row.revoked_at = utc_now()
            session.commit()
        responses.append(
            client.get("/private", headers={"Authorization": f"Bearer {issued.key}"})
        )
    assert calls == [unknown, wrong, issued.key]
    assert all(response.status_code == 401 for response in responses)
    assert len({str(response.json()) for response in responses}) == 1


def test_missing_short_or_placeholder_pepper_refuses_both_apps(
    database, monkeypatch
) -> None:
    from llm_policy_gateway.admin import create_admin_app
    from llm_policy_gateway.keys import PEPPER_PLACEHOLDER

    url, _ = database
    for pepper in ("", "short", PEPPER_PLACEHOLDER):
        monkeypatch.setenv("KEY_PEPPER", pepper)
        with pytest.raises(ValueError, match="KEY_PEPPER"):
            with TestClient(create_app(database_url=url)):
                pass
        with pytest.raises(ValueError, match="KEY_PEPPER"):
            with TestClient(
                create_admin_app(database_url=url, admin_token="valid-admin-token")
            ):
                pass


def test_authentication_revocation_disabled_tenant_and_uniform_errors(database) -> None:
    url, factory = database
    with factory() as session:
        tenant = create_tenant(session, "clinical-team")
        issued = issue_key(session, tenant.id, "primary")
    app = create_app(database_url=url)

    @app.get("/private")
    def private(context: Annotated[AuthContext, Depends(authenticate_request)]) -> dict:
        return {"tenant_id": context.tenant_id}

    with TestClient(app) as client:
        good = client.get("/private", headers={"Authorization": f"Bearer {issued.key}"})
        assert good.status_code == 200
        assert good.json()["tenant_id"] == tenant.id

        bad_keys = [
            issued.key[:-1] + ("A" if issued.key[-1] != "A" else "B"),
            "lpg_deadbeef_" + "a" * 43,
        ]
        failures = [
            client.get("/private", headers={"Authorization": f"Bearer {key}"})
            for key in bad_keys
        ]
        failures.append(client.get("/private"))
        failures.append(client.get("/private", headers={"Authorization": "Token abc"}))
        with factory() as session:
            row = session.get(ApiKey, issued.id)
            first_used = row.last_used_at
            assert first_used is not None
            verify_key(session, issued.key)
            session.refresh(row)
            assert row.last_used_at == first_used
            row.last_used_at = utc_now() - timedelta(minutes=2)
            session.commit()
            verify_key(session, issued.key)
            session.refresh(row)
            assert row.last_used_at > first_used
            row.revoked_at = utc_now()
            session.commit()
        failures.append(
            client.get("/private", headers={"Authorization": f"Bearer {issued.key}"})
        )
        with factory() as session:
            row = session.get(ApiKey, issued.id)
            row.revoked_at = None
            session.get(Tenant, tenant.id).disabled_at = utc_now()
            session.commit()
        failures.append(
            client.get("/private", headers={"Authorization": f"Bearer {issued.key}"})
        )
        assert all(response.status_code == 401 for response in failures)
        assert len({str(response.json()) for response in failures}) == 1


def test_tenant_without_policy_is_denied(database) -> None:
    url, factory = database
    with factory() as session:
        tenant = create_tenant(session, "unlisted")
        issued = issue_key(session, tenant.id, "primary")
    app = create_app(database_url=url)

    @app.get("/private")
    def private(context: Annotated[AuthContext, Depends(authenticate_request)]) -> dict:
        return {"tenant_id": context.tenant_id}

    with TestClient(app) as client:
        response = client.get(
            "/private", headers={"Authorization": f"Bearer {issued.key}"}
        )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "policy_denied"
