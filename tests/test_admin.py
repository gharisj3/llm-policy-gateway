import json

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from llm_policy_gateway.admin import PLACEHOLDER_TOKEN, create_admin_app
from llm_policy_gateway.app import create_app
from llm_policy_gateway.cli import main


@pytest.fixture
def database_url(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'admin.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(Config("alembic.ini"), "head")
    return url


def test_admin_auth_and_one_time_key_response(database_url) -> None:
    app = create_admin_app(database_url=database_url, admin_token="secret-admin-token")
    auth = {"Authorization": "Bearer secret-admin-token"}
    with TestClient(app) as client:
        missing = client.post("/admin/tenants", json={"name": "clinical-team"})
        wrong = client.post(
            "/admin/tenants",
            json={"name": "clinical-team"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert missing.status_code == wrong.status_code == 401
        assert missing.json() == wrong.json()

        tenant = client.post(
            "/admin/tenants", json={"name": "clinical-team"}, headers=auth
        )
        assert tenant.status_code == 201
        tenant_id = tenant.json()["id"]
        key = client.post(
            f"/admin/tenants/{tenant_id}/keys",
            json={"label": "primary"},
            headers=auth,
        )
        assert key.status_code == 201
        full_key = key.json()["key"]
        listed = client.get(f"/admin/tenants/{tenant_id}/keys", headers=auth)
        assert listed.status_code == 200
        assert len(listed.json()) == 1
        assert "key" not in listed.json()[0]
        assert "key_hash" not in listed.json()[0]
        assert full_key not in listed.text
        revoked = client.post(f"/admin/keys/{key.json()['id']}/revoke", headers=auth)
        assert revoked.status_code == 200
        assert revoked.json()["revoked_at"] is not None
        assert "key" not in revoked.json()


def test_placeholder_token_refuses_startup_and_public_has_no_admin(
    database_url,
) -> None:
    for token in ("", PLACEHOLDER_TOKEN):
        with pytest.raises(ValueError, match="ADMIN_TOKEN"):
            create_admin_app(database_url=database_url, admin_token=token)
    with TestClient(create_app(database_url=database_url)) as client:
        assert (
            client.post("/admin/tenants", json={"name": "clinical-team"}).status_code
            == 404
        )


def test_cli_tenant_and_key_commands(database_url, capsys) -> None:
    assert main(["tenant", "create", "clinical-team"], database_url) == 0
    tenant = json.loads(capsys.readouterr().out)
    assert main(["key", "create", tenant["id"], "--label", "cli"], database_url) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["key"].startswith("lpg_")
    assert main(["key", "list", tenant["id"]], database_url) == 0
    listed = json.loads(capsys.readouterr().out)
    assert "key" not in listed[0]
    assert main(["key", "revoke", created["id"]], database_url) == 0
    revoked = json.loads(capsys.readouterr().out)
    assert revoked["revoked_at"] is not None
