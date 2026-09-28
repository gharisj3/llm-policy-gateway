import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import inspect

from llm_policy_gateway.admin import create_admin_app
from llm_policy_gateway.app import create_app
from llm_policy_gateway.db import make_engine, normalize_database_url


def test_initial_migration_creates_tenants_and_keys(tmp_path, monkeypatch) -> None:
    database_url = f"sqlite:///{tmp_path / 'fresh.db'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = Config("alembic.ini")

    command.upgrade(config, "head")

    inspector = inspect(make_engine(database_url))
    assert {"tenants", "api_keys", "alembic_version"} <= set(
        inspector.get_table_names()
    )
    assert {"id", "name", "created_at", "disabled_at"} == {
        column["name"] for column in inspector.get_columns("tenants")
    }
    assert {
        "id",
        "tenant_id",
        "prefix",
        "key_hash",
        "label",
        "created_at",
        "last_used_at",
        "revoked_at",
    } == {column["name"] for column in inspector.get_columns("api_keys")}


def test_postgres_urls_use_the_pinned_driver() -> None:
    assert normalize_database_url("postgresql://host/db") == (
        "postgresql+psycopg://host/db"
    )
    assert normalize_database_url("postgres://host/db") == (
        "postgresql+psycopg://host/db"
    )
    assert normalize_database_url("sqlite:///local.db") == "sqlite:///local.db"


def test_both_apps_refuse_an_unmigrated_database(tmp_path, monkeypatch) -> None:
    url = f"sqlite:///{tmp_path / 'unmigrated.db'}"
    for app in (
        create_app(database_url=url),
        create_admin_app(database_url=url, admin_token="valid-admin-token"),
    ):
        with pytest.raises(
            RuntimeError, match="database not migrated — run make migrate"
        ):
            with TestClient(app):
                pass
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(Config("alembic.ini"), "head")
    for app in (
        create_app(database_url=url),
        create_admin_app(database_url=url, admin_token="valid-admin-token"),
    ):
        with TestClient(app) as client:
            assert client is not None
