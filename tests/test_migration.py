from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

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
