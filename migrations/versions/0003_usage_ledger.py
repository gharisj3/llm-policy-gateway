"""Add the daily usage ledger.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "usage_ledger",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id", sa.String(36), sa.ForeignKey("tenants.id"), nullable=False
        ),
        sa.Column("request_id", sa.String(36), nullable=False, unique=True),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(14, 6), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_usage_ledger_tenant_id", "usage_ledger", ["tenant_id"])
    op.create_index("ix_usage_ledger_day", "usage_ledger", ["day"])


def downgrade() -> None:
    op.drop_index("ix_usage_ledger_day", table_name="usage_ledger")
    op.drop_index("ix_usage_ledger_tenant_id", table_name="usage_ledger")
    op.drop_table("usage_ledger")
