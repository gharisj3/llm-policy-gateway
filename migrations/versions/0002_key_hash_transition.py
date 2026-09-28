"""Mark the API key hash transition.

Existing argon2 hashes are invalidated. This affects development databases only;
nothing has been released. Reissue development keys after applying this migration.

Revision ID: 0002
Revises: 0001
"""

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
