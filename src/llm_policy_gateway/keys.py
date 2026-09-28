"""Tenant management and one-time API key issuance."""

import re
import secrets
from dataclasses import dataclass
from datetime import timedelta

from argon2 import PasswordHasher, Type
from argon2.exceptions import VerificationError, VerifyMismatchError
from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from llm_policy_gateway.db import ApiKey, Tenant, utc_now

KEY_PATTERN = re.compile(r"^lpg_([0-9a-f]{8})_([A-Za-z0-9_-]{32,})$")
HASHER = PasswordHasher(type=Type.ID)


class InvalidKey(Exception):
    pass


@dataclass(frozen=True)
class IssuedKey:
    id: str
    key: str
    prefix: str


@dataclass(frozen=True)
class VerifiedKey:
    tenant: Tenant
    api_key: ApiKey


def create_tenant(session: Session, name: str) -> Tenant:
    tenant = Tenant(name=name)
    session.add(tenant)
    session.commit()
    session.refresh(tenant)
    return tenant


def issue_key(session: Session, tenant_id: str, label: str) -> IssuedKey:
    tenant = session.get(Tenant, tenant_id)
    if tenant is None or tenant.disabled_at is not None:
        raise ValueError("Tenant does not exist or is disabled")
    while True:
        prefix = secrets.token_hex(4)
        if session.scalar(select(ApiKey.id).where(ApiKey.prefix == prefix)) is None:
            break
    full_key = f"lpg_{prefix}_{secrets.token_urlsafe(32)}"
    row = ApiKey(
        tenant_id=tenant_id,
        prefix=prefix,
        key_hash=HASHER.hash(full_key),
        label=label,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return IssuedKey(id=row.id, key=full_key, prefix=prefix)


def revoke_key(session: Session, key_id: str) -> ApiKey:
    row = session.get(ApiKey, key_id)
    if row is None:
        raise ValueError("Key does not exist")
    if row.revoked_at is None:
        row.revoked_at = utc_now()
        session.commit()
        session.refresh(row)
    return row


def list_keys(session: Session, tenant_id: str) -> list[ApiKey]:
    if session.get(Tenant, tenant_id) is None:
        raise ValueError("Tenant does not exist")
    return list(
        session.scalars(
            select(ApiKey)
            .where(ApiKey.tenant_id == tenant_id)
            .order_by(ApiKey.created_at, ApiKey.id)
        )
    )


def verify_key(session: Session, presented: str) -> VerifiedKey:
    match = KEY_PATTERN.fullmatch(presented)
    if match is None:
        raise InvalidKey
    row = session.scalar(select(ApiKey).where(ApiKey.prefix == match.group(1)))
    if row is None:
        raise InvalidKey
    try:
        valid = HASHER.verify(row.key_hash, presented)
    except (VerificationError, VerifyMismatchError) as error:
        raise InvalidKey from error
    if not valid or row.revoked_at is not None:
        raise InvalidKey
    tenant = session.get(Tenant, row.tenant_id)
    if tenant is None or tenant.disabled_at is not None:
        raise InvalidKey
    cutoff = utc_now() - timedelta(minutes=1)
    session.execute(
        update(ApiKey)
        .where(
            ApiKey.id == row.id,
            or_(ApiKey.last_used_at.is_(None), ApiKey.last_used_at <= cutoff),
        )
        .values(last_used_at=utc_now())
        .execution_options(synchronize_session=False)
    )
    session.commit()
    return VerifiedKey(tenant=tenant, api_key=row)
