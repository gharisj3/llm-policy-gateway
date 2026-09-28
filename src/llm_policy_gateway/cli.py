"""Local administration and dual-port service launcher."""

import argparse
import asyncio
import json
import os
import sys

import uvicorn

from llm_policy_gateway.admin import create_admin_app, parse_bind
from llm_policy_gateway.app import create_app
from llm_policy_gateway.db import make_engine, make_session_factory
from llm_policy_gateway.keys import (
    create_tenant,
    disable_tenant,
    issue_key,
    list_keys,
    list_tenants,
    revoke_key,
)


async def _serve() -> None:
    admin_app = create_admin_app()
    public_app = create_app()
    gateway_host, gateway_port = parse_bind(os.getenv("GATEWAY_BIND", "127.0.0.1:8080"))
    admin_host, admin_port = parse_bind(os.getenv("ADMIN_BIND", "127.0.0.1:8081"))
    gateway = uvicorn.Server(
        uvicorn.Config(public_app, host=gateway_host, port=gateway_port)
    )
    admin = uvicorn.Server(uvicorn.Config(admin_app, host=admin_host, port=admin_port))
    await asyncio.gather(gateway.serve(), admin.serve())


def main(argv: list[str] | None = None, database_url: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lpg")
    root = parser.add_subparsers(dest="group", required=True)
    root.add_parser("serve")
    tenant_parser = root.add_parser("tenant")
    tenant_actions = tenant_parser.add_subparsers(dest="action", required=True)
    tenant_create = tenant_actions.add_parser("create")
    tenant_create.add_argument("name")
    tenant_disable = tenant_actions.add_parser("disable")
    tenant_disable.add_argument("tenant_id")
    tenant_actions.add_parser("list")
    key_parser = root.add_parser("key")
    key_actions = key_parser.add_subparsers(dest="action", required=True)
    key_create = key_actions.add_parser("create")
    key_create.add_argument("tenant_id")
    key_create.add_argument("--label", required=True)
    key_revoke = key_actions.add_parser("revoke")
    key_revoke.add_argument("key_id")
    key_list = key_actions.add_parser("list")
    key_list.add_argument("tenant_id")
    args = parser.parse_args(argv)

    try:
        if args.group == "serve":
            asyncio.run(_serve())
            return 0

        url = database_url or os.getenv("DATABASE_URL", "sqlite:///./gateway.db")
        engine = make_engine(url)
        factory = make_session_factory(engine)
        try:
            with factory() as session:
                if args.group == "tenant" and args.action == "create":
                    row = create_tenant(session, args.name)
                    print(json.dumps({"id": row.id, "name": row.name}))
                elif args.group == "tenant" and args.action == "disable":
                    row = disable_tenant(session, args.tenant_id)
                    print(
                        json.dumps(
                            {"id": row.id, "disabled_at": row.disabled_at.isoformat()}
                        )
                    )
                elif args.group == "tenant":
                    print(
                        json.dumps(
                            [
                                {
                                    "id": row.id,
                                    "name": row.name,
                                    "created_at": row.created_at.isoformat(),
                                    "disabled_at": row.disabled_at.isoformat()
                                    if row.disabled_at
                                    else None,
                                }
                                for row in list_tenants(session)
                            ]
                        )
                    )
                elif args.action == "create":
                    issued = issue_key(session, args.tenant_id, args.label)
                    print(
                        json.dumps(
                            {
                                "id": issued.id,
                                "prefix": issued.prefix,
                                "key": issued.key,
                            }
                        )
                    )
                elif args.action == "revoke":
                    row = revoke_key(session, args.key_id)
                    print(
                        json.dumps(
                            {"id": row.id, "revoked_at": row.revoked_at.isoformat()}
                        )
                    )
                else:
                    rows = list_keys(session, args.tenant_id)
                    print(
                        json.dumps(
                            [
                                {
                                    "id": row.id,
                                    "prefix": row.prefix,
                                    "label": row.label,
                                    "created_at": row.created_at.isoformat(),
                                    "last_used_at": row.last_used_at.isoformat()
                                    if row.last_used_at
                                    else None,
                                    "revoked_at": row.revoked_at.isoformat()
                                    if row.revoked_at
                                    else None,
                                }
                                for row in rows
                            ]
                        )
                    )
        finally:
            engine.dispose()
        return 0
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
