"""Authentication placeholder and the Principal used for permission filtering.

The Principal is passed all the way down into retrieval SQL so that access
control is enforced *while* retrieving candidates, never after generation.
Replace `get_principal` with real auth (JWT/OIDC) when multi-user support lands.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings


@dataclass(frozen=True)
class Principal:
    user_id: str | None = None
    organization_id: str | None = None
    is_admin: bool = False

    @classmethod
    def anonymous(cls) -> Principal:
        return cls()

    @classmethod
    def system(cls) -> Principal:
        return cls(user_id="system", is_admin=True)


def verify_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    keys = settings.api_key_list
    if not keys:
        return
    if not x_api_key or not any(secrets.compare_digest(x_api_key, k) for k in keys):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


def get_principal(
    x_user_id: str | None = Header(default=None),
    x_org_id: str | None = Header(default=None),
) -> Principal:
    # NOTE: headers are trusted only because this is a placeholder; in production
    # derive identity from a verified token instead.
    return Principal(user_id=x_user_id, organization_id=x_org_id)
