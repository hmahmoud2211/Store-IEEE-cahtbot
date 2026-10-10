from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.container import AppContainer
from app.core.security import verify_api_key


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_session(container: AppContainer = Depends(get_container)) -> Iterator[Session]:
    session = container.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


auth = [Depends(verify_api_key)]
