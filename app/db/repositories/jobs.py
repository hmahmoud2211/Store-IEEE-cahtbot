from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import IngestionJob


class JobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, **fields: Any) -> IngestionJob:
        job = IngestionJob(id=uuid.uuid4(), status=fields.pop("status", "running"), **fields)
        self.session.add(job)
        self.session.flush()
        return job

    def get(self, job_id: uuid.UUID) -> IngestionJob | None:
        return self.session.get(IngestionJob, job_id)

    def list_recent(self, limit: int = 20) -> list[IngestionJob]:
        return list(self.session.scalars(select(IngestionJob).order_by(IngestionJob.started_at.desc()).limit(limit)))

    def finish(self, job: IngestionJob, *, status: str, error: str | None = None, **fields: Any) -> None:
        job.status = status
        job.error = error
        job.finished_at = datetime.now(UTC)
        for key, value in fields.items():
            setattr(job, key, value)
        self.session.flush()
