"""Initial schema: documents, chunks (pgvector + full-text), components, prices, jobs.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-02

The embedding column dimension is taken from EMBEDDING_DIMENSION at migration
time. Switching to a model with a different dimension requires a new migration
that alters the column (and drops/recreates the HNSW index) followed by
`POST /api/v1/reindex`.
"""

from __future__ import annotations

from alembic import op

from app.core.config import get_settings
from app.db.models import TSV_EXPRESSION

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    dim = int(get_settings().embedding_dimension)

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.execute(
        """
        CREATE TABLE documents (
            id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title            TEXT NOT NULL,
            source_type      VARCHAR(32) NOT NULL,
            source_name      TEXT NOT NULL,
            source_url       TEXT,
            file_path        TEXT,
            content_hash     VARCHAR(64) NOT NULL UNIQUE,
            status           VARCHAR(32) NOT NULL DEFAULT 'active',
            owner_user_id    TEXT,
            organization_id  TEXT,
            visibility       VARCHAR(16) NOT NULL DEFAULT 'public'
                             CHECK (visibility IN ('public', 'organization', 'private')),
            metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_documents_source ON documents (source_type, source_name)")
    op.execute("CREATE INDEX ix_documents_acl ON documents (organization_id, visibility)")

    op.execute(
        f"""
        CREATE TABLE document_chunks (
            id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id        UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index        INTEGER NOT NULL,
            content            TEXT NOT NULL,
            search_keys        TEXT,
            content_tsv        TSVECTOR GENERATED ALWAYS AS ({TSV_EXPRESSION}) STORED,
            embedding          VECTOR({dim}),
            embedding_model    TEXT,
            embedding_version  TEXT,
            page_number        INTEGER,
            section            TEXT,
            token_count        INTEGER,
            owner_user_id      TEXT,
            organization_id    TEXT,
            visibility         VARCHAR(16) NOT NULL DEFAULT 'public',
            metadata           JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (document_id, chunk_index)
        )
        """
    )
    # Approximate nearest neighbour index, cosine distance (<=>).
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    op.execute("CREATE INDEX ix_chunks_tsv ON document_chunks USING gin (content_tsv)")
    op.execute("CREATE INDEX ix_chunks_metadata ON document_chunks USING gin (metadata jsonb_path_ops)")
    op.execute("CREATE INDEX ix_chunks_document ON document_chunks (document_id)")
    op.execute("CREATE INDEX ix_chunks_acl ON document_chunks (organization_id, visibility)")
    op.execute("CREATE INDEX ix_chunks_category ON document_chunks ((lower(metadata->>'category')))")
    op.execute("CREATE INDEX ix_chunks_component ON document_chunks ((metadata->>'component_id'))")

    op.execute(
        """
        CREATE TABLE components (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name                TEXT NOT NULL,
            normalized_name     TEXT NOT NULL UNIQUE,
            category            TEXT NOT NULL,
            aliases             TEXT[] NOT NULL DEFAULT '{}',
            search_keys         TEXT,
            price_min           NUMERIC(12, 2),
            price_max           NUMERIC(12, 2),
            currency            CHAR(3),
            price_unit          TEXT,
            price_text          TEXT,
            description         TEXT,
            typical_uses        TEXT[] NOT NULL DEFAULT '{}',
            source_document_id  UUID REFERENCES documents(id) ON DELETE SET NULL,
            metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (price_min IS NULL OR price_max IS NULL OR price_min <= price_max)
        )
        """
    )
    op.execute("CREATE INDEX ix_components_category ON components (lower(category))")
    op.execute("CREATE INDEX ix_components_price_min ON components (price_min)")
    op.execute("CREATE INDEX ix_components_price_max ON components (price_max)")
    op.execute("CREATE INDEX ix_components_aliases ON components USING gin (aliases)")
    op.execute("CREATE INDEX ix_components_name_trgm ON components USING gin (normalized_name gin_trgm_ops)")
    op.execute("CREATE INDEX ix_components_keys_trgm ON components USING gin (search_keys gin_trgm_ops)")

    op.execute(
        """
        CREATE TABLE component_prices (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            component_id        UUID NOT NULL REFERENCES components(id) ON DELETE CASCADE,
            source_document_id  UUID REFERENCES documents(id) ON DELETE CASCADE,
            price_min           NUMERIC(12, 2),
            price_max           NUMERIC(12, 2),
            currency            CHAR(3),
            price_unit          TEXT,
            price_text          TEXT,
            supplier            TEXT,
            observed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
            metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
            UNIQUE (component_id, source_document_id)
        )
        """
    )
    op.execute("CREATE INDEX ix_component_prices_component ON component_prices (component_id)")

    op.execute(
        """
        CREATE TABLE ingestion_jobs (
            id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_type             VARCHAR(16) NOT NULL DEFAULT 'ingest',
            document_id          UUID REFERENCES documents(id) ON DELETE SET NULL,
            source               TEXT,
            source_type          VARCHAR(32),
            status               VARCHAR(16) NOT NULL DEFAULT 'pending',
            embedding_model      TEXT,
            embedding_dimension  INTEGER,
            embedding_version    TEXT,
            chunks_created       INTEGER NOT NULL DEFAULT 0,
            error                TEXT,
            stats                JSONB NOT NULL DEFAULT '{}'::jsonb,
            started_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at          TIMESTAMPTZ
        )
        """
    )
    op.execute("CREATE INDEX ix_jobs_status ON ingestion_jobs (status, started_at DESC)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ingestion_jobs")
    op.execute("DROP TABLE IF EXISTS component_prices")
    op.execute("DROP TABLE IF EXISTS components")
    op.execute("DROP TABLE IF EXISTS document_chunks")
    op.execute("DROP TABLE IF EXISTS documents")
