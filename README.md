# Hardware Component RAG System

A Retrieval-Augmented Generation (RAG) service for an electronics component knowledge base, with prices in EGP. It answers price lookups, numeric filters, comparisons, project recommendations and cost estimates. Every answer is grounded in retrieved data and returned with structured citations.

The service is built on FastAPI and PostgreSQL. Inside PostgreSQL, pgvector holds the embeddings, full-text search handles exact identifiers, and ordinary SQL tables hold the structured prices. The models are pluggable:

| Role       | Default                                    | Where it runs |
| ---------- | ------------------------------------------ | ------------- |
| Embeddings | `BAAI/bge-small-en-v1.5`                 | local         |
| Reranker   | `BAAI/bge-reranker-base` (cross-encoder) | local         |
| LLM        | Gemini                                     | API           |

```
Question → Query understanding → SQL (when it can answer exactly)
         → Dense (pgvector) + Keyword (tsvector) retrieval → RRF → metadata/ACL filter
         → Cross-encoder rerank → best 3–8 → grounded prompt → LLM
         → citation validation + numeric grounding check → answer + sources
```

---

## 1. Architecture

### Offline ingestion

```
PDF / DOCX / CSV / Excel / Web / DB / API / Text
        │  LoaderRegistry → BaseDocumentLoader.load() → LoadedDocument[Element]
        ▼        (heading | paragraph | list | table | record)
Clean + normalize           NFKC, whitespace, boilerplate; keeps 74LS85, HC-SR04, 5V, "4–10 EGP"
        ▼
Hash + dedup                sha256(raw bytes): same hash → skip; same source name + new hash → re-index
        ▼
Component extraction        tables/rows with name + price columns → normalized name, aliases,
        ▼                   category, PriceRange(min, max, currency, unit, original)
Structure-aware chunking    one chunk per record; prose grouped by heading path (CHUNK_SIZE / CHUNK_OVERLAP)
        ▼
Metadata                    source, page, section, component, category, price, country, supplier, ACL…
        ▼
Embeddings                  EmbeddingProvider.embed_documents (batched)
        ▼
PostgreSQL + pgvector       documents · document_chunks (text + tsvector + vector) · components · component_prices · ingestion_jobs
```

### Online query

```
                      USER QUESTION
                            │
                    QueryProcessor  (identifiers, aliases, categories, price bounds, sort,
                            │        quantities, comparison/recommendation/calculation intent)
                            │
              Component resolution (alias phrases → part-number keys; unknown parts → NOT FOUND)
                            │
                       QueryRouter
          ┌─────────────────┴──────────────────┐
  Structured SQL                         Semantic retrieval
  FACT_LOOKUP / PRICE_FILTER /           Vector search (HNSW, cosine)  ┐
  COMPARISON / CALCULATION               Keyword search (GIN tsvector) ┴─ RRF → metadata + ACL filter
  (PricingService does the math)         → top 20 → cross-encoder → threshold → best 3–8
          └─────────────────┬──────────────────┘
                     ContextBuilder  ([SOURCE n] blocks + [COMPUTED] + [NOT FOUND])
                            │
                 System prompt + context → LLM (Gemini) / extractive fallback
                            │
           Citation validation + numeric grounding check → Answer + Sources (JSON)
```

| Query type          | Example                                                                   | What runs                                                                    |
| ------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `FACT_LOOKUP`     | "What is the price of a 74LS85 comparator?"                               | SQL lookup of resolved components + hybrid retrieval for extra context       |
| `PRICE_FILTER`    | "Which counter ICs cost less than 25 EGP?" / "cheapest 7-segment display" | SQL only (no embeddings): full matches + "range straddles the limit" matches |
| `COMPARISON`      | "Compare the DS1307 and DS3231."                                          | SQL facts + programmatic comparison statements + retrieval                   |
| `CALCULATION`     | "How much would 2 × NE555, 1 × breadboard, 5 × LED cost?"              | Item resolution + Decimal arithmetic (38–60 EGP)                            |
| `RECOMMENDATION`  | "What components can I use to build a digital clock?"                     | Hybrid retrieval, reranked against the topic phrase ("digital clock")        |
| `SEMANTIC_SEARCH` | "Explain how the CD4511 works." / "components related to counters"        | Hybrid retrieval (+ SQL category listing when a category is named)           |

---

## 2. Database schema

All of this lives in one PostgreSQL database. It is created by `migrations/versions/0001_initial.py`.

| Table                | Purpose                                            | Key columns / indexes                                                                                                                                                                                                                        |
| -------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `documents`        | One row per ingested source                        | `content_hash` UNIQUE, `source_type`, `source_name`, `source_url`, `owner_user_id`, `organization_id`, `visibility`, `metadata` JSONB                                                                                        |
| `document_chunks`  | Original chunk text + embedding + full-text vector | `content`, `embedding VECTOR(384)` (**HNSW, `vector_cosine_ops`**), `content_tsv` generated column (**GIN**), `search_keys`, `page_number`, `section`, `metadata` JSONB (GIN), denormalized ACL columns          |
| `components`       | Structured, parsed component records               | `normalized_name` UNIQUE, `category`, `aliases TEXT[]` (GIN), `search_keys` (trigram GIN), `price_min` / `price_max NUMERIC` (btree), `currency`, `price_unit`, `price_text` (original), `description`, `typical_uses` |
| `component_prices` | Every observed price per source and supplier       | lets the system report**disagreeing sources** instead of silently picking one                                                                                                                                                          |
| `ingestion_jobs`   | Ingest and re-index bookkeeping                    | `status`, `embedding_model`, `embedding_dimension`, `embedding_version`, `chunks_created`, `stats`                                                                                                                               |

`content_tsv` is built from three weighted vectors:

| Weight      | Text config | Source                      | Example terms                   |
| ----------- | ----------- | --------------------------- | ------------------------------- |
| **A** | `simple`  | compact identifier keys     | `hcsr04`, `7485`, `ne555` |
| **B** | `simple`  | exact tokens in the content | `74ls85`, `hc-sr04`         |
| **C** | `english` | stemmed content             | "counters" matches "counter"    |

Prices are always stored as numbers plus the original text. For example, `4–10 EGP` becomes `price_min=4, price_max=10, currency=EGP, price_text="4–10 EGP"`. That is why "cheaper than 20 EGP" is answered by SQL, not by embeddings.

---

## 3. Project layout

```
app/
  main.py                    FastAPI app factory, lifespan, error handler
  container.py               composition root (providers chosen from settings)
  api/routes/                query.py (query, search) · ingest.py (ingest, upload, reindex, jobs) · documents.py · components.py · health.py
  core/                      config.py (env settings) · logging.py (structlog + timers) · security.py (API key, Principal) · cache.py
  db/                        models.py · session.py · repositories/{documents,chunks,components,jobs}.py
  ingestion/
    loaders/                 base.py (interface) · registry.py · pdf · docx · csv · excel · web · database · api · text
    cleaning.py · price_parser.py · component_normalizer.py · categorizer.py
    component_extractor.py · chunking.py · metadata.py · pipeline.py
  embeddings/                base.py (EmbeddingProvider) · provider.py (sentence-transformers, Gemini, hashing, cache wrapper)
  retrieval/                 vector_search.py · keyword_search.py · hybrid_search.py (RRF) · metadata_filter.py (filters + ACL) · reranker.py
  rag/                       query_processor.py · query_router.py · context_builder.py · prompt_builder.py · citations.py · pipeline.py
    llm/                     base.py · gemini.py · extractive.py (deterministic fallback)
  services/                  component_service.py (alias index, resolution) · pricing_service.py · document_service.py
  schemas/                   Pydantic API models
data/
  hardware_components.csv    initial dataset (32 rows, prices verbatim + curated description/typical_uses)
  component_aliases.yaml     curated aliases ("quad AND gate", "hex inverter", "Tiny RTC" …)
  eval/eval_set.yaml         evaluation questions + ground truth
migrations/                  Alembic (0001_initial)
scripts/                     ingest_initial_data.py · ingest.py · evaluate_rag.py
tests/                       unit/ (no DB) · integration/ (PostgreSQL) · retrieval/ (real models, -m models)
Dockerfile · docker-compose.yml · requirements*.txt · pyproject.toml · .env.example
```

---

## 4. Running locally (macOS + Homebrew PostgreSQL)

```bash
# 1. PostgreSQL 17 + pgvector
#    (Homebrew's pgvector bottle targets PostgreSQL 17/18. For PostgreSQL 14 build pgvector
#     from source with PG_CONFIG=/opt/homebrew/opt/postgresql@14/bin/pg_config.)
brew install postgresql@17 pgvector
brew services start postgresql@17

# 2. Databases + pgvector extension (the migration also runs CREATE EXTENSION)
/opt/homebrew/opt/postgresql@17/bin/createdb rag_hardware
/opt/homebrew/opt/postgresql@17/bin/createdb rag_hardware_test
/opt/homebrew/opt/postgresql@17/bin/psql rag_hardware -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3. Python environment (3.11 recommended for torch / sentence-transformers wheels)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# 4. Configuration
cp .env.example .env          # set GEMINI_API_KEY; without it answers use the extractive fallback

# 5. Migrations
alembic upgrade head

# 6. Load the hardware dataset (first run downloads the bge embedding model, ~130 MB)
python scripts/ingest_initial_data.py

# 7. Start the API (http://localhost:8000/docs for Swagger UI)
uvicorn app.main:app --reload
```

Then open **http://localhost:8000/** (redirects to `/ui/`). This is the **test console**, a single page in `app/static/index.html` served by the API itself. It has five tabs:

| Tab                  | What you can do                                                                                                                                                                                                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ask**        | Chat with follow-ups and example questions. Click any answer to inspect its query type, routing, sources, SQL results, calculation, timings, reranked chunks, and the exact context sent to the LLM. Turn off**Use LLM** to compare against the deterministic answers. |
| **Retrieval**  | Run hybrid search without generation. Shows the vector, keyword, RRF and rerank scores for every candidate.                                                                                                                                                                  |
| **Components** | Browse and filter the structured`components` table (category, price range, alias / part-number search).                                                                                                                                                                    |
| **Documents**  | Upload a PDF, DOCX, CSV, Excel or TXT file to ingest it. View chunks, or delete documents.                                                                                                                                                                                   |
| **System**     | Health and configuration. Set an API key or user / organization ids to test permissions.                                                                                                                                                                                     |

### Query the API

```bash
curl -s -X POST localhost:8000/api/v1/query -H 'Content-Type: application/json' \
  -d '{"question": "Which counter ICs cost less than 25 EGP?", "filters": {"currency": "EGP"}}'

curl -s -X POST localhost:8000/api/v1/query -H 'Content-Type: application/json' \
  -d '{"question": "How much would 2 x NE555, 1 x breadboard and 5 x LED cost?"}'

# Follow-up questions: pass the conversation in `history`
curl -s -X POST localhost:8000/api/v1/query -H 'Content-Type: application/json' \
  -d '{"question": "Show the source of the price you used.",
       "history": [{"role": "user", "content": "What is the price of the DS3231?"}]}'

curl -s -X POST localhost:8000/api/v1/search -H 'Content-Type: application/json' -d '{"query": "HC SR04", "top_k": 3}'
curl -s "localhost:8000/api/v1/components?category=counter&max_price=20"
curl -s localhost:8000/api/v1/health
```

Response shape (abridged):

```json
{
  "answer": "Matching counter components priced less than 25 EGP:\n- 74LS161 Counter: 18 EGP [1]\n\nPossibly within the limit (the price range straddles it):\n- CD4017 Counter: 15–25 EGP [2]\n...\n\nSources:\n[1] Hardware Components Price List — 74LS161 Counter\n[2] Hardware Components Price List — CD4017 Counter",
  "query_type": "PRICE_FILTER",
  "sources": [{"source_id": 1, "kind": "component", "document": "hardware_components.csv",
               "document_title": "Hardware Components Price List", "component": "74LS161 Counter",
               "page": null, "price_text": "18 EGP"}],
  "structured_results": [{"name": "74LS161 Counter", "price_min": "18.00", "price_max": "18.00", "currency": "EGP", "partial_match": false}],
  "calculation": null,
  "retrieval_metadata": {"query_type": "PRICE_FILTER", "route_reasons": ["numeric price constraint"],
                         "timings_ms": {"structured_ms": 10.2, "total_ms": 22.1}}
}
```

### Ingest more sources

```bash
# CLI (any loader)
python scripts/ingest.py data/catalogs/electronics_catalog.pdf
python scripts/ingest.py data/datasheets/ds3231_notes.docx --title "DS3231 notes"
python scripts/ingest.py data/prices/supplier_y.xlsx --supplier "Supplier Y"
python scripts/ingest.py https://supplier.example/timers --type web --supplier "Supplier X"
python scripts/ingest.py '{"url": "https://api.example.com/items", "records_path": "data.items"}' --type api
python scripts/ingest.py '{"url": "postgresql+psycopg://…", "query": "SELECT name AS component, price FROM items"}' --type database

# API
curl -s -X POST localhost:8000/api/v1/ingest -H 'Content-Type: application/json' \
  -d '{"source": "data/hardware_components.csv", "source_type": "csv", "force": true}'
curl -s -X POST localhost:8000/api/v1/ingest/upload -F file=@catalog.pdf -F run_in_background=true
curl -s localhost:8000/api/v1/jobs/<job_id>
```

Any table (CSV/Excel rows, PDF/DOCX/web tables) that has a name-like column and a price-like column is also loaded into `components`. Its prices then become filterable by SQL. If a second source lists a different price, both appear in `component_prices`, and answers mention the disagreement.

**API endpoints**: every route is under `/api/v1`.

| Method | Route                                                                                                   |
| ------ | ------------------------------------------------------------------------------------------------------- |
| POST   | `/ingest`, `/ingest/upload`, `/query`, `/search`, `/reindex`                                  |
| GET    | `/documents`, `/documents/{id}`, `/components`, `/components/{id}`, `/jobs/{id}`, `/health` |
| DELETE | `/documents/{id}`                                                                                     |

---

## 5. Docker

```bash
cp .env.example .env              # add GEMINI_API_KEY
docker compose up --build -d      # PostgreSQL+pgvector (host port 5433) + API (port 8000); runs migrations on start
docker compose exec api python scripts/ingest_initial_data.py
curl -s localhost:8000/api/v1/health
# optional Redis cache: docker compose --profile cache up -d   and set REDIS_URL=redis://redis:6379/0
```

Hugging Face models are cached in the `models` volume. To bake them into the image instead, use `docker build --build-arg PRELOAD_MODELS=1 .`.

---

## 6. Tests and evaluation

```bash
pytest tests/unit                       # 169 tests, no DB, no model downloads (~1 s)
pytest tests/integration                # 39 tests against TEST_DATABASE_URL (default: local rag_hardware_test)
pytest -m models tests/retrieval        # retrieval quality with the real bge models
pytest                                  # unit + integration (model tests are opt-in)

python scripts/evaluate_rag.py --no-llm   # deterministic answers, no API calls
python scripts/evaluate_rag.py            # with Gemini (needs GEMINI_API_KEY)
python scripts/evaluate_rag.py --judge    # + Gemini LLM-as-judge faithfulness / relevance
```

The unit and integration tests use a deterministic hashing embedder, a pass-through reranker and the extractive answerer. They never download models or call APIs.

The tests cover:

- **Ingestion:** price parsing, name normalization, alias matching ("555 timer", "7485", "HC SR04"), categorization of all 32 rows, chunking, metadata and loaders.
- **Retrieval:** vector, keyword and hybrid RRF search, metadata filters, ACL filtering inside SQL, and reranking.
- **Generation:** query routing, citation validation, numeric grounding, the LLM-error and hallucination fallbacks, and no-answer behavior.
- **API:** every endpoint.

Latest evaluation over the 19 questions in `data/eval/eval_set.yaml` (`k=5`, real local models):

| Metric                                           | Gemini (`gemini-3.8-flash`) | Extractive (`--no-llm`) |
| ------------------------------------------------ | ----------------------------- | ------------------------- |
| Hit-rate@5 / MRR                                 | 1.00 / 1.00                   | 1.00 / 1.00               |
| Recall@5 / Precision@5                           | 0.965 / 0.78                  | 0.965 / 0.78              |
| Query-type accuracy                              | 1.00                          | 1.00                      |
| Answer relevance (required facts present)        | 1.00                          | 1.00                      |
| Faithfulness (numeric grounding)                 | 1.00                          | 1.00                      |
| Citation correctness                             | 1.00                          | 1.00                      |
| Refusal accuracy (ESP32, Arduino Uno, off-topic) | 1.00                          | 1.00                      |
| Hallucination rate                               | 0.00                          | 0.00                      |
| Answers replaced by the grounding fallback       | 0.00                          | n/a                       |
| Latency p50 / p95                                | ~3.0 s / ~5.1 s               | ~56 ms / ~254 ms          |

Why some metrics are below 1.0:

- **Precision@5 is 0.78** because lookups legitimately include related items, such as the NE555 module next to the NE555 IC.
- **Recall is below 1** only where more than five components are expected.

The refusals (ESP32, off-topic questions) return in under 1 s because they never reach the LLM.

Gemini latency is dominated by the API call. `LLM_THINKING_LEVEL` (`low` / `medium` / `high`) trades reasoning effort for speed; `gemini-3.8-flash` rejects `minimal`.

---

## 7. Design decisions

- **SQL first for exact questions.** Price filters, cheapest/most-expensive, comparisons and totals come from numeric columns and `PricingService` (Decimal arithmetic). The LLM is told to copy the `[COMPUTED]` numbers verbatim, and the grounding check enforces this.
- **Range-aware filtering.** "Under 25 EGP" returns *full* matches (`price_max < 25`) separately from *partial* matches whose range straddles the limit (CD4017 at 15–25 EGP).
- **Identifiers are matched three ways.**

  1. An in-memory alias index does longest-phrase matching.
  2. Compact equivalence keys cover variants: `74LS85 ≡ 7485`, `HC SR04 ≡ HC-SR04`, `NE555 → 555`, `LM7805 ≡ 7805`.
  3. Full-text search adds weight-A `search_keys`.

  Unknown part numbers are never fuzzy-matched to a different part. They are reported as not found.
- **Hybrid retrieval with RRF** (`k=60`). RRF uses only ranks, so cosine similarities and `ts_rank_cd` never need normalizing against each other.
- **Filters inside the SQL.** Metadata and ACL filters are part of both retrieval queries, with HNSW `iterative_scan` on pgvector ≥ 0.8. A Python post-filter re-checks the same constraints as defense in depth.
- **Topic-aware reranking.** Cross-encoders score "does this passage answer the question?". For "what do I need to build a digital clock" no single part does, and scores stay around 0.01. Reranking against the extracted topic ("digital clock") lifts relevant parts to 0.1–0.6 while off-topic items stay near 0.
- **Calibrated thresholds.** On bge-small, off-topic questions scored a cosine similarity of 0.45–0.50 and relevant ones 0.58–0.74. Hence `MIN_RETRIEVAL_SCORE=0.55` (lexical matches always pass) and `MIN_RERANK_SCORE=0.05`.
- **No-answer without the LLM.** Unknown parts or low-confidence retrieval return the fixed refusal message, and the LLM is never called.
- **Generation guardrails.**

  - Invalid `[n]` citations are removed.
  - Any price-like number not present in the context triggers the extractive fallback (`GROUNDING_POLICY=fallback`), or a warning when set to `warn`.
  - The `Sources:` footer is generated by code, not by the LLM.
- **Provenance of curated text.** Descriptions and typical uses in the seed CSV are curated annotations and are labeled as such in the context. Prices only ever come from the price column.
- **Replaceable providers.** `EmbeddingProvider`, `Reranker`, `LLMProvider` and `BaseDocumentLoader` are interfaces wired up in `app/container.py`. The embedding model, version and dimension are stored per chunk and per job. `/health` reports chunks needing re-index, and `POST /reindex` re-embeds them.
- **Built to scale.**

  - Lookups use HNSW, GIN and btree indexes, so no stage scans the whole knowledge base.
  - Only the 3–8 best chunks reach the LLM.
  - Ingestion embeds in batches.
  - The DB session is released before the LLM call.

### Changing the embedding model

1. Set `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` and `EMBEDDING_VERSION`.
2. If the dimension changed, add a migration that alters `document_chunks.embedding` to `VECTOR(<new dim>)` and recreates the HNSW index.
3. Call `POST /api/v1/reindex`, then check that `/health` reports `chunks_needing_reindex: 0`.

### Security notes

- `X-User-Id` / `X-Org-Id` headers are a **placeholder** identity. Replace `get_principal` with verified tokens before multi-user use.
- Setting `API_KEYS` enables `X-API-Key` checks.
- Logs record query fingerprints, ids and timings, never raw document content or secrets.
- `POST /ingest` only accepts local paths under `data/`. Use `/ingest/upload` for anything else.
