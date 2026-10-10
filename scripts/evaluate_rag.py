#!/usr/bin/env python
"""RAG evaluation harness.

Runs every question in data/eval/eval_set.yaml through the full pipeline and reports:

  Retrieval     precision@k, recall@k, MRR, hit-rate@k  (over components placed in the context)
  Routing       query-type accuracy
  Answers       answer relevance (required facts present), faithfulness (numeric grounding),
                citation correctness, refusal accuracy, hallucination rate
  Latency       p50 / p95 / max

Usage:
    python scripts/evaluate_rag.py                 # uses the configured LLM (Gemini if GEMINI_API_KEY is set)
    python scripts/evaluate_rag.py --no-llm        # deterministic extractive answers (no API calls)
    python scripts/evaluate_rag.py --judge         # add Gemini LLM-as-judge scores (needs GEMINI_API_KEY)
    python scripts/evaluate_rag.py -k 5 --out reports/
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.container import AppContainer  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.rag.citations import check_answer  # noqa: E402
from app.rag.llm.base import LLMProvider  # noqa: E402

REFUSAL_RE = re.compile(r"could not find|not (?:available|present|listed) in the (?:current )?knowledge base", re.I)


@dataclass
class ItemResult:
    id: str
    question: str
    expected_query_type: str | None
    query_type: str
    route_ok: bool
    retrieved: list[str]
    expected: list[str]
    precision_at_k: float | None
    recall_at_k: float | None
    reciprocal_rank: float | None
    hit_at_k: float | None
    must_refuse: bool
    refused: bool
    refusal_ok: bool
    answer_relevant: bool
    grounded: bool
    ungrounded_numbers: list[str]
    citation_ok: bool | None
    hallucination: bool
    latency_ms: float
    answer: str
    llm_fallback: bool = False
    judge: dict | None = None
    notes: list[str] = field(default_factory=list)


def context_components(result) -> list[str]:
    if result.context is None:
        return []
    names: list[str] = []
    for src in result.context.sources:
        name = src.component.name if src.component else (src.hit.metadata.get("component") if src.hit else None)
        if name and name not in names:
            names.append(name)
    return names


def retrieval_metrics(retrieved: list[str], expected: list[str], k: int) -> tuple[float, float, float, float]:
    top = retrieved[:k]
    relevant = [r for r in top if r in expected]
    precision = len(relevant) / len(top) if top else 0.0
    recall = len(set(relevant)) / len(expected)
    rr = next((1.0 / i for i, r in enumerate(retrieved, start=1) if r in expected), 0.0)
    hit = 1.0 if relevant else 0.0
    return precision, recall, rr, hit


JUDGE_PROMPT = """You are grading a retrieval-augmented answer.

Question: {question}

Retrieved context:
{context}

Answer:
{answer}

Score two things from 1 (bad) to 5 (perfect) and reply ONLY with JSON:
{{"faithfulness": <1-5, every claim supported by the context>, "relevance": <1-5, answers the question>, "reason": "<short>"}}"""


def judge(llm: LLMProvider, question: str, context: str, answer: str) -> dict | None:
    try:
        response = llm.generate(
            "You are a strict, fair evaluator.",
            JUDGE_PROMPT.format(question=question, context=context[:6000], answer=answer),
        )
        match = re.search(r"\{[\s\S]*\}", response.text)
        return json.loads(match.group(0)) if match else None
    except Exception as exc:  # judge failures never fail the evaluation
        return {"error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-file", default=str(ROOT / "data" / "eval" / "eval_set.yaml"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic extractive answers")
    parser.add_argument("--judge", action="store_true", help="LLM-as-judge for faithfulness/relevance (Gemini)")
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("ERROR", json_logs=False)
    container = AppContainer(settings)
    items = yaml.safe_load(Path(args.eval_file).read_text(encoding="utf-8"))

    judge_llm = None
    if args.judge:
        from app.container import build_llm
        from app.rag.llm.extractive import ExtractiveLLM

        judge_llm = build_llm(settings)
        if isinstance(judge_llm, ExtractiveLLM):
            print("--judge requires GEMINI_API_KEY; skipping LLM judge.")
            judge_llm = None

    container.warmup()  # load embedding + reranker models before timing
    results: list[ItemResult] = []
    for item in items:
        expected = item.get("expected_components") or []
        start = time.perf_counter()
        result = container.rag.answer(item["question"], use_llm=not args.no_llm)
        latency = (time.perf_counter() - start) * 1000

        retrieved = context_components(result)
        p = r = rr = hit = None
        if expected:
            p, r, rr, hit = retrieval_metrics(retrieved, expected, args.k)

        answer_body = result.answer.split("\n\nSources:")[0]
        refused = result.no_answer or bool(REFUSAL_RE.search(answer_body) and not result.structured_results)
        must_refuse = bool(item.get("must_refuse"))
        refusal_ok = refused if must_refuse else not result.no_answer

        must = item.get("answer_must_contain") or []
        must_not = item.get("answer_must_not_contain") or []
        relevant = all(m.lower() in answer_body.lower() for m in must) and not any(
            m.lower() in answer_body.lower() for m in must_not
        )

        grounded, ungrounded = True, []
        if result.context is not None:
            _, report = check_answer(answer_body, result.context)
            grounded, ungrounded = report.grounded, report.ungrounded_numbers

        citation_ok = None
        if expected and not must_refuse:
            cited = {s.get("component") for s in result.sources}
            citation_ok = bool(result.sources) and bool(cited & set(expected))

        hallucination = (not grounded) or (must_refuse and not refused)
        judged = None
        if judge_llm is not None and result.context is not None:
            judged = judge(judge_llm, item["question"], result.context.render(), answer_body)

        results.append(
            ItemResult(
                id=item["id"],
                question=item["question"],
                expected_query_type=item.get("expected_query_type"),
                query_type=result.query_type.value,
                route_ok=item.get("expected_query_type") in (None, result.query_type.value),
                retrieved=retrieved,
                expected=expected,
                precision_at_k=p,
                recall_at_k=r,
                reciprocal_rank=rr,
                hit_at_k=hit,
                must_refuse=must_refuse,
                refused=refused,
                refusal_ok=refusal_ok,
                answer_relevant=relevant,
                grounded=grounded,
                ungrounded_numbers=ungrounded,
                citation_ok=citation_ok,
                hallucination=hallucination,
                latency_ms=round(latency, 1),
                answer=result.answer,
                llm_fallback=bool(result.retrieval_metadata.get("llm_fallback")),
                judge=judged,
            )
        )

    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 3) if values else None

    latencies = sorted(r.latency_ms for r in results)
    p95_index = max(0, int(round(0.95 * len(latencies))) - 1)
    summary = {
        "k": args.k,
        "items": len(results),
        "llm": "extractive" if args.no_llm else container.llm.model_name,
        "embedding_model": container.embedder.model_name,
        "reranker": container.reranker.name,
        f"precision@{args.k}": mean(r.precision_at_k for r in results),
        f"recall@{args.k}": mean(r.recall_at_k for r in results),
        "mrr": mean(r.reciprocal_rank for r in results),
        f"hit_rate@{args.k}": mean(r.hit_at_k for r in results),
        "query_type_accuracy": mean(float(r.route_ok) for r in results),
        "answer_relevance": mean(float(r.answer_relevant) for r in results),
        "faithfulness_grounded": mean(float(r.grounded) for r in results),
        "citation_correctness": mean(None if r.citation_ok is None else float(r.citation_ok) for r in results),
        "refusal_accuracy": mean(float(r.refusal_ok) for r in results),
        "hallucination_rate": mean(float(r.hallucination) for r in results),
        "llm_fallback_rate": mean(float(r.llm_fallback) for r in results),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": latencies[p95_index],
        "latency_max_ms": latencies[-1],
    }
    if judge_llm is not None:
        summary["judge_faithfulness"] = mean(
            (r.judge or {}).get("faithfulness") for r in results if r.judge and "error" not in r.judge
        )
        summary["judge_relevance"] = mean(
            (r.judge or {}).get("relevance") for r in results if r.judge and "error" not in r.judge
        )

    # ----------------------------------------------------------------- print
    print(
        f"\n{'id':4} {'type':16} {'route':5} {'P@k':>5} {'R@k':>5} {'RR':>5} {'refuse':6} {'rel':4} {'grnd':4} {'cite':4} {'ms':>8}  question"
    )
    print("-" * 130)
    fmt = lambda v: "  -  " if v is None else f"{v:5.2f}"  # noqa: E731
    flag = lambda b: " -  " if b is None else (" ok " if b else "FAIL")  # noqa: E731
    for r in results:
        print(
            f"{r.id:4} {r.query_type:16} {flag(r.route_ok):5} {fmt(r.precision_at_k)} {fmt(r.recall_at_k)} "
            f"{fmt(r.reciprocal_rank)} {flag(r.refusal_ok):6} {flag(r.answer_relevant):4} {flag(r.grounded):4} "
            f"{flag(r.citation_ok):4} {r.latency_ms:8.1f}  {r.question[:60]}"
        )
    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key:24} {value}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_file.write_text(
        json.dumps({"summary": summary, "items": [asdict(r) for r in results]}, indent=2, ensure_ascii=False)
    )
    print(f"\nReport written to {out_file}")
    container.close()
    failures = [r.id for r in results if not (r.route_ok and r.refusal_ok and r.grounded)]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
