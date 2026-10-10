"""Query understanding: deterministic extraction of identifiers, categories,
price constraints, quantities and intents from the user question.

This stage is intentionally rule-based: it is fast, testable and predictable,
and its output drives SQL filters, which must be exact.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from app.ingestion.categorizer import detect_categories
from app.ingestion.component_normalizer import extract_identifiers
from app.ingestion.price_parser import normalize_currency
from app.rag.types import PriceConstraint, QuantityItem, QueryAnalysis

_NUM = r"(\d+(?:\.\d+)?)"
_CUR = r"\s*(egp|l\.e\.?|le|pounds?|usd|us\$|\$|eur|€)?"
_CUR_REQUIRED = r"\s*(egp|l\.e\.?|le|pounds?|usd|us\$|\$|eur|€)"

_MAX_EXCLUSIVE = re.compile(rf"(?:less than|lower than|cheaper than|under|below|beneath|<)\s*{_NUM}{_CUR}")
_MAX_INCLUSIVE = re.compile(
    rf"(?:at most|up to|no more than|not more than|maximum of|max(?:imum)?|<=|≤|within a budget of|budget of|"
    rf"within)\s*{_NUM}{_CUR}"
)
_MAX_OR_LESS = re.compile(rf"{_NUM}{_CUR}\s*or (?:less|cheaper|below)")
_MIN_EXCLUSIVE = re.compile(rf"(?:more than|greater than|higher than|more expensive than|above|over|>)\s*{_NUM}{_CUR}")
_MIN_INCLUSIVE = re.compile(rf"(?:at least|minimum of|min(?:imum)?|>=|≥)\s*{_NUM}{_CUR}")
_MIN_OR_MORE = re.compile(rf"{_NUM}{_CUR}\s*or (?:more|above)")
_BETWEEN = re.compile(rf"(?:between|from)\s*{_NUM}{_CUR}\s*(?:and|to|-|–)\s*{_NUM}{_CUR}")
_APPROX_WITH_CURRENCY = re.compile(
    rf"(?:approximately|approx\.?|around|about|roughly|~|close to)\s*{_NUM}{_CUR_REQUIRED}"
)
_APPROX_AFTER_COST = re.compile(
    rf"(?:costs?|priced|price of|for)\s+(?:approximately|approx\.?|around|about|roughly)\s*{_NUM}"
)
_EXACT_PRICE = re.compile(rf"(?:costs?|priced at|for)\s+{_NUM}{_CUR_REQUIRED}")
_PRICE_SPAN = re.compile(rf"{_NUM}\s*(?:-|–|to)?\s*(?:\d+(?:\.\d+)?)?{_CUR_REQUIRED}")

_SORT_ASC = re.compile(
    r"\b(cheapest|least expensive|lowest[- ]priced?|lowest price|lowest cost|most affordable|minimum price|"
    r"min price)\b"
)
_SORT_DESC = re.compile(r"\b(most expensive|priciest|highest[- ]priced?|highest price|costliest|maximum price)\b")

_COMPARISON = re.compile(
    r"\b(compare|comparison|compared|versus|vs\.?|difference|differences|differ|which is better|"
    r"pros and cons)\b"
)
_WHICH_CHEAPER = re.compile(r"\b(cheaper|more expensive|pricier|better|less expensive)\b")
_CALCULATION = re.compile(
    r"\b(total|in total|altogether|overall cost|sum|estimate|estimated|estimation|budget for|bill of materials|bom|"
    r"how much (?:would|will) (?:these|this|they|it|all|the|those|i|my)|how much for|cost of (?:these|all|the following)|"
    r"(?:these|those|the following) (?:components|parts|items)? ?(?:would )?(?:approximately )?cost)\b"
)
_RECOMMENDATION = re.compile(
    r"\b(do i need|should i use|can i use|could i use|to build|for building|to make|to design|recommend|suggest|"
    r"needed for|required for|need for|parts for|components for|what (?:components|parts) (?:are )?(?:needed|required)|"
    r"how (?:should|do|can|could|would) i (?:build|make|create|design|do|start|implement|assemble)|"
    r"how to (?:build|make|create|design|implement|assemble)|i (?:want|would like|need|plan) to (?:build|make|create|design)|"
    r"what do i need)\b"
)
# Explicit request for a full build procedure.
_BUILD_GUIDE = re.compile(
    r"\b(step[- ]by[- ]step|steps|instructions?|guide me|build guide|walk me through|tutorial|procedure|"
    r"how to (?:assemble|wire|connect) (?:it|this|the project|everything))\b"
)
# Topic phrases that only point back at the conversation ("this project", "it").
_VAGUE_TOPIC = re.compile(
    r"^(?:(?:this|that|it|these|those|the|my|our|same)\s*)+(?:project|circuit|build|thing|one|idea|system|device)?$"
)
_TOPIC_NOISE = re.compile(
    r"\s+(?:step[- ]by[- ]step|with (?:full |complete |detailed )?(?:steps|instructions)|please|from scratch)\s*$"
)
_LIST = re.compile(r"\b(all|list|every|show me|give me|which ones)\b")
_LIST_TARGET = re.compile(r"\b(components?|parts|items|options|ics|chips|modules)\b|\brelated to\b")
_RELATED = re.compile(r"\b(related to|associated with|used (?:for|in|with))\b")
_SOURCE = re.compile(r"\b(source|sources|cite|citation|reference|where (?:did|does) .* come from)\b")
_EXPLANATION = re.compile(
    r"\b(explain|how does|how do|what does|how .* works?|working principle|what is an?|describe|purpose of|"
    r"function of|pinout)\b"
)
_PRICE_WORDS = re.compile(r"\b(price|prices|priced|cost|costs|costing|how much|egp|cheap|cheaper|cheapest|expensive)\b")
_REFERS_BACK = re.compile(
    r"\b(these|those|them|it|its|that one|the same|above|previous|you used|you mentioned|the price you)\b"
)
# Last "build a X" / "for a X" / "related to X" phrase in the question.
_TOPIC = re.compile(
    r".*\b(?:to build|for building|to make|to design|build|make|design|implement|create|needed for|required for|"
    r"need for|use for|"
    r"for|related to|associated with|used in)\s+(?:an?\s+|the\s+|my\s+|some\s+)?([^?.!,;]+?)\s*[?.!]*$"
)
MAX_HISTORY_TOPIC_CHARS = 600
MIN_SPEC_WORDS = 10  # shorter turns ("how should I do this?") are follow-ups, not project specs

_LIMIT = re.compile(r"\b(?:top|first)\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:cheapest|most expensive)\b")

# Quantity patterns: "2 x NE555", "2 × NE555", "2 pcs NE555", "NE555 x2", "2 NE555"
_ITEM_END = r"(?=\s*(?:,|;|\+|\n|\band\b|\bplus\b|\bwith\b|$|\?|\.(?:\s|$)))"
_QTY_EXPLICIT = re.compile(
    rf"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:x|\*|pcs?\.?|pieces?(?: of)?|units?(?: of)?)\s+([^\n,;+?]+?){_ITEM_END}"
)
_QTY_SUFFIX = re.compile(rf"([a-z][\w\- ]*?)\s*(?:x|\*)\s*(\d+){_ITEM_END}")
_QTY_IMPLICIT = re.compile(rf"(?<![\w.\-])(\d{{1,3}})\s+([a-z0-9][^\n,;+?]*?){_ITEM_END}")
_MENTION_NOISE = re.compile(
    r"^(?:of|the|a|an)\s+|\s+(?:each|please|approximately|approx|would|will|cost|costs|total|in total|and)$"
)
_CURRENCY_WORD = re.compile(r"^(egp|le|pounds?|usd|eur)\b")


@dataclass
class ChatMessage:
    role: str
    content: str


def _normalize(question: str) -> str:
    text = unicodedata.normalize("NFKC", question)
    text = text.replace("×", " x ")
    return re.sub(r"[ \t]+", " ", text).strip()


def _dec(raw: str) -> Decimal:
    return Decimal(raw)


class QueryProcessor:
    def __init__(self, default_currency: str = "EGP") -> None:
        self.default_currency = default_currency

    # ---------------------------------------------------------------- public
    def analyze(self, question: str, history: list[ChatMessage] | None = None) -> QueryAnalysis:
        text = _normalize(question)
        lower = text.lower()
        analysis = QueryAnalysis(original=question, normalized=text)

        price, price_spans = self._price_constraint(lower)
        analysis.price = price
        analysis.currency = price.currency or normalize_currency(lower)
        if price.active and price.currency is None:
            price.currency = analysis.currency or self.default_currency
        analysis.currency = analysis.currency or (self.default_currency if price.active else None)

        if _SORT_ASC.search(lower):
            analysis.sort = "asc"
        elif _SORT_DESC.search(lower):
            analysis.sort = "desc"

        limit = _LIMIT.search(lower)
        if limit:
            analysis.limit = int(limit.group(1) or limit.group(2))

        analysis.is_price_question = bool(_PRICE_WORDS.search(lower))
        analysis.is_explanation = bool(_EXPLANATION.search(lower)) and not lower.startswith("what is the price")
        analysis.is_source_request = bool(_SOURCE.search(lower))
        analysis.is_recommendation = bool(_RECOMMENDATION.search(lower))
        analysis.is_list_request = bool((_LIST.search(lower) and _LIST_TARGET.search(lower)) or _RELATED.search(lower))

        # Identifiers: ignore price expressions ("100 EGP" is not part number 100).
        stripped = text
        for start, end in sorted(price_spans, reverse=True):
            stripped = stripped[:start] + " " + stripped[end:]
        stripped = _PRICE_SPAN.sub(" ", stripped)
        analysis.identifiers = extract_identifiers(stripped)

        analysis.categories = [c.value for c in detect_categories(stripped)]

        has_explicit_qty = bool(_QTY_EXPLICIT.search(lower) or _QTY_SUFFIX.search(lower))
        analysis.is_calculation = bool(_CALCULATION.search(lower)) and not (
            # Treat "give me all the components for X" as a recommendation, not a calculation.
            _LIST.search(lower) and _LIST_TARGET.search(lower) and not _PRICE_WORDS.search(lower)
        )
        if analysis.is_calculation and not price.active:
            analysis.quantities = self._quantities(stripped)

        mentions_two = len(analysis.identifiers) >= 2 or bool(re.search(r"\bor\b|,", lower))
        analysis.is_comparison = bool(_COMPARISON.search(lower)) and not (
            # Avoid triggering comparison if it's just a list request.
            _LIST.search(lower) and _LIST_TARGET.search(lower)
        )

        # "Explain step by step how the CD4511 works" is an explanation, not a project build.
        analysis.is_build_guide = bool(_BUILD_GUIDE.search(lower)) and not (
            analysis.identifiers and analysis.is_explanation and not analysis.is_recommendation
        )

        if analysis.is_recommendation or analysis.is_list_request or analysis.is_build_guide:
            analysis.topic = self._topic(lower)

        # "Give me the steps" / "how should I do this?" → the project named in an earlier turn.
        if (analysis.is_recommendation or analysis.is_build_guide) and not analysis.topic and history:
            analysis.topic = self._topic_from_history(history)
            analysis.topic_from_history = analysis.topic is not None

        analysis.refers_to_previous = bool(_REFERS_BACK.search(lower)) and not analysis.identifiers
        if history and (self._needs_history(analysis) or analysis.topic_from_history):
            analysis.inherited_from_history = True
        return analysis

    @staticmethod
    def _topic(lower: str) -> str | None:
        match = _TOPIC.match(lower)
        if not match or len(match.group(1).split()) > 8:
            return None
        topic = _TOPIC_NOISE.sub("", match.group(1)).strip()
        return None if not topic or _VAGUE_TOPIC.match(topic) else topic

    def _topic_from_history(self, history: list[ChatMessage]) -> str | None:
        for message in reversed(history):
            if message.role != "user":
                continue
            text = _normalize(message.content)
            lower = text.lower()
            if _RECOMMENDATION.search(lower) or _BUILD_GUIDE.search(lower):
                topic = self._topic(lower)
                if topic:
                    return topic
                # A long project spec has no short "build a X" phrase; the spec itself is the topic.
                if len(lower.split()) >= MIN_SPEC_WORDS:
                    return text[:MAX_HISTORY_TOPIC_CHARS]
        return None

    def history_text(self, history: list[ChatMessage] | None, max_turns: int = 3) -> str:
        """Recent conversation text used to resolve "these"/"it" references."""
        if not history:
            return ""
        recent = [m.content for m in history if m.role in ("user", "assistant")][-max_turns * 2 :]
        return "\n".join(recent)

    @staticmethod
    def _needs_history(a: QueryAnalysis) -> bool:
        # Expand history needs to include recommendation/list requests that lack a topic.
        no_subject = not a.identifiers and not a.categories and not a.quantities and not a.topic
        return no_subject and (a.refers_to_previous or a.is_source_request or a.is_calculation or a.is_recommendation or a.is_list_request)

    def _price_constraint(self, lower: str) -> tuple[PriceConstraint, list[tuple[int, int]]]:
        pc = PriceConstraint()
        spans: list[tuple[int, int]] = []
        currencies: list[str | None] = []

        between = _BETWEEN.search(lower)
        if between:
            low, high = _dec(between.group(1)), _dec(between.group(3))
            pc.min_price, pc.max_price = min(low, high), max(low, high)
            pc.min_inclusive = pc.max_inclusive = True
            currencies += [between.group(2), between.group(4)]
            spans.append(between.span())
        else:
            for pattern, inclusive, is_max in (
                (_MAX_INCLUSIVE, True, True),
                (_MAX_OR_LESS, True, True),
                (_MAX_EXCLUSIVE, False, True),
                (_MIN_INCLUSIVE, True, False),
                (_MIN_OR_MORE, True, False),
                (_MIN_EXCLUSIVE, False, False),
            ):
                m = pattern.search(lower)
                if not m:
                    continue
                if is_max and pc.max_price is None:
                    pc.max_price, pc.max_inclusive = _dec(m.group(1)), inclusive
                elif not is_max and pc.min_price is None:
                    pc.min_price, pc.min_inclusive = _dec(m.group(1)), inclusive
                else:
                    continue
                currencies.append(m.group(2))
                spans.append(m.span())

        if pc.max_price is None and pc.min_price is None:
            m = _APPROX_WITH_CURRENCY.search(lower) or _APPROX_AFTER_COST.search(lower) or _EXACT_PRICE.search(lower)
            if m:
                pc.approx_price = _dec(m.group(1))
                if m.lastindex and m.lastindex >= 2:
                    currencies.append(m.group(2))
                spans.append(m.span())

        found = next((normalize_currency(c) for c in currencies if c), None)
        pc.currency = found
        return pc, spans

    def _quantities(self, text: str) -> list[QuantityItem]:
        lower = text.lower()
        items: list[QuantityItem] = []
        taken: list[tuple[int, int]] = []

        def add(qty: str, mention: str, span: tuple[int, int]) -> None:
            if any(s < span[1] and span[0] < e for s, e in taken):
                return
            mention = mention.strip(" .:-")
            for _ in range(3):
                mention = _MENTION_NOISE.sub("", mention).strip()
            if not mention or not re.search(r"[a-z]", mention, re.I) or _CURRENCY_WORD.match(mention.lower()):
                return
            if len(mention) > 60:
                return
            q = Decimal(qty)
            if q <= 0 or q > 10000:
                return
            taken.append(span)
            items.append(QuantityItem(quantity=q, mention=mention))

        for m in _QTY_EXPLICIT.finditer(lower):
            add(m.group(1), text[m.start(2) : m.end(2)], m.span())
        for m in _QTY_SUFFIX.finditer(lower):
            add(m.group(2), text[m.start(1) : m.end(1)], m.span())
        for m in _QTY_IMPLICIT.finditer(lower):
            add(m.group(1), text[m.start(2) : m.end(2)], m.span())
        return items
