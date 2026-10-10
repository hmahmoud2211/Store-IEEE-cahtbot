from decimal import Decimal

import pytest

from app.rag.query_processor import ChatMessage, QueryProcessor
from app.rag.query_router import QueryRouter
from app.rag.types import QueryType

qp = QueryProcessor()


def test_price_filter_extraction():
    a = qp.analyze("Which counter costs less than 25 EGP?")
    assert a.categories == ["Counter"]
    assert a.price.max_price == Decimal(25)
    assert a.price.max_inclusive is False
    assert a.price.currency == "EGP"
    assert a.identifiers == []  # "25" is a price, not a part number


def test_inclusive_and_between_constraints():
    a = qp.analyze("components up to 30 EGP")
    assert a.price.max_price == Decimal(30) and a.price.max_inclusive
    b = qp.analyze("parts between 10 and 20 EGP")
    assert (b.price.min_price, b.price.max_price) == (Decimal(10), Decimal(20))
    c = qp.analyze("Give me all components costing below 15 EGP.")
    assert c.price.max_price == Decimal(15)


def test_approximate_price():
    a = qp.analyze("Which sensor costs approximately 40 EGP?")
    assert a.price.approx_price == Decimal(40)
    assert a.categories == ["Sensor"]


def test_comparison_extraction():
    a = qp.analyze("Compare DS1307 and DS3231.")
    assert a.identifiers == ["DS1307", "DS3231"]
    assert a.is_comparison


def test_which_is_cheaper_is_comparison():
    a = qp.analyze("Which sensor is cheaper, HC-SR04 or IR obstacle sensor?")
    assert a.is_comparison
    assert not a.price.active


def test_cheapest_extraction():
    a = qp.analyze("What is the cheapest 7-segment display?")
    assert a.sort == "asc"
    assert a.categories == ["Display"]


def test_quantities_extraction():
    a = qp.analyze("How much would 2 × NE555, 1 × 400-point breadboard and 5 × LED cost?")
    assert a.is_calculation
    assert [(q.quantity, q.mention) for q in a.quantities] == [
        (Decimal(2), "NE555"),
        (Decimal(1), "400-point breadboard"),
        (Decimal(5), "LED"),
    ]


def test_multiline_quantities():
    a = qp.analyze("Estimate the cost:\n2 x NE555\n1 x breadboard\n5 x LED")
    assert [q.mention for q in a.quantities] == ["NE555", "breadboard", "LED"]


def test_implicit_quantities():
    a = qp.analyze("How much would 2 NE555 and 5 LEDs cost in total?")
    assert [(q.quantity, q.mention) for q in a.quantities] == [(Decimal(2), "NE555"), (Decimal(5), "LEDs")]


def test_recommendation_and_list_intents():
    assert qp.analyze("What components can I use to build a digital clock?").is_recommendation
    assert qp.analyze("What components do I need for a timer circuit?").is_recommendation
    assert qp.analyze("Give me all components related to counters.").is_list_request


def test_how_to_build_is_recommendation():
    a = qp.analyze("How should I build a digital clock?")
    assert a.is_recommendation and not a.is_build_guide
    assert a.topic == "digital clock"
    assert qp.analyze("I want to make a line follower robot").topic == "line follower robot"


def test_build_guide_intent():
    a = qp.analyze("How do I build a digital clock step by step?")
    assert a.is_build_guide and a.topic == "digital clock"
    assert not qp.analyze("Explain step by step how the CD4511 works.").is_build_guide


@pytest.mark.parametrize(
    "follow_up", ["Give me instructions step by step", "How should I do this?", "Give me the steps for this project"]
)
def test_project_follow_up_inherits_topic(follow_up):
    history = [
        ChatMessage("user", "What components do I need to build a digital clock?"),
        ChatMessage("assistant", "You need a DS3231 RTC module and a 7-segment display."),
    ]
    a = qp.analyze(follow_up, history)
    assert a.topic == "digital clock"
    assert a.topic_from_history and a.inherited_from_history


def test_follow_up_uses_long_project_spec_as_topic():
    spec = (
        "I want to build a digital circuit for a safe that accepts a binary passcode, compares it to a preset "
        "code and locks after three wrong trials."
    )
    history = [ChatMessage("user", spec), ChatMessage("user", "How should I do this?")]
    a = qp.analyze("Give me instructions step by step", history)
    assert a.topic == spec and a.topic_from_history


def test_history_reference():
    history = [ChatMessage("user", "What is the price of the DS3231?")]
    a = qp.analyze("Show the source of the price you used.", history)
    assert a.is_source_request and a.inherited_from_history


# --------------------------------------------------------------- routing
router = QueryRouter()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the price of a 74LS85 comparator?", QueryType.FACT_LOOKUP),
        ("Which counter ICs cost less than 25 EGP?", QueryType.PRICE_FILTER),
        ("What components can I use to build a digital clock?", QueryType.RECOMMENDATION),
        ("Compare the DS1307 and DS3231.", QueryType.COMPARISON),
        ("What is the cheapest 7-segment display?", QueryType.PRICE_FILTER),
        ("Give me all components related to counters.", QueryType.SEMANTIC_SEARCH),
        ("What components do I need for a timer circuit?", QueryType.RECOMMENDATION),
        ("Which sensor is cheaper, HC-SR04 or IR obstacle sensor?", QueryType.COMPARISON),
        ("How much would 2 x NE555 and 5 x LED cost?", QueryType.CALCULATION),
        ("Which components are below 20 EGP?", QueryType.PRICE_FILTER),
        ("Explain how the CD4511 works.", QueryType.SEMANTIC_SEARCH),
        ("What is the price of an ESP32?", QueryType.FACT_LOOKUP),
        ("How should I build a digital clock?", QueryType.RECOMMENDATION),
        ("Give me step by step instructions to build a digital clock", QueryType.PROJECT_GUIDE),
    ],
)
def test_router(alias_index, question, expected):
    analysis = qp.analyze(question)
    resolution = alias_index.resolve(analysis.normalized, analysis.identifiers)
    assert router.route(analysis, resolution).query_type is expected


def test_router_prefers_sql_for_numeric_filters(alias_index):
    analysis = qp.analyze("Which components are below 20 EGP?")
    plan = router.route(analysis, alias_index.resolve(analysis.normalized, analysis.identifiers))
    assert plan.use_structured and not plan.use_retrieval
