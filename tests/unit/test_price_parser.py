from decimal import Decimal

import pytest

from app.ingestion.price_parser import PriceRange, format_amount, normalize_currency, parse_price, sum_ranges


@pytest.mark.parametrize(
    ("text", "low", "high", "currency", "unit"),
    [
        ("4–10 EGP", "4", "10", "EGP", "piece"),
        ("4-10 EGP", "4", "10", "EGP", "piece"),
        ("4 to 10 EGP", "4", "10", "EGP", "piece"),
        ("25 EGP", "25", "25", "EGP", "piece"),
        ("0.5–2 EGP each", "0.5", "2", "EGP", "piece"),
        ("1–5 EGP each", "1", "5", "EGP", "piece"),
        ("135 EGP", "135", "135", "EGP", "piece"),
        ("25 LE", "25", "25", "EGP", "piece"),
        ("30 L.E.", "30", "30", "EGP", "piece"),
        ("1,200 EGP", "1200", "1200", "EGP", "piece"),
        ("0,5 EGP", "0.5", "0.5", "EGP", "piece"),
        ("$3.50", "3.50", "3.50", "USD", "piece"),
        ("40 EGP per set", "40", "40", "EGP", "set"),
        ("10–4 EGP", "4", "10", "EGP", "piece"),  # reversed range is normalized
    ],
)
def test_parse_price(text, low, high, currency, unit):
    price = parse_price(text)
    assert price is not None
    assert price.min == Decimal(low)
    assert price.max == Decimal(high)
    assert price.currency == currency
    assert price.unit == unit
    assert price.original == text  # original text is always preserved


def test_parse_price_default_currency_and_invalid():
    assert parse_price("15", default_currency="EGP").currency == "EGP"
    assert parse_price("call for price") is None
    assert parse_price("") is None
    assert parse_price(None) is None


def test_normalize_currency():
    assert normalize_currency("25 EGP") == "EGP"
    assert normalize_currency("25 le") == "EGP"
    assert normalize_currency("25 جنيه") == "EGP"
    assert normalize_currency("about 3 USD") == "USD"
    # 'le' inside a word must not be treated as a currency
    assert normalize_currency("available") is None


def test_price_range_arithmetic():
    ne555 = parse_price("4–10 EGP")
    assert ne555.multiply(2).min == Decimal(8)
    assert ne555.multiply(2).max == Decimal(20)
    total = sum_ranges([ne555.multiply(2), parse_price("25 EGP"), parse_price("1–3 EGP").multiply(5)])
    assert (total.min, total.max) == (Decimal(38), Decimal(60))
    assert total.format() == "38–60 EGP"


def test_price_range_mixed_currency_rejected():
    with pytest.raises(ValueError):
        _ = PriceRange(Decimal(1), Decimal(1), "EGP") + PriceRange(Decimal(1), Decimal(1), "USD")


def test_format_amount():
    assert format_amount(Decimal("4.00")) == "4"
    assert format_amount(Decimal("0.50")) == "0.5"
    assert format_amount(Decimal("12.25")) == "12.25"
    assert parse_price("25 EGP").format() == "25 EGP"
    assert parse_price("4–10 EGP").format() == "4–10 EGP"
