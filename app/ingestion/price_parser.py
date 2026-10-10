"""Parse human price strings into structured, numeric price ranges.

    "4–10 EGP"        -> PriceRange(min=4,   max=10,  currency="EGP", unit="piece")
    "0.5–2 EGP each"  -> PriceRange(min=0.5, max=2,   currency="EGP", unit="piece")
    "25 LE"           -> PriceRange(min=25,  max=25,  currency="EGP")

The original text is always preserved in `PriceRange.original`.
All arithmetic uses Decimal so calculated totals never suffer float drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation

# Longest aliases first so "l.e." wins over "le".
_CURRENCY_ALIASES: list[tuple[str, str]] = [
    (r"egyptian\s+pounds?", "EGP"),
    (r"ج\.?\s?م\.?", "EGP"),
    (r"جنيه(?:ات)?", "EGP"),
    (r"l\.\s?e\.?", "EGP"),
    (r"e£", "EGP"),
    (r"egp", "EGP"),
    (r"le", "EGP"),
    (r"pounds?", "EGP"),
    (r"usd", "USD"),
    (r"us\$", "USD"),
    (r"\$", "USD"),
    (r"dollars?", "USD"),
    (r"eur(?:os?)?", "EUR"),
    (r"€", "EUR"),
]
_CURRENCY_RE = re.compile(r"(?<![a-z])(" + "|".join(p for p, _ in _CURRENCY_ALIASES) + r")(?![a-z])", re.IGNORECASE)

_NUMBER = r"\d+(?:[.,]\d+)*"
_RANGE_RE = re.compile(rf"(?P<min>{_NUMBER})\s*(?:-|–|—|~|to|until)\s*(?P<max>{_NUMBER})", re.IGNORECASE)
_SINGLE_RE = re.compile(rf"(?P<value>{_NUMBER})")

_UNIT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(each|per\s+(?:piece|pc|unit|item)|/\s*(?:pc|pcs|piece|unit)|a\s+piece)\b", re.I), "piece"),
    (re.compile(r"\b(per\s+set|/\s*set|a\s+set)\b", re.I), "set"),
    (re.compile(r"\b(per\s+pack|/\s*pack)\b", re.I), "pack"),
    (re.compile(r"\b(per\s+meter|/\s*m)\b", re.I), "meter"),
]


def normalize_currency(text: str | None) -> str | None:
    """Map any recognised currency token in `text` to its ISO code."""
    if not text:
        return None
    match = _CURRENCY_RE.search(text)
    if not match:
        return None
    token = match.group(1)
    for pattern, code in _CURRENCY_ALIASES:
        if re.fullmatch(pattern, token, re.IGNORECASE):
            return code
    return None


def _to_decimal(raw: str) -> Decimal | None:
    raw = raw.strip()
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", raw):  # thousands separators: 1,200.50
        raw = raw.replace(",", "")
    else:  # decimal comma: 0,5
        raw = raw.replace(",", ".")
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def format_amount(value: Decimal) -> str:
    """4.00 -> '4', 0.50 -> '0.5', 12.25 -> '12.25'."""
    normalized = value.quantize(Decimal("0.01")).normalize()
    text = format(normalized, "f")
    return text


@dataclass(frozen=True)
class PriceRange:
    min: Decimal
    max: Decimal
    currency: str | None
    unit: str = "piece"
    original: str = ""

    @property
    def is_range(self) -> bool:
        return self.min != self.max

    @property
    def midpoint(self) -> Decimal:
        return (self.min + self.max) / 2

    def multiply(self, quantity: Decimal | int) -> PriceRange:
        q = Decimal(quantity)
        return replace(self, min=self.min * q, max=self.max * q, original="")

    def __add__(self, other: PriceRange) -> PriceRange:
        if self.currency and other.currency and self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return PriceRange(
            min=self.min + other.min,
            max=self.max + other.max,
            currency=self.currency or other.currency,
            unit="total",
        )

    def format(self, with_currency: bool = True) -> str:
        amount = f"{format_amount(self.min)}–{format_amount(self.max)}" if self.is_range else format_amount(self.min)
        return f"{amount} {self.currency}" if with_currency and self.currency else amount

    def as_dict(self) -> dict[str, object]:
        return {
            "price_min": float(self.min),
            "price_max": float(self.max),
            "currency": self.currency,
            "price_unit": self.unit,
            "price_text": self.original or self.format(),
        }


def parse_price(text: str | None, default_currency: str | None = None) -> PriceRange | None:
    """Parse a price/price-range string. Returns None when no number is present."""
    if text is None:
        return None
    original = str(text).strip()
    if not original:
        return None

    currency = normalize_currency(original) or default_currency

    unit = "piece"
    for pattern, unit_name in _UNIT_PATTERNS:
        if pattern.search(original):
            unit = unit_name
            break

    range_match = _RANGE_RE.search(original)
    if range_match:
        low = _to_decimal(range_match.group("min"))
        high = _to_decimal(range_match.group("max"))
        if low is None or high is None:
            return None
        if low > high:
            low, high = high, low
        return PriceRange(min=low, max=high, currency=currency, unit=unit, original=original)

    single = _SINGLE_RE.search(original)
    if single:
        value = _to_decimal(single.group("value"))
        if value is None:
            return None
        return PriceRange(min=value, max=value, currency=currency, unit=unit, original=original)
    return None


def sum_ranges(ranges: list[PriceRange]) -> PriceRange | None:
    if not ranges:
        return None
    total = ranges[0].multiply(1)
    for r in ranges[1:]:
        total = total + r
    return replace(total, unit="total")
