"""Rule-based component categorization (ingestion side) and category detection in queries.

Rules are ORDERED: more specific rules come first, e.g. "Breadboard 5V power
module" must be a Power Supply (not a Breadboard) and "7-Segment Driver" must be
a Display Driver (not a Display).
"""

from __future__ import annotations

import re
from enum import Enum


class Category(str, Enum):
    TIMER = "Timer"
    COUNTER = "Counter"
    LOGIC_GATE = "Logic Gate"
    COMPARATOR = "Comparator"
    FLIP_FLOP = "Flip-Flop"
    DISPLAY = "Display"
    DISPLAY_DRIVER = "Display Driver"
    SENSOR = "Sensor"
    RTC = "RTC"
    INPUT_DEVICE = "Input Device"
    OUTPUT_DEVICE = "Output Device"
    TRANSISTOR = "Transistor"
    PASSIVE = "Passive Component"
    BREADBOARD = "Breadboard"
    POWER_SUPPLY = "Power Supply"
    VOLTAGE_REGULATOR = "Voltage Regulator"
    WIRING = "Wiring"
    MICROCONTROLLER = "Microcontroller"
    OTHER = "Other"


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


# Ingestion-side rules: applied to the component *name* (and optionally description).
_COMPONENT_RULES: list[tuple[Category, re.Pattern[str]]] = [
    (Category.POWER_SUPPLY, _rx(r"\bpower\s+(module|supply)\b", r"\bpsu\b", r"\bmb102\b")),
    (
        Category.VOLTAGE_REGULATOR,
        _rx(r"\bregulator\b", r"\b(?:lm|l)?78\d\d\b", r"\b(?:lm|l)?79\d\d\b", r"\blm317\b", r"\bams1117\b"),
    ),
    (Category.DISPLAY_DRIVER, _rx(r"\bdriver\b", r"\bdecoder\b", r"\bmax7219\b")),
    (Category.DISPLAY, _rx(r"\bsegment\s+display\b", r"\bdisplay\b", r"\blcd\b", r"\boled\b")),
    (Category.RTC, _rx(r"\brtc\b", r"\bds1307\b", r"\bds3231\b", r"\breal[\s-]?time\s+clock\b")),
    (Category.TIMER, _rx(r"\btimer\b", r"\b(?:ne|lm)?555\b", r"\b556\b")),
    (Category.COUNTER, _rx(r"\bcounter\b")),
    (Category.COMPARATOR, _rx(r"\bcomparator\b")),
    (Category.FLIP_FLOP, _rx(r"\bflip[\s-]?flop\b", r"\blatch\b")),
    (Category.LOGIC_GATE, _rx(r"\b(?:and|or|not|nand|nor|xor|xnor)\s+gate\b", r"\binverter\b", r"\blogic\s+gate\b")),
    (Category.SENSOR, _rx(r"\bsensor\b", r"\bhc-?sr04\b", r"\bdht1[12]\b", r"\bldr\b")),
    (
        Category.MICROCONTROLLER,
        _rx(r"\barduino\b", r"\besp32\b", r"\besp8266\b", r"\batmega\d+\b", r"\bmicrocontroller\b"),
    ),
    (Category.TRANSISTOR, _rx(r"\btransistor\b", r"\bmosfet\b", r"\b2n\d{4}\b", r"\bbc5\d\d\b")),
    (
        Category.PASSIVE,
        _rx(r"\bresistors?\b", r"\bcapacitors?\b", r"\binductors?\b", r"\bpotentiometer\b", r"\bcrystal\b"),
    ),
    (Category.BREADBOARD, _rx(r"\bbreadboard\b", r"\bprotoboard\b", r"\bperfboard\b")),
    (Category.WIRING, _rx(r"\bjumper\b", r"\bwires?\b", r"\bcables?\b", r"\bdupont\b")),
    (Category.INPUT_DEVICE, _rx(r"\bswitch(?:es)?\b", r"\bbuttons?\b", r"\bkeypad\b", r"\bjoystick\b")),
    (Category.OUTPUT_DEVICE, _rx(r"\bleds?\b", r"\bbuzzer\b", r"\bspeaker\b", r"\bmotor\b", r"\brelay\b")),
]


def categorize(name: str, description: str | None = None) -> Category:
    for category, pattern in _COMPONENT_RULES:
        if pattern.search(name):
            return category
    if description:
        for category, pattern in _COMPONENT_RULES:
            if pattern.search(description):
                return category
    return Category.OTHER


def parse_category(value: str | None) -> Category | None:
    if not value:
        return None
    key = re.sub(r"[\s_-]+", " ", value).strip().lower()
    for cat in Category:
        if re.sub(r"[\s_-]+", " ", cat.value).lower() == key or cat.name.lower().replace("_", " ") == key:
            return cat
    # Allow free-text category columns ("Timers", "7-seg displays"…).
    detected = detect_categories(value)
    return detected[0] if detected else None


# Query-side detection: tolerant to plurals and colloquial phrasing.
_QUERY_RULES: list[tuple[Category, re.Pattern[str]]] = [
    (Category.POWER_SUPPLY, _rx(r"\bpower\s+(modules?|supply|supplies)\b")),
    (Category.VOLTAGE_REGULATOR, _rx(r"\b(voltage\s+)?regulators?\b")),
    (Category.DISPLAY_DRIVER, _rx(r"\b(display\s+)?drivers?\b", r"\bdecoders?\b")),
    (Category.DISPLAY, _rx(r"\b(7|seven)[\s-]?seg(ment)?s?\b", r"\bdisplays?\b", r"\blcds?\b")),
    (Category.RTC, _rx(r"\brtcs?\b", r"\breal[\s-]?time[\s-]?clocks?\b")),
    (Category.TIMER, _rx(r"\btimers?\b")),
    (Category.COUNTER, _rx(r"\bcounters?\b", r"\bcounting\b")),
    (Category.COMPARATOR, _rx(r"\bcomparators?\b")),
    (Category.FLIP_FLOP, _rx(r"\bflip[\s-]?flops?\b", r"\blatches\b")),
    (Category.LOGIC_GATE, _rx(r"\b(logic\s+)?gates?\b", r"\binverters?\b")),
    (Category.SENSOR, _rx(r"\bsensors?\b")),
    (Category.MICROCONTROLLER, _rx(r"\bmicrocontrollers?\b", r"\bmcus?\b")),
    (Category.TRANSISTOR, _rx(r"\btransistors?\b")),
    (Category.PASSIVE, _rx(r"\bpassives?\b", r"\bresistors?\b", r"\bcapacitors?\b")),
    (Category.BREADBOARD, _rx(r"\bbreadboards?\b")),
    (Category.WIRING, _rx(r"\bjumpers?\b", r"\bwires?\b", r"\bwiring\b")),
    (Category.INPUT_DEVICE, _rx(r"\binput\s+devices?\b", r"\bswitch(es)?\b", r"\bbuttons?\b")),
    (Category.OUTPUT_DEVICE, _rx(r"\boutput\s+devices?\b", r"\bleds?\b", r"\bbuzzers?\b")),
]


def detect_categories(text: str) -> list[Category]:
    """Detect categories mentioned in free text (query side). Order-preserving, deduped."""
    found: list[Category] = []
    lowered = text.lower()
    for category, pattern in _QUERY_RULES:
        if pattern.search(lowered) and category not in found:
            # "breadboard power module" is a power supply, not a breadboard.
            if category is Category.BREADBOARD and Category.POWER_SUPPLY in found:
                continue
            # "7-segment driver" -> driver only.
            if (
                category is Category.DISPLAY
                and Category.DISPLAY_DRIVER in found
                and not re.search(r"\bdisplays?\b", lowered)
            ):
                continue
            found.append(category)
    return found
