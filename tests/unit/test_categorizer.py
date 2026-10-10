import pytest

from app.ingestion.categorizer import Category, categorize, detect_categories, parse_category

EXPECTED = {
    "NE555 Timer IC": Category.TIMER,
    "NE555 adjustable timer module": Category.TIMER,
    "74LS85 / 7485 Comparator": Category.COMPARATOR,
    "74LS161 Counter": Category.COUNTER,
    "7493 Binary Counter": Category.COUNTER,
    "74LS08 / 7408 AND Gate": Category.LOGIC_GATE,
    "74LS32 / 7432 OR Gate": Category.LOGIC_GATE,
    "74HC04 / NOT Gate": Category.LOGIC_GATE,
    "74LS74 Flip-Flop": Category.FLIP_FLOP,
    "CD4017 Counter": Category.COUNTER,
    "CD4511 BCD-to-7-Segment Decoder": Category.DISPLAY_DRIVER,
    "7447 / 7448 7-Segment Driver": Category.DISPLAY_DRIVER,
    "Single 7-segment display": Category.DISPLAY,
    "Dual 7-segment display": Category.DISPLAY,
    "4-digit 7-segment display": Category.DISPLAY,
    "HC-SR04 Ultrasonic Sensor": Category.SENSOR,
    "IR obstacle sensor": Category.SENSOR,
    "DS1307 RTC IC": Category.RTC,
    "DS1307 RTC Module + battery": Category.RTC,
    "DS3231 RTC Module": Category.RTC,
    "DIP switch": Category.INPUT_DEVICE,
    "Push button": Category.INPUT_DEVICE,
    "Standard LED": Category.OUTPUT_DEVICE,
    "Buzzer": Category.OUTPUT_DEVICE,
    "2N2222 / similar transistor": Category.TRANSISTOR,
    "Resistors": Category.PASSIVE,
    "Capacitors": Category.PASSIVE,
    "400-point breadboard": Category.BREADBOARD,
    "830-point breadboard": Category.BREADBOARD,
    "Jumper wire set": Category.WIRING,
    "Breadboard 5V power module": Category.POWER_SUPPLY,
    "7805 voltage regulator": Category.VOLTAGE_REGULATOR,
}


@pytest.mark.parametrize(("name", "category"), EXPECTED.items())
def test_categorize_seed_components(name, category):
    assert categorize(name) is category


def test_all_seed_rows_categorized(seed_records):
    assert len(seed_records) == 32
    for rec in seed_records:
        assert rec.category is EXPECTED[rec.name], rec.name


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which counter ICs cost less than 25 EGP?", [Category.COUNTER]),
        ("What is the cheapest 7-segment display?", [Category.DISPLAY]),
        ("What components are related to seven-segment displays?", [Category.DISPLAY]),
        ("Which sensor is cheaper?", [Category.SENSOR]),
        ("price of a breadboard power module", [Category.POWER_SUPPLY]),
        ("7 segment driver options", [Category.DISPLAY_DRIVER]),
        ("What components can I use to build a digital clock?", []),
    ],
)
def test_detect_categories(question, expected):
    assert detect_categories(question) == expected


def test_parse_category():
    assert parse_category("Logic Gate") is Category.LOGIC_GATE
    assert parse_category("logic_gate") is Category.LOGIC_GATE
    assert parse_category("Timers") is Category.TIMER
    assert parse_category(None) is None
