"""Alias / identifier resolution — the examples the spec requires to work."""

import pytest


def _names(resolution):
    return [r.name for r in resolution.resolved]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("555 timer", "NE555 Timer IC"),
        ("7485", "74LS85 / 7485 Comparator"),
        ("HC SR04", "HC-SR04 Ultrasonic Sensor"),
        ("hcsr04", "HC-SR04 Ultrasonic Sensor"),
        ("HC-SR04", "HC-SR04 Ultrasonic Sensor"),
        ("74LS08", "74LS08 / 7408 AND Gate"),
        ("7408", "74LS08 / 7408 AND Gate"),
        ("AND gate", "74LS08 / 7408 AND Gate"),
        ("quad AND gate", "74LS08 / 7408 AND Gate"),
        ("hex inverter", "74HC04 / NOT Gate"),
        ("4017", "CD4017 Counter"),
        ("LM7805", "7805 voltage regulator"),
        ("IR obstacle sensor", "IR obstacle sensor"),
        ("LEDs", "Standard LED"),
        ("74HC85", "74LS85 / 7485 Comparator"),  # logic-family equivalent
    ],
)
def test_resolves_aliases(alias_index, query, expected):
    resolution = alias_index.resolve(query)
    assert expected in _names(resolution), (query, _names(resolution))


def test_555_timer_prefers_ic_over_module(alias_index):
    names = _names(alias_index.resolve("555 timer"))
    assert names == ["NE555 Timer IC"]


def test_ne555_matches_both_ic_and_module(alias_index):
    names = set(_names(alias_index.resolve("NE555")))
    assert names == {"NE555 Timer IC", "NE555 adjustable timer module"}


def test_comparison_resolves_all_mentions(alias_index):
    names = set(_names(alias_index.resolve("Compare the DS1307 and DS3231.")))
    assert {"DS1307 RTC IC", "DS1307 RTC Module + battery", "DS3231 RTC Module"} <= names


def test_mixed_question(alias_index):
    names = set(_names(alias_index.resolve("Which sensor is cheaper, HC-SR04 or IR obstacle sensor?")))
    assert names == {"HC-SR04 Ultrasonic Sensor", "IR obstacle sensor"}


def test_unknown_identifier_is_unresolved(alias_index):
    resolution = alias_index.resolve("What is the price of an ESP32?")
    assert resolution.resolved == []
    assert resolution.unresolved == ["ESP32"]


def test_generic_words_do_not_resolve(alias_index):
    assert alias_index.resolve("What components can I use to build a digital clock?").resolved == []
    assert alias_index.resolve("Which counter ICs cost less than 25 EGP?").resolved == []


def test_name_overlap_prefers_specific_item(alias_index):
    ic = alias_index.test_ids["NE555 Timer IC"]
    module = alias_index.test_ids["NE555 adjustable timer module"]
    assert alias_index.name_overlap("NE555", ic) > alias_index.name_overlap("NE555", module)
