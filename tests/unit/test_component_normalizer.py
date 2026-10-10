import pytest

from app.ingestion.component_normalizer import (
    build_search_keys,
    compact_key,
    extract_identifiers,
    generate_aliases,
    identifier_variants,
    normalize_name,
    phrase_key,
    split_name,
)


def test_normalize_name():
    assert normalize_name("NE555 Timer IC") == "ne555 timer ic"
    assert normalize_name("  74LS85 /  7485   Comparator ") == "74ls85 / 7485 comparator"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("NE555 Timer IC", ["NE555"]),
        ("74LS85 / 7485 Comparator", ["74LS85", "7485"]),
        ("HC-SR04 Ultrasonic Sensor", ["HC-SR04"]),
        ("2N2222 / similar transistor", ["2N2222"]),
        ("7447 / 7448 7-Segment Driver", ["7447", "7448"]),
        ("CD4511 BCD-to-7-Segment Decoder", ["CD4511"]),
        ("Breadboard 5V power module", []),
        ("400-point breadboard", []),
        ("4-digit 7-segment display", []),
        ("What is the price of an ESP32?", ["ESP32"]),
        ("is HC SR04 cheaper", ["HC-SR04"]),
    ],
)
def test_extract_identifiers(text, expected):
    assert extract_identifiers(text) == expected


def test_technical_tokens_are_preserved():
    # Normalization must not damage identifiers or values.
    for token in ("74LS85", "HC-SR04", "5V", "4–10 EGP"):
        assert token.lower().replace("–", "-") in normalize_name(f"x {token} y").replace("–", "-")


def test_identifier_variants():
    assert identifier_variants("74LS85") == {"74ls85", "7485"}
    assert "7404" in identifier_variants("74HC04")
    assert "555" in identifier_variants("NE555")
    assert "4017" in identifier_variants("CD4017")
    assert identifier_variants("HC-SR04") == {"hcsr04"}
    assert {"7805", "l7805", "lm7805"} <= identifier_variants("7805")
    assert "pn2222" in identifier_variants("2N2222")
    # Manufacturer prefixes that are part of the part number are kept.
    assert identifier_variants("DS1307") == {"ds1307"}


def test_compact_and_phrase_keys():
    assert compact_key("HC-SR04") == compact_key("HC SR04") == compact_key("hcsr04")
    assert phrase_key("HC-SR04") == phrase_key("HC SR04") == "hc sr04"
    assert phrase_key("7-segment display") == "7 segment display"


def test_split_name():
    parsed = split_name("74LS85 / 7485 Comparator")
    assert parsed.identifiers == ["74LS85", "7485"]
    assert parsed.descriptor == "Comparator"
    assert split_name("74LS08 / 7408 AND Gate").descriptor == "AND Gate"


def test_generate_aliases_ne555():
    aliases = [a.lower() for a in generate_aliases("NE555 Timer IC", curated={})]
    assert "ne555" in aliases
    assert "555 timer" in aliases
    assert "555" in aliases


def test_generate_aliases_logic_gate_with_curated():
    aliases = [a.lower() for a in generate_aliases("74LS08 / 7408 AND Gate")]
    for expected in ("74ls08", "7408", "and gate", "quad and gate"):
        assert expected in aliases


def test_module_aliases_do_not_steal_ic_alias():
    module_aliases = [a.lower() for a in generate_aliases("NE555 adjustable timer module", curated={})]
    assert "555 timer" not in module_aliases


def test_build_search_keys():
    keys = build_search_keys("74LS85 / 7485 Comparator", ["74LS85"]).split()
    assert set(keys) == {"74ls85", "7485"}
