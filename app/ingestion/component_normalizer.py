"""Component name normalization, part-number extraction and alias generation.

Technical identifiers (74LS85, HC-SR04, 2N2222, DS3231…) are preserved as-is in
stored text. For matching we derive *compact keys* (lowercase, alphanumerics
only) plus equivalence variants:

    HC-SR04  -> hcsr04
    74LS85   -> 74ls85, 7485            (logic-family stripped)
    NE555    -> ne555, 555              (manufacturer prefix stripped)
    7805     -> 7805, l7805, lm7805
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

# Tokens that look like part numbers but are actually units / measurements.
_MEASUREMENT_RE = re.compile(
    r"^\d+(?:\.\d+)?(?:v|mv|kv|a|ma|ua|w|mw|kw|ohm|ohms|r|k|m|uf|nf|pf|mf|mhz|khz|hz|ghz|mm|cm|m|"
    r"pcs|pc|pin|pins|bit|bits|x|egp|le|usd|v\d*|mah|ah|db|c|f|s|ms|us|ns|h|min|sec|%)$",
    re.IGNORECASE,
)
_MEASUREMENT_SUFFIX_RE = re.compile(
    r"-(?:point|points|segment|segments|digit|digits|pin|pins|bit|bits|way|channel|channels|input|inputs)$",
    re.IGNORECASE,
)
_TOKEN_SPLIT_RE = re.compile(r"[\s/,;:()\[\]{}|+]+")
_TOKEN_SHAPE_RE = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")

_LOGIC_FAMILY_RE = re.compile(
    r"^(?:sn)?(74|54)(ls|hc|hct|als|as|f|s|l|h|ac|act|lv|lvc|ahc|abt|bct|fct)?(\d{2,4})(?:n|p|d|dr)?$"
)
_STRIPPABLE_PREFIXES = ("ne", "sa", "lm", "sn", "cd", "hef", "mc", "ua")
_REGULATOR_RE = re.compile(r"^(?:lm|l|ua|mc)?(78|79)(\d{2})(?:ct|cv)?$")
_TRANSISTOR_RE = re.compile(r"^(?:pn|2n)(\d{4})a?$")


def normalize_text(text: str) -> str:
    """Unicode-normalize, lowercase and collapse whitespace. Keeps -/./+ characters."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_name(name: str) -> str:
    """Canonical key for a component name, e.g. 'NE555 Timer IC' -> 'ne555 timer ic'."""
    text = normalize_text(name)
    text = re.sub(r"[^\w\s\-/.+]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_key(text: str) -> str:
    """Normalization used for alias phrase matching: hyphens/slashes become spaces.

    'HC-SR04' and 'HC SR04' -> 'hc sr04'; '7-segment' -> '7 segment'.
    """
    text = normalize_text(text)
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"[^\w\s.]", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(text))


def is_identifier_token(token: str) -> bool:
    if len(token) < 3 or not _TOKEN_SHAPE_RE.match(token):
        return False
    if not any(ch.isdigit() for ch in token):
        return False
    if _MEASUREMENT_RE.match(token) or _MEASUREMENT_SUFFIX_RE.search(token):
        return False
    has_alpha = any(ch.isalpha() for ch in token)
    if not has_alpha:
        # Pure numbers are part numbers only when >= 3 digits (555, 7485, 4017).
        return len(token) >= 3
    # Alphanumerics must contain at least 2 digits (DS3231, NE555, 2N2222, HC-SR04).
    return sum(ch.isdigit() for ch in token) >= 2


def extract_identifiers(text: str) -> list[str]:
    """Extract part-number-like identifiers, preserving their original spelling."""
    found: list[str] = []
    seen: set[str] = set()
    tokens = [t.strip(".?!'\"") for t in _TOKEN_SPLIT_RE.split(unicodedata.normalize("NFKC", text))]
    for i, token in enumerate(tokens):
        candidates = [token]
        # "HC SR04" -> also consider "HC-SR04" (letters followed by a part number).
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if (
            re.fullmatch(r"[A-Za-z]{2,4}", token)
            and nxt[:1].isalpha()
            and is_identifier_token(nxt)
            and (token.isupper() or token.lower() in {"hc", "ds", "cd", "ne", "lm", "sn"})
        ):
            candidates = [f"{token}-{nxt}"]
        for cand in candidates:
            if is_identifier_token(cand):
                key = compact_key(cand)
                if key not in seen:
                    seen.add(key)
                    found.append(cand)
    # Drop a bare suffix captured both standalone and as part of a joined token.
    joined = {compact_key(f) for f in found if "-" in f}
    return [f for f in found if not any(compact_key(f) != j and j.endswith(compact_key(f)) for j in joined)]


def identifier_variants(identifier: str) -> set[str]:
    """Compact equivalence keys for an identifier (used for exact matching)."""
    key = compact_key(identifier)
    variants = {key}
    family = _LOGIC_FAMILY_RE.match(key)
    if family:
        series, _, number = family.groups()
        variants.add(f"{series}{number}")
    for prefix in _STRIPPABLE_PREFIXES:
        if key.startswith(prefix) and key[len(prefix) :].isdigit() and len(key) - len(prefix) >= 3:
            variants.add(key[len(prefix) :])
    reg = _REGULATOR_RE.match(key)
    if reg:
        base = f"{reg.group(1)}{reg.group(2)}"
        variants |= {base, f"l{base}", f"lm{base}"}
    tr = _TRANSISTOR_RE.match(key)
    if tr:
        variants |= {f"2n{tr.group(1)}", f"pn{tr.group(1)}"}
    return variants


@dataclass
class ParsedName:
    identifiers: list[str]
    descriptor: str  # name without the identifier part, e.g. "Comparator"


def split_name(name: str) -> ParsedName:
    """'74LS85 / 7485 Comparator' -> identifiers [74LS85, 7485], descriptor 'Comparator'."""
    identifiers = extract_identifiers(name)
    descriptor = name
    for ident in identifiers:
        descriptor = re.sub(re.escape(ident), " ", descriptor, flags=re.IGNORECASE)
    descriptor = re.sub(r"\s*/\s*", " ", descriptor)
    descriptor = re.sub(r"\bsimilar\b", " ", descriptor, flags=re.IGNORECASE)
    descriptor = re.sub(r"\s+", " ", descriptor).strip(" -/+,")
    return ParsedName(identifiers=identifiers, descriptor=descriptor)


@lru_cache
def load_curated_aliases(path: str | None = None) -> dict[str, list[str]]:
    """Curated aliases keyed by normalized component name."""
    from app.core.config import get_settings

    file = Path(path) if path else get_settings().data_dir / "component_aliases.yaml"
    if not file.exists():
        return {}
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return {normalize_name(k): [str(a) for a in (v or [])] for k, v in data.items()}


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = re.sub(r"\s+", " ", item).strip()
        key = phrase_key(item)
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def generate_aliases(name: str, extra: Iterable[str] = (), curated: dict[str, list[str]] | None = None) -> list[str]:
    """Generate searchable aliases for a component name.

    NE555 Timer IC -> [NE555, NE555 timer, 555, 555 timer, NE555 Timer IC ...]
    """
    parsed = split_name(name)
    descriptor = parsed.descriptor
    # "Timer IC" -> "timer"; "BCD-to-7-Segment Decoder" stays.
    short_desc = re.sub(r"\b(ic|chip|module|board)\b", " ", descriptor, flags=re.IGNORECASE)
    short_desc = re.sub(r"\s+", " ", short_desc).strip()
    is_module = bool(re.search(r"\bmodule\b", name, re.IGNORECASE))

    aliases: list[str] = []
    for ident in parsed.identifiers:
        aliases.append(ident)
        bare_variants = sorted(v for v in identifier_variants(ident) if v != compact_key(ident) and v.isdigit())
        forms = [ident, *bare_variants]
        for form in forms:
            if descriptor:
                aliases.append(f"{form} {descriptor}")
            if short_desc and short_desc.lower() != descriptor.lower():
                aliases.append(f"{form} {short_desc}" + (" module" if is_module else ""))
        aliases.extend(bare_variants)

    if len(parsed.identifiers) > 1 and descriptor:
        aliases.append(f"{parsed.identifiers[0]} {descriptor}")

    curated = load_curated_aliases() if curated is None else curated
    aliases.extend(curated.get(normalize_name(name), []))
    aliases.extend(extra)
    # Never alias a component to its own full name (that's already searchable).
    return [a for a in _dedupe(aliases) if phrase_key(a) != phrase_key(name)]


def build_search_keys(name: str, aliases: Iterable[str]) -> str:
    """Space-separated compact keys indexed with weight 'A' in full-text search."""
    keys: set[str] = set()
    for text in [name, *aliases]:
        for ident in extract_identifiers(text):
            keys |= identifier_variants(ident)
    return " ".join(sorted(keys))


@dataclass
class NormalizedComponentName:
    name: str
    normalized_name: str
    identifiers: list[str]
    aliases: list[str] = field(default_factory=list)
    search_keys: str = ""


def normalize_component(name: str, extra_aliases: Iterable[str] = ()) -> NormalizedComponentName:
    clean = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", name)).strip()
    aliases = generate_aliases(clean, extra_aliases)
    return NormalizedComponentName(
        name=clean,
        normalized_name=normalize_name(clean),
        identifiers=split_name(clean).identifiers,
        aliases=aliases,
        search_keys=build_search_keys(clean, aliases),
    )
