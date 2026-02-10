"""
ED Normalisation (Logic-First)

Implements the 4-case protocol:

CASE 1: Exact taxonomy match  -> MAPPED_EXACT
CASE 2: Alias/phrase variant  -> MAPPED_ALIAS (Category-scoped, Theme-filtered)
CASE 3: Legit new ED          -> CANDIDATE (staged, not canonized)
CASE 4: Invalid/corrupted     -> REJECTED_RECORD

Core invariants
---------------
1) Theme is treated as immutable placement / firewall, not identity.
2) Similarity search happens only within the Theme + Category candidate set.
3) Canonical registry entries are the source of truth (stable IDs + stable labels).
4) Candidate EDs are staged for review; this normaliser never mutates the registry.

MANDATORY REPO BEHAVIOR (from stress-test learnings)
---------------------------------------------------
Your registry repo MUST perform lookups case-insensitively for:
- theme
- category
- label (Category → Subcategory)

i.e., the following must resolve to the same canonical ED:
- "Billing Experience → Bill Clarity"
- "BILLING EXPERIENCE → BILL CLARITY"
- " billing   experience  ->  bill clarity "

The normaliser keeps parsing/normalization deterministic, but relies on the repo to
treat keys case-insensitively while returning the canonical-cased CanonicalED.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple
import re


# -----------------------------
# Types & contracts
# -----------------------------

class EDStatus(str, Enum):
    MAPPED_EXACT = "MAPPED_EXACT"          # Case 1
    MAPPED_ALIAS = "MAPPED_ALIAS"          # Case 2
    CANDIDATE = "CANDIDATE"                # Case 3
    REJECTED_RECORD = "REJECTED_RECORD"    # Case 4


@dataclass(frozen=True)
class CanonicalED:
    """
    Canonical Experience Driver entity from the registry.

    label MUST be in the canonical form:
        "Category → Subcategory"
    """
    ed_id: str
    theme: str
    category: str
    subcategory: str
    label: str  # "Category → Subcategory"


@dataclass(frozen=True)
class EDParse:
    """Parsed ED from model output after structural normalization."""
    category: str
    subcategory: str
    label: str  # normalized "Category → Subcategory"


@dataclass
class EDResolution:
    """Resolution payload for one (theme, ed_label_raw)."""
    status: EDStatus
    input_label_raw: str
    theme: str

    # If mapped:
    canonical_ed: Optional[CanonicalED] = None
    alias_label: Optional[str] = None

    # If candidate:
    proposed: Optional[EDParse] = None
    nearest_neighbors: Optional[List[Dict[str, Any]]] = None  # [{"ed_id":..., "label":..., "score":...}, ...]

    # If rejected:
    reject_reason: Optional[str] = None


class EDRegistryRepo(Protocol):
    """
    External registry repository (DB/service/etc).

    IMPORTANT:
    - Implementations MUST treat lookups case-insensitively for theme, category, and label.
    - Implementations MUST return CanonicalED with canonical casing and stable IDs.
    """

    def get_exact(self, theme: str, label: str) -> Optional[CanonicalED]:
        """Return canonical ED if label is an exact match in the registry (theme-scoped; case-insensitive)."""
        ...

    def list_by_theme_category(self, theme: str, category: str) -> List[CanonicalED]:
        """Return canonical EDs under (theme, category) (case-insensitively)."""
        ...


# -----------------------------
# Normalisation helpers
# -----------------------------

# supports "→" or "->" or ">"
_ARROW_PATTERN = re.compile(r"\s*(.+?)\s*[→>-]+\s*(.+?)\s*$")


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _canonical_arrow(label: str) -> str:
    """Normalize various arrow forms to 'Category → Subcategory'."""
    m = _ARROW_PATTERN.match(label or "")
    if not m:
        raise ValueError("ED label missing/invalid arrow separator (expected 'Category → Subcategory').")
    cat = _norm_space(m.group(1))
    sub = _norm_space(m.group(2))
    if not cat or not sub:
        raise ValueError("ED label has empty Category or Subcategory.")
    return f"{cat} → {sub}"


def parse_ed_label(ed_label_raw: str) -> EDParse:
    """Parse and normalize ED label. Raises ValueError if invalid (Case 4 trigger)."""
    canon = _canonical_arrow(ed_label_raw)
    cat, sub = [x.strip() for x in canon.split("→", 1)]
    cat = _norm_space(cat)
    sub = _norm_space(sub)
    return EDParse(category=cat, subcategory=sub, label=f"{cat} → {sub}")


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization for similarity. Replace later with:
    - alias dictionary normaliser (deterministic), or
    - character n-gram similarity (deterministic), or
    - embeddings (only within theme+category search space).
    """
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    parts = re.split(r"[\s\-]+", text)
    return [p for p in parts if p]


def jaccard(a: str, b: str) -> float:
    """
    Deterministic similarity measure (baseline).
    Note: this can miss synonym variants like "Upfront" vs "Transparent".
    """
    A = set(tokenize(a))
    B = set(tokenize(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# -----------------------------
# Core ED normaliser logic
# -----------------------------

@dataclass(frozen=True)
class EDNormaliserConfig:
    alias_threshold: float = 0.82  # tune per vertical
    neighbors_k: int = 5


class EDNormaliser:
    """
    Logic-first Experience Driver normaliser.

    - Deterministic parsing + candidate restriction.
    - Registry repo is responsible for case-insensitive key matching.
    """

    def __init__(self, repo: EDRegistryRepo, config: Optional[EDNormaliserConfig] = None):
        self.repo = repo
        self.config = config or EDNormaliserConfig()

    def resolve(self, *, theme: str, ed_label_raw: str) -> EDResolution:
        """
        Resolve one extracted ED label under a known Theme.
        Theme is a firewall/placement input, not something we compute.
        """
        theme_n = _norm_space(theme)

        # Case 4: invalid/corrupt ED label
        try:
            parsed = parse_ed_label(ed_label_raw)
        except ValueError as e:
            return EDResolution(
                status=EDStatus.REJECTED_RECORD,
                input_label_raw=ed_label_raw,
                theme=theme_n,
                reject_reason=str(e),
            )

        ed_label_norm = parsed.label

        # Case 1: exact match (repo must be case-insensitive)
        exact = self.repo.get_exact(theme=theme_n, label=ed_label_norm)
        if exact is not None:
            return EDResolution(
                status=EDStatus.MAPPED_EXACT,
                input_label_raw=ed_label_raw,
                theme=theme_n,
                canonical_ed=exact,
            )

        # Candidate restriction: Theme fixed + Category scoped (repo must be case-insensitive)
        candidates = self.repo.list_by_theme_category(theme=theme_n, category=parsed.category)

        # Case 2: alias match within same category
        best: Optional[Tuple[CanonicalED, float]] = None
        scored: List[Tuple[CanonicalED, float]] = []

        for c in candidates:
            score = jaccard(parsed.subcategory, c.subcategory)
            scored.append((c, score))
            if best is None or score > best[1]:
                best = (c, score)

        scored.sort(key=lambda t: t[1], reverse=True)
        neighbors = [
            {"ed_id": c.ed_id, "label": c.label, "score": round(s, 4)}
            for c, s in scored[: self.config.neighbors_k]
        ]

        if best is not None and best[1] >= self.config.alias_threshold:
            canonical, _ = best
            return EDResolution(
                status=EDStatus.MAPPED_ALIAS,
                input_label_raw=ed_label_raw,
                theme=theme_n,
                canonical_ed=canonical,
                alias_label=ed_label_norm,
                nearest_neighbors=neighbors,
            )

        # Case 3: legit new ED (candidate lane)
        return EDResolution(
            status=EDStatus.CANDIDATE,
            input_label_raw=ed_label_raw,
            theme=theme_n,
            proposed=parsed,
            nearest_neighbors=neighbors,
        )
