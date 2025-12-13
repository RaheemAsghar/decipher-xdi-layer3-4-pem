from __future__ import annotations
import pandas as pd

# We keep core computation logic sourced from the authoritative implementation.
from Layer_2_V8 import Layer2Computer as _MonolithLayer2Computer

class Layer2CoreMixin:
    """Core ERI/EVI/RF computations + classification logic.

    Import this mixin if you want to reuse the core math independently.
    """

    _compute_eri_from_counts = _MonolithLayer2Computer._compute_eri_from_counts
    _compute_evi = _MonolithLayer2Computer._compute_evi
    _compute_evi_from_counts = _MonolithLayer2Computer._compute_evi_from_counts

    _map_loyalty_tier = _MonolithLayer2Computer._map_loyalty_tier
    _map_rf_tier = _MonolithLayer2Computer._map_rf_tier
    _normalize_rf_tier = _MonolithLayer2Computer._normalize_rf_tier
    _map_evi_tier = _MonolithLayer2Computer._map_evi_tier

    _classify_emotional_state_band = _MonolithLayer2Computer._classify_emotional_state_band
    _classify_priority = _MonolithLayer2Computer._classify_priority
    _classify_purpose = _MonolithLayer2Computer._classify_purpose

    _safe_qcut = _MonolithLayer2Computer._safe_qcut
    _compute_dynamic_component_tiers = _MonolithLayer2Computer._compute_dynamic_component_tiers
