from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Layer2Config:
    """Central config for Layer 2.

    NOTE: This is a thin config surface so consumers can tweak knobs without
    touching the full engine. The underlying engine still validates/normalizes
    weights internally.
    """
    window_days: int | None = None
    tau_days: float = 30.0
    rf_weight_r: float = 0.6
    rf_weight_f: float = 0.4
    verbose: bool = False

DEFAULT_CONFIG = Layer2Config()
