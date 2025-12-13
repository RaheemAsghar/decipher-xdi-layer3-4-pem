from __future__ import annotations
import pandas as pd

from .layer2_config import Layer2Config, DEFAULT_CONFIG
from .layer2_core import Layer2CoreMixin
from .layer2_recency_profiling import Layer2RecencyProfilingMixin
from .layer2_temporal_intelligence import Layer2TemporalIntelligenceMixin

# Authoritative implementation (compute() orchestration)
from Layer_2_V8 import Layer2Computer as _MonolithLayer2Computer

class Layer2Computer(_MonolithLayer2Computer, Layer2CoreMixin, Layer2RecencyProfilingMixin, Layer2TemporalIntelligenceMixin):
    """Master orchestrator (thin wrapper).

    This class preserves the original Layer_2_V8.Layer2Computer behavior while
    providing modular import surfaces via the mixins + Layer2Config.
    """
    def __init__(self, df: pd.DataFrame, config: Layer2Config | None = None, **kwargs):
        cfg = config or DEFAULT_CONFIG

        # Allow explicit overrides via kwargs, but default to config
        window_days = kwargs.pop("window_days", cfg.window_days)
        verbose = kwargs.pop("verbose", cfg.verbose)
        tau_days = kwargs.pop("tau_days", cfg.tau_days)
        rf_weight_r = kwargs.pop("rf_weight_r", cfg.rf_weight_r)
        rf_weight_f = kwargs.pop("rf_weight_f", cfg.rf_weight_f)

        super().__init__(
            df=df,
            window_days=window_days,
            verbose=verbose,
            tau_days=tau_days,
            rf_weight_r=rf_weight_r,
            rf_weight_f=rf_weight_f,
            **kwargs
        )
