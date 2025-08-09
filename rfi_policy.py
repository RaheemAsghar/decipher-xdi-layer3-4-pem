# rfi_policy.py
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional, Dict

@dataclass
class RFIPolicyConfig:
    # master kill‑switches
    enable_rfi_tiebreak: bool = True          # only used when RF tiers tie
    enable_rf_blend: bool = False              # OFF by default (keeps today's behavior)
    # blend + gating
    alpha: float = 0.80                        # RF′ = α·RF + (1−α)·RFI
    tau: float = 0.15                          # min absolute gap to consider RFI material (|RFI−RF|)
    gate_priorities: tuple = ("P0", "P1", "P2")# only allow nudges within these priority bands
    # sanity/guardrails
    never_downrank_for: tuple = ("P0", "P1")   # P0/P1 can't be pushed down by RFI influence
    require_corroboration: bool = True         # when nudging up, require PEM/Trigger corroboration if available

class RFIPolicy:
    """
    Non‑intrusive RFI hooks:
      1) optional RF′ blend behind a flag
      2) deterministic tie‑break using RFI when RF ties
      3) scoreboard utilities to compare RF vs RFI vs RF′ after outcomes land
    Expected input df columns (from your Layer 2):
      ['experience_driver','RF','Priority_Status','RFI']
    Optional columns for corroboration gate (Layer 3+):
      ['pem_block.trajectory_forecast.pem_trajectory', 'signal_strength_block.qssi_summary.qssi_tier']
    """

    def __init__(self, cfg: Optional[RFIPolicyConfig] = None):
        self.cfg = cfg or RFIPolicyConfig()

    # ---------- public API ----------
    def enrich_with_rfp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds RF′ (blended) and an OrderHint (tiebreaker) without changing existing RF/Priority.
        Safe to call and ignore.
        """
        out = df.copy()

        # RF′ (blended) – stays inert unless enable_rf_blend=True and gating conditions met
        out["RF_prime"] = out["RF"]
        if self.cfg.enable_rf_blend:
            mask_gated = out["Priority_Status"].isin(self.cfg.gate_priorities)
            gap = (out["RFI"] - out["RF"]).abs()
            mask_gap = gap >= self.cfg.tau
            candidate = mask_gated & mask_gap

            rf_prime = self.cfg.alpha * out["RF"] + (1 - self.cfg.alpha) * out["RFI"]

            # never downrank P0/P1
            safe = candidate & ~out["Priority_Status"].isin(self.cfg.never_downrank_for)
            out.loc[safe, "RF_prime"] = rf_prime[safe]

            # optional corroboration (nudge only when we have strong signal)
            if self.cfg.require_corroboration:
                has_qssi = out.get("signal_strength_block.qssi_summary.qssi_tier")
                has_pem = out.get("pem_block.trajectory_forecast.pem_trajectory")
                if has_qssi is not None and has_pem is not None:
                    strong_qssi = out["signal_strength_block.qssi_summary.qssi_tier"].isin(["💥 Critical Signal","🔥 Strong Signal"])
                    pem_ok = out["pem_block.trajectory_forecast.pem_trajectory"].isin(["Likely Escalation","At Risk of Decay"])
                    corroborated = strong_qssi & pem_ok
                    out.loc[~corroborated, "RF_prime"] = out["RF"]  # revert if not corroborated

        # deterministic tie‑break hint using RFI (ascending sort key; lower = earlier)
        out["OrderHint"] = 0
        if self.cfg.enable_rfi_tiebreak:
            # higher RFI should come earlier when RF ties
            # we encode as a negative hint so default ascending sorts work: (-RFI)
            out["OrderHint"] = -out["RFI"].fillna(0)

        return out

    def rank_for_execution(self, df: pd.DataFrame, use_blend: Optional[bool] = None) -> pd.DataFrame:
        """
        Returns a ranked view WITHOUT mutating original priority labels.
        Current behavior preserved unless you pass use_blend=True or config enables it.
        """
        use_blend = self.cfg.enable_rf_blend if use_blend is None else use_blend
        scored = self.enrich_with_rfp(df)

        # numeric priority for sorting (P0 highest)
        priomap = {"P0":0,"P1":1,"P2":2,"P3":3,"P4":4,"P5":5}
        scored["PriorityNum"] = scored["Priority_Status"].map(priomap).fillna(99)

        key_rf = "RF_prime" if use_blend else "RF"

        return (
            scored
            .sort_values(["PriorityNum", key_rf, "OrderHint"], ascending=[True, False, True])
            .reset_index(drop=True)
        )

    # ---------- scoreboard (offline evaluation, later) ----------
    def scoreboard(self, df_outcomes: pd.DataFrame) -> Dict[str, float]:
        """
        Compare RF vs RFI vs RF′ against realized outcomes.
        Expects df_outcomes with columns:
          ['RF','RFI','RF_prime', 'Priority_Status', 
           'post_eri_shift',           # numeric (post‑window ERI − pre‑ERI)
           'recurrence_delta',         # negative = good (less recurrence)
           'time_to_intervention_days' # lower = better
          ]
        Returns simple Kendall tau correlations as a quick potency proxy.
        """
        from scipy.stats import kendalltau

        metrics = {
            "RF~ERI_shift": np.nan,
            "RFI~ERI_shift": np.nan,
            "RFp~ERI_shift": np.nan,
            "RF~recurrence": np.nan,
            "RFI~recurrence": np.nan,
            "RFp~recurrence": np.nan,
            "RF~speed": np.nan,
            "RFI~speed": np.nan,
            "RFp~speed": np.nan
        }

        def tau(x, y):
            try:
                t, _ = kendalltau(x, y)
                return float(t)
            except Exception:
                return np.nan

        if "RF_prime" not in df_outcomes.columns:
            tmp = self.enrich_with_rfp(df_outcomes)
        else:
            tmp = df_outcomes.copy()

        # higher score should predict better ERI_shift (positive), more recurrence drop (negative), and faster speed (negative)
        metrics["RF~ERI_shift"]   = tau(tmp["RF"],        tmp["post_eri_shift"])
        metrics["RFI~ERI_shift"]  = tau(tmp["RFI"],       tmp["post_eri_shift"])
        metrics["RFp~ERI_shift"]  = tau(tmp["RF_prime"],  tmp["post_eri_shift"])

        metrics["RF~recurrence"]  = tau(tmp["RF"],        -tmp["recurrence_delta"])
        metrics["RFI~recurrence"] = tau(tmp["RFI"],       -tmp["recurrence_delta"])
        metrics["RFp~recurrence"] = tau(tmp["RF_prime"],  -tmp["recurrence_delta"])

        metrics["RF~speed"]       = tau(tmp["RF"],        -tmp["time_to_intervention_days"])
        metrics["RFI~speed"]      = tau(tmp["RFI"],       -tmp["time_to_intervention_days"])
        metrics["RFp~speed"]      = tau(tmp["RF_prime"],  -tmp["time_to_intervention_days"])

        return metrics