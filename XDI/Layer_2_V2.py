# 1. Compute ERI (from emotion counts)
# 2. Compute EVI (from emotion counts)  
# 3. Apply ERI × EVI Matrix → Get Emotional State (ES1-ES5)
# 4. Compute RF (from recency + frequency)
# 5. Apply ES × RF Grid → Get Final Priority (P0-P5)

import os, numpy as np, pandas as pd

# === Layer 2 Logic ===
class Layer2Computer:
    def __init__(
        self,
        df,
        window_days=None,
        today_anchor=None,
        verbose=False,
        tau_days: float = 30.0,
        rf_weight_r: float = 0.6,
        rf_weight_f: float = 0.4,
    ):
        self.df = df.copy()

        # ✅ Require and use only the plain 'date' column (YYYY-MM-DD)
        if "date" not in self.df.columns:
            raise ValueError("Layer2 requires a 'date' column in YYYY-MM-DD format.")
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce").dt.normalize()
        if self.df["date"].isna().all():
            raise ValueError("All values in 'date' are invalid/NaT.")

        # ✅ Store 'today' as a normalized pandas Timestamp (no tz)
        self.today = (
            pd.to_datetime(today_anchor).normalize()
            if today_anchor is not None
            else pd.Timestamp.today().normalize()
        )

        self.layer2_df = None
        self.verbose = verbose
        self.window_days = window_days
        self.timeframe_days = window_days

        self.emotion_scores = {
            "Adoration": 3,
            "Appreciation": 1,
            "Ambivalence": 0,
            "Agitation": -1,
            "Anger": -3,
        }

        # Recency decay constant (τ)
        self.tau_days = float(tau_days)
        assert self.tau_days > 0

        # RF weights (normalize to sum = 1)
        total = float(rf_weight_r) + float(rf_weight_f)
        if total <= 0:
            self.rf_weight_r, self.rf_weight_f = 0.5, 0.5
        else:
            self.rf_weight_r = float(rf_weight_r) / total
            self.rf_weight_f = float(rf_weight_f) / total

        
    # ---------------------------------------------------
    # NEW LAYER 2 DECISION LOGIC (ERI × EVI × RF)
    # ---------------------------------------------------

    def _classify_emotional_state_band(self, eri_tier, evi_tier):
        """
        Returns:
            ES1_Secure_Loyalty
            ES2_Growth_Opportunity
            ES3_At_Risk
            ES4_Active_Crisis
            ES5_Critical_Failure
        """

        et = eri_tier
        vt = evi_tier

        # ES1 – Secure Loyalty
        if et in {"Very Positive", "Positive"} and vt in {"Stable Emotion", "Low Volatility"}:
            return "ES1_Secure_Loyalty"

        # ES2 – Growth Opportunity
        if (et in {"Very Positive", "Positive"} and vt == "Moderate Volatility") or \
        (et == "Neutral" and vt == "Stable Emotion"):
            return "ES2_Growth_Opportunity"

        # ES5 – Critical Failure
        if et == "Very Negative":
            return "ES5_Critical_Failure"
        if et == "Negative" and vt == "Extreme Volatility":
            return "ES5_Critical_Failure"

        # ES4 – Active Crisis
        if (et == "Neutral" and vt in {"High Volatility", "Extreme Volatility"}) or \
        (et == "Negative" and vt in {"Moderate Volatility", "High Volatility"}):
            return "ES4_Active_Crisis"

        # ES3 – At Risk
        if (et in {"Very Positive", "Positive"} and vt in {"High Volatility", "Extreme Volatility"}) or \
        (et == "Neutral" and vt in {"Low Volatility", "Moderate Volatility"}) or \
        (et == "Negative" and vt in {"Stable Emotion", "Low Volatility"}):
            return "ES3_At_Risk"

        return "ES3_At_Risk"


    PRIORITY_GRID = {
        "ES1_Secure_Loyalty": {
            "Dormant":  "P5",
            "Low":      "P4",
            "Moderate": "P3",
            "High":     "P2",
            "Very High":"P2",
        },
        "ES2_Growth_Opportunity": {
            "Dormant":  "P5",
            "Low":      "P4",
            "Moderate": "P3",
            "High":     "P3",
            "Very High":"P2",
        },
        "ES3_At_Risk": {
            "Dormant":  "P4",
            "Low":      "P3",
            "Moderate": "P2",
            "High":     "P1",
            "Very High":"P0",
        },
        "ES4_Active_Crisis": {
            "Dormant":  "P3",
            "Low":      "P2",
            "Moderate": "P1",
            "High":     "P1",
            "Very High":"P0",
        },
        "ES5_Critical_Failure": {
            "Dormant":  "P2",
            "Low":      "P2",
            "Moderate": "P1",
            "High":     "P0",
            "Very High":"P0",
        },
    }

    def _classify_priority(self, emotional_state_band, rf_tier_short):
        return self.PRIORITY_GRID.get(emotional_state_band, {}).get(rf_tier_short, "P3")

    PURPOSE_MATRIX = {
    ("ES1_Secure_Loyalty", "P5"): "Monitor Stable Advocacy",
    ("ES1_Secure_Loyalty", "P4"): "Maintain Positive Momentum",
    ("ES1_Secure_Loyalty", "P3"): "Nurture Loyalty",
    ("ES1_Secure_Loyalty", "P2"): "Reinforce High-Value Advocates",

    ("ES2_Growth_Opportunity", "P5"): "Background Growth Watch",
    ("ES2_Growth_Opportunity", "P4"): "Develop Emerging Loyalty",
    ("ES2_Growth_Opportunity", "P3"): "Strengthen Growth Drivers",
    ("ES2_Growth_Opportunity", "P2"): "Activate Growth Levers",

    ("ES3_At_Risk", "P4"): "Monitor Weak Signals",
    ("ES3_At_Risk", "P3"): "Investigate Weak Risks",
    ("ES3_At_Risk", "P2"): "Address Early Risks",
    ("ES3_At_Risk", "P1"): "Contain At-Risk Emotion",
    ("ES3_At_Risk", "P0"): "Rapid Recovery Action",

    ("ES4_Active_Crisis", "P3"): "Early Crisis Watch",
    ("ES4_Active_Crisis", "P2"): "Crisis Containment",
    ("ES4_Active_Crisis", "P1"): "Active Crisis Intervention",
    ("ES4_Active_Crisis", "P0"): "Emergency Crisis Response",

    ("ES5_Critical_Failure", "P2"): "Structural Failure Monitoring",
    ("ES5_Critical_Failure", "P1"): "Contain Structural Breakdown",
    ("ES5_Critical_Failure", "P0"): "Critical Failure Response",
    }

    def _classify_purpose(self, emotional_state_band, priority_tier):
        return self.PURPOSE_MATRIX.get((emotional_state_band, priority_tier), "General Signal Monitoring")


    def compute(self):
        """
        Layer 2 (ERI + EVI + RF):
        - Compute ERI (unchanged)
        - Compute EVI (new volatility metric)
        - Map ERI → ERI_Tier
        - Map EVI → EVI_Tier
        - ERI_Tier x EVI_Tier → Emotional_State_Band (ES1..ES5)
        - Compute RF (unchanged)
        - RF → RF_Urgency_Category (long) + RF_Tier_Short
        - Emotional_State_Band x RF_Tier_Short → Priority_Tier (P0..P5)
        - Emotional_State_Band x Priority_Tier → Priority_Purpose
        """
        

        # Use ONLY the 'date' column (YYYY-MM-DD) — normalized
        ts_col = "date"
        self.df[ts_col] = pd.to_datetime(self.df[ts_col], errors="coerce").dt.normalize()

        # 'today' is already normalized in __init__; keep as-is
        today_ts = self.today

        grouped = self.df.groupby("experience_driver", dropna=False)
        total_rows = len(self.df)
        entity_counts = self.df["experience_driver"].value_counts()
        max_mentions = max(1, entity_counts.max())

        # Priority score map remains the same (P0..P5 -> numeric score)
        p_score = self._priority_score_map()

        results = []

        for entity, group in grouped:
            entry = {"experience_driver": entity}

            # --- counts & dates ---------------------------------------------------
            total_mentions = int(len(group))
            times = group[ts_col]
            first_seen = times.min() if total_mentions else None
            most_recent = times.max() if total_mentions else None

            # Age in days from MOST RECENT date
            if most_recent is None or pd.isna(most_recent):
                age_days = None
            else:
                age_days = int(max(0, (today_ts - most_recent).days))

            # --- ERI (unchanged) --------------------------------------------------
            # ERI = weighted mean of emotional polarity, linearly normalized to a standard comparative scale.
            emotion_counts = group["emotion_primary"].value_counts(dropna=False).to_dict()
            eri_raw = sum(
                float(self.emotion_scores.get(em, 0.0)) * int(ct)
                for em, ct in emotion_counts.items()
            )
            eri_mean = (eri_raw / total_mentions) if total_mentions else 0.0
            eri_norm = ((float(eri_mean) + 3.0) / 6.0) * 200.0 - 100.0
            eri_norm = float(np.clip(eri_norm, -100.0, 100.0))
            eri_tier = self._map_loyalty_tier(eri_norm)

            # --- EVI (NEW: Emotional Volatility Index) ----------------------------
            # Use the same emotion_counts and emotion_scores
            n_adore       = int(emotion_counts.get("Adoration", 0))
            n_appreciate  = int(emotion_counts.get("Appreciation", 0))
            n_ambivalent  = int(emotion_counts.get("Ambivalence", 0))
            n_agitate     = int(emotion_counts.get("Agitation", 0))
            n_anger       = int(emotion_counts.get("Anger", 0))

            evi_score = self._compute_evi(
                n_adore,
                n_appreciate,
                n_ambivalent,
                n_agitate,
                n_anger,
            )
            evi_score = float(np.clip(evi_score, 0.0, 100.0))
            evi_tier = self._map_evi_tier(evi_score)

            # --- Emotional State Band (ERI_Tier × EVI_Tier) ----------------------
            emotional_state_band = self._classify_emotional_state_band(eri_tier, evi_tier)

            # --- RF (Recency & Frequency) – unchanged -----------------------------
            R = 0.0 if age_days is None else 100.0 * np.exp(-float(age_days) / self.tau_days)
            F = min(
                100.0,
                (np.log1p(total_mentions) / np.log1p(max_mentions)) * 100.0,
            )

            rf_r_abs = self.rf_weight_r * float(R)
            rf_f_abs = self.rf_weight_f * float(F)
            RF = rf_r_abs + rf_f_abs
            rf_tier_long = self._map_rf_tier(RF)  # e.g. "High Activity"

            # % contribution (guard against divide-by-zero)
            if RF > 0:
                rf_r_pct = round(100.0 * rf_r_abs / RF, 1)
                rf_f_pct = round(100.0 - rf_r_pct, 1)
            else:
                rf_r_pct = 0.0
                rf_f_pct = 0.0

            # --- RF short tier (for decision grid) --------------------------------
            rf_tier_short = normalize_rf_tier(rf_tier_long)  # "High", "Moderate", etc.

            # --- Priority via Emotional_State × RF --------------------------------
            priority_tier = self._classify_priority(emotional_state_band, rf_tier_short)
            priority_purpose = self._classify_purpose(emotional_state_band, priority_tier)
            priority_score = p_score.get(priority_tier, 0)

            # --- Entities & coverage ---------------------------------------------
            associated_names = (
                sorted(group["entity_name"].dropna().unique().tolist())
                if "entity_name" in group.columns
                else []
            )
            mention_share_pct = round(
                (total_mentions / max(1, total_rows)) * 100.0,
                2,
            )

            # --- record -----------------------------------------------------------
            entry.update(
                {
                    "Today_Anchor": self.today.strftime("%Y-%m-%d"),
                    "Window_Days": int(self.window_days)
                    if self.window_days is not None
                    else None,

                    # ERI (unchanged)
                    "ERI": round(eri_norm, 2),
                    "ERI_Tier": eri_tier,
                    "Loyalty_State": eri_tier,

                    # EVI (new)
                    "EVI": round(evi_score, 2),
                    "EVI_Tier": evi_tier,
                    "Emotional_State_Band": emotional_state_band,

                    # Recency-Frequency (unchanged)
                    "Recency_Decay_Days": float(self.tau_days),
                    "R": round(float(R), 2),
                    "F": round(float(F), 2),
                    "RF_Weight_R": float(self.rf_weight_r),
                    "RF_Weight_F": float(self.rf_weight_f),
                    "RF": round(float(RF), 2),
                    "RF_R_Component": round(rf_r_abs, 2),
                    "RF_R_ContributionPct": rf_r_pct,
                    "RF_F_Component": round(rf_f_abs, 2),
                    "RF_F_ContributionPct": rf_f_pct,
                    "RF_Urgency_Category": rf_tier_long,
                    "RF_Tier_Short": rf_tier_short,

                    # New priority model
                    "Priority_Tier": priority_tier,          # P0..P5
                    "Priority_Score": priority_score,        # numeric via _priority_score_map()
                    "Priority_Purpose": priority_purpose,    # ES×P interpretation

                    # Backward compatible debug-style fields (optional)
                    "ERI_RF_Quadrant": f"{eri_tier} x {rf_tier_long}",
                    "Quadrant_Purpose": priority_purpose,
                    "Priority_Status": priority_tier,
                    "Quadrant_Key": f"{eri_tier}|{rf_tier_long}",

                    # Association / counts
                    "Associated_Entity_Names": associated_names,
                    "No_of_Mentions": total_mentions,
                    "Mention_Share_%": mention_share_pct,
                    "First_Seen_Date": first_seen.strftime("%Y-%m-%d")
                    if first_seen is not None and not pd.isna(first_seen)
                    else None,
                    "Most_Recent_Date": most_recent.strftime("%Y-%m-%d")
                    if most_recent is not None and not pd.isna(most_recent)
                    else None,
                    "Age_Days": age_days,

                    # Debug
                    "Debug_ERI_Raw": round(float(eri_mean), 4),
                    "Debug_Emotion_Counts": emotion_counts,
                }
            )

            results.append(entry)

        # Build DF once
        layer2_df = pd.DataFrame(results)

        # Assign dynamic tiers for R and F (5 bins each) and persist them
        layer2_df = self._compute_dynamic_component_tiers(layer2_df)

        # Save + bind
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/layer2_output_debug.csv"
        layer2_df.to_csv(output_path, index=False, encoding="utf-8")
        self.layer2_df = layer2_df
        print(f"✅ Layer 2 output saved to: {output_path}")

        return self.layer2_df

    def _compute_evi(self, n_adore, n_appreciate, n_ambivalent, n_agitate, n_anger):
        """
        Computes Emotional Volatility Index (EVI) based on emotion intensity distribution.
        Returns EVI in range [0, 100].
        """
        
        # Build list of values according to intensity scale
        values = (
            [3]  * int(n_adore) +
            [1]  * int(n_appreciate) +
            [0]  * int(n_ambivalent) +
            [-1] * int(n_agitate) +
            [-3] * int(n_anger)
        )

        if len(values) == 0:
            return 0.0  # default: no volatility

        arr = np.array(values, dtype=float)

        # population standard deviation
        sigma = arr.std(ddof=0)   # ddof=0 → population σ

        # max sigma on {-3, -1, 0, 1, 3} scale = 3
        evi = (sigma / 3.0) * 100.0

        return float(evi)
       
    # --- Tier mapper -------------------------------------------------
    def _map_evi_tier(self, evi_score):
        """
        Emotional Volatility Tiers:
        - [80,100] Extreme Volatility
        - [60,79]  High Volatility
        - [40,59]  Moderate Volatility
        - [20,39]  Low Volatility
        - [0,19]   Stable Emotion
        """
        try:
            v = float(evi_score)
        except (TypeError, ValueError):
            v = float('nan')
        if not pd.notna(v):
            return "Stable Emotion"  # safe default
        v = max(0.0, min(100.0, v))

        if v >= 80: return "Extreme Volatility"
        if v >= 60: return "High Volatility"
        if v >= 40: return "Moderate Volatility"
        if v >= 20: return "Low Volatility"
        
        return "Stable Emotion"  
          
    def _map_loyalty_tier(self, eri_normalized):
        """
        ERI ∈ [-100, 100] → {Very Positive, Positive, Neutral, Negative, Very Negative}
        Boundaries (inclusive): [80,100], [30,79], [-10,29], [-50,-11], [-100,-51]
        """
        try:
            x = float(eri_normalized)
        except (TypeError, ValueError):
            x = float('nan')
        if not pd.notna(x):
            return "Neutral"  # safe default
        x = max(-100.0, min(100.0, x))

        if x >= 80:    return "Very Positive"
        if x >= 30:    return "Positive"
        if x >= -10:   return "Neutral"
        if x >= -50:   return "Negative"
        return "Very Negative"

    def _map_rf_tier(self, RF):
        """
        RF ∈ [0, 100] → {Very High Activity, High Activity, Moderate Activity, Low Activity, Activity Dormant}
        Boundaries (inclusive): [80,100], [60,79], [40,59], [20,39], [0,19]
        """
        try:
            r = float(RF)
        except (TypeError, ValueError):
            r = float('nan')
        if not pd.notna(r):
            return "Activity Dormant"  # safe default
        r = max(0.0, min(100.0, r))

        if r >= 80: return "Very High Activity"
        if r >= 60: return "High Activity"
        if r >= 40: return "Moderate Activity"
        if r >= 20: return "Low Activity"
        return "Activity Dormant"

    def _compute_dynamic_component_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Post-Layer2: assign 5-tier labels to R and F using distribution-aware quantiles.
        Returns df with: R_Tier, R_Tier_Rank, F_Tier, F_Tier_Rank
        Also stores the cutpoints on self for transparency/debug.
        """
        df = df.copy()

        # Labels + ranks (descending = higher is 'stronger')
        r_labels = ["No Recent Activity", "Low Recency", "Moderate Recency", "High Recency", "Very High Recency"]
        f_labels = ["Sparse Mentions", "Low Frequency", "Moderate Frequency", "High Frequency", "Very High Frequency"]
        ranks   = [1, 2, 3, 4, 5]

        # Work only on valid values
        r_valid = df["R"].dropna().astype(float)
        f_valid = df["F"].dropna().astype(float)

        def _safe_qcut(series, labels):
            # If too few unique values for qcut, fall back to rank-based percentiles
            try:
                return pd.qcut(series, q=5, labels=labels, duplicates="drop")
            except ValueError:
                # rank to [0,1], then bucket
                pct = series.rank(method="average", pct=True)
                bins = pd.cut(pct, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=labels, include_lowest=True)
                return bins

        # Compute tiers
        r_tier = _safe_qcut(r_valid, r_labels)
        f_tier = _safe_qcut(f_valid, f_labels)

        # Write back to df
        df.loc[r_tier.index, "R_Tier"] = r_tier.astype(str)
        df.loc[f_tier.index, "F_Tier"] = f_tier.astype(str)

        # Numeric ranks aligned with labels (map label->rank)
        r_map = dict(zip(r_labels, ranks))
        f_map = dict(zip(f_labels, ranks))
        df["R_Tier_Rank"] = df["R_Tier"].map(r_map).fillna(0).astype(int)
        df["F_Tier_Rank"] = df["F_Tier"].map(f_map).fillna(0).astype(int)

        # Keep the actual cutpoints for transparency
        # (best-effort: if qcut succeeded, retrieve bin edges via quantiles)
        try:
            self.meta_r_quantiles = r_valid.quantile([0.0, .2, .4, .6, .8, 1.0]).round(2).to_dict()
        except Exception:
            self.meta_r_quantiles = {}
        try:
            self.meta_f_quantiles = f_valid.quantile([0.0, .2, .4, .6, .8, 1.0]).round(2).to_dict()
        except Exception:
            self.meta_f_quantiles = {}

        return df


    

