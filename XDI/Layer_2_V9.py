import os, numpy as np, pandas as pd
import uuid, hashlib
from collections import Counter

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

            # ✅ [EXISTING CODE - Date validation]
            if "date" not in self.df.columns:
                raise ValueError("Layer2 requires a 'date' column in YYYY-MM-DD format.")
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce").dt.normalize()
            if self.df["date"].isna().all():
                raise ValueError("All values in 'date' are invalid/NaT.")

            self.today = (
                pd.to_datetime(today_anchor).normalize()
                if today_anchor is not None
                else pd.Timestamp.today().normalize()
            )

            self.layer2_df = None
            self.verbose = verbose
            self.window_days = window_days
            self.timeframe_days = window_days

            # ========================================================================
            # 🆕 SEU PROVENANCE / VERSIONING (V9)
            # ========================================================================
            self.schema_version = "SEU_L2_V9"
            self.pipeline_version = "Layer2Computer_v9"
            self.run_id = str(uuid.uuid4())
            self.computed_at_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            # ✅ [EXISTING CODE - Emotion scores]
            self.emotion_scores = {
                "Adoration": 3,
                "Appreciation": 1,
                "Ambivalence": 0,
                "Agitation": -1,
                "Anger": -3,
            }

            # ✅ [EXISTING CODE - Recency decay]
            self.tau_days = float(tau_days)
            assert self.tau_days > 0

            # ✅ [EXISTING CODE - RF weights]
            total = float(rf_weight_r) + float(rf_weight_f)
            if total <= 0:
                self.rf_weight_r, self.rf_weight_f = 0.5, 0.5
            else:
                self.rf_weight_r = float(rf_weight_r) / total
                self.rf_weight_f = float(rf_weight_f) / total

            # ========================================================================
            # 🆕 TIER BOUNDARY CONSTANTS
            # ========================================================================
            
            # --- ERI Tier Boundaries (normalized -100 to +100) ---
            self.ERI_VERY_POSITIVE_MIN = 80
            self.ERI_POSITIVE_MIN = 30
            self.ERI_NEUTRAL_MIN = -10
            self.ERI_NEGATIVE_MIN = -50
            # Below -50 = Very Negative
            
            # --- EVI Tier Boundaries (0 to 100) ---
            self.EVI_EXTREME_MIN = 80
            self.EVI_HIGH_MIN = 60
            self.EVI_MODERATE_MIN = 40
            self.EVI_LOW_MIN = 20
            # Below 20 = Stable Emotion
            
            # --- RF Tier Boundaries (0 to 100) ---
            self.RF_VERY_HIGH_MIN = 80
            self.RF_HIGH_MIN = 60
            self.RF_MODERATE_MIN = 40
            self.RF_LOW_MIN = 20
            # Below 20 = Dormant
            
            # ========================================================================
            # 🆕 TEMPORAL INTELLIGENCE THRESHOLDS
            # ========================================================================
            
            # --- ERI Trajectory Thresholds (tier-crossing based) ---
            self.ERI_STRONG_IMPROVEMENT_TIERS = 2   # Cross 2+ tiers upward
            self.ERI_IMPROVEMENT_TIERS = 1          # Cross 1 tier upward
            self.ERI_STRONG_DECLINE_TIERS = -2      # Cross 2+ tiers downward
            self.ERI_DECLINE_TIERS = -1             # Cross 1 tier downward
            # Between -1 and +1 = Stable
            
            # --- EVI Trajectory Thresholds (tier-crossing based) ---
            self.EVI_STRONG_FRAGMENTATION_TIERS = 2   # Cross 2+ tiers upward
            self.EVI_FRAGMENTATION_TIERS = 1          # Cross 1 tier upward
            self.EVI_STRONG_CONSOLIDATION_TIERS = -2  # Cross 2+ tiers downward
            self.EVI_CONSOLIDATION_TIERS = -1         # Cross 1 tier downward
            # Between -1 and +1 = Stable Volatility
            
            # --- Emotion Momentum Thresholds (percentage point shifts) ---
            self.EMOTION_STRONG_RISE_PCT = 20     # 20+ pct point increase
            self.EMOTION_RISE_PCT = 10            # 10+ pct point increase
            self.EMOTION_MINOR_RISE_PCT = 5       # 5+ pct point increase
            self.EMOTION_STRONG_FADE_PCT = -20    # 20+ pct point decrease
            self.EMOTION_FADE_PCT = -10           # 10+ pct point decrease
            self.EMOTION_MINOR_FADE_PCT = -5      # 5+ pct point decrease
            # Between -5 and +5 = Stable
            
            # --- Pattern Interpretation Thresholds ---
            self.PATTERN_ERI_CRISIS_THRESHOLD   =  20   # 🔥 FIXED: was -20
            self.PATTERN_EVI_FRAGMENT_THRESHOLD =  15   # EVI delta for fragmentation (unchanged)
            self.PATTERN_ERI_RECOVERY_THRESHOLD =  20   # ERI delta for recovery (unchanged)
            self.PATTERN_EVI_CONSOLIDATE_THRESHOLD = -10 # EVI delta for consolidation (unchanged)
            self.PATTERN_ERI_STABLE_RANGE       =   5   # +/- range for stability (unchanged)
            self.PATTERN_EVI_STABLE_RANGE       =   5   # +/- range for stability (unchanged)
            self.PATTERN_ERI_POSITIVE_THRESHOLD =  30   # ERI level for "positive" (unchanged)
            self.PATTERN_ERI_NEGATIVE_THRESHOLD = -30   # ERI level for "negative" (unchanged)
            self.PATTERN_EVI_CHAOS_THRESHOLD    =  20   # EVI delta for chaos (unchanged)

    def _classify_emotional_state_band(self, eri_tier, evi_tier):
        """
        ERI_Tier x EVI_Tier → Emotional State Band (ES1–ES5)

        ERI_Tier  : Very Positive | Positive | Neutral | Negative | Very Negative
        EVI_Tier  : Stable Emotion | Low Volatility | Moderate Volatility | High Volatility | Extreme Volatility

        Full 5x5 matrix (25 combinations), all explicitly resolved:

        ┌────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
        │ │ ERI \\ EVI   │ Stable   │ Low      │ Moderate │ High     │ Extreme  │
        ├────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
        │ Very Positive  │ ES1      │ ES1      │ ES2      │ ES2      │ ES2      │
        │ Positive       │ ES1      │ ES1      │ ES2      │ ES2      │ ES2      │
        │ Neutral        │ ES3      │ ES3      │ ES3      │ ES4      │ ES4      │
        │ Negative       │ ES4      │ ES4      │ ES4      │ ES4      │ ES4      │
        │ Very Negative  │ ES5      │ ES5      │ ES5      │ ES5      │ ES5      │
        └────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

        Logic rationale:
        - ES1: Happy + consistent = advocacy foundation
        - ES2: Happy + polarized = engaged but inconsistent, room to stabilize
        - ES3: Neutral + low/moderate polarization = disengaged, silent churn risk
        - ES4: Neutral + high/extreme polarization = unresolved conflict masking as neutral
               Negative (any polarization) = active confirmed crisis
        - ES5: Very Negative (any polarization) = structural failure, polarity dominates
        """
        et = eri_tier
        vt = evi_tier

        # ES1 – Secure Loyalty
        if et in {"Very Positive", "Positive"} and vt in {"Stable Emotion", "Low Volatility"}:
            return "ES1_Secure_Loyalty"

        # ES2 – Growth Opportunity
        if et in {"Very Positive", "Positive"} and vt in {"Moderate Volatility", "High Volatility", "Extreme Volatility"}:
            return "ES2_Growth_Opportunity"

        # ES5 – Critical Failure (checked before ES4 — polarity dominates)
        if et == "Very Negative":
            return "ES5_Critical_Failure"

        # ES4 – Active Crisis
        if et == "Negative":
            return "ES4_Active_Crisis"
        if et == "Neutral" and vt in {"High Volatility", "Extreme Volatility"}:
            return "ES4_Active_Crisis"

        # ES3 – At Risk
        if et == "Neutral" and vt in {"Stable Emotion", "Low Volatility", "Moderate Volatility"}:
            return "ES3_At_Risk"

        # Fallback — should never be reached given exhaustive matrix above
        return "ES3_At_Risk"

    # ----------- Priority Grid -----------
    # ES Band x RF_Tier → Priority (P0–P5)
    #
    # Full 5x5 matrix (25 combinations), all explicitly resolved:
    #
    # ┌──────────────────────┬─────────┬─────────┬──────────┬──────────┬───────────┐
    # │ ES Band \\ RF        │ Dormant │ Low     │ Moderate │ High     │ Very High │
    # ├──────────────────────┼─────────┼─────────┼──────────┼──────────┼───────────┤
    # │ ES1_Secure_Loyalty   │ P5      │ P5      │ P4       │ P4       │ P3        │
    # │ ES2_Growth_Opp.      │ P4      │ P4      │ P3       │ P3       │ P2        │
    # │ ES3_At_Risk          │ P2      │ P3      │ P2       │ P1       │ P1        │
    # │ ES4_Active_Crisis    │ P2      │ P2      │ P1       │ P0       │ P0        │
    # │ ES5_Critical_Failure │ P1      │ P1      │ P0       │ P0       │ P0        │
    # └──────────────────────┴─────────┴─────────┴──────────┴──────────┴───────────┘
    #
    # Design principles:
    # - ES5 is strictly worse than ES4 at every RF tier (P1 where ES4=P2, P0 where ES4=P1)
    # - No serious band (ES3/ES4/ES5) drops below P2 when Dormant — silence is never safe
    # - ES2 Dormant = P4 (volatile-positive gone silent = real warning, not background noise)
    # - RF modulates urgency, ES Band determines severity floor
    PRIORITY_GRID = {
        "ES1_Secure_Loyalty": {
            "Dormant":   "P5",
            "Low":       "P5",
            "Moderate":  "P4",
            "High":      "P4",
            "Very High": "P3",
        },
        "ES2_Growth_Opportunity": {
            "Dormant":   "P4",  # Volatile-positive gone silent = warning, not background
            "Low":       "P4",
            "Moderate":  "P3",
            "High":      "P3",
            "Very High": "P2",
        },
        "ES3_At_Risk": {
            "Dormant":   "P2",  # Silent at-risk = urgent — silence is a red flag
            "Low":       "P3",
            "Moderate":  "P2",
            "High":      "P1",
            "Very High": "P1",
        },
        "ES4_Active_Crisis": {
            "Dormant":   "P2",  # Crisis + silence = likely already churning
            "Low":       "P2",
            "Moderate":  "P1",
            "High":      "P0",
            "Very High": "P0",
        },
        "ES5_Critical_Failure": {
            "Dormant":   "P1",  # Structural failure + silence = reputational time bomb
            "Low":       "P1",  # Quiet critical failure still needs structural response
            "Moderate":  "P0",  # Active critical failure = no buffer, straight to emergency
            "High":      "P0",
            "Very High": "P0",
        },
    }

    def _classify_priority(self, emotional_state_band, rf_tier_short):
        return self.PRIORITY_GRID.get(emotional_state_band, {}).get(rf_tier_short, "P3")

    # ----------- Purpose Matrix -----------
    # ES Band x Priority → Purpose label
    #
    # Only reachable combinations are mapped (14 live entries).
    # Every combination that can occur in production resolves to an explicit label.
    # Unreachable combinations (e.g. ES1+P0, ES5+P2) are not mapped — they
    # cannot occur given the Priority Grid above, so no defensive entries needed.
    #
    # Reachability map:
    # ES1 → produces P5, P4, P3 only
    # ES2 → produces P4, P3, P2 only
    # ES3 → produces P3, P2, P1 only
    # ES4 → produces P2, P1, P0 only
    # ES5 → produces P1, P0 only
    
    PURPOSE_MATRIX = {
        # ES1 – Secure Loyalty: advocacy activation, not intervention
        ("ES1_Secure_Loyalty", "P5"): "Monitor Stable Advocacy",
        ("ES1_Secure_Loyalty", "P4"): "Maintain Positive Momentum",
        ("ES1_Secure_Loyalty", "P3"): "Activate Advocacy Programs",

        # ES2 – Growth Opportunity: nurture, stabilize, don't lose the signal
        ("ES2_Growth_Opportunity", "P4"): "Develop Emerging Loyalty",
        ("ES2_Growth_Opportunity", "P3"): "Strengthen Growth Drivers",
        ("ES2_Growth_Opportunity", "P2"): "Activate Growth Levers",

        # ES3 – At Risk: investigate and intervene before churn
        ("ES3_At_Risk", "P3"): "Investigate Low-Activity Risk",
        ("ES3_At_Risk", "P2"): "Address Early Warning Signs",
        ("ES3_At_Risk", "P1"): "Contain At-Risk Emotion",

        # ES4 – Active Crisis: containment and immediate intervention
        ("ES4_Active_Crisis", "P2"): "Crisis Containment",
        ("ES4_Active_Crisis", "P1"): "Active Crisis Intervention",
        ("ES4_Active_Crisis", "P0"): "Emergency Crisis Response",

        # ES5 – Critical Failure: structural response, no soft options
        ("ES5_Critical_Failure", "P1"): "Structural Failure Assessment",
        ("ES5_Critical_Failure", "P0"): "Critical Failure Response",
    }

    def _classify_purpose(self, emotional_state_band, priority_tier):
        return self.PURPOSE_MATRIX.get(
            (emotional_state_band, priority_tier),
            "Unclassified — Review Manually",
        )

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
            - NEW: Compute Emotion Recency Profile per Experience Driver
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
        
            results = []

            for entity, group in grouped:
                entry = {"experience_driver": entity}

                # --- SEU identity (stable per driver + anchor + window) -----------
                today_anchor_str = self.today.strftime("%Y-%m-%d")
                window_days_int = int(self.window_days) if self.window_days is not None else None
                seu_id_raw = f"{entity}||{today_anchor_str}||{window_days_int}"
                seu_id = hashlib.md5(seu_id_raw.encode("utf-8")).hexdigest()

                # --- counts & dates ---------------------------------------------------
                total_mentions = int(len(group))
                times = group[ts_col]
                observed_days = int(times.nunique()) if total_mentions else 0
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
                eri_norm = self._compute_eri_from_counts(emotion_counts, total_mentions)
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
                rf_tier_short = self._normalize_rf_tier(rf_tier_long)  # "High", "Moderate", etc.

                # --- Priority via Emotional_State × RF --------------------------------
                priority_tier = self._classify_priority(emotional_state_band, rf_tier_short)
                priority_purpose = self._classify_purpose(emotional_state_band, priority_tier)

                # --- Emotion Recency Profile ------------------------------------
                # Recency-of-emotion distribution for this experience_driver
                emotion_recency_profile = self._compute_emotion_recency_profile(group)

                # --- Temporal intelligence availability (explicit) ----------------
                temporal_available = None
                sufficiency_reason = None

                if total_mentions < 2 or observed_days < 2:
                    temporal_available = False
                    sufficiency_reason = "INSUFFICIENT_DAYS_OR_MENTIONS"
                else:
                    tvs = (emotion_recency_profile or {}).get("tier_volume_summary") or {}
                    early_mentions = ((tvs.get("early") or {}).get("mentions"))
                    recent_mentions = ((tvs.get("recent") or {}).get("mentions"))

                    if early_mentions is None or recent_mentions is None:
                        temporal_available = False
                        sufficiency_reason = "TEMPORAL_SUMMARY_MISSING"
                    elif early_mentions == 0 and recent_mentions > 0:
                        temporal_available = False
                        sufficiency_reason = "ONLY_RECENT_DATA"
                    elif recent_mentions == 0 and early_mentions > 0:
                        temporal_available = False
                        sufficiency_reason = "ONLY_EARLY_DATA"
                    elif early_mentions == 0 and recent_mentions == 0:
                        temporal_available = False
                        sufficiency_reason = "NO_TIER_DATA"
                    else:
                        temporal_available = True
                        sufficiency_reason = None

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
                        "seu_id": seu_id,
                        "schema_version": self.schema_version,
                        "pipeline_version": self.pipeline_version,
                        "run_id": self.run_id,
                        "computed_at_utc": self.computed_at_utc,

                        "Today_Anchor": self.today.strftime("%Y-%m-%d"),
                        "Window_Days": int(self.window_days)
                        if self.window_days is not None
                        else None,

                        "Observed_Days": observed_days,
                        "Temporal_Intelligence_Available": temporal_available,
                        "Sufficiency_Reason": sufficiency_reason,

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

                        # NEW: Emotion Recency Profile (dict: by_emotion + by_tier)
                        "Emotion_Recency_Profile": emotion_recency_profile,

                        # Debug
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
    
    def _compute_eri_from_counts(self, emotion_counts: dict, total_mentions: int) -> float:
                """
                Compute ERI from emotion_counts dict.
                Returns ERI in range [-100, +100].
                """
                if total_mentions == 0:
                    return 0.0
                
                eri_raw = sum(
                    float(self.emotion_scores.get(em, 0.0)) * int(ct)
                    for em, ct in emotion_counts.items()
                )
                eri_mean = eri_raw / total_mentions  # Range: [-3, +3]
                
                # Normalize to [-100, +100]
                eri_norm = ((float(eri_mean) + 3.0) / 6.0) * 200.0 - 100.0
                eri_norm = float(np.clip(eri_norm, -100.0, 100.0))
                
                return eri_norm
        
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

    def _normalize_rf_tier(self, rf_tier_long):
            """
            Convert long RF tier names to short format for PRIORITY_GRID lookup.
            
            Args:
                rf_tier_long (str): Output from _map_rf_tier() (e.g., "High Activity")
            
            Returns:
                str: Short tier name for PRIORITY_GRID (e.g., "High")
            
            Examples:
                >>> self._normalize_rf_tier("High Activity")
                "High"
                >>> self._normalize_rf_tier("Activity Dormant")
                "Dormant"
            """
            mapping = {
                "Very High Activity": "Very High",
                "High Activity": "High",
                "Moderate Activity": "Moderate",
                "Low Activity": "Low",
                "Activity Dormant": "Dormant",
            }
            return mapping.get(rf_tier_long, "Moderate")  # Safe default
        
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

    # ----------- Emotion Recency Tier -----------
    def _map_emotion_recency_tier(self, age_days, window_days) -> str:
            """
            Timeframe-agnostic recency tiers.
            Divides window into 5 equal buckets + 'Beyond_Window'.
            """
            
            try:
                d = int(age_days)
            except (TypeError, ValueError):
                return "Beyond_Window"
            
            if d < 0 or not pd.notna(d):
                return "Beyond_Window"
            
            w = float(window_days) if window_days and window_days > 0 else 30.0
            tier_width = w / 5.0
            
            if d <= tier_width:
                return "Tier_1"
            if d <= tier_width * 2:
                return "Tier_2"
            if d <= tier_width * 3:
                return "Tier_3"
            if d <= tier_width * 4:
                return "Tier_4"
            if d <= w:
                return "Tier_5"
            
            return "Beyond_Window"

    # ----------- EVI from Counts -----------
    def _compute_evi_from_counts(self, counts: dict) -> float:
            return self._compute_evi(
                counts.get("Adoration", 0),
                counts.get("Appreciation", 0),
                counts.get("Ambivalence", 0),
                counts.get("Agitation", 0),
                counts.get("Anger", 0),
            )

    # ----------- Emotion Recency Profile -----------
    def _compute_emotion_recency_profile(self, group: pd.DataFrame) -> dict:
        """
        For a single experience_driver group:
        Returns:
        1. Distribution of each emotion across all recency tiers (by_emotion)
        2. FULL emotion distribution within each tier (by_tier)
        3. ERI per tier (normalized -100..+100) using _compute_eri_from_counts
        4. EVI per tier (0..100) using _compute_evi_from_counts → _compute_evi
        5. ES_Band per tier using existing decision matrix
        6. Temporal intelligence (auto-derived from tier comparisons),
           with guard rails for sparse / incomplete tier coverage.
        7. Tier volume profile → share of total mentions per tier
        8. Tier volume summary → full accounting across early / mid / recent / beyond_window
           (early + mid + recent + beyond_window always equals total_mentions exactly)

        Tier structure (window divided into 5 equal buckets):
            Tier_1      → most recent  (0   to window/5 days)
            Tier_2      → recent       (window/5+1 to 2*window/5 days)
            Tier_3      → mid          (2*window/5+1 to 3*window/5 days)
            Tier_4      → early        (3*window/5+1 to 4*window/5 days)
            Tier_5      → oldest       (4*window/5+1 to window days)
            Beyond_Window → outside the analysis window entirely

        Temporal comparison groups:
            recent = Tier_1 + Tier_2
            mid    = Tier_3
            early  = Tier_4 + Tier_5
        """

        # --- Safety checks ---
        if "date" not in group.columns or "emotion_primary" not in group.columns:
            return {}

        sub = group.dropna(subset=["date"])
        if sub.empty:
            return {}

        # --- Age in days → recency tier ---
        ages = (self.today - sub["date"]).dt.days.clip(lower=0)

        tmp = pd.DataFrame({
            "emotion": sub["emotion_primary"],
            "age_days": ages,
        })
        tmp["tier"] = tmp["age_days"].apply(
            lambda x: self._map_emotion_recency_tier(x, self.window_days)
        )

        all_tiers = ["Tier_1", "Tier_2", "Tier_3", "Tier_4", "Tier_5", "Beyond_Window"]

        # ── Part 1: By Emotion ────────────────────────────────────────────
        # How each emotion is spread across recency tiers.
        # Every emotion gets a full 6-tier distribution, zeroed where absent.
        by_emotion: dict = {}
        for emotion, g_em in tmp.groupby("emotion"):
            total = len(g_em)
            tier_counts = g_em["tier"].value_counts().to_dict()
            tiers_dist = {}
            for t in all_tiers:
                ct = int(tier_counts.get(t, 0))
                tiers_dist[t] = {
                    "count": ct,
                    "pct": round(100.0 * ct / max(1, total), 2),
                }
            by_emotion[str(emotion)] = {
                "total_count": int(total),
                "tiers": tiers_dist,
            }

        # ── Part 2: By Tier ───────────────────────────────────────────────
        # Full emotion distribution + ERI / EVI / ES_Band per tier.
        # Empty tiers are explicitly represented with null intelligence fields.
        by_tier: dict = {}
        for t in all_tiers:
            tier_data = tmp[tmp["tier"] == t]
            total_in_tier = len(tier_data)

            if total_in_tier == 0:
                by_tier[t] = {
                    "total_mentions":  0,
                    "emotions":        {},
                    "dominant_emotion": None,
                    "dominant_count":  0,
                    "dominant_pct":    0.0,
                    "runner_up":       None,
                    # Intelligence fields — null for empty tiers (guard rail)
                    "ERI":             None,
                    "ERI_Tier":        None,
                    "EVI":             None,
                    "EVI_Tier":        None,
                    "ES_Band":         None,
                }
                continue

            counts             = tier_data["emotion"].value_counts()
            total_in_tier      = int(counts.sum())
            tier_emotion_counts = counts.to_dict()

            # Emotion distribution
            emotions_dist = {
                emotion: {
                    "count": int(ct),
                    "pct":   round(100.0 * int(ct) / max(1, total_in_tier), 2),
                }
                for emotion, ct in tier_emotion_counts.items()
            }

            # Dominant + runner-up
            dominant    = counts.index[0]
            dominant_ct = int(counts.iloc[0])
            runner_up   = counts.index[1] if len(counts) > 1 else None

            # ERI, EVI, ES_Band for this tier
            eri_value      = self._compute_eri_from_counts(tier_emotion_counts, total_in_tier)
            eri_tier_label = self._map_loyalty_tier(eri_value)
            evi_value      = self._compute_evi_from_counts(tier_emotion_counts)
            evi_tier_label = self._map_evi_tier(evi_value)
            es_band_tier   = self._classify_emotional_state_band(eri_tier_label, evi_tier_label)

            by_tier[t] = {
                "total_mentions":  total_in_tier,
                "emotions":        emotions_dist,
                "dominant_emotion": dominant,
                "dominant_count":  dominant_ct,
                "dominant_pct":    round(100.0 * dominant_ct / max(1, total_in_tier), 2),
                "runner_up":       runner_up,
                # Tier-level intelligence
                "ERI":             round(eri_value, 2),
                "ERI_Tier":        eri_tier_label,
                "EVI":             round(evi_value, 2),
                "EVI_Tier":        evi_tier_label,
                "ES_Band":         es_band_tier,
            }

        # ── Part 2b: Tier Volume Profile ──────────────────────────────────
        # Share of total mentions per tier (all 6 tiers, sums to 100%).
        # Also enriches by_tier with share_of_total_pct for convenience.
        total_all_mentions = sum(info["total_mentions"] for info in by_tier.values())

        tier_volume_profile: dict = {}
        for t, info in by_tier.items():
            ct        = int(info["total_mentions"])
            share_pct = round(100.0 * ct / max(1, total_all_mentions), 2)
            info["share_of_total_pct"] = share_pct          # enrich by_tier in-place
            tier_volume_profile[t] = {
                "total_mentions": ct,
                "share_pct":      share_pct,
            }

        # ── Part 2c: Tier Volume Summary ──────────────────────────────────
        # Full accounting: recent + mid + early + beyond_window = total_mentions exactly.
        # FIX: original code omitted mid (Tier_3) and beyond_window, causing
        # early + recent to never equal total when either had data.
        recent_tiers = ["Tier_1", "Tier_2"]
        mid_tiers    = ["Tier_3"]
        early_tiers  = ["Tier_4", "Tier_5"]

        recent_total = sum(by_tier[t]["total_mentions"] for t in recent_tiers)
        mid_total    = by_tier["Tier_3"]["total_mentions"]
        early_total  = sum(by_tier[t]["total_mentions"] for t in early_tiers)
        bw_total     = by_tier["Beyond_Window"]["total_mentions"]

        # Sanity: these four must always sum to total_all_mentions
        assert recent_total + mid_total + early_total + bw_total == total_all_mentions, (
            "tier_volume_summary accounting error: segments do not sum to total"
        )

        tier_volume_summary = {
            "total_mentions": int(total_all_mentions),
            "recent": {
                "tiers":    recent_tiers,
                "mentions": int(recent_total),
                "share_pct": round(100.0 * recent_total / max(1, total_all_mentions), 2),
            },
            "mid": {
                "tiers":    mid_tiers,
                "mentions": int(mid_total),
                "share_pct": round(100.0 * mid_total / max(1, total_all_mentions), 2),
            },
            "early": {
                "tiers":    early_tiers,
                "mentions": int(early_total),
                "share_pct": round(100.0 * early_total / max(1, total_all_mentions), 2),
            },
            "beyond_window": {
                "tiers":    ["Beyond_Window"],
                "mentions": int(bw_total),
                "share_pct": round(100.0 * bw_total / max(1, total_all_mentions), 2),
            },
        }

        # --- Part 3: Temporal Intelligence (with guard rails) ---
        temporal_intel = self._derive_temporal_intelligence_from_tiers(by_tier, all_tiers)

        return {
            "by_emotion":          by_emotion,
            "by_tier":             by_tier,
            "tier_volume_profile": tier_volume_profile,
            "tier_volume_summary": tier_volume_summary,
            "temporal_intelligence": temporal_intel,
        }

    # ----------- Temporal Intelligence -----------
    def _derive_temporal_intelligence_from_tiers(self, by_tier: dict, all_tiers: list) -> dict:
        """
        Assemble a structured Temporal Intelligence block from tier-level data.

        Uses:
        - ERI per tier         → polarity flow (ERI_Flow)
        - EVI per tier         → volatility flow (EVI_Flow)
        - ES_Band per tier     → relationship state flow (ES_Flow)
        - Emotion distributions → composition flow (Emotion_Flow)

        Compares early tiers (Tier_4, Tier_5) vs recent tiers (Tier_1, Tier_2)
        and returns a single dict with named flows:

            {
                "ERI_Flow": {...},
                "EVI_Flow": {...},
                "ES_Flow": {...},
                "Emotion_Flow": {...},
                "Pattern_Flow": {...},
                "Context_Flow": {...},
            }

        Guard rails:
        - Works even if some tiers are empty
        - Individual flows may be None if insufficient data
        """

        # Define recent vs early tier groups
        recent_tiers = ["Tier_1", "Tier_2"]
        early_tiers = ["Tier_4", "Tier_5"]

        # --- Core flows --------------------------------------------------------
        eri_trajectory = self._calculate_eri_trajectory(
            by_tier, recent_tiers, early_tiers, all_tiers
        )
        evi_trajectory = self._calculate_evi_trajectory(
            by_tier, recent_tiers, early_tiers, all_tiers
        )
        es_migration = self._calculate_es_band_migration(
            by_tier, recent_tiers, early_tiers
        )

        # Emotion composition flow (ENHANCED with polarity tracking)
        emotion_flow = self._compute_emotion_flow(
            by_tier=by_tier,
            recent_tiers=recent_tiers,
            early_tiers=early_tiers,
            eri_traj=eri_trajectory,
            evi_traj=evi_trajectory,
        )

        # High-level pattern label (existing interpreter, now treated as pattern)
        pattern_label = self._interpret_temporal_pattern(
            eri_trajectory, evi_trajectory, es_migration, emotion_flow
        )

        # Pattern_Flow bundle
        pattern_flow = {
            "pattern_label": pattern_label,
            "eri_trend": eri_trajectory.get("trend") if eri_trajectory else None,
            "evi_trend": evi_trajectory.get("trend") if evi_trajectory else None,
            "es_trajectory": (
                es_migration.get("trajectory") if es_migration else None
            ),
            "dominant_emotion_shift": (
                emotion_flow.get("dominant_shift") if emotion_flow else None
            ),
        }

                # Context_Flow: short, structured narrative
            context_summary = self._build_temporal_context_summary(
            eri_traj=eri_trajectory,
            evi_traj=evi_trajectory,
            es_mig=es_migration,
            emotion_flow=emotion_flow,
            pattern_flow=pattern_flow,
        )

        context_flow = {
            "summary": context_summary,
        }

        # NEW: PEM_Guidance – derived guidance layer (no new math)
        pem_guidance = self._derive_pem_guidance(
            eri_traj=eri_trajectory,
            evi_traj=evi_trajectory,
            es_mig=es_migration,
            emotion_flow=emotion_flow,
            pattern_flow=pattern_flow,
        )

        return {
            "ERI_Flow": eri_trajectory,
            "EVI_Flow": evi_trajectory,
            "ES_Flow": es_migration,
            "Emotion_Flow": emotion_flow,
            "Pattern_Flow": pattern_flow,
            "Context_Flow": context_flow,
            "PEM_Guidance": pem_guidance,
        }

    def _derive_pem_guidance(
        self,
        eri_traj: dict | None,
        evi_traj: dict | None,
        es_mig: dict | None,
        emotion_flow: dict | None,
        pattern_flow: dict | None,
    ) -> dict:
        """
        PEM_Guidance:
        A small, derived guidance block sitting on top of Temporal Intelligence.
        NO NEW MATH - pure repackaging of existing flows.

        Fields:
            - mode: high-level emotional mode (CRISIS / RECOVERY / STABLE / etc.)
            - recommended_stream_bias: list of streams to bias towards
            - urgency_band: Immediate / Near-Term / Watchful / Dormant / Unknown
            - emotional_driver: dominant recent emotion (fallback to early)
            - pattern_label: underlying temporal pattern label text
            - es_start / es_end: ES band migration endpoints
            - eri_end / eri_delta / evi_delta: core numeric hooks
        """
        pattern_label = None
        if pattern_flow:
            pattern_label = pattern_flow.get("pattern_label")

        label_upper = (pattern_label or "").upper()

        # ---------------------------------------------------------------------
        # 1) MODE + URGENCY
        # ---------------------------------------------------------------------
        if not eri_traj and not evi_traj and not es_mig:
            mode = "INSUFFICIENT_DATA"
            urgency_band = "Unknown"
        else:
            if label_upper.startswith((
                "CRISIS",
                "CATASTROPH",
                "RELATIONSHIP COLLAPSE",
                "ENGAGEMENT COLLAPSE",
                "CHURN ACTUALIZED",
                "RISK ACTUALIZED",
            )):
                mode = "CRISIS"
                urgency_band = "Immediate"
            elif label_upper.startswith((
                "FULL RECOVERY",
                "RECOVERY",
                "RE-ENGAGEMENT SUCCESS",
                "POSITIVE SHIFT",
                "CRISIS STABILIZING",
                "STABILIZATION",
                "MINOR IMPROVEMENT",
            )):
                mode = "RECOVERY"
                urgency_band = "Near-Term"
            elif label_upper.startswith(("STABLE LOYALTY",)):
                mode = "STABLE"
                urgency_band = "Dormant"
            elif label_upper.startswith((
                "ENTRENCHED NEGATIVITY",
                "DANGER PATTERN",
            )):
                mode = "NEGATIVE_STABLE"
                urgency_band = "Near-Term"
            elif label_upper.startswith((
                "VOLATILE GROWTH",
                "FRAGMENTING RELATIONSHIP",
                "VOLATILITY EMERGING",
            )):
                mode = "VOLATILE"
                urgency_band = "Watchful"
            elif "INSUFFICIENT TEMPORAL DATA" in label_upper:
                mode = "INSUFFICIENT_DATA"
                urgency_band = "Unknown"
            else:
                mode = "AMBIGUOUS"
                urgency_band = "Watchful"

        # ---------------------------------------------------------------------
        # 2) RECOMMENDED STREAM BIAS (purely derived from mode + ERI)
        # ---------------------------------------------------------------------
        eri_end = eri_traj.get("end") if eri_traj else None
        recommended_stream_bias: list[str] = []

        if mode in {"CRISIS", "NEGATIVE_STABLE"}:
            recommended_stream_bias = ["Fix", "Optimize"]
        elif mode == "RECOVERY":
            recommended_stream_bias = ["Fix", "Optimize"]
        elif mode == "STABLE":
            if eri_end is not None and eri_end > self.PATTERN_ERI_POSITIVE_THRESHOLD:
                recommended_stream_bias = ["Amplify", "Innovate"]
            else:
                recommended_stream_bias = ["Optimize"]
        elif mode == "VOLATILE":
            if eri_end is not None and eri_end >= 0:
                recommended_stream_bias = ["Optimize", "Innovate"]
            else:
                recommended_stream_bias = ["Fix", "Optimize"]
        else:
            # AMBIGUOUS / INSUFFICIENT_DATA → no strong bias
            recommended_stream_bias = []

        # ---------------------------------------------------------------------
        # 3) EMOTIONAL DRIVER
        # ---------------------------------------------------------------------
        emotional_driver = None
        if emotion_flow:
            emotional_driver = (
                emotion_flow.get("dominant_emotion_recent")
                or emotion_flow.get("dominant_emotion_early")
            )

        # ---------------------------------------------------------------------
        # 4) NUMERIC HOOKS (for downstream usage / agents)
        # ---------------------------------------------------------------------
        eri_delta = eri_traj.get("delta") if eri_traj else None
        evi_delta = evi_traj.get("delta") if evi_traj else None

        es_start = es_mig.get("start") if es_mig else None
        es_end = es_mig.get("end") if es_mig else None

        return {
            "mode": mode,
            "recommended_stream_bias": recommended_stream_bias,
            "urgency_band": urgency_band,
            "emotional_driver": emotional_driver,
            "pattern_label": pattern_label,
            "es_start": es_start,
            "es_end": es_end,
            "eri_end": eri_end,
            "eri_delta": eri_delta,
            "evi_delta": evi_delta,
        }
    
    # ----------- ERI Trajectory -----------
    def _calculate_eri_trajectory(
        self, by_tier: dict, recent_tiers: list, early_tiers: list, all_tiers: list
    ) -> dict:
        """
        Calculate ERI trajectory using tier-crossing logic.
        
        ENHANCED: Now includes percentage change calculation
        
        Returns:
            dict with keys: start, end, delta, delta_pct, tier_shift, trend, by_tier
            OR None if insufficient data
        """
        recent_eris = self._get_tier_values(by_tier, recent_tiers, "ERI")
        early_eris = self._get_tier_values(by_tier, early_tiers, "ERI")

        if not recent_eris or not early_eris:
            return None

        eri_start = round(float(np.mean(early_eris)), 2)
        eri_end = round(float(np.mean(recent_eris)), 2)
        eri_delta = round(eri_end - eri_start, 2)

        # NEW: Calculate percentage change
        # Handle zero/near-zero baseline carefully
        if abs(eri_start) < 0.01:
            # If starting from near-zero, report as "N/A" or use alternative metric
            eri_delta_pct = None  # Can't meaningfully compute % from zero
        else:
            eri_delta_pct = round((eri_delta / abs(eri_start)) * 100.0, 2)

        # EVI scale is 0 → 100, so max swing = 100 points
        full_scale_move_pct = round((abs(eri_delta) / 200.0) * 100.0, 2)

        # Calculate tier-crossing distance
        start_tier_idx = self._eri_tier_index(eri_start)
        end_tier_idx = self._eri_tier_index(eri_end)
        tier_shift = end_tier_idx - start_tier_idx

        # Determine trend based on tier crossings (using constants)
        if tier_shift >= self.ERI_STRONG_IMPROVEMENT_TIERS:
            eri_trend = "↑↑ Strongly Improving"
        elif tier_shift >= self.ERI_IMPROVEMENT_TIERS:
            eri_trend = "↑ Improving"
        elif tier_shift <= self.ERI_STRONG_DECLINE_TIERS:
            eri_trend = "↓↓ Strongly Declining"
        elif tier_shift <= self.ERI_DECLINE_TIERS:
            eri_trend = "↓ Declining"
        else:
            eri_trend = "→ Stable"

        return {
            "start": eri_start,
            "end": eri_end,
            "delta": eri_delta,
            "delta_pct": eri_delta_pct,  # NEW: Percentage change
            "full_scale_move_pct": full_scale_move_pct,
            "tier_shift": tier_shift,
            "trend": eri_trend,
            "by_tier": {t: by_tier[t].get("ERI") for t in all_tiers},
        }


    # ----------- EVI Trajectory -----------
    def _calculate_evi_trajectory(
        self, by_tier: dict, recent_tiers: list, early_tiers: list, all_tiers: list
    ) -> dict:
        """
        Calculate EVI trajectory using tier-crossing logic.
        
        ENHANCED: Now includes percentage change calculation
        
        Returns:
            dict with keys: start, end, delta, delta_pct, tier_shift, trend, by_tier
            OR None if insufficient data
        """
        recent_evis = self._get_tier_values(by_tier, recent_tiers, "EVI")
        early_evis = self._get_tier_values(by_tier, early_tiers, "EVI")

        if not recent_evis or not early_evis:
            return None

        evi_start = round(float(np.mean(early_evis)), 2)
        evi_end = round(float(np.mean(recent_evis)), 2)
        evi_delta = round(evi_end - evi_start, 2)

        # NEW: Calculate percentage change
        if abs(evi_start) < 0.01:
            # Starting from near-zero volatility
            evi_delta_pct = None
        else:
            evi_delta_pct = round((evi_delta / evi_start) * 100.0, 2)

        # EVI scale spans 100 points total: 0 → 100
        full_scale_move_pct = round((abs(evi_delta) / 100.0) * 100.0, 2)
        
        # Calculate tier-crossing distance
        start_tier_idx = self._evi_tier_index(evi_start)
        end_tier_idx = self._evi_tier_index(evi_end)
        tier_shift = end_tier_idx - start_tier_idx

        # Determine trend based on tier crossings (using constants)
        if tier_shift >= self.EVI_STRONG_FRAGMENTATION_TIERS:
            evi_trend = "↑↑ Rapidly Fragmenting"
        elif tier_shift >= self.EVI_FRAGMENTATION_TIERS:
            evi_trend = "↑ Fragmenting"
        elif tier_shift <= self.EVI_STRONG_CONSOLIDATION_TIERS:
            evi_trend = "↓↓ Rapidly Consolidating"
        elif tier_shift <= self.EVI_CONSOLIDATION_TIERS:
            evi_trend = "↓ Consolidating"
        else:
            evi_trend = "→ Stable Volatility"

        return {
            "start": evi_start,
            "end": evi_end,
            "delta": evi_delta,
            "delta_pct": evi_delta_pct,  
            "full_scale_move_pct": full_scale_move_pct,
            "tier_shift": tier_shift,
            "trend": evi_trend,
            "by_tier": {t: by_tier[t].get("EVI") for t in all_tiers},
        }

    # ----------- Emotion Flow -----------
    def _compute_emotion_flow(
        self,
        by_tier: dict,
        recent_tiers: list,
        early_tiers: list,
        eri_traj: dict | None,
        evi_traj: dict | None,
    ) -> dict | None:
        """
        Build the Emotion_Flow object.

        Fields:
        - early_distribution:          % per emotion in early tiers (Tier_4+Tier_5)
        - recent_distribution:         % per emotion in recent tiers (Tier_1+Tier_2)
        - combined_distribution:       % per emotion across both early+recent
        - shifts:                      per-emotion delta (momentum logic)
        - dominant_shift:              emotion with largest absolute delta
        - top_early_emotions:          top 3 emotions in early period
        - top_recent_emotions:         top 3 emotions in recent period
        - ambivalence_type:            "true", "false", or "none"
        - polarity_shift:              aggregate positive/negative group flows
        - dominant_emotion_early:      most common emotion in early period
        - dominant_emotion_recent:     most common emotion in recent period
        - dominant_emotion_transition: "Anger → Appreciation" style label
        - data_coverage:               flags one-sided or partial data

        🔥 FIX: Previously returned None only when BOTH early and recent were
        empty. If only one side had data, the function continued and produced
        a misleadingly structured comparative object with empty distributions
        on one side. Now explicitly detects one-sided data and sets
        data_coverage accordingly so downstream consumers know the flow is
        non-comparative.
        """
        early_dist   = self._aggregate_emotions_from_tiers(by_tier, early_tiers)
        recent_dist  = self._aggregate_emotions_from_tiers(by_tier, recent_tiers)
        combined_dist = self._aggregate_emotions_from_tiers(
            by_tier, early_tiers + recent_tiers
        )

        # ── Guard rails ───────────────────────────────────────────────────
        # Both empty → nothing to return
        if not early_dist and not recent_dist:
            return None

        # 🔥 FIX: Detect one-sided data explicitly
        has_early  = bool(early_dist)
        has_recent = bool(recent_dist)

        if has_early and has_recent:
            data_coverage = "both"        # Full comparative flow
        elif has_recent and not has_early:
            data_coverage = "recent_only" # Only recent data — no early baseline
        else:
            data_coverage = "early_only"  # Only early data — no recent signal

        # Momentum and top-K only meaningful when both sides exist
        if data_coverage == "both":
            momentum = self._calculate_emotion_momentum(
                by_tier=by_tier,
                recent_tiers=recent_tiers,
                early_tiers=early_tiers,
            )
        else:
            momentum = None

        def _top_k(dist: dict, k: int = 3):
            return [
                {"emotion": em, "pct": pct}
                for em, pct in sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]
            ]

        top_early  = _top_k(early_dist)  if early_dist  else []
        top_recent = _top_k(recent_dist) if recent_dist else []

        # Ambivalence classification requires both trajectories
        ambivalence_type = self._classify_ambivalence_type(
            combined_dist=combined_dist,
            eri_traj=eri_traj,
            evi_traj=evi_traj,
        )

        # ── Polarity shift (meaningful only when both sides present) ──────
        negative_emotions = {"Anger", "Agitation"}
        positive_emotions = {"Adoration", "Appreciation"}

        neg_early_pct  = sum(early_dist.get(e, 0.0)  for e in negative_emotions)
        pos_early_pct  = sum(early_dist.get(e, 0.0)  for e in positive_emotions)
        neut_early_pct = early_dist.get("Ambivalence", 0.0)

        neg_recent_pct  = sum(recent_dist.get(e, 0.0) for e in negative_emotions)
        pos_recent_pct  = sum(recent_dist.get(e, 0.0) for e in positive_emotions)
        neut_recent_pct = recent_dist.get("Ambivalence", 0.0)

        neg_delta  = round(neg_recent_pct  - neg_early_pct,  2)
        pos_delta  = round(pos_recent_pct  - pos_early_pct,  2)
        neut_delta = round(neut_recent_pct - neut_early_pct, 2)

        # Polarity flow direction only meaningful with both sides
        if data_coverage == "both":
            threshold = 15.0
            if pos_delta > threshold and neg_delta < -threshold:
                polarity_flow_direction = "Negative → Positive"
            elif neg_delta > threshold and pos_delta < -threshold:
                polarity_flow_direction = "Positive → Negative"
            elif neut_delta > threshold and (pos_delta < -10 or neg_delta < -10):
                polarity_flow_direction = "Polarized → Neutral"
            elif neut_delta < -threshold and (pos_delta > 10 or neg_delta > 10):
                polarity_flow_direction = "Neutral → Polarized"
            elif abs(pos_delta) < 10 and abs(neg_delta) < 10:
                polarity_flow_direction = "Stable"
            else:
                polarity_flow_direction = "Mixed/Fragmenting"
        else:
            polarity_flow_direction = "Insufficient Data"

        polarity_shift = {
            "negative_early_pct":       round(neg_early_pct,  2),
            "negative_recent_pct":      round(neg_recent_pct, 2),
            "negative_delta":           neg_delta,
            "positive_early_pct":       round(pos_early_pct,  2),
            "positive_recent_pct":      round(pos_recent_pct, 2),
            "positive_delta":           pos_delta,
            "neutral_early_pct":        round(neut_early_pct,  2),
            "neutral_recent_pct":       round(neut_recent_pct, 2),
            "neutral_delta":            neut_delta,
            "polarity_flow_direction":  polarity_flow_direction,
        }

        # ── Dominant emotion tracking ─────────────────────────────────────
        if early_dist:
            dom_early = max(early_dist.items(), key=lambda x: x[1])
            dominant_emotion_early     = dom_early[0]
            dominant_emotion_early_pct = dom_early[1]
        else:
            dominant_emotion_early     = None
            dominant_emotion_early_pct = 0.0

        if recent_dist:
            dom_recent = max(recent_dist.items(), key=lambda x: x[1])
            dominant_emotion_recent     = dom_recent[0]
            dominant_emotion_recent_pct = dom_recent[1]
        else:
            dominant_emotion_recent     = None
            dominant_emotion_recent_pct = 0.0

        # Transition label
        if dominant_emotion_early and dominant_emotion_recent:
            if dominant_emotion_early != dominant_emotion_recent:
                dominant_emotion_transition = f"{dominant_emotion_early} → {dominant_emotion_recent}"
            else:
                dominant_emotion_transition = dominant_emotion_early  # stable
        else:
            dominant_emotion_transition = None

        return {
            # Distributions
            "early_distribution":          early_dist,
            "recent_distribution":         recent_dist,
            "combined_distribution":       combined_dist,
            # Momentum (None when one-sided)
            "shifts":                      momentum.get("shifts")        if momentum else {},
            "dominant_shift":              momentum.get("dominant_shift") if momentum else None,
            # Top emotions
            "top_early_emotions":          top_early,
            "top_recent_emotions":         top_recent,
            # Ambivalence
            "ambivalence_type":            ambivalence_type,
            # Polarity
            "polarity_shift":              polarity_shift,
            # Dominant emotion
            "dominant_emotion_early":      dominant_emotion_early,
            "dominant_emotion_early_pct":  round(dominant_emotion_early_pct,  2),
            "dominant_emotion_recent":     dominant_emotion_recent,
            "dominant_emotion_recent_pct": round(dominant_emotion_recent_pct, 2),
            "dominant_emotion_transition": dominant_emotion_transition,
            # 🔥 NEW: Coverage flag so downstream knows if flow is one-sided
            "data_coverage":               data_coverage,
        }

    # ----------- Context Summary -----------
    def _build_temporal_context_summary(
        self,
        eri_traj: dict | None,
        evi_traj: dict | None,
        es_mig: dict | None,
        emotion_flow: dict | None,
        pattern_flow: dict | None,
    ) -> str:
        """
        Build a short narrative summarizing the main temporal signals from Temporal Intelligence.
        
        ENHANCED: Now includes polarity flow and dominant emotion transition in narrative
        
        This is NOT an action plan, just an explanation of what changed.
        """
        # If we don't even have ERI/EVI, we can't say much
        if not eri_traj or not evi_traj:
            return "Insufficient temporal data to summarize emotional evolution."

        parts = []

        # 1) ERI movement (ENHANCED with % change)
        eri_start = eri_traj["start"]
        eri_end = eri_traj["end"]
        eri_delta = eri_traj["delta"]
        eri_delta_pct = eri_traj.get("delta_pct")
        eri_trend = eri_traj["trend"]
        
        if eri_delta_pct is not None:
            parts.append(
                f"Sentiment moved from ERI {eri_start} to {eri_end} "
                f"(Δ={eri_delta}, {eri_delta_pct:+.1f}%), classified as {eri_trend}."
            )
        else:
            parts.append(
                f"Sentiment moved from ERI {eri_start} to {eri_end} "
                f"(Δ={eri_delta}), classified as {eri_trend}."
            )

        # 2) EVI movement (ENHANCED with % change)
        evi_start = evi_traj["start"]
        evi_end = evi_traj["end"]
        evi_delta = evi_traj["delta"]
        evi_delta_pct = evi_traj.get("delta_pct")
        evi_trend = evi_traj["trend"]
        
        if evi_delta_pct is not None:
            parts.append(
                f"Volatility (EVI) shifted from {evi_start} to {evi_end} "
                f"(Δ={evi_delta}, {evi_delta_pct:+.1f}%), indicating {evi_trend}."
            )
        else:
            parts.append(
                f"Volatility (EVI) shifted from {evi_start} to {evi_end} "
                f"(Δ={evi_delta}), indicating {evi_trend}."
            )

        # 3) ES Band migration
        if es_mig:
            parts.append(
                f"Relationship state moved from {es_mig['start']} to {es_mig['end']} "
                f"({es_mig['trajectory']})."
            )

        # 4) NEW: Polarity flow direction
        if emotion_flow and emotion_flow.get("polarity_shift"):
            polarity = emotion_flow["polarity_shift"]
            pol_direction = polarity.get("polarity_flow_direction")
            
            if pol_direction and pol_direction != "Stable":
                parts.append(
                    f"Emotional polarity shifted: {pol_direction} "
                    f"(Positive: {polarity['positive_early_pct']}%→{polarity['positive_recent_pct']}%, "
                    f"Negative: {polarity['negative_early_pct']}%→{polarity['negative_recent_pct']}%)."
                )

        # 5) NEW: Dominant emotion transition
        if emotion_flow:
            dom_trans = emotion_flow.get("dominant_emotion_transition")
            if dom_trans and "→" in dom_trans:  # Only if it changed
                parts.append(
                    f"Dominant emotion transitioned: {dom_trans}."
                )

        # 6) Per-emotion shifts (existing)
        if emotion_flow and emotion_flow.get("dominant_shift"):
            dom = emotion_flow["dominant_shift"]
            parts.append(
                f"The largest individual emotion shift was {dom['emotion']} "
                f"(Δ={dom['delta']} percentage points; {dom['trend']})."
            )

        # 7) Ambivalence flag (existing)
        if emotion_flow and emotion_flow.get("ambivalence_type") in {"true", "false"}:
            ambi = emotion_flow["ambivalence_type"]
            if ambi == "true":
                parts.append(
                    "Ambivalence appears to be TRUE (genuinely neutral, low-volatility emotion)."
                )
            else:
                parts.append(
                    "Ambivalence appears to be FALSE (neutral ERI masking a high-volatility mix of positive and negative emotions)."
                )

        # 8) Named pattern (existing)
        if pattern_flow and pattern_flow.get("pattern_label"):
            parts.append(f"Overall temporal pattern: {pattern_flow['pattern_label']}.")

        return " ".join(parts) if parts else "Temporal signals are present but inconclusive."

    # ----------- Ambivalence Type -----------
    def _classify_ambivalence_type(
            self,
            combined_dist: dict,
            eri_traj: dict | None,
            evi_traj: dict | None,
        ) -> str:
            """
            Coarse classification of ambivalence:

            - "true"  → genuinely neutral, low volatility, high Ambivalence emotion
            - "false" → ERI near 0 but high volatility with strong positive+negative mix
            - "none"  → no meaningful ambivalence pattern

            This uses only existing signals:
              - combined emotion distribution
              - ERI trajectory end point
              - EVI trajectory end point
            """
            if not combined_dist or not eri_traj or not evi_traj:
                return "none"

            eri_end = eri_traj.get("end", 0.0)
            evi_end = evi_traj.get("end", 0.0)

            # Overall mix
            ambiv_pct = combined_dist.get("Ambivalence", 0.0)
            pos_pct = (
                combined_dist.get("Adoration", 0.0)
                + combined_dist.get("Appreciation", 0.0)
            )
            neg_pct = (
                combined_dist.get("Agitation", 0.0)
                + combined_dist.get("Anger", 0.0)
            )

            # Heuristics grounded in your existing constants:
            # "Near neutral" ERI
            is_eri_neutral = abs(eri_end) <= getattr(
                self, "PATTERN_ERI_STABLE_RANGE", 10
            )

            # Low vs high volatility (EVI)
            is_low_vol = evi_end < getattr(self, "EVI_LOW_MIN", 20)
            is_high_vol = evi_end >= getattr(self, "EVI_HIGH_MIN", 40)

            # Strong positive + negative presence
            strong_posneg = (pos_pct >= 30.0) and (neg_pct >= 30.0)

            # True ambivalence: genuinely neutral, low volatility, high Ambivalence
            if is_eri_neutral and is_low_vol and ambiv_pct >= 40.0 and not strong_posneg:
                return "true"

            # False ambivalence: neutral ERI but high volatility with strong pos+neg mix
            if is_eri_neutral and is_high_vol and strong_posneg:
                return "false"

            return "none"

    # ----------- Emotional State Band Migration -----------
    def _calculate_es_band_migration(
    self, by_tier: dict, recent_tiers: list, early_tiers: list) -> dict:
        """
        Calculate Emotional State Band migration pattern.
        
        Returns:
            dict with keys: start, end, trajectory, changed
            OR None if insufficient data
        """
        recent_es_bands = [
            by_tier.get(t, {}).get("ES_Band")
            for t in recent_tiers
            if by_tier.get(t, {}).get("ES_Band") is not None
        ]
        early_es_bands = [
            by_tier.get(t, {}).get("ES_Band")
            for t in early_tiers
            if by_tier.get(t, {}).get("ES_Band") is not None
        ]

        if not recent_es_bands or not early_es_bands:
            return None

        es_start = Counter(early_es_bands).most_common(1)[0][0]
        es_end = Counter(recent_es_bands).most_common(1)[0][0]

        return {
            "start": es_start,
            "end": es_end,
            "trajectory": f"{es_start} → {es_end}" if es_start != es_end else es_start,
            "changed": es_start != es_end,
        }

    # ----------- Emotion Momentum -----------
    def _calculate_emotion_momentum(
    self, by_tier: dict, recent_tiers: list, early_tiers: list) -> dict:

        """
        Calculate emotion momentum (percentage point shifts).
        
        Returns:
            dict with keys: shifts (dict), dominant_shift (dict)
        """
        early_emot_dist = self._aggregate_emotions_from_tiers(by_tier, early_tiers)
        recent_emot_dist = self._aggregate_emotions_from_tiers(by_tier, recent_tiers)

        all_emotions = set(early_emot_dist.keys()) | set(recent_emot_dist.keys())
        emotion_shifts = {}
        
        for em in all_emotions:
            old_pct = early_emot_dist.get(em, 0.0)
            new_pct = recent_emot_dist.get(em, 0.0)
            delta = round(new_pct - old_pct, 2)

            # Determine trend using constants
            if delta >= self.EMOTION_STRONG_RISE_PCT:
                trend = "↑↑ Strongly Emerging"
            elif delta >= self.EMOTION_RISE_PCT:
                trend = "↑ Rising"
            elif delta >= self.EMOTION_MINOR_RISE_PCT:
                trend = "↑ Minor Rise"
            elif delta <= self.EMOTION_STRONG_FADE_PCT:
                trend = "↓↓ Rapidly Fading"
            elif delta <= self.EMOTION_FADE_PCT:
                trend = "↓ Declining"
            elif delta <= self.EMOTION_MINOR_FADE_PCT:
                trend = "↓ Minor Decline"
            else:
                trend = "→ Stable"

            emotion_shifts[em] = {
                "old_pct": old_pct,
                "new_pct": new_pct,
                "delta": delta,
                "trend": trend,
            }

        # Find dominant shift
        if emotion_shifts:
            max_shift = max(emotion_shifts.items(), key=lambda x: abs(x[1]["delta"]))
            dominant_shift = {
                "emotion": max_shift[0],
                "delta": max_shift[1]["delta"],
                "trend": max_shift[1]["trend"],
            }
        else:
            dominant_shift = None

        return {
            "shifts": emotion_shifts,
            "dominant_shift": dominant_shift,
        }

    # -----------Temporal Intelligence Helper Methods-----------
    def _get_tier_values(self, by_tier: dict, tier_list: list, key: str) -> list:
        """
        Extract non-null values for a given key from specified tiers.
        
        Args:
            by_tier: Dictionary of tier data
            tier_list: List of tier names to extract from
            key: Key to extract (e.g., "ERI", "EVI")
        
        Returns:
            List of non-null values
        """
        vals = []
        for t in tier_list:
            tier_info = by_tier.get(t, {})
            val = tier_info.get(key)
            if val is not None:
                vals.append(val)
        return vals
    
    def _eri_tier_index(self, eri_value: float) -> int:
        """
        Map ERI value to tier index using defined boundaries.
        
        Returns:
            0 (Very Negative), 1 (Negative), 2 (Neutral), 3 (Positive), 4 (Very Positive)
        """
        if eri_value < self.ERI_NEGATIVE_MIN:
            return 0  # Very Negative
        elif eri_value < self.ERI_NEUTRAL_MIN:
            return 1  # Negative
        elif eri_value < self.ERI_POSITIVE_MIN:
            return 2  # Neutral
        elif eri_value < self.ERI_VERY_POSITIVE_MIN:
            return 3  # Positive
        else:
            return 4  # Very Positive

    def _evi_tier_index(self, evi_value: float) -> int:
        """
        Map EVI value to tier index using defined boundaries.
        
        Returns:
            0 (Stable), 1 (Low Vol), 2 (Moderate Vol), 3 (High Vol), 4 (Extreme Vol)
        """
        if evi_value < self.EVI_LOW_MIN:
            return 0  # Stable Emotion
        elif evi_value < self.EVI_MODERATE_MIN:
            return 1  # Low Volatility
        elif evi_value < self.EVI_HIGH_MIN:
            return 2  # Moderate Volatility
        elif evi_value < self.EVI_EXTREME_MIN:
            return 3  # High Volatility
        else:
            return 4  # Extreme Volatility

    def _aggregate_emotions_from_tiers(self, by_tier: dict, tier_list: list) -> dict:
        """
        Aggregate emotion percentages across specified tiers.
        
        Returns:
            Dict of {emotion: percentage}
        """
        total = 0
        counts = {}
        
        for t in tier_list:
            tier_emots = by_tier.get(t, {}).get("emotions", {})
            for em, data in tier_emots.items():
                ct = int(data.get("count", 0))
                counts[em] = counts.get(em, 0) + ct
                total += ct
        
        if total == 0:
            return {}
        
        return {em: round(100.0 * ct / total, 2) for em, ct in counts.items()}
      
    # ------------ Temporal Intelligence Interpretation -----------
    def _interpret_temporal_pattern(self, eri_traj, evi_traj, es_mig, emotion_flow) -> str:
        """
        Generate human-readable interpretation of temporal intelligence.
        Prioritizes ES Band migration patterns as the primary signal.

        Priority 1: ES Band migration (most semantically meaningful)
        Priority 2: Combined ERI + EVI numeric patterns (fallback)

        COMPLETE ES migration matrix — all 20 valid directed pairs covered:

        Deterioration (5):
            ES1→ES4  ES1→ES5  ES2→ES4  ES2→ES5  ES3→ES4  ES3→ES5

        Improvement (9):
            ES4→ES1  ES4→ES2  ES4→ES3  ← 🔥 NEW: ES4→ES3
            ES5→ES1  ES5→ES2  ← 🔥 NEW: ES5→ES1, ES5→ES2
            ES5→ES3  ES5→ES4
            ES3→ES1  ES3→ES2

        Lateral (6):
            ES1→ES2  ES2→ES1
            ES1→ES3  ← 🔥 NEW: ES1→ES3
            ES2→ES3  ← 🔥 NEW: ES2→ES3
            ES3→ES4 (covered in deterioration)
            ES4→ES5  (covered in deterioration)

        🔥 FIX: PATTERN_ERI_CRISIS_THRESHOLD sign — was -20 (double negation bug),
        now +20 so comparisons correctly fire only when ERI drops 20+ points.
        """
        if not eri_traj or not evi_traj:
            return "Insufficient temporal data for temporal pattern interpretation."

        eri_delta = eri_traj["delta"]
        evi_delta = evi_traj["delta"]
        eri_end   = eri_traj["end"]

        # ── PRIORITY 1: ES Band Migration ────────────────────────────────
        if es_mig and es_mig.get("changed"):
            es_start = es_mig["start"]
            es_end   = es_mig["end"]

            # --- Deterioration ---
            if "ES1" in es_start and "ES4" in es_end:
                return "RELATIONSHIP COLLAPSE: Secure Loyalty → Active Crisis (rapid deterioration)"
            if "ES1" in es_start and "ES5" in es_end:
                return "CATASTROPHIC FAILURE: Secure Loyalty → Critical Failure (complete breakdown)"
            if "ES2" in es_start and "ES4" in es_end:
                return "GROWTH DERAILED: Growth Opportunity → Active Crisis (volatility turned negative)"
            if "ES2" in es_start and "ES5" in es_end:
                return "ENGAGEMENT COLLAPSE: Growth Opportunity → Critical Failure (complete reversal)"
            if "ES3" in es_start and "ES4" in es_end:
                return "RISK ACTUALIZED: At Risk → Active Crisis (disengagement turned to crisis)"
            if "ES3" in es_start and "ES5" in es_end:
                return "CHURN ACTUALIZED: At Risk → Critical Failure (silent deterioration)"

            # --- Improvement ---
            if "ES4" in es_start and "ES1" in es_end:
                return "FULL RECOVERY: Active Crisis → Secure Loyalty (successful intervention)"
            if "ES4" in es_start and "ES2" in es_end:
                return "CRISIS STABILIZING: Active Crisis → Growth Opportunity (containment working)"
            if "ES4" in es_start and "ES3" in es_end:                                      # 🔥 NEW
                return "CRISIS DE-ESCALATING: Active Crisis → At Risk (situation improving)"
            if "ES5" in es_start and "ES1" in es_end:                                      # 🔥 NEW
                return "FULL TURNAROUND: Critical Failure → Secure Loyalty (remarkable recovery)"
            if "ES5" in es_start and "ES2" in es_end:                                      # 🔥 NEW
                return "STRONG RECOVERY: Critical Failure → Growth Opportunity (significant improvement)"
            if "ES5" in es_start and "ES3" in es_end:
                return "STABILIZATION: Critical Failure → At Risk (containment working)"
            if "ES5" in es_start and "ES4" in es_end:
                return "MINOR IMPROVEMENT: Critical Failure → Active Crisis (still critical)"
            if "ES3" in es_start and "ES1" in es_end:
                return "RE-ENGAGEMENT SUCCESS: At Risk → Secure Loyalty (relationship rebuilt)"
            if "ES3" in es_start and "ES2" in es_end:
                return "POSITIVE SHIFT: At Risk → Growth Opportunity (re-engagement emerging)"

            # --- Lateral ---
            if "ES1" in es_start and "ES2" in es_end:
                return "VOLATILITY EMERGING: Secure Loyalty → Growth Opportunity (emotions destabilizing)"
            if "ES2" in es_start and "ES1" in es_end:
                return "STABILIZATION: Growth Opportunity → Secure Loyalty (volatility resolved)"
            if "ES1" in es_start and "ES3" in es_end:                                      # 🔥 NEW
                return "SILENT DRIFT: Secure Loyalty → At Risk (quiet disengagement emerging)"
            if "ES2" in es_start and "ES3" in es_end:                                      # 🔥 NEW
                return "GROWTH STALLING: Growth Opportunity → At Risk (momentum lost)"
            if "ES4" in es_start and "ES5" in es_end:
                return "CRISIS HARDENING: Active Crisis → Critical Failure (situation worsening)"

            # Safety net: changed=True but no pair matched (should not occur
            # given exhaustive matrix above, but guard against future ES band additions)
            return f"TRANSITION DETECTED: {es_start} → {es_end} (review manually)"

        # ── PRIORITY 2: Combined ERI + EVI numeric patterns ──────────────
        # 🔥 FIX: PATTERN_ERI_CRISIS_THRESHOLD is now +20 (was -20).
        # Comparisons use -self.PATTERN_ERI_CRISIS_THRESHOLD = -20, which
        # correctly fires only when ERI drops 20+ points.

        # Deterioration patterns
        if eri_delta < -self.PATTERN_ERI_CRISIS_THRESHOLD and evi_delta > self.PATTERN_EVI_FRAGMENT_THRESHOLD:
            return "CRISIS PATTERN: Sentiment collapsing while emotions fragmenting rapidly."

        if eri_delta < -self.PATTERN_ERI_CRISIS_THRESHOLD and evi_delta < self.PATTERN_EVI_CONSOLIDATE_THRESHOLD:
            return "DANGER PATTERN: Negative consolidation - dissatisfaction hardening."

        # Improvement patterns
        if eri_delta > self.PATTERN_ERI_RECOVERY_THRESHOLD and evi_delta < self.PATTERN_EVI_CONSOLIDATE_THRESHOLD:
            return "RECOVERY PATTERN: Sentiment improving and emotions stabilizing."

        if eri_delta > self.PATTERN_ERI_RECOVERY_THRESHOLD and evi_delta > self.PATTERN_EVI_CONSOLIDATE_THRESHOLD:
            return "VOLATILE GROWTH: Sentiment improving but emotional consistency lacking."

        # Stagnation patterns
        if abs(eri_delta) < self.PATTERN_ERI_STABLE_RANGE and abs(evi_delta) < self.PATTERN_EVI_STABLE_RANGE:
            if eri_end > self.PATTERN_ERI_POSITIVE_THRESHOLD:
                return "STABLE LOYALTY: Consistently positive and emotionally stable."
            elif eri_end < self.PATTERN_ERI_NEGATIVE_THRESHOLD:
                return "ENTRENCHED NEGATIVITY: Stable dissatisfaction with churn risk."
            else:
                return "STAGNANT RELATIONSHIP: Neither improving nor declining meaningfully."

        # Chaos patterns
        if evi_delta > self.PATTERN_EVI_CHAOS_THRESHOLD:
            return "FRAGMENTING RELATIONSHIP: Emotional complexity and instability rising."

        return "Mixed or ambiguous temporal signals - review detailed tier-level data."
