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


    def _classify_emotional_state_band(self, eri_tier, evi_tier):
        """
        Returns:
            ES1_Secure_Loyalty
            ES2_Growth_Opportunity
            ES3_At_Risk
            ES4_Active_Crisis
            ES5_Critical_Failure
        
        Revised Logic:
        - Neutral + Stable/Low Vol → ES3 (stagnation = risk, not growth)
        - Positive + High/Extreme Vol → ES2 (engaged but inconsistent, not yet at-risk)
        - Negative + Stable/Low Vol → ES4 (entrenched negativity is worse than volatile)
        """

        et = eri_tier
        vt = evi_tier

        # ES1 – Secure Loyalty
        # Happy + consistent = advocacy foundation
        if et in {"Very Positive", "Positive"} and vt in {"Stable Emotion", "Low Volatility"}:
            return "ES1_Secure_Loyalty"

        # ES2 – Growth Opportunity
        # Positive with some volatility = engaged, room to stabilize
        # High/Extreme volatility on positive sentiment = they care, just inconsistent
        if et in {"Very Positive", "Positive"} and vt in {"Moderate Volatility", "High Volatility", "Extreme Volatility"}:
            return "ES2_Growth_Opportunity"

        # ES5 – Critical Failure
        # Very Negative always = critical (polarity dominates)
        # Negative + Extreme Volatility = chaotic negativity
        if et == "Very Negative":
            return "ES5_Critical_Failure"
        if et == "Negative" and vt == "Extreme Volatility":
            return "ES5_Critical_Failure"

        # ES4 – Active Crisis
        # Negative + Stable/Low/Moderate/High Vol = entrenched dissatisfaction
        # (Stable negativity is arguably WORSE than volatile - no recovery signals)
        # Neutral + Extreme Volatility = chaotic indifference, crisis territory
        if et == "Negative" and vt in {"Stable Emotion", "Low Volatility", "Moderate Volatility", "High Volatility"}:
            return "ES4_Active_Crisis"
        if et == "Neutral" and vt == "Extreme Volatility":
            return "ES4_Active_Crisis"

        # ES3 – At Risk
        # Neutral + Stable/Low/Moderate/High Vol = disengaged, stagnant, churn risk
        # (Previously Neutral+Stable was ES2, but stable indifference ≠ growth opportunity)
        if et == "Neutral" and vt in {"Stable Emotion", "Low Volatility", "Moderate Volatility", "High Volatility"}:
            return "ES3_At_Risk"

        # Fallback
        return "ES3_At_Risk"


    PRIORITY_GRID = {
        # ES1: Happy customers - lower priority for intervention, higher for advocacy programs
        # Reduced priority inflation for active happy customers (they don't need rescue)
        "ES1_Secure_Loyalty": {
            "Dormant":  "P5",
            "Low":      "P5",
            "Moderate": "P4",
            "High":     "P4",
            "Very High":"P3",  # Was P2 - these are advocates, not emergencies
        },
        # ES2: Growth potential - moderate attention to nurture
        "ES2_Growth_Opportunity": {
            "Dormant":  "P5",
            "Low":      "P4",
            "Moderate": "P3",
            "High":     "P3",
            "Very High":"P2",
        },
        # ES3: At Risk - silence is a warning sign, bump dormant priority
        "ES3_At_Risk": {
            "Dormant":  "P3",  # Was P4 - silent at-risk = churn signal
            "Low":      "P3",
            "Moderate": "P2",
            "High":     "P1",
            "Very High":"P0",
        },
        # ES4: Active Crisis - dormant crisis = likely churning, needs attention
        "ES4_Active_Crisis": {
            "Dormant":  "P2",  # Was P3 - crisis + silence = urgent
            "Low":      "P2",
            "Moderate": "P1",
            "High":     "P0",  # Was P1 - active loud crisis = immediate
            "Very High":"P0",
        },
        # ES5: Critical Failure - dormant may be lost already, deprioritize vs recoverable
        "ES5_Critical_Failure": {
            "Dormant":  "P3",  # Was P2 - likely already churned, focus elsewhere
            "Low":      "P2",
            "Moderate": "P1",
            "High":     "P0",
            "Very High":"P0",
        },
    }

    def _classify_priority(self, emotional_state_band, rf_tier_short):
        return self.PRIORITY_GRID.get(emotional_state_band, {}).get(rf_tier_short, "P3")

    PURPOSE_MATRIX = {
        # ES1 - Secure Loyalty: Focus on advocacy, not intervention
        ("ES1_Secure_Loyalty", "P5"): "Monitor Stable Advocacy",
        ("ES1_Secure_Loyalty", "P4"): "Maintain Positive Momentum",
        ("ES1_Secure_Loyalty", "P3"): "Activate Advocacy Programs",  # Renamed - these are advocates to leverage

        # ES2 - Growth Opportunity: Nurture and stabilize
        ("ES2_Growth_Opportunity", "P5"): "Background Growth Watch",
        ("ES2_Growth_Opportunity", "P4"): "Develop Emerging Loyalty",
        ("ES2_Growth_Opportunity", "P3"): "Strengthen Growth Drivers",
        ("ES2_Growth_Opportunity", "P2"): "Activate Growth Levers",

        # ES3 - At Risk: Investigate and intervene before churn
        ("ES3_At_Risk", "P3"): "Investigate Silent Risk",  # New - dormant at-risk
        ("ES3_At_Risk", "P2"): "Address Early Risks",
        ("ES3_At_Risk", "P1"): "Contain At-Risk Emotion",
        ("ES3_At_Risk", "P0"): "Rapid Recovery Action",

        # ES4 - Active Crisis: Immediate containment
        ("ES4_Active_Crisis", "P2"): "Crisis Containment",
        ("ES4_Active_Crisis", "P1"): "Active Crisis Intervention",
        ("ES4_Active_Crisis", "P0"): "Emergency Crisis Response",

        # ES5 - Critical Failure: Triage - focus on recoverable cases
        ("ES5_Critical_Failure", "P3"): "Evaluate Recovery Viability",  # New - dormant critical
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
            rf_tier_short = normalize_rf_tier(rf_tier_long)  # "High", "Moderate", etc.

            # --- Priority via Emotional_State × RF --------------------------------
            priority_tier = self._classify_priority(emotional_state_band, rf_tier_short)
            priority_purpose = self._classify_purpose(emotional_state_band, priority_tier)
            priority_score = p_score.get(priority_tier, 0)

            # --- Emotion Recency Profile ------------------------------------
            # Recency-of-emotion distribution for this experience_driver
            emotion_recency_profile = self._compute_emotion_recency_profile(group)

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

                    # NEW: Emotion Recency Profile (dict: by_emotion + by_tier)
                    "Emotion_Recency_Profile": emotion_recency_profile,

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

    def _map_emotion_recency_tier(self, age_days, window_days):
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

    def _compute_emotion_recency_profile(self, group: pd.DataFrame):
        """
        For a single experience_driver group:
        Returns:
        1. Distribution of each emotion across all recency tiers
        2. FULL emotion distribution within each tier
        3. ERI per tier (normalized -100..+100) using _compute_eri_from_counts
        4. EVI per tier (0..100) using _compute_evi_from_counts → _compute_evi
        5. ES_Band per tier using existing decision matrix
        6. Temporal intelligence (auto-derived from tier comparisons),
            with guard rails for sparse / incomplete tier coverage.
        """

        # Basic safety checks
        if "date" not in group.columns or "emotion_primary" not in group.columns:
            return {}

        sub = group.dropna(subset=["date"])
        if sub.empty:
            return {}

        # Compute age in days and map to recency tiers
        ages = (self.today - sub["date"]).dt.days.clip(lower=0)

        tmp = pd.DataFrame({
            "emotion": sub["emotion_primary"],
            "age_days": ages,
        })
        tmp["tier"] = tmp["age_days"].apply(
            lambda x: self._map_emotion_recency_tier(x, self.window_days)
        )

        all_tiers = ["Tier_1", "Tier_2", "Tier_3", "Tier_4", "Tier_5", "Beyond_Window"]

        # --- Part 1: By Emotion (how each emotion is spread across tiers) ---
        by_emotion = {}
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
                "total_count": total,
                "tiers": tiers_dist,
            }

        # --- Part 2: By Tier (FULL emotion distribution + ERI/EVI/ES_Band per tier) ---
        by_tier = {}
        for t in all_tiers:
            tier_data = tmp[tmp["tier"] == t]
            total_in_tier = len(tier_data)

            if total_in_tier == 0:
                by_tier[t] = {
                    "total_mentions": 0,
                    "emotions": {},
                    "dominant_emotion": None,
                    "dominant_count": 0,
                    "dominant_pct": 0.0,
                    "runner_up": None,
                    # Tier-level intelligence (null for empty tiers)
                    "ERI": None,
                    "ERI_Tier": None,
                    "EVI": None,
                    "EVI_Tier": None,
                    "ES_Band": None,
                }
                continue

            # Emotion counts in this tier
            counts = tier_data["emotion"].value_counts()
            total_in_tier = int(counts.sum())
            tier_emotion_counts = counts.to_dict()

            # Build emotions distribution
            emotions_dist = {}
            for emotion, ct in tier_emotion_counts.items():
                ct_int = int(ct)
                emotions_dist[emotion] = {
                    "count": ct_int,
                    "pct": round(100.0 * ct_int / max(1, total_in_tier), 2),
                }

            # Dominant / runner_up for quick reference
            dominant = counts.index[0]
            dominant_ct = int(counts.iloc[0])
            runner_up = counts.index[1] if len(counts) > 1 else None

            # ERI for this tier (DRY helper)
            eri_value = self._compute_eri_from_counts(tier_emotion_counts, total_in_tier)
            eri_tier_label = self._map_loyalty_tier(eri_value)

            # EVI for this tier (DRY wrapper -> _compute_evi)
            evi_value = self._compute_evi_from_counts(tier_emotion_counts)
            evi_tier_label = self._map_evi_tier(evi_value)

            # ES Band from existing decision matrix
            es_band_tier = self._classify_emotional_state_band(eri_tier_label, evi_tier_label)

            by_tier[t] = {
                "total_mentions": total_in_tier,
                "emotions": emotions_dist,
                "dominant_emotion": dominant,
                "dominant_count": dominant_ct,
                "dominant_pct": round(100.0 * dominant_ct / max(1, total_in_tier), 2),
                "runner_up": runner_up,

                # Tier-level intelligence
                "ERI": round(eri_value, 2),
                "ERI_Tier": eri_tier_label,
                "EVI": round(evi_value, 2),
                "EVI_Tier": evi_tier_label,
                "ES_Band": es_band_tier,
            }

        # --- Part 3: Temporal Intelligence (with guard rails) ---
        temporal_intel = self._derive_temporal_intelligence_from_tiers(by_tier, all_tiers)

        return {
            "by_emotion": by_emotion,
            "by_tier": by_tier,
            "temporal_intelligence": temporal_intel,
        }


    def _derive_temporal_intelligence_from_tiers(self, by_tier: dict, all_tiers: list) -> dict:
        """
        AUTO-GENERATE TEMPORAL INTELLIGENCE from tier-level data.

        Compares early tiers (Tier_4, Tier_5) vs recent tiers (Tier_1, Tier_2)
        to derive trajectory, momentum, and consolidation patterns.

        Guard rails:
        - Works even if some tiers are empty
        - Returns None for specific blocks when data is insufficient
        - Never crashes on empty emotion shifts
        """
        # Define early vs recent tier groups
        recent_tiers = ["Tier_1", "Tier_2"]
        early_tiers = ["Tier_4", "Tier_5"]

        # Helper: extract non-null values from tiers
        def get_values(tier_list, key):
            vals = []
            for t in tier_list:
                tier_info = by_tier.get(t, {})
                val = tier_info.get(key)
                if val is not None:
                    vals.append(val)
            return vals

        # --- ERI Trajectory ---
        recent_eris = get_values(recent_tiers, "ERI")
        early_eris = get_values(early_tiers, "ERI")

        if recent_eris and early_eris:
            eri_start = round(float(np.mean(early_eris)), 2)
            eri_end = round(float(np.mean(recent_eris)), 2)
            eri_delta = round(eri_end - eri_start, 2)

            if eri_delta >= 20:
                eri_trend = "↑↑ Strongly Improving"
            elif eri_delta >= 5:
                eri_trend = "↑ Improving"
            elif eri_delta <= -20:
                eri_trend = "↓↓ Strongly Declining"
            elif eri_delta <= -5:
                eri_trend = "↓ Declining"
            else:
                eri_trend = "→ Stable"

            eri_trajectory = {
                "start": eri_start,
                "end": eri_end,
                "delta": eri_delta,
                "trend": eri_trend,
                "by_tier": {t: by_tier[t].get("ERI") for t in all_tiers},
            }
        else:
            eri_trajectory = None

        # --- EVI Trajectory ---
        recent_evis = get_values(recent_tiers, "EVI")
        early_evis = get_values(early_tiers, "EVI")

        if recent_evis and early_evis:
            evi_start = round(float(np.mean(early_evis)), 2)
            evi_end = round(float(np.mean(recent_evis)), 2)
            evi_delta = round(evi_end - evi_start, 2)

            if evi_delta >= 20:
                evi_trend = "↑↑ Rapidly Fragmenting"
            elif evi_delta >= 10:
                evi_trend = "↑ Fragmenting"
            elif evi_delta <= -20:
                evi_trend = "↓↓ Rapidly Consolidating"
            elif evi_delta <= -10:
                evi_trend = "↓ Consolidating"
            else:
                evi_trend = "→ Stable Volatility"

            evi_trajectory = {
                "start": evi_start,
                "end": evi_end,
                "delta": evi_delta,
                "trend": evi_trend,
                "by_tier": {t: by_tier[t].get("EVI") for t in all_tiers},
            }
        else:
            evi_trajectory = None

        # --- ES Band Migration ---
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

        if recent_es_bands and early_es_bands:
            from collections import Counter

            es_start = Counter(early_es_bands).most_common(1)[0][0]
            es_end = Counter(recent_es_bands).most_common(1)[0][0]

            es_migration = {
                "start": es_start,
                "end": es_end,
                "trajectory": f"{es_start} → {es_end}" if es_start != es_end else es_start,
                "changed": es_start != es_end,
            }
        else:
            es_migration = None

        # --- Emotion Momentum (which emotions gained/lost share) ---
        def aggregate_emotions(tier_list):
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

        early_emot_dist = aggregate_emotions(early_tiers)
        recent_emot_dist = aggregate_emotions(recent_tiers)

        all_emotions = set(early_emot_dist.keys()) | set(recent_emot_dist.keys())
        emotion_shifts = {}
        for em in all_emotions:
            old_pct = early_emot_dist.get(em, 0.0)
            new_pct = recent_emot_dist.get(em, 0.0)
            delta = round(new_pct - old_pct, 2)

            if delta >= 15:
                trend = "↑↑ Strongly Emerging"
            elif delta >= 5:
                trend = "↑ Rising"
            elif delta <= -15:
                trend = "↓↓ Rapidly Fading"
            elif delta <= -5:
                trend = "↓ Declining"
            else:
                trend = "→ Stable"

            emotion_shifts[em] = {
                "old_pct": old_pct,
                "new_pct": new_pct,
                "delta": delta,
                "trend": trend,
            }

        if emotion_shifts:
            max_shift = max(emotion_shifts.items(), key=lambda x: abs(x[1]["delta"]))
            dominant_shift = {
                "emotion": max_shift[0],
                "delta": max_shift[1]["delta"],
                "trend": max_shift[1]["trend"],
            }
        else:
            dominant_shift = None

        emotion_momentum = {
            "shifts": emotion_shifts,
            "dominant_shift": dominant_shift,
        }

        # --- Overall interpretation (guarded) ---
        interpretation = self._interpret_temporal_pattern(
            eri_trajectory, evi_trajectory, es_migration, emotion_momentum
        )

        return {
            "ERI_trajectory": eri_trajectory,
            "EVI_trajectory": evi_trajectory,
            "ES_Band_migration": es_migration,
            "emotion_momentum": emotion_momentum,
            "interpretation": interpretation,
        }

    def _interpret_temporal_pattern(self, eri_traj, evi_traj, es_mig, emot_mom) -> str:
        """
        Generate human-readable interpretation of temporal intelligence.
        Guarded: returns a safe message if trajectories are missing.
        """
        if not eri_traj or not evi_traj:
            return "Insufficient temporal data for temporal pattern interpretation."

        eri_delta = eri_traj["delta"]
        evi_delta = evi_traj["delta"]
        eri_end = eri_traj["end"]

        # Deterioration patterns
        if eri_delta < -20 and evi_delta > 15:
            return "CRISIS PATTERN: Sentiment collapsing while emotions are fragmenting rapidly."

        if eri_delta < -20 and evi_delta < -10:
            return "DANGER PATTERN: Negative consolidation – dissatisfaction is hardening."

        # Improvement patterns
        if eri_delta > 20 and evi_delta < -10:
            return "RECOVERY PATTERN: Sentiment improving and emotions stabilizing."

        if eri_delta > 20 and evi_delta > 10:
            return "VOLATILE GROWTH: Sentiment improving but emotional consistency is lacking."

        # Stagnation patterns
        if abs(eri_delta) < 5 and abs(evi_delta) < 5:
            if eri_end > 30:
                return "STABLE LOYALTY: Consistently positive and emotionally stable."
            elif eri_end < -30:
                return "ENTRENCHED NEGATIVITY: Stable dissatisfaction with churn risk."
            else:
                return "STAGNANT RELATIONSHIP: Neither improving nor declining in a meaningful way."

        # Chaos patterns
        if evi_delta > 20:
            return "FRAGMENTING RELATIONSHIP: Emotional complexity and instability are rising."

        return "Mixed or ambiguous temporal signals – review detailed tier-level data."
