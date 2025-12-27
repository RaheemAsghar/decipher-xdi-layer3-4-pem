def compute(self):
    """
    Layer 2 (ERI + EVI + RF) - DUAL ARCHITECTURE VERSION
    
    Computes:
    1. GLOBAL view: SEU for each experience_driver across all customers
    2. SEGMENT view: For EACH driver, SEU breakdown by customer_tier
    
    Key logic:
    - ERI/EVI: Calculated from segment's emotions only
    - R: Based on segment's most recent mention date
    - F: Normalized against GLOBAL mention count for THAT DRIVER
    
    Output: Single unified dataframe with Cohort_Type column
    """
    
    # Date normalization already done in __init__
    ts_col = "date"
    today_ts = self.today
    
    # ========================================================================
    # STEP 0: VALIDATE CUSTOMER TIER COLUMN
    # ========================================================================
    has_customer_tier = "customer_tier" in self.df.columns
    
    if not has_customer_tier:
        print("⚠️  'customer_tier' column not found in dataframe.")
        print("   Will compute GLOBAL view only. Add 'customer_tier' column for segment stratification.")
    else:
        tier_counts = self.df["customer_tier"].value_counts()
        print(f"✅ Found 'customer_tier' column with {len(tier_counts)} tiers:")
        print(f"   {dict(tier_counts)}")
    
    # ========================================================================
    # STEP 1: CALCULATE GLOBAL METRICS FOR NORMALIZATION
    # ========================================================================
    total_rows = len(self.df)
    driver_mention_counts = self.df["experience_driver"].value_counts().to_dict()
    
    # ========================================================================
    # STEP 2: COMPUTE GLOBAL VIEW (all customers)
    # ========================================================================
    print("\n" + "="*80)
    print("COMPUTING GLOBAL PROBLEM MANAGEMENT VIEW")
    print("="*80)
    
    all_results = []
    
    for driver, driver_group in self.df.groupby("experience_driver", dropna=False):
        # Global mention count for THIS driver (used for F normalization in all segments)
        driver_global_mentions = driver_mention_counts.get(driver, 0)
        
        # Compute GLOBAL SEU for this driver
        global_seu = self._compute_seu_for_group(
            group=driver_group,
            driver_name=driver,
            cohort_label="GLOBAL",
            driver_global_mentions=driver_global_mentions,
            context_total_rows=total_rows
        )
        all_results.append(global_seu)
    
    print(f"   Computed {len(all_results)} global driver SEU records")
    
    # ========================================================================
    # STEP 3: COMPUTE SEGMENT VIEWS (if customer_tier exists)
    # ========================================================================
    if has_customer_tier:
        print("\n" + "="*80)
        print("COMPUTING SEGMENT-STRATIFIED VIEWS")
        print("="*80)
        
        # For each driver, break down by customer_tier
        for driver, driver_group in self.df.groupby("experience_driver", dropna=False):
            driver_global_mentions = driver_mention_counts.get(driver, 0)
            
            # Get segment-level total rows (all rows for this driver, used for mention_share calculation)
            driver_total_rows = len(driver_group)
            
            # Group by tier within this driver
            for tier, tier_group in driver_group.groupby("customer_tier", dropna=False):
                segment_seu = self._compute_seu_for_group(
                    group=tier_group,
                    driver_name=driver,
                    cohort_label=str(tier),
                    driver_global_mentions=driver_global_mentions,
                    context_total_rows=driver_total_rows  # Segment share relative to driver total
                )
                all_results.append(segment_seu)
        
        segment_count = len(all_results) - len(driver_mention_counts)
        print(f"   Computed {segment_count} segment-level SEU records")
    
    # ========================================================================
    # STEP 4: BUILD UNIFIED DATAFRAME
    # ========================================================================
    layer2_df = pd.DataFrame(all_results)
    
    # Assign dynamic component tiers (R and F)
    layer2_df = self._compute_dynamic_component_tiers(layer2_df)
    
    # ========================================================================
    # STEP 5: SAVE UNIFIED OUTPUT
    # ========================================================================
    os.makedirs("outputs", exist_ok=True)
    
    unified_path = "outputs/layer2_unified_output.csv"
    layer2_df.to_csv(unified_path, index=False, encoding="utf-8")
    print(f"\n✅ Unified Layer 2 output saved to: {unified_path}")
    print(f"   Total rows: {len(layer2_df)}")
    
    # Legacy alias for backwards compatibility
    legacy_path = "outputs/layer2_output_debug.csv"
    layer2_df.to_csv(legacy_path, index=False, encoding="utf-8")
    
    # Breakdown by cohort
    cohort_counts = layer2_df["Cohort_Type"].value_counts().to_dict()
    print(f"\n   Breakdown by Cohort_Type:")
    for cohort, count in sorted(cohort_counts.items()):
        print(f"      {cohort}: {count} drivers")
    
    # Bind to instance
    self.layer2_df = layer2_df
    
    print("\n" + "="*80)
    print("DUAL ARCHITECTURE COMPUTATION COMPLETE")
    print("="*80)
    
    return self.layer2_df


def _compute_seu_for_group(
    self, 
    group, 
    driver_name, 
    cohort_label, 
    driver_global_mentions, 
    context_total_rows
):
    """
    Compute SEU metrics for a single group (driver × cohort combination).
    
    Args:
        group: DataFrame subset (e.g., "Slow Checkout" mentions from Gold customers)
        driver_name: Experience driver name
        cohort_label: "GLOBAL" or tier name (e.g., "Gold")
        driver_global_mentions: Total mentions of this driver across ALL customers
        context_total_rows: Total rows in context (global total or driver total for segments)
    
    Returns:
        Dictionary with all SEU metrics
    """
    ts_col = "date"
    today_ts = self.today
    
    entry = {
        "experience_driver": driver_name,
        "Cohort_Type": cohort_label,
    }
    
    # --- Counts & Dates ---------------------------------------------------
    total_mentions = int(len(group))
    times = group[ts_col]
    first_seen = times.min() if total_mentions else None
    most_recent = times.max() if total_mentions else None
    
    # Age in days from MOST RECENT date (segment-specific)
    if most_recent is None or pd.isna(most_recent):
        age_days = None
    else:
        age_days = int(max(0, (today_ts - most_recent).days))
    
    # --- ERI (from segment's emotions only) -------------------------------
    emotion_counts = group["emotion_primary"].value_counts(dropna=False).to_dict()
    eri_norm = self._compute_eri_from_counts(emotion_counts, total_mentions)
    eri_tier = self._map_loyalty_tier(eri_norm)
    
    # --- EVI (from segment's emotions only) -------------------------------
    n_adore       = int(emotion_counts.get("Adoration", 0))
    n_appreciate  = int(emotion_counts.get("Appreciation", 0))
    n_ambivalent  = int(emotion_counts.get("Ambivalence", 0))
    n_agitate     = int(emotion_counts.get("Agitation", 0))
    n_anger       = int(emotion_counts.get("Anger", 0))
    
    evi_score = self._compute_evi(n_adore, n_appreciate, n_ambivalent, n_agitate, n_anger)
    evi_score = float(np.clip(evi_score, 0.0, 100.0))
    evi_tier = self._map_evi_tier(evi_score)
    
    # --- Emotional State Band (ERI × EVI) ---------------------------------
    emotional_state_band = self._classify_emotional_state_band(eri_tier, evi_tier)
    
    # --- R (Recency - segment-specific) -----------------------------------
    R = 0.0 if age_days is None else 100.0 * np.exp(-float(age_days) / self.tau_days)
    
    # --- F (Frequency - normalized to DRIVER'S global mentions) -----------
    # CRITICAL: Use driver_global_mentions as max, NOT all-driver max
    F = min(
        100.0,
        (np.log1p(total_mentions) / np.log1p(driver_global_mentions)) * 100.0,
    )
    
    # --- RF (Recency-Frequency composite) ---------------------------------
    rf_r_abs = self.rf_weight_r * float(R)
    rf_f_abs = self.rf_weight_f * float(F)
    RF = rf_r_abs + rf_f_abs
    rf_tier_long = self._map_rf_tier(RF)
    
    # % contribution
    if RF > 0:
        rf_r_pct = round(100.0 * rf_r_abs / RF, 1)
        rf_f_pct = round(100.0 - rf_r_pct, 1)
    else:
        rf_r_pct = 0.0
        rf_f_pct = 0.0
    
    rf_tier_short = self._normalize_rf_tier(rf_tier_long)
    
    # --- Priority (ES Band × RF) ------------------------------------------
    priority_tier = self._classify_priority(emotional_state_band, rf_tier_short)
    priority_purpose = self._classify_purpose(emotional_state_band, priority_tier)
    
    # --- Emotion Recency Profile ------------------------------------------
    emotion_recency_profile = self._compute_emotion_recency_profile(group)
    
    # --- Entities & Coverage ----------------------------------------------
    associated_names = sorted(group["entity_name"].dropna().unique().tolist()) if "entity_name" in group.columns else []
    
    # Mention share relative to context (global for GLOBAL, driver total for segments)
    mention_share_pct = round((total_mentions / max(1, context_total_rows)) * 100.0, 2)
    
    # --- Build Entry ------------------------------------------------------
    entry.update({
        "Today_Anchor": self.today.strftime("%Y-%m-%d"),
        "Window_Days": int(self.window_days) if self.window_days is not None else None,
        
        # ERI
        "ERI": round(eri_norm, 2),
        "ERI_Tier": eri_tier,
        "Loyalty_State": eri_tier,
        
        # EVI
        "EVI": round(evi_score, 2),
        "EVI_Tier": evi_tier,
        "Emotional_State_Band": emotional_state_band,
        
        # Recency-Frequency
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
        
        # Priority
        "Priority_Tier": priority_tier,
        "Priority_Purpose": priority_purpose,
        
        # Backward compatible
        "ERI_RF_Quadrant": f"{eri_tier} x {rf_tier_long}",
        "Quadrant_Purpose": priority_purpose,
        "Priority_Status": priority_tier,
        "Quadrant_Key": f"{eri_tier}|{rf_tier_long}",
        
        # Counts & Dates
        "Associated_Entity_Names": associated_names,
        "No_of_Mentions": total_mentions,
        "Mention_Share_%": mention_share_pct,
        "First_Seen_Date": first_seen.strftime("%Y-%m-%d") if first_seen is not None and not pd.isna(first_seen) else None,
        "Most_Recent_Date": most_recent.strftime("%Y-%m-%d") if most_recent is not None and not pd.isna(most_recent) else None,
        "Age_Days": age_days,
        
        # Temporal Intelligence
        "Emotion_Recency_Profile": emotion_recency_profile,
        
        # Debug
        "Debug_Emotion_Counts": emotion_counts,
    })
    
    return entry
