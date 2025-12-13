"""
Layer 2 Configuration Module
=============================
All constants, tier boundaries, thresholds, grids, and matrices.

This module contains NO logic - only configuration data.
Modify this file to adjust system behavior without touching core logic.
"""


class Layer2Config:
    """
    Central configuration for Layer 2 emotional intelligence system.
    """
    
    # ========================================================================
    # EMOTION SCORING
    # ========================================================================
    EMOTION_SCORES = {
        "Adoration": 3,
        "Appreciation": 1,
        "Ambivalence": 0,
        "Agitation": -1,
        "Anger": -3,
    }
    
    # ========================================================================
    # RECENCY DECAY DEFAULTS
    # ========================================================================
    DEFAULT_TAU_DAYS = 30.0  # Exponential decay half-life
    DEFAULT_RF_WEIGHT_R = 0.6  # Recency weight
    DEFAULT_RF_WEIGHT_F = 0.4  # Frequency weight
    
    # ========================================================================
    # ERI TIER BOUNDARIES (normalized -100 to +100)
    # ========================================================================
    ERI_VERY_POSITIVE_MIN = 80
    ERI_POSITIVE_MIN = 30
    ERI_NEUTRAL_MIN = -10
    ERI_NEGATIVE_MIN = -50
    # Below -50 = Very Negative
    
    # ========================================================================
    # EVI TIER BOUNDARIES (0 to 100)
    # ========================================================================
    EVI_EXTREME_MIN = 80
    EVI_HIGH_MIN = 60
    EVI_MODERATE_MIN = 40
    EVI_LOW_MIN = 20
    # Below 20 = Stable Emotion
    
    # ========================================================================
    # RF TIER BOUNDARIES (0 to 100)
    # ========================================================================
    RF_VERY_HIGH_MIN = 80
    RF_HIGH_MIN = 60
    RF_MODERATE_MIN = 40
    RF_LOW_MIN = 20
    # Below 20 = Dormant
    
    # ========================================================================
    # TEMPORAL INTELLIGENCE THRESHOLDS
    # ========================================================================
    
    # --- ERI Trajectory Thresholds (tier-crossing based) ---
    ERI_STRONG_IMPROVEMENT_TIERS = 2   # Cross 2+ tiers upward
    ERI_IMPROVEMENT_TIERS = 1          # Cross 1 tier upward
    ERI_STRONG_DECLINE_TIERS = -2      # Cross 2+ tiers downward
    ERI_DECLINE_TIERS = -1             # Cross 1 tier downward
    # Between -1 and +1 = Stable
    
    # --- EVI Trajectory Thresholds (tier-crossing based) ---
    EVI_STRONG_FRAGMENTATION_TIERS = 2   # Cross 2+ tiers upward
    EVI_FRAGMENTATION_TIERS = 1          # Cross 1 tier upward
    EVI_STRONG_CONSOLIDATION_TIERS = -2  # Cross 2+ tiers downward
    EVI_CONSOLIDATION_TIERS = -1         # Cross 1 tier downward
    # Between -1 and +1 = Stable Volatility
    
    # --- Emotion Momentum Thresholds (percentage point shifts) ---
    EMOTION_STRONG_RISE_PCT = 20     # 20+ pct point increase
    EMOTION_RISE_PCT = 10            # 10+ pct point increase
    EMOTION_MINOR_RISE_PCT = 5       # 5+ pct point increase
    EMOTION_STRONG_FADE_PCT = -20    # 20+ pct point decrease
    EMOTION_FADE_PCT = -10           # 10+ pct point decrease
    EMOTION_MINOR_FADE_PCT = -5      # 5+ pct point decrease
    # Between -5 and +5 = Stable
    
    # --- Pattern Interpretation Thresholds ---
    PATTERN_ERI_CRISIS_THRESHOLD = 20      # ERI delta for crisis
    PATTERN_EVI_FRAGMENT_THRESHOLD = 15     # EVI delta for fragmentation
    PATTERN_ERI_RECOVERY_THRESHOLD = 20     # ERI delta for recovery
    PATTERN_EVI_CONSOLIDATE_THRESHOLD = -10 # EVI delta for consolidation
    PATTERN_ERI_STABLE_RANGE = 5            # +/- range for stability
    PATTERN_EVI_STABLE_RANGE = 5            # +/- range for stability
    PATTERN_ERI_POSITIVE_THRESHOLD = 30     # ERI level for "positive"
    PATTERN_ERI_NEGATIVE_THRESHOLD = -30    # ERI level for "negative"
    PATTERN_EVI_CHAOS_THRESHOLD = 20        # EVI delta for chaos
    
    # ========================================================================
    # PRIORITY GRID (Emotional State × RF Tier → Priority)
    # ========================================================================
    PRIORITY_GRID = {
        # ES1: Happy customers - lower priority for intervention
        "ES1_Secure_Loyalty": {
            "Dormant":  "P5",
            "Low":      "P5",
            "Moderate": "P4",
            "High":     "P4",
            "Very High":"P3",
        },
        
        # ES2: Growth potential - moderate attention
        "ES2_Growth_Opportunity": {
            "Dormant":  "P5",
            "Low":      "P4",
            "Moderate": "P3",
            "High":     "P3",
            "Very High":"P2",
        },
        
        # ES3: At Risk - silence is a warning sign
        "ES3_At_Risk": {
            "Dormant":  "P2",
            "Low":      "P3",
            "Moderate": "P2",
            "High":     "P1",
            "Very High":"P1",
        },
        
        # ES4: Active Crisis - immediate response
        "ES4_Active_Crisis": {
            "Dormant":  "P2",
            "Low":      "P2",
            "Moderate": "P1",
            "High":     "P0",
            "Very High":"P0",
        },
        
        # ES5: Critical Failure - triage focus
        "ES5_Critical_Failure": {
            "Dormant":  "P2",
            "Low":      "P2",
            "Moderate": "P1",
            "High":     "P0",
            "Very High":"P0",
        },
    }
    
    # ========================================================================
    # PURPOSE MATRIX (Emotional State × Priority → Purpose)
    # ========================================================================
    PURPOSE_MATRIX = {
        # ES1 - Secure Loyalty
        ("ES1_Secure_Loyalty", "P5"): "Monitor Stable Advocacy",
        ("ES1_Secure_Loyalty", "P4"): "Maintain Positive Momentum",
        ("ES1_Secure_Loyalty", "P3"): "Activate Advocacy Programs",

        # ES2 - Growth Opportunity
        ("ES2_Growth_Opportunity", "P5"): "Background Growth Watch",
        ("ES2_Growth_Opportunity", "P4"): "Develop Emerging Loyalty",
        ("ES2_Growth_Opportunity", "P3"): "Strengthen Growth Drivers",
        ("ES2_Growth_Opportunity", "P2"): "Activate Growth Levers",

        # ES3 - At Risk
        ("ES3_At_Risk", "P3"): "Investigate Silent Risk",
        ("ES3_At_Risk", "P2"): "Address Early Warning Signs",
        ("ES3_At_Risk", "P1"): "Contain At-Risk Emotion",

        # ES4 - Active Crisis
        ("ES4_Active_Crisis", "P2"): "Crisis Containment",
        ("ES4_Active_Crisis", "P1"): "Active Crisis Intervention",
        ("ES4_Active_Crisis", "P0"): "Emergency Crisis Response",

        # ES5 - Critical Failure
        ("ES5_Critical_Failure", "P2"): "Critical Failure Containment",
        ("ES5_Critical_Failure", "P1"): "Contain Structural Breakdown",
        ("ES5_Critical_Failure", "P0"): "Critical Failure Response",
    }
    
    # ========================================================================
    # RF TIER NAME MAPPING (long → short)
    # ========================================================================
    RF_TIER_NORMALIZATION = {
        "Very High Activity": "Very High",
        "High Activity": "High",
        "Moderate Activity": "Moderate",
        "Low Activity": "Low",
        "Activity Dormant": "Dormant",
    }
    
    # ========================================================================
    # RECENCY TIER LABELS
    # ========================================================================
    RECENCY_TIER_LABELS = [
        "No Recent Activity",
        "Low Recency",
        "Moderate Recency",
        "High Recency",
        "Very High Recency"
    ]
    
    FREQUENCY_TIER_LABELS = [
        "Sparse Mentions",
        "Low Frequency",
        "Moderate Frequency",
        "High Frequency",
        "Very High Frequency"
    ]
    
    TIER_RANKS = [1, 2, 3, 4, 5]
    
    # ========================================================================
    # TEMPORAL TIER NAMES
    # ========================================================================
    ALL_TEMPORAL_TIERS = [
        "Tier_1",
        "Tier_2",
        "Tier_3",
        "Tier_4",
        "Tier_5",
        "Beyond_Window"
    ]
    
    RECENT_TIERS = ["Tier_1", "Tier_2"]
    EARLY_TIERS = ["Tier_4", "Tier_5"]
    
    # ========================================================================
    # EMOTION GROUPS (for polarity tracking)
    # ========================================================================
    NEGATIVE_EMOTIONS = {"Anger", "Agitation"}
    POSITIVE_EMOTIONS = {"Adoration", "Appreciation"}
    NEUTRAL_EMOTION = {"Ambivalence"}
    
    # ========================================================================
    # POLARITY SHIFT THRESHOLD
    # ========================================================================
    POLARITY_SHIFT_THRESHOLD = 15.0  # Percentage points for "meaningful" shift
