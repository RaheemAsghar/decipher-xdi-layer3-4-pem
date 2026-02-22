"""
================================================================================
SEU_Surface1_V1.py
Decipher — Predictive Intelligence Layer
Surface 1: SEU Temporal State Evolution & Structural Migration Prediction
================================================================================

ARCHITECTURAL POSITION
-----------------------
This module sits ABOVE the SEU computation engine (Layer_2_V9).
It does not modify, extend, or touch V9 in any way.
V9 computes current state. This module predicts future state.

The relationship is strictly:
    V9 Output (SEU Snapshot) → Surface1 (Prediction Layer)

This separation is non-negotiable and intentional.
"Prediction sits above state. It advises. It does not rewrite."
— Decipher Predictive Intelligence Layer Specification

WHAT THIS MODULE DOES
----------------------
Surface 1 treats the Experience Driver as a state machine.
It does not predict sentiment. It predicts structural state transitions.

Specifically it forecasts forward probability of:
    1. Emotional State Band migration (ES1–ES5 transitions)
    2. Crisis pattern emergence
    3. Volatility escalation or consolidation
    4. Polarity reversal or entrenchment
    5. Urgency band escalation
    6. ERI directional continuation
    7. EVI fragmentation acceleration
    8. Structural relationship collapse or stabilization

HOW IT WORKS — FOUR STAGES
----------------------------

STAGE 1 — SNAPSHOT PERSISTENCE
    Every V9 run produces one SEU per driver per window.
    This module takes that output and writes it as a versioned,
    time-stamped row to a SQLite snapshot store.
    Over time, this store becomes the training dataset.
    Without accumulation, prediction is impossible.
    Accumulation costs nothing. It just requires discipline.

STAGE 2 — FEATURE ENGINEERING
    Takes the raw V9 output and adds five predictive features
    that V9 does not compute because they are not state features —
    they are predictive features:

    a) Velocity Features
       Rate of ERI/EVI tier change per unit time.
       A 2-tier ERI drop in 7 days is categorically different
       from a 2-tier drop in 60 days. V9 captures the magnitude.
       Surface1 captures the speed.

    b) Silence Duration Signal
       RF acceleration/deceleration between windows.
       A driver going suddenly dormant after high activity
       is a leading indicator of churn, not recovery.

    c) Cross-Driver Context Score
       How many other drivers in the same dataset are currently
       in ES4 or ES5. Crisis rarely affects one driver in isolation.
       Multi-driver deterioration compounds individual crisis probability.

    d) Structural Stability Score (SSS)
       A single composite meta-feature synthesising:
           - EVI trend direction (encoded)
           - ES Band change flag
           - Polarity flow direction (encoded)
           - Pattern label severity (encoded)
       This gives gradient boosting a powerful root split feature
       without requiring new computation — it synthesises
       features already present in V9 output.

    e) Window Sequence Index
       How many snapshots exist for this driver.
       Early snapshots are less reliable than mature ones.
       The model needs to know where in the accumulation curve it is.

STAGE 3 — LABEL GENERATION
    For each historical snapshot, looks forward N days to the
    next available snapshot and computes prediction targets:

    Binary Targets:
        - ES band downgrade within X days (Y/N)
        - ES band upgrade within X days (Y/N)
        - Crisis pattern trigger within X days (Y/N)
        - Urgency escalation to Immediate within X days (Y/N)
        - EVI enters High/Extreme tier within X days (Y/N)

    Multiclass Targets:
        - Next ES Band state (ES1–ES5)
        - Next Pattern classification
        - Next Urgency band

    Regression Targets:
        - ERI delta next window
        - EVI delta next window

    Time-to-Event Targets:
        - Days until ES4 (Active Crisis)
        - Days until ES5 (Critical Failure)
        - Days until P0 priority
        Note: ES4 and ES5 are treated as competing risks.
        A driver cannot reach ES5 without passing ES4 within
        the same window granularity. Survival models must
        account for this or they will underestimate ES5 probability
        for drivers already in ES4.

STAGE 4 — MODEL TRAINING & INFERENCE
    Gradient Boosting (LightGBM) on tabular features.
    One model per target type (binary / multiclass / regression).
    Survival models (lifelines) for time-to-event targets.

    Minimum snapshot governance:
        A driver must have at least MIN_SNAPSHOTS_REQUIRED
        historical snapshots before it is eligible for prediction.
        Below this threshold, the system returns None rather than
        a low-confidence prediction dressed as truth.
        This is a hard governance rule, not a soft warning.

WHAT THIS MODULE DOES NOT DO
------------------------------
    - It does not modify V9 output
    - It does not recompute ERI, EVI, or RF
    - It does not redefine Experience Drivers
    - It does not override OU logic
    - It does not replace governance rules
    - It does not self-report accuracy — validation is external

DEPENDENCIES
-------------
    Layer_2_V9.py       — SEU computation engine (upstream)
    pandas              — data handling
    numpy               — numerical operations
    sqlite3             — snapshot persistence
    lightgbm            — gradient boosting models
    lifelines           — survival/time-to-event models
    sklearn             — validation utilities
    json                — feature serialisation

ENGINEERS: READ THIS BEFORE TOUCHING ANYTHING
----------------------------------------------
    The snapshot store is append-only.
    Never delete rows. Never update rows retroactively.
    If a recomputation is needed, add a new row with a new run_id.
    Referential integrity across time is a first-class law here,
    identical to the Memory Law in the ERM.

    The MIN_SNAPSHOTS_REQUIRED constant is a governance parameter.
    Do not lower it to get more training data.
    Raise it if you find predictions unstable at the current threshold.

    Model files are versioned by training date and driver count.
    Never overwrite a model file. Archive the old one first.

VERSION
--------
    SEU_Surface1_V1.py
    Decipher Predictive Intelligence Layer
    Authored: 2025–2026
    Status: V1 — Production-ready architecture, requires snapshot
            accumulation period before model training is meaningful.

================================================================================
"""

from __future__ import annotations

import sqlite3
import json
import os
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Optional ML dependencies (graceful degradation if not installed) ─────────
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logging.warning("LightGBM not installed. Model training unavailable. pip install lightgbm")

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logging.warning("lifelines not installed. Survival models unavailable. pip install lifelines")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, roc_auc_score,
        mean_absolute_error, mean_squared_error
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not installed. Validation metrics unavailable.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# GOVERNANCE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MIN_SNAPSHOTS_REQUIRED = 5       # Hard minimum before a driver is prediction-eligible
DEFAULT_HORIZON_DAYS   = 30      # Default forward prediction horizon
SNAPSHOT_DB_VERSION    = "S1_V1" # Version tag on all persisted rows

# ES Band ordinal encoding (lower = better health)
ES_BAND_ORDINAL = {
    "ES1_Secure_Loyalty":    1,
    "ES2_Growth_Opportunity": 2,
    "ES3_At_Risk":            3,
    "ES4_Active_Crisis":      4,
    "ES5_Critical_Failure":   5,
}

# Polarity flow direction encoding
POLARITY_FLOW_ORDINAL = {
    "Negative → Positive":   2,   # Strong improvement
    "Neutral → Polarized":   1,   # Mixed improvement signal
    "Stable":                0,   # No change
    "Mixed/Fragmenting":    -1,   # Unstable
    "Polarized → Neutral":  -1,   # Losing signal
    "Positive → Negative":  -2,   # Strong deterioration
}

# Pattern label severity encoding
PATTERN_SEVERITY = {
    "STABLE LOYALTY":               -2,  # Very healthy — no crisis signal
    "FULL RECOVERY":                -1,  # Recovering
    "CRISIS STABILIZING":           -1,
    "RECOVERY PATTERN":             -1,
    "RE-ENGAGEMENT SUCCESS":        -1,
    "POSITIVE SHIFT":               -1,
    "STABILIZATION":                -1,
    "MINOR IMPROVEMENT":             0,
    "VOLATILE GROWTH":               0,
    "STAGNANT RELATIONSHIP":         0,
    "VOLATILITY EMERGING":           1,
    "FRAGMENTING RELATIONSHIP":      1,
    "ENTRENCHED NEGATIVITY":         2,
    "DANGER PATTERN":                2,
    "GROWTH DERAILED":               2,
    "RISK ACTUALIZED":               3,
    "CRISIS PATTERN":                3,
    "CRISIS HARDENING":              3,
    "CHURN ACTUALIZED":              3,
    "ENGAGEMENT COLLAPSE":           4,
    "RELATIONSHIP COLLAPSE":         4,
    "CATASTROPHIC FAILURE":          4,
}

# Urgency band encoding
URGENCY_ORDINAL = {
    "Dormant":    0,
    "Watchful":   1,
    "Near-Term":  2,
    "Immediate":  3,
    "Unknown":   -1,
}

# EVI trend encoding
EVI_TREND_ORDINAL = {
    "↓↓ Rapidly Consolidating": -2,
    "↓ Consolidating":          -1,
    "→ Stable Volatility":       0,
    "↑ Fragmenting":             1,
    "↑↑ Rapidly Fragmenting":    2,
}

# ERI trend encoding
ERI_TREND_ORDINAL = {
    "↓↓ Strongly Declining":  -2,
    "↓ Declining":            -1,
    "→ Stable":                0,
    "↑ Improving":             1,
    "↑↑ Strongly Improving":   2,
}


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: SNAPSHOT PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

class SnapshotStore:
    """
    Append-only time-series store for SEU snapshots.

    Every V9 run produces one SEU per driver. This store accumulates
    those snapshots over time, building the dataset that Surface 1
    prediction depends on.

    RULES:
        - Rows are never deleted
        - Rows are never updated retroactively
        - Each row is identified by (seu_id, run_id, persisted_at)
        - Schema version is stamped on every row
    """

    def __init__(self, db_path: str = "outputs/seu_snapshots.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialise_schema()

    def _initialise_schema(self):
        """Create snapshot table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS seu_snapshots (
                    -- Identity
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    seu_id                  TEXT NOT NULL,
                    experience_driver       TEXT NOT NULL,
                    run_id                  TEXT NOT NULL,
                    schema_version          TEXT NOT NULL,
                    snapshot_db_version     TEXT NOT NULL,
                    persisted_at            TEXT NOT NULL,

                    -- Window
                    today_anchor            TEXT,
                    window_days             INTEGER,

                    -- Core SEU State (V9 outputs)
                    eri                     REAL,
                    eri_tier                TEXT,
                    evi                     REAL,
                    evi_tier                TEXT,
                    emotional_state_band    TEXT,
                    es_band_ordinal         INTEGER,
                    rf                      REAL,
                    rf_tier_short           TEXT,
                    priority_tier           TEXT,
                    priority_purpose        TEXT,
                    no_of_mentions          INTEGER,
                    age_days                INTEGER,
                    observed_days           INTEGER,
                    temporal_available      INTEGER,  -- boolean as 0/1

                    -- Temporal Intelligence (flattened from nested dicts)
                    eri_flow_start          REAL,
                    eri_flow_end            REAL,
                    eri_flow_delta          REAL,
                    eri_flow_delta_pct      REAL,
                    eri_flow_full_scale_pct REAL,
                    eri_flow_tier_shift     INTEGER,
                    eri_flow_trend          TEXT,

                    evi_flow_start          REAL,
                    evi_flow_end            REAL,
                    evi_flow_delta          REAL,
                    evi_flow_delta_pct      REAL,
                    evi_flow_tier_shift     INTEGER,
                    evi_flow_trend          TEXT,

                    es_flow_start           TEXT,
                    es_flow_end             TEXT,
                    es_flow_changed         INTEGER,  -- boolean as 0/1
                    es_flow_trajectory      TEXT,

                    polarity_flow_direction TEXT,
                    dominant_emotion_transition TEXT,
                    ambivalence_type        TEXT,

                    pattern_label           TEXT,
                    pem_mode                TEXT,
                    pem_urgency_band        TEXT,
                    pem_eri_delta           REAL,
                    pem_evi_delta           REAL,
                    pem_es_start            TEXT,
                    pem_es_end              TEXT,
                    pem_stream_bias         TEXT,   -- JSON list as string

                    -- Full raw snapshot (for future schema evolution)
                    raw_snapshot_json       TEXT,

                    -- Uniqueness guard
                    UNIQUE(seu_id, run_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_driver_anchor "
                "ON seu_snapshots(experience_driver, today_anchor)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_persisted "
                "ON seu_snapshots(persisted_at)"
            )
            conn.commit()

    def persist(self, seu_row: Dict[str, Any]) -> Optional[str]:
        """
        Persist a single SEU row from V9 output.

        Args:
            seu_row: One row from Layer2Computer.compute() output,
                     as a dict (from df.to_dict('records')).

        Returns:
            snapshot_id (str) if persisted, None if already exists.
        """
        erp = seu_row.get("Emotion_Recency_Profile") or {}
        ti  = erp.get("temporal_intelligence") or {}

        eri_flow     = ti.get("ERI_Flow") or {}
        evi_flow     = ti.get("EVI_Flow") or {}
        es_flow      = ti.get("ES_Flow") or {}
        emotion_flow = ti.get("Emotion_Flow") or {}
        pattern_flow = ti.get("Pattern_Flow") or {}
        pem          = ti.get("PEM_Guidance") or {}

        now = datetime.utcnow().isoformat()

        row = {
            "seu_id":                  seu_row.get("seu_id"),
            "experience_driver":       str(seu_row.get("experience_driver", "")),
            "run_id":                  seu_row.get("run_id"),
            "schema_version":          seu_row.get("schema_version", "unknown"),
            "snapshot_db_version":     SNAPSHOT_DB_VERSION,
            "persisted_at":            now,
            "today_anchor":            seu_row.get("Today_Anchor"),
            "window_days":             seu_row.get("Window_Days"),

            # Core state
            "eri":                     seu_row.get("ERI"),
            "eri_tier":                seu_row.get("ERI_Tier"),
            "evi":                     seu_row.get("EVI"),
            "evi_tier":                seu_row.get("EVI_Tier"),
            "emotional_state_band":    seu_row.get("Emotional_State_Band"),
            "es_band_ordinal":         ES_BAND_ORDINAL.get(
                                           seu_row.get("Emotional_State_Band"), None),
            "rf":                      seu_row.get("RF"),
            "rf_tier_short":           seu_row.get("RF_Tier_Short"),
            "priority_tier":           seu_row.get("Priority_Tier"),
            "priority_purpose":        seu_row.get("Priority_Purpose"),
            "no_of_mentions":          seu_row.get("No_of_Mentions"),
            "age_days":                seu_row.get("Age_Days"),
            "observed_days":           seu_row.get("Observed_Days"),
            "temporal_available":      1 if seu_row.get(
                                           "Temporal_Intelligence_Available") else 0,

            # ERI Flow
            "eri_flow_start":          eri_flow.get("start"),
            "eri_flow_end":            eri_flow.get("end"),
            "eri_flow_delta":          eri_flow.get("delta"),
            "eri_flow_delta_pct":      eri_flow.get("delta_pct"),
            "eri_flow_full_scale_pct": eri_flow.get("full_scale_move_pct"),
            "eri_flow_tier_shift":     eri_flow.get("tier_shift"),
            "eri_flow_trend":          eri_flow.get("trend"),

            # EVI Flow
            "evi_flow_start":          evi_flow.get("start"),
            "evi_flow_end":            evi_flow.get("end"),
            "evi_flow_delta":          evi_flow.get("delta"),
            "evi_flow_delta_pct":      evi_flow.get("delta_pct"),
            "evi_flow_tier_shift":     evi_flow.get("tier_shift"),
            "evi_flow_trend":          evi_flow.get("trend"),

            # ES Flow
            "es_flow_start":           es_flow.get("start"),
            "es_flow_end":             es_flow.get("end"),
            "es_flow_changed":         1 if es_flow.get("changed") else 0,
            "es_flow_trajectory":      es_flow.get("trajectory"),

            # Emotion Flow
            "polarity_flow_direction":      emotion_flow.get("polarity_flow_direction"),
            "dominant_emotion_transition":  emotion_flow.get("dominant_emotion_transition"),
            "ambivalence_type":             emotion_flow.get("ambivalence_type"),

            # Pattern & PEM
            "pattern_label":           pattern_flow.get("pattern_label"),
            "pem_mode":                pem.get("mode"),
            "pem_urgency_band":        pem.get("urgency_band"),
            "pem_eri_delta":           pem.get("eri_delta"),
            "pem_evi_delta":           pem.get("evi_delta"),
            "pem_es_start":            pem.get("es_start"),
            "pem_es_end":              pem.get("es_end"),
            "pem_stream_bias":         json.dumps(pem.get("recommended_stream_bias", [])),

            # Full raw snapshot
            "raw_snapshot_json":       json.dumps(seu_row, default=str),
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                cols = ", ".join(row.keys())
                placeholders = ", ".join(["?"] * len(row))
                conn.execute(
                    f"INSERT OR IGNORE INTO seu_snapshots ({cols}) VALUES ({placeholders})",
                    list(row.values())
                )
                conn.commit()
            logger.info(f"✅ Persisted snapshot: {row['experience_driver']} | {row['today_anchor']}")
            return row["seu_id"]
        except Exception as e:
            logger.error(f"❌ Snapshot persistence failed: {e}")
            return None

    def persist_batch(self, seu_df: pd.DataFrame) -> int:
        """
        Persist all rows from a V9 output DataFrame.

        Args:
            seu_df: Full output from Layer2Computer.compute()

        Returns:
            Count of rows successfully persisted.
        """
        count = 0
        for _, row in seu_df.iterrows():
            result = self.persist(row.to_dict())
            if result:
                count += 1
        logger.info(f"✅ Batch persist complete: {count}/{len(seu_df)} rows stored.")
        return count

    def load_snapshots(
        self,
        driver: Optional[str] = None,
        min_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load historical snapshots from store.

        Args:
            driver:   Filter to a specific experience_driver (or None for all).
            min_date: ISO date string — only return snapshots from this date onward.

        Returns:
            DataFrame of snapshot rows, sorted by driver + anchor date.
        """
        query = "SELECT * FROM seu_snapshots WHERE 1=1"
        params = []

        if driver:
            query += " AND experience_driver = ?"
            params.append(driver)
        if min_date:
            query += " AND today_anchor >= ?"
            params.append(min_date)

        query += " ORDER BY experience_driver, today_anchor ASC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        logger.info(f"📦 Loaded {len(df)} snapshots from store.")
        return df

    def driver_snapshot_counts(self) -> pd.Series:
        """Return snapshot count per driver — useful for eligibility checks."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT experience_driver, COUNT(*) as cnt "
                "FROM seu_snapshots GROUP BY experience_driver",
                conn
            )
        return df.set_index("experience_driver")["cnt"]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

class Surface1FeatureEngineer:
    """
    Builds the full predictive feature set for Surface 1.

    Takes raw snapshot rows (from SnapshotStore) and adds:
        - Velocity features (rate of change per unit time)
        - Silence duration signal (RF acceleration/deceleration)
        - Cross-driver context score
        - Structural Stability Score (SSS) composite
        - Window sequence index

    All features are computed from existing V9 outputs.
    No new data sources are required.
    """

    def build_features(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """
        Build full feature set from snapshot history.

        Args:
            snapshots: Output of SnapshotStore.load_snapshots()
                       Must be sorted by experience_driver + today_anchor.

        Returns:
            DataFrame with all engineered features appended.
        """
        if snapshots.empty:
            return snapshots

        df = snapshots.copy()
        df["today_anchor"] = pd.to_datetime(df["today_anchor"])
        df = df.sort_values(["experience_driver", "today_anchor"])

        df = self._add_velocity_features(df)
        df = self._add_silence_signal(df)
        df = self._add_window_sequence(df)
        df = self._add_structural_stability_score(df)
        df = self._add_cross_driver_context(df)
        df = self._encode_categoricals(df)

        logger.info(f"✅ Feature engineering complete: {len(df)} rows, {len(df.columns)} columns.")
        return df

    # ── Velocity Features ──────────────────────────────────────────────────────

    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rate of ERI/EVI tier change per unit time.

        A 2-tier ERI drop in 7 days is categorically different
        from a 2-tier drop in 60 days.
        V9 captures magnitude. Surface1 captures speed.
        """
        df = df.copy()

        df["days_since_prev"] = df.groupby("experience_driver")["today_anchor"].diff().dt.days
        df["prev_eri"]        = df.groupby("experience_driver")["eri"].shift(1)
        df["prev_evi"]        = df.groupby("experience_driver")["evi"].shift(1)
        df["prev_es_ordinal"] = df.groupby("experience_driver")["es_band_ordinal"].shift(1)

        df["eri_delta_abs"]   = df["eri"] - df["prev_eri"]
        df["evi_delta_abs"]   = df["evi"] - df["prev_evi"]
        df["es_ordinal_delta"]= df["es_band_ordinal"] - df["prev_es_ordinal"]

        # Velocity = magnitude / time (tier points per day)
        safe_days = df["days_since_prev"].replace(0, np.nan)
        df["eri_velocity"]    = df["eri_delta_abs"] / safe_days
        df["evi_velocity"]    = df["evi_delta_abs"] / safe_days
        df["es_velocity"]     = df["es_ordinal_delta"] / safe_days

        # Rolling 3-window velocity trend
        df["eri_velocity_3w"] = (
            df.groupby("experience_driver")["eri_velocity"]
              .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
        df["evi_velocity_3w"] = (
            df.groupby("experience_driver")["evi_velocity"]
              .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )

        return df

    # ── Silence Duration Signal ────────────────────────────────────────────────

    def _add_silence_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        RF acceleration and sudden dormancy detection.

        A driver going suddenly dormant after high activity
        is a leading indicator of churn actualisation, not recovery.
        """
        df = df.copy()

        df["prev_rf"]          = df.groupby("experience_driver")["rf"].shift(1)
        df["prev_mentions"]    = df.groupby("experience_driver")["no_of_mentions"].shift(1)

        df["rf_delta"]         = df["rf"] - df["prev_rf"]
        df["mention_delta"]    = df["no_of_mentions"] - df["prev_mentions"]

        # Sudden dormancy flag: RF dropped > 30 points in one window
        df["sudden_dormancy"]  = (df["rf_delta"] < -30).astype(int)

        # Silence acceleration: 3-window rolling RF decline
        df["rf_rolling_3w"]    = (
            df.groupby("experience_driver")["rf"]
              .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
        df["rf_acceleration"]  = df["rf"] - df["rf_rolling_3w"]

        # Age momentum: is the driver getting older (less recent signal)?
        df["prev_age_days"]    = df.groupby("experience_driver")["age_days"].shift(1)
        df["age_acceleration"] = df["age_days"] - df["prev_age_days"]

        return df

    # ── Window Sequence Index ──────────────────────────────────────────────────

    def _add_window_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank of this snapshot within the driver's history.
        Early snapshots are less reliable than mature ones.
        The model needs to weight this appropriately.
        """
        df = df.copy()
        df["window_seq"] = df.groupby("experience_driver").cumcount() + 1
        df["is_mature"]  = (df["window_seq"] >= MIN_SNAPSHOTS_REQUIRED).astype(int)
        return df

    # ── Structural Stability Score ─────────────────────────────────────────────

    def _add_structural_stability_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Composite meta-feature synthesising directional momentum.

        Combines:
            - EVI trend direction (encoded ordinal)
            - ES Band change flag (0/1)
            - Polarity flow direction (encoded ordinal)
            - Pattern label severity (encoded ordinal)

        Higher SSS = more structurally unstable / at risk.
        Lower/negative SSS = stabilising or healthy.

        This gives gradient boosting a powerful root split
        without requiring any new computation.
        """
        df = df.copy()

        df["evi_trend_enc"] = df["evi_flow_trend"].map(EVI_TREND_ORDINAL).fillna(0)
        df["polarity_enc"]  = df["polarity_flow_direction"].map(POLARITY_FLOW_ORDINAL).fillna(0)
        df["pattern_sev"]   = df["pattern_label"].map(
            lambda x: next(
                (v for k, v in PATTERN_SEVERITY.items() if str(x).upper().startswith(k)),
                0
            ) if pd.notna(x) else 0
        )

        # Weighted composite:
        # Pattern severity carries most signal (weight 3)
        # EVI trend is leading indicator (weight 2)
        # ES changed is structural confirmation (weight 2)
        # Polarity direction provides direction (weight 1)
        df["structural_stability_score"] = (
            df["pattern_sev"]    * 3 +
            df["evi_trend_enc"]  * 2 +
            df["es_flow_changed"]* 2 +
            df["polarity_enc"]   * 1
        )

        # Normalise to -10 / +10 range for interpretability
        max_possible = 3*4 + 2*2 + 2*1 + 1*2  # = 20
        min_possible = 3*(-2) + 2*(-2) + 0 + 1*(-2)  # = -12
        range_size   = max_possible - min_possible
        df["sss_normalised"] = (
            (df["structural_stability_score"] - min_possible) / range_size * 20 - 10
        ).round(2)

        return df

    # ── Cross-Driver Context Score ─────────────────────────────────────────────

    def _add_cross_driver_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Count of other drivers in crisis state at the same anchor point.

        Crisis rarely affects one driver in isolation.
        Multi-driver deterioration compounds individual crisis probability.
        """
        df = df.copy()

        crisis_bands = {"ES4_Active_Crisis", "ES5_Critical_Failure"}

        context = (
            df.groupby("today_anchor")
              .apply(lambda g: g["emotional_state_band"].isin(crisis_bands).sum())
              .rename("total_crisis_drivers_in_window")
              .reset_index()
        )
        df = df.merge(context, on="today_anchor", how="left")

        # Subtract self from count
        df["cross_driver_crisis_count"] = (
            df["total_crisis_drivers_in_window"] -
            df["emotional_state_band"].isin(crisis_bands).astype(int)
        ).clip(lower=0)

        df.drop(columns=["total_crisis_drivers_in_window"], inplace=True)
        return df

    # ── Categorical Encoding ───────────────────────────────────────────────────

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode remaining categorical features as ordinals for modelling."""
        df = df.copy()

        df["eri_trend_enc"]     = df["eri_flow_trend"].map(ERI_TREND_ORDINAL).fillna(0)
        df["urgency_enc"]       = df["pem_urgency_band"].map(URGENCY_ORDINAL).fillna(-1)
        df["priority_enc"]      = df["priority_tier"].str.replace("P", "").apply(
                                      pd.to_numeric, errors="coerce"
                                  ).fillna(3)

        # ES band ordinal already added at persist time
        df["es_start_ordinal"]  = df["es_flow_start"].map(ES_BAND_ORDINAL).fillna(3)
        df["es_end_ordinal"]    = df["es_flow_end"].map(ES_BAND_ORDINAL).fillna(3)

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: LABEL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

class Surface1LabelGenerator:
    """
    Generates prediction targets by looking forward N days in snapshot history.

    For each snapshot row, finds the next available snapshot for the
    same driver and computes what changed.

    COMPETING RISKS NOTE (Engineers):
        ES4 and ES5 targets are not independent.
        Do not model them as separate binary classifiers.
        Use a competing risks survival model (lifelines CoxPHFitter
        with event type encoding) for time-to-event targets.
    """

    def __init__(self, horizon_days: int = DEFAULT_HORIZON_DAYS):
        self.horizon_days = horizon_days

    def generate(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all Surface 1 prediction targets to the feature DataFrame.

        Args:
            features_df: Output of Surface1FeatureEngineer.build_features()

        Returns:
            DataFrame with target columns appended.
            Rows without a future snapshot within horizon have NaN targets
            and should be excluded from training (but kept for inference).
        """
        df = features_df.copy()
        df["today_anchor"] = pd.to_datetime(df["today_anchor"])
        df = df.sort_values(["experience_driver", "today_anchor"])

        targets = []
        for driver, group in df.groupby("experience_driver"):
            group = group.reset_index(drop=True)
            driver_targets = self._generate_driver_targets(group)
            targets.append(driver_targets)

        target_df = pd.concat(targets, ignore_index=True)
        df = df.reset_index(drop=True)
        target_cols = [c for c in target_df.columns if c.startswith("target_")]
        df[target_cols] = target_df[target_cols]

        labelled = df[df[target_cols].notna().any(axis=1)]
        logger.info(
            f"✅ Label generation complete: {len(labelled)}/{len(df)} rows have targets."
        )
        return df

    def _generate_driver_targets(self, group: pd.DataFrame) -> pd.DataFrame:
        """Generate targets for one driver's snapshot sequence."""
        n = len(group)
        target_rows = []

        for i in range(n):
            row = group.iloc[i]
            current_anchor = row["today_anchor"]
            horizon_cutoff = current_anchor + timedelta(days=self.horizon_days)

            # Find next snapshot within horizon
            future = group[
                (group["today_anchor"] > current_anchor) &
                (group["today_anchor"] <= horizon_cutoff)
            ]

            if future.empty:
                target_rows.append(self._empty_targets())
                continue

            next_row = future.iloc[0]
            target_rows.append(self._compute_targets(row, next_row))

        return pd.DataFrame(target_rows)

    def _compute_targets(
        self, current: pd.Series, next_snap: pd.Series
    ) -> Dict[str, Any]:
        """Compute all targets given current and next snapshot."""

        curr_es  = current.get("es_band_ordinal", 3)
        next_es  = next_snap.get("es_band_ordinal", 3)
        next_urg = next_snap.get("pem_urgency_band", "Unknown")
        next_evi = next_snap.get("evi", 0.0) or 0.0
        next_pat = str(next_snap.get("pattern_label", ""))

        days_forward = (
            pd.to_datetime(next_snap["today_anchor"]) -
            pd.to_datetime(current["today_anchor"])
        ).days

        return {
            # ── Binary Targets ────────────────────────────────────────────────
            "target_es_downgrade":
                int(next_es > curr_es) if pd.notna(next_es) else None,

            "target_es_upgrade":
                int(next_es < curr_es) if pd.notna(next_es) else None,

            "target_crisis_trigger":
                int(next_es >= 4) if pd.notna(next_es) else None,

            "target_urgency_immediate":
                int(next_urg == "Immediate"),

            "target_evi_high":
                int(next_evi >= 60) if pd.notna(next_evi) else None,

            # ── Multiclass Targets ────────────────────────────────────────────
            "target_next_es_band":
                int(next_es) if pd.notna(next_es) else None,

            "target_next_urgency_band":
                URGENCY_ORDINAL.get(next_urg, -1),

            # ── Regression Targets ────────────────────────────────────────────
            "target_eri_delta_next":
                round(float(next_snap.get("eri", 0)) -
                      float(current.get("eri", 0)), 2),

            "target_evi_delta_next":
                round(float(next_snap.get("evi", 0)) -
                      float(current.get("evi", 0)), 2),

            # ── Time-to-Event (days forward to next snapshot, event encoded) ──
            # Engineers: use these with lifelines CoxPHFitter + event_col
            # event_type: 0=no event, 1=ES4 crisis, 2=ES5 critical failure
            "target_tte_days":          days_forward,
            "target_tte_event_type":    self._encode_tte_event(next_es),
        }

    def _encode_tte_event(self, next_es_ordinal) -> int:
        """
        Competing risks encoding for time-to-event targets.
            0 = No crisis event in horizon
            1 = Reached ES4 (Active Crisis)
            2 = Reached ES5 (Critical Failure)
        """
        if next_es_ordinal is None or pd.isna(next_es_ordinal):
            return 0
        if int(next_es_ordinal) == 5:
            return 2
        if int(next_es_ordinal) == 4:
            return 1
        return 0

    def _empty_targets(self) -> Dict[str, Any]:
        """Return NaN targets for rows with no future snapshot in horizon."""
        return {
            "target_es_downgrade":      None,
            "target_es_upgrade":        None,
            "target_crisis_trigger":    None,
            "target_urgency_immediate": None,
            "target_evi_high":          None,
            "target_next_es_band":      None,
            "target_next_urgency_band": None,
            "target_eri_delta_next":    None,
            "target_evi_delta_next":    None,
            "target_tte_days":          None,
            "target_tte_event_type":    None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: MODEL TRAINING & INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

# Core feature columns used for model training
SURFACE1_FEATURE_COLS = [
    # Core state
    "eri", "evi", "rf", "es_band_ordinal",
    "no_of_mentions", "age_days", "observed_days",

    # ERI Flow
    "eri_flow_start", "eri_flow_end", "eri_flow_delta",
    "eri_flow_delta_pct", "eri_flow_full_scale_pct",
    "eri_flow_tier_shift", "eri_trend_enc",

    # EVI Flow
    "evi_flow_start", "evi_flow_end", "evi_flow_delta",
    "evi_flow_delta_pct", "evi_flow_tier_shift", "evi_trend_enc",

    # ES Flow
    "es_flow_changed", "es_start_ordinal", "es_end_ordinal",

    # Emotion Flow
    "polarity_enc",

    # Pattern & PEM
    "pattern_sev", "urgency_enc", "priority_enc",
    "pem_eri_delta", "pem_evi_delta",

    # Engineered: Velocity
    "eri_velocity", "evi_velocity", "es_velocity",
    "eri_velocity_3w", "evi_velocity_3w",

    # Engineered: Silence
    "rf_delta", "sudden_dormancy", "rf_acceleration", "age_acceleration",

    # Engineered: Composite
    "structural_stability_score", "sss_normalised",

    # Engineered: Context
    "cross_driver_crisis_count",

    # Engineered: Sequence
    "window_seq",
]


class Surface1ModelTrainer:
    """
    Trains and evaluates Surface 1 prediction models.

    One model per target type.
    Eligibility check enforced before training.

    ENGINEERS:
        Do not lower MIN_SNAPSHOTS_REQUIRED to get more training data.
        Ineligible drivers are excluded from training, not imputed.
        A confident wrong prediction is worse than no prediction.
    """

    def __init__(self, model_output_dir: str = "outputs/surface1_models"):
        self.model_output_dir = model_output_dir
        os.makedirs(model_output_dir, exist_ok=True)
        self.models = {}

    def _eligibility_check(
        self, df: pd.DataFrame, driver_counts: pd.Series
    ) -> pd.DataFrame:
        """
        Filter to drivers with sufficient snapshot history.
        Hard governance rule — not a soft warning.
        """
        eligible_drivers = driver_counts[
            driver_counts >= MIN_SNAPSHOTS_REQUIRED
        ].index.tolist()

        excluded = set(df["experience_driver"].unique()) - set(eligible_drivers)
        if excluded:
            logger.warning(
                f"⚠️  {len(excluded)} drivers excluded (insufficient snapshots): "
                f"{list(excluded)[:5]}{'...' if len(excluded) > 5 else ''}"
            )

        return df[df["experience_driver"].isin(eligible_drivers)]

    def prepare_training_data(
        self,
        labelled_df: pd.DataFrame,
        driver_counts: pd.Series,
        target_col: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare X, y for a specific target.

        Args:
            labelled_df:   Output of Surface1LabelGenerator.generate()
            driver_counts: From SnapshotStore.driver_snapshot_counts()
            target_col:    Name of the target column to train on.

        Returns:
            (X, y) tuple ready for model training.
        """
        df = self._eligibility_check(labelled_df, driver_counts)
        df = df[df[target_col].notna()].copy()

        available_features = [
            c for c in SURFACE1_FEATURE_COLS if c in df.columns
        ]
        X = df[available_features].fillna(0)
        y = df[target_col]

        logger.info(
            f"📊 Training data: {len(X)} rows | "
            f"{len(available_features)} features | target: {target_col}"
        )
        return X, y

    def train_binary_classifier(
        self, X: pd.DataFrame, y: pd.Series, target_name: str
    ) -> Optional[Any]:
        """Train a LightGBM binary classifier."""
        if not LGBM_AVAILABLE:
            logger.error("LightGBM not available. Cannot train.")
            return None
        if len(X) < 20:
            logger.warning(f"Insufficient data for {target_name}: {len(X)} rows.")
            return None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_samples=5,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )

        if SKLEARN_AVAILABLE:
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            logger.info(f"\n{classification_report(y_val, y_pred)}")
            try:
                auc = roc_auc_score(y_val, y_prob)
                logger.info(f"AUC-ROC: {auc:.4f}")
            except Exception:
                pass

        self.models[target_name] = model
        return model

    def train_regression_model(
        self, X: pd.DataFrame, y: pd.Series, target_name: str
    ) -> Optional[Any]:
        """Train a LightGBM regression model."""
        if not LGBM_AVAILABLE:
            logger.error("LightGBM not available.")
            return None
        if len(X) < 20:
            logger.warning(f"Insufficient data for {target_name}: {len(X)} rows.")
            return None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_samples=5,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )

        if SKLEARN_AVAILABLE:
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            logger.info(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        self.models[target_name] = model
        return model

    def train_survival_model(
        self,
        X: pd.DataFrame,
        duration_col: str = "target_tte_days",
        event_col: str    = "target_tte_event_type",
        target_name: str  = "tte_es4_es5"
    ) -> Optional[Any]:
        """
        Train a Cox Proportional Hazards survival model for time-to-event.

        COMPETING RISKS:
            event_type 0 = censored (no crisis)
            event_type 1 = ES4 Active Crisis
            event_type 2 = ES5 Critical Failure

        Engineers: For full competing risks, train two separate models:
            Model A: event = ES4 (type 1), censor ES5 as 0
            Model B: event = ES5 (type 2), censor ES4 as 0
        """
        if not LIFELINES_AVAILABLE:
            logger.error("lifelines not available. Cannot train survival model.")
            return None

        survival_df = X.copy()
        survival_df["duration"]   = X[duration_col] if duration_col in X.columns else 30
        survival_df["event_obs"]  = (
            X[event_col].isin([1, 2]).astype(int)
            if event_col in X.columns else 0
        )

        feature_cols = [c for c in SURFACE1_FEATURE_COLS if c in survival_df.columns]

        cox = CoxPHFitter(penalizer=0.1)
        try:
            cox.fit(
                survival_df[feature_cols + ["duration", "event_obs"]],
                duration_col="duration",
                event_col="event_obs",
            )
            cox.print_summary()
            self.models[target_name] = cox
            return cox
        except Exception as e:
            logger.error(f"Survival model training failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class Surface1Predictor:
    """
    Runs Surface 1 inference on a new SEU snapshot.

    Checks eligibility, builds features from snapshot history,
    and returns predictions with confidence scores.

    Returns None for ineligible drivers rather than
    generating a low-confidence prediction dressed as truth.
    """

    def __init__(
        self,
        store: SnapshotStore,
        engineer: Surface1FeatureEngineer,
        trainer: Surface1ModelTrainer,
    ):
        self.store    = store
        self.engineer = engineer
        self.trainer  = trainer

    def predict(
        self,
        driver: str,
        horizon_days: int = DEFAULT_HORIZON_DAYS
    ) -> Optional[Dict[str, Any]]:
        """
        Generate Surface 1 predictions for a single Experience Driver.

        Args:
            driver:       Canonical driver name (Category → Subcategory)
            horizon_days: Forward prediction horizon in days.

        Returns:
            Dict of predictions, or None if driver is ineligible.
        """
        counts = self.store.driver_snapshot_counts()
        driver_count = counts.get(driver, 0)

        if driver_count < MIN_SNAPSHOTS_REQUIRED:
            logger.warning(
                f"⚠️  {driver} ineligible: {driver_count} snapshots "
                f"(minimum {MIN_SNAPSHOTS_REQUIRED} required). "
                f"Returning None — not a low-confidence prediction."
            )
            return None

        snapshots = self.store.load_snapshots(driver=driver)
        features  = self.engineer.build_features(snapshots)

        latest = features.sort_values("today_anchor").iloc[[-1]]
        feature_cols = [c for c in SURFACE1_FEATURE_COLS if c in latest.columns]
        X = latest[feature_cols].fillna(0)

        predictions = {
            "driver":        driver,
            "as_of":         str(latest["today_anchor"].iloc[0]),
            "horizon_days":  horizon_days,
            "snapshot_count": int(driver_count),
            "eligible":      True,
        }

        for target_name, model in self.trainer.models.items():
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X)[0][1]
                    pred = model.predict(X)[0]
                    predictions[target_name] = {
                        "prediction": int(pred),
                        "probability": round(float(prob), 4)
                    }
                elif hasattr(model, "predict"):
                    pred = model.predict(X)[0]
                    predictions[target_name] = {
                        "prediction": round(float(pred), 4)
                    }
            except Exception as e:
                predictions[target_name] = {"error": str(e)}

        return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION — END TO END
# ═══════════════════════════════════════════════════════════════════════════════

class Surface1Pipeline:
    """
    End-to-end orchestrator for Surface 1.

    Usage:
        pipeline = Surface1Pipeline()

        # After every V9 run:
        pipeline.ingest(seu_df)

        # When enough data has accumulated:
        pipeline.train()

        # For any driver:
        result = pipeline.predict("Emergency Services → Triage Efficiency")
    """

    def __init__(self, db_path: str = "outputs/seu_snapshots.db"):
        self.store    = SnapshotStore(db_path=db_path)
        self.engineer = Surface1FeatureEngineer()
        self.labeller = Surface1LabelGenerator()
        self.trainer  = Surface1ModelTrainer()
        self.predictor = Surface1Predictor(self.store, self.engineer, self.trainer)

    def ingest(self, seu_df: pd.DataFrame) -> int:
        """
        Ingest a V9 output DataFrame into the snapshot store.
        Call this after every V9 run.
        """
        return self.store.persist_batch(seu_df)

    def train(self, horizon_days: int = DEFAULT_HORIZON_DAYS):
        """
        Build features, generate labels, and train all Surface 1 models.
        Call this once sufficient snapshots have accumulated.
        """
        logger.info("🔄 Loading all snapshots...")
        snapshots = self.store.load_snapshots()

        logger.info("🔄 Engineering features...")
        features = self.engineer.build_features(snapshots)

        logger.info("🔄 Generating labels...")
        labelled = self.labeller.generate(features)

        counts = self.store.driver_snapshot_counts()

        logger.info("🔄 Training binary classifiers...")
        binary_targets = [
            "target_es_downgrade",
            "target_es_upgrade",
            "target_crisis_trigger",
            "target_urgency_immediate",
            "target_evi_high",
        ]
        for t in binary_targets:
            X, y = self.trainer.prepare_training_data(labelled, counts, t)
            if len(X) >= 20:
                self.trainer.train_binary_classifier(X, y, t)

        logger.info("🔄 Training regression models...")
        regression_targets = [
            "target_eri_delta_next",
            "target_evi_delta_next",
        ]
        for t in regression_targets:
            X, y = self.trainer.prepare_training_data(labelled, counts, t)
            if len(X) >= 20:
                self.trainer.train_regression_model(X, y, t)

        logger.info("🔄 Training survival model...")
        survival_features = [c for c in SURFACE1_FEATURE_COLS if c in labelled.columns]
        survival_data = labelled[
            labelled["target_tte_days"].notna()
        ][survival_features + ["target_tte_days", "target_tte_event_type"]].copy()
        if len(survival_data) >= 20:
            self.trainer.train_survival_model(survival_data)

        logger.info("✅ Surface 1 training complete.")

    def predict(
        self, driver: str, horizon_days: int = DEFAULT_HORIZON_DAYS
    ) -> Optional[Dict[str, Any]]:
        """
        Generate predictions for a single driver.
        Returns None if driver is ineligible.
        """
        return self.predictor.predict(driver, horizon_days)

    def predict_all_eligible(
        self, horizon_days: int = DEFAULT_HORIZON_DAYS
    ) -> List[Dict[str, Any]]:
        """Generate predictions for all eligible drivers."""
        counts  = self.store.driver_snapshot_counts()
        eligible = counts[counts >= MIN_SNAPSHOTS_REQUIRED].index.tolist()

        results = []
        for driver in eligible:
            result = self.predict(driver, horizon_days)
            if result:
                results.append(result)

        logger.info(f"✅ Predictions generated for {len(results)} eligible drivers.")
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE (for engineers)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Minimal example showing the full pipeline.

    In production:
        1. Replace mock_seu_df with actual Layer2Computer.compute() output
        2. Call pipeline.ingest() after every V9 run (daily / weekly)
        3. Call pipeline.train() once MIN_SNAPSHOTS_REQUIRED is reached
        4. Call pipeline.predict() on demand for any driver
    """

    print("=" * 70)
    print("Decipher — Surface 1 Predictive Layer")
    print("SEU Temporal State Evolution & Structural Migration Prediction")
    print("=" * 70)

    # Initialise pipeline
    pipeline = Surface1Pipeline(db_path="outputs/seu_snapshots_demo.db")

    # ── Simulate V9 output (replace with real Layer2Computer output) ──────────
    mock_rows = []
    drivers = [
        "Emergency Services → Triage Efficiency",
        "Medication Safety → E-Prescription Accuracy",
        "Support Quality → Communication Style",
    ]

    import random
    random.seed(42)

    for anchor_offset in range(10):  # 10 simulated windows
        anchor = (datetime.today() - timedelta(days=anchor_offset * 14)).strftime("%Y-%m-%d")
        for driver in drivers:
            seu_id_raw = f"{driver}||{anchor}||75"
            seu_id = hashlib.md5(seu_id_raw.encode()).hexdigest()
            eri = round(random.uniform(-60, 60), 2)
            evi = round(random.uniform(10, 80), 2)

            mock_rows.append({
                "seu_id": seu_id,
                "experience_driver": driver,
                "run_id": str(anchor_offset),
                "schema_version": "SEU_L2_V9",
                "Today_Anchor": anchor,
                "Window_Days": 75,
                "ERI": eri,
                "ERI_Tier": "Positive" if eri > 30 else ("Negative" if eri < -10 else "Neutral"),
                "EVI": evi,
                "EVI_Tier": "High Volatility" if evi > 60 else "Moderate Volatility",
                "Emotional_State_Band": "ES3_At_Risk" if eri < 0 else "ES2_Growth_Opportunity",
                "RF": round(random.uniform(20, 80), 2),
                "RF_Tier_Short": "Moderate",
                "Priority_Tier": "P2",
                "Priority_Purpose": "Address Early Warning Signs",
                "No_of_Mentions": random.randint(5, 50),
                "Age_Days": random.randint(1, 30),
                "Observed_Days": random.randint(5, 20),
                "Temporal_Intelligence_Available": True,
                "Emotion_Recency_Profile": {},
            })

    mock_df = pd.DataFrame(mock_rows)

    # ── Ingest into snapshot store ────────────────────────────────────────────
    print("\n📥 Ingesting simulated V9 snapshots...")
    count = pipeline.ingest(mock_df)
    print(f"   {count} snapshots persisted.")

    # ── Check eligibility ─────────────────────────────────────────────────────
    driver_counts = pipeline.store.driver_snapshot_counts()
    print(f"\n📊 Snapshot counts per driver:")
    print(driver_counts.to_string())

    # ── Train models (requires ML dependencies) ───────────────────────────────
    if LGBM_AVAILABLE and SKLEARN_AVAILABLE:
        print("\n🔄 Training Surface 1 models...")
        pipeline.train()
    else:
        print("\n⚠️  ML dependencies not installed.")
        print("   Install: pip install lightgbm scikit-learn lifelines")
        print("   Snapshot store and feature engineering are fully operational.")
        print("   Model training will activate once dependencies are installed.")

    print("\n✅ Surface 1 Pipeline demonstration complete.")
    print("=" * 70)
