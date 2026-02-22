"""
Surface 1B — Statistical Calibration Layer (SEU V10-Pure)

Purpose
-------
This module provides a time-aware statistical calibration layer
that estimates forward structural migration risk for an Experience Driver.

It operates strictly on canonical SEU V10 snapshots and does not
recompute or mutate any state elements.

Scope
-----
The model estimates probability of:

    • ES downgrade within 30 days
    • Entry into ES4 (Active Crisis) within 30 days
    • Entry into ES5 (Critical Failure) within 60 days

These predictions are advisory only and do not override deterministic
SEU state, priority tiers, or execution routing logic.

Design Principles
-----------------
• Consumes only canonical V10 outputs (ERI, EVI, ES, RF, Priority,
  and Temporal Intelligence flows).

• No invented metrics, synthetic risk scores, or composite indices.

• Time-aware split (no random shuffling) to prevent temporal leakage.

• Separate models for ES4 and ES5 to avoid competing risk collapse.

• Enforces minimum snapshot and positive case thresholds.

• Append-only architecture: does not mutate historical SEU records.

Architecture Role
-----------------
Surface 1B acts as a thin statistical calibration layer
above the deterministic SEU state engine.

It enhances forward visibility using historical state transitions,
while preserving full alignment with the SEU ontology and governance rules.

If insufficient history is available, predictive outputs are disabled
rather than inferred.

This module is intentionally minimal and auditable.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from datetime import timedelta


class SEUSurface1Statistical:
    """
    Surface 1B — Statistical Calibration Layer (V10-Pure)

    Predicts:
        - ES downgrade within 30 days
        - ES4 entry within 30 days
        - ES5 entry within 60 days

    Strictly consumes canonical SEU V10 outputs.
    """

    MIN_SNAPSHOTS_REQUIRED = 20
    MIN_POSITIVE_CASES = 5

    def __init__(self):
        self.models = {}
        self.encoders = {}

    # --------------------------------------------------
    # Snapshot Preparation
    # --------------------------------------------------

    def prepare_snapshots(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df = df[
            (df["Temporal_Intelligence_Available"] == True) &
            (df["Observed_Days"] >= 2)
        ]

        df = df.sort_values(["experience_driver", "today_anchor"])

        return df

    # --------------------------------------------------
    # Feature Extraction (STRICTLY V10 FIELDS)
    # --------------------------------------------------

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:

        X = pd.DataFrame()

        # Core State
        X["ERI"] = df["ERI"]
        X["EVI"] = df["EVI"]
        X["ES_Ordinal"] = df["Emotional_State_Band_Ordinal"]
        X["RF_Ordinal"] = df["RF_Tier_Ordinal"]
        X["Priority_Ordinal"] = df["Priority_Tier_Ordinal"]

        # Temporal Flow
        X["ERI_Delta"] = df["ERI_Flow.delta"]
        X["ERI_Tier_Shift"] = df["ERI_Flow.tier_shift"]
        X["EVI_Delta"] = df["EVI_Flow.delta"]
        X["EVI_Tier_Shift"] = df["EVI_Flow.tier_shift"]
        X["ES_Changed"] = df["ES_Flow.changed"].astype(int)

        # Pattern
        le_pattern = LabelEncoder()
        X["Pattern_Label"] = le_pattern.fit_transform(df["Pattern_Flow.pattern_label"])
        self.encoders["pattern"] = le_pattern

        # Emotion Polarity Shift
        le_polarity = LabelEncoder()
        X["Polarity_Shift"] = le_polarity.fit_transform(df["Emotion_Flow.polarity_shift"])
        self.encoders["polarity"] = le_polarity

        # PEM Mode
        le_pem = LabelEncoder()
        X["PEM_Mode"] = le_pem.fit_transform(df["PEM_Guidance.mode"])
        self.encoders["pem"] = le_pem

        return X

    # --------------------------------------------------
    # Target Generation (Time-Aware)
    # --------------------------------------------------

    def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df["today_anchor"] = pd.to_datetime(df["today_anchor"])

        downgrade = []
        es4_entry = []
        es5_entry = []

        for idx, row in df.iterrows():

            driver = row["experience_driver"]
            current_date = row["today_anchor"]

            future = df[
                (df["experience_driver"] == driver) &
                (df["today_anchor"] > current_date)
            ]

            future_30 = future[
                future["today_anchor"] <= current_date + timedelta(days=30)
            ]

            future_60 = future[
                future["today_anchor"] <= current_date + timedelta(days=60)
            ]

            # Downgrade = ES ordinal increases
            downgrade_flag = (
                (future_30["Emotional_State_Band_Ordinal"] >
                 row["Emotional_State_Band_Ordinal"]).any()
            )

            es4_flag = (
                (future_30["Emotional_State_Band"] == "ES4").any()
            )

            es5_flag = (
                (future_60["Emotional_State_Band"] == "ES5").any()
            )

            downgrade.append(int(downgrade_flag))
            es4_entry.append(int(es4_flag))
            es5_entry.append(int(es5_flag))

        df["target_downgrade_30d"] = downgrade
        df["target_es4_30d"] = es4_entry
        df["target_es5_60d"] = es5_entry

        return df

    # --------------------------------------------------
    # Time-Based Split
    # --------------------------------------------------

    def time_split(self, df: pd.DataFrame):

        cutoff = df["today_anchor"].quantile(0.8)

        train = df[df["today_anchor"] <= cutoff]
        test = df[df["today_anchor"] > cutoff]

        return train, test

    # --------------------------------------------------
    # Train Models
    # --------------------------------------------------

    def train(self, df: pd.DataFrame):

        if len(df) < self.MIN_SNAPSHOTS_REQUIRED:
            raise ValueError("Insufficient snapshot history for training.")

        df = self.generate_targets(df)

        train_df, test_df = self.time_split(df)

        X_train = self.extract_features(train_df)
        X_test = self.extract_features(test_df)

        targets = [
            "target_downgrade_30d",
            "target_es4_30d",
            "target_es5_60d"
        ]

        for target in targets:

            y_train = train_df[target]
            y_test = test_df[target]

            if y_train.sum() < self.MIN_POSITIVE_CASES:
                continue

            model = GradientBoostingClassifier()
            model.fit(X_train, y_train)

            if len(y_test.unique()) > 1:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                print(f"{target} AUC:", round(auc, 3))

            self.models[target] = model

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    def predict(self, seu_snapshot: dict) -> dict:

        if not self.models:
            return {"predictive_available": False}

        df = pd.DataFrame([seu_snapshot])

        X = self.extract_features(df)

        results = {}

        for target, model in self.models.items():
            prob = model.predict_proba(X)[0][1]
            results[target.replace("target_", "")] = round(float(prob), 4)

        results["predictive_available"] = True

        return results