from __future__ import annotations
    
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from OU_prioritisation_V2 import TemporalIntelligenceFilter
    
import numpy as np
import math
import sqlite3
import json
import os
import re
import ast
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import acf
from typing import List, Dict, Any, Tuple
from collections import Counter
from uuid import uuid4     
from statsmodels.tsa.stattools import acf

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# === Flexible Date Range Analyzer ===
class FlexibleTimeframeAnalyzer:
    def __init__(self, input_path, output_dir="outputs", timeframe_days=75, compute_granular=True, verbose=True):
        """
        Flexible analyzer with plug-and-play timeframe logic and behavioral clustering setup.
        Updated to Universal Execution Contract Three-Axis Framework:
        - INTENT AXIS (formerly feedback_type)
        - AFFECT AXIS (formerly emotion/primary_emotion)  
        - ACTION AXIS (formerly opportunity_stream)
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.date_range_type = "window"
        self.timeframe_days = timeframe_days
        self.compute_granular = compute_granular
        self.verbose = verbose
        
        os.makedirs(self.output_dir, exist_ok=True)

        # ✅ Load and parse input
        self.df = pd.read_csv(self.input_path)
        self.df["date"] = pd.to_datetime(self.df["date"]).dt.date

        # ✅ Use max date as anchor
        self.latest_data_date = self.df["date"].max()
        self.cutoff_date = self.latest_data_date - timedelta(days=self.timeframe_days)
        self.start_date = self.cutoff_date
        self.end_date = self.latest_data_date

        # ✅ Filter the time window
        self.raw_df = self.df[self.df["date"].between(self.cutoff_date, self.latest_data_date)]

        if self.verbose:
            print(f"📅 Anchored to dataset's max date: {self.latest_data_date}")
            print(f"🪟 Timeframe applied: {self.cutoff_date} → {self.latest_data_date}")
            print(f"🧮 Filtered {len(self.raw_df)} rows within this range")

        # Holders for layered dataframes
        self.layer2_df = None
        self.layer3_df = None
        self.details_df = None

        # SQLite output paths
        suffix = self._get_timeframe_suffix()
        self.summary_sqlite_path = os.path.join(self.output_dir, f"sentientsignal_longitudinal_{suffix}.sqlite")
        self.granular_sqlite_path = os.path.join(self.output_dir, f"sentientsignal_granular_{suffix}.sqlite")

        # 🗺️ INTENT AXIS - Canonical intent labels (vertical-centric capture, universal computation)
        # Maps domain-specific variations to canonical intent labels
        self.INTENT_MAP = {
            "compliment": "Compliment",
            "complaint": "Complaint",
            "question": "Question",
            "suggestion": "Suggestion",
            "request": "Request",
            **{k: "Product Usage Insight" for k in ("usage insight", "product usage insight")},
            **{k: "Emerging Trends / Market Insight" for k in ("emerging trends", "market insight", "emerging trends / market insight")},
        }

        # Derived valid set (no need to maintain separately)
        self.VALID_INTENTS = set(self.INTENT_MAP.values())
 
        # 🗺️ ACTION AXIS - Canonical action/response classes
        # Maps variations to the 4 canonical strategic response classes
        self.ACTION_MAP = {
            "fix": "Fix",
            "optimize": "Optimize",
            "optimise": "Optimize",   # UK variant
            "amplify": "Amplify",
            "innovate": "Innovate",
        }
        
        # Helpful enums for validation/logging
        self.VALID_ACTIONS = {"Fix", "Optimize", "Amplify", "Innovate"}
        
        # 🗺️ AFFECT AXIS - Emotion scoring config (5-tier pressure lattice for ERI)
        # Universal pressure scoring regardless of vertical-specific emotion labels
        self.emotion_scores = {
            "Adoration": 3,
            "Appreciation": 1,
            "Ambivalence": 0,
            "Agitation": -1,
            "Anger": -3
        }

        # 🧠 Clustering parameters (define before setdefault usage)
        self.OU_CFG = {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # fully-qualified name

            # optional guard you already had — keep as-is
            "min_cluster_size": 8,
            "adaptive_min_cluster": {  # per-slice adaptive min cluster size
                "enabled": True,
                "pct": 0.08,   # 8% of rows in this ED→Intent→Action slice
                "floor": 3     # never below 3
            },

            "bcs_cumu_threshold": 0.80,
            "action_threshold": 0.80,  # renamed from stream_threshold
            "skip_singletons": False,
            "preview_location_style": "short",

            # 🔧 Granularity control — single knob with at most one adjustment
            # Start thresholds per signature (simple + agnostic)
            "signature_threshold_start": {
                "default_K1": 0.30,   # K1: matters + (moment & stage)
                "fallback_K3": 0.32,  # K3: context/exception-forward
            },
            # Bounds and step for the one-bump correction
            "bcs_distance_threshold_bounds": (0.25, 0.35),
            "bcs_distance_threshold_step": 0.02,
            "bcs_threshold_max_adjustments": 1,     # at most one bump
            "cluster_band_target": (3, 8),          # keep clusters in this band
            "singleton_cap": 0.40,                  # aim to stay at/under this

            # Back-compat: we initialize with the start value; runtime may bump once
            "bcs_distance_threshold": 0.30,         # will be set from signature_threshold_start[…]
        }

        self.ALLOWED_FOR_SIGNATURE = frozenset({
            "context","customer_journey","interaction_moment","customer_journey_stage",
            "semantic_customer_reality","matters","action_justification","behavioral_impact"
        })
        
        # ✍️ Signature weights (using semantic_customer_reality, NOT full SAS)
        self.SIGNATURE_LIBRARY = {
            "default_K1": {
                "matters": 6,
                "interaction_moment": 4,
                "customer_journey_stage": 4,
                "context": 3,
                "semantic_customer_reality": 2,
                "behavioral_impact": 1,
                "action_justification": 1,  # renamed from stream_justification
                "customer_journey": 1,
            },
            "fallback_K3": {
                "matters": 6,
                "context": 4,
                "semantic_customer_reality": 3,
                "interaction_moment": 3,
                "customer_journey_stage": 2,
                "behavioral_impact": 2,
                "action_justification": 1,  # renamed from stream_justification
                "customer_journey": 1,
            },
        }

        # ✅ Default choice (and set starting τ accordingly)
        self.OU_CFG["signature_config_name"] = "default_K1"  # echo in OU meta for reproducibility
        self.OU_CFG["signature_weights"] = self.SIGNATURE_LIBRARY[self.OU_CFG["signature_config_name"]]
        self.OU_CFG["bcs_distance_threshold"] = self.OU_CFG["signature_threshold_start"][self.OU_CFG["signature_config_name"]]

        # (optional but nice): make intent explicit — we only use the first half of SAS
        self.OU_CFG["semantic_statement_mode"] = "customer_reality_only"

    def _get_timeframe_suffix(self) -> str:
        """Generate filename suffix based on timeframe configuration."""
        return f"{self.cutoff_date.strftime('%Y%m%d')}_{self.latest_data_date.strftime('%Y%m%d')}"

    # ========== NORMALIZATION & VALIDATION ==========

    def normalize_intent_axis(self, val: Any) -> str:
        """
        Normalize INTENT AXIS values using the canonical intent map.
        Formerly: normalize_feedback_type
        """
        if pd.isna(val):
            return "unknown"
        s = str(val).strip().lower()
        return self.INTENT_MAP.get(s, s.title())

    def normalize_action_axis(self, val: Any) -> str:
        """
        Normalize ACTION AXIS values using the canonical action map.
        Formerly: normalize_stream
        """
        if pd.isna(val):
            return "unknown"
        s = str(val).strip().lower()
        return self.ACTION_MAP.get(s, s.title())

    def validate_intent_axis(self, series: pd.Series) -> pd.Series:
        """
        Validate INTENT AXIS values against canonical set.
        Formerly: validate_feedback_types
        """
        valid_mask = series.isin(self.VALID_INTENTS)
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum()
            invalid_vals = series[~valid_mask].unique()[:5]
            if self.verbose:
                print(f"⚠️  Found {invalid_count} invalid intent_axis values. Sample: {invalid_vals}")
        return series

    def validate_action_axis(self, series: pd.Series) -> pd.Series:
        """
        Validate ACTION AXIS values against canonical set.
        Formerly: validate_streams
        """
        valid_mask = series.isin(self.VALID_ACTIONS)
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum()
            invalid_vals = series[~valid_mask].unique()[:5]
            if self.verbose:
                print(f"⚠️  Found {invalid_count} invalid action_axis values. Sample: {invalid_vals}")
        return series

    def _prepare_raw_for_snapshot(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize THREE AXES before snapshot generation.
        Updates column names and values to new framework.
        """
        rdf = raw_df.copy()
        
        # Handle legacy column names (backward compatibility)
        column_mapping = {
            'feedback_type': 'intent_axis',
            'opportunity_stream': 'action_axis',
            'emotion_primary': 'affect_axis'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in rdf.columns and new_name not in rdf.columns:
                rdf.rename(columns={old_name: new_name}, inplace=True)
                if self.verbose:
                    print(f"🔄 Renamed legacy column: {old_name} → {new_name}")
        
        # Normalize axes
        if "intent_axis" in rdf.columns:
            rdf["intent_axis"] = rdf["intent_axis"].apply(self.normalize_intent_axis)
            self.validate_intent_axis(rdf["intent_axis"])
        
        if "action_axis" in rdf.columns:
            rdf["action_axis"] = rdf["action_axis"].apply(self.normalize_action_axis)
            self.validate_action_axis(rdf["action_axis"])
        
        return rdf

    def _safe_list(self, val: Any, allow_strings: bool = True) -> List[str]:
        """
        Convert various inputs to a clean list of lowercase strings.
        """
        if pd.isna(val) or val is None:
            return []
        if isinstance(val, str):
            if allow_strings:
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return [str(x).strip().lower() for x in parsed if str(x).strip()]
                except (ValueError, SyntaxError):
                    pass
            return [val.strip().lower()] if val.strip() else []
        if isinstance(val, (list, tuple)):
            return [str(x).strip().lower() for x in val if str(x).strip()]
        return []

    # ========== INTENT AXIS FOCUS LOGIC ==========

    def intent_focus_from_rows(self, rows: pd.DataFrame, threshold: float = 0.80) -> Tuple[List[str], Dict[str, float]]:
        """
        Determine dominant intent(s) from a set of rows using threshold logic.
        Formerly: feedbacktype_focus_from_rows
        
        Returns:
            - List of dominant intent labels (lowercase)
            - Distribution dict of all intents
        """
        if "intent_axis" not in rows.columns or rows.empty:
            return [], {}

        # Count and normalize
        counts = rows["intent_axis"].value_counts()
        total = counts.sum()
        if total == 0:
            return [], {}

        dist = (counts / total).to_dict()
        dist_lower = {k.lower(): v for k, v in dist.items()}

        # Sort by frequency descending
        sorted_intents = sorted(dist_lower.items(), key=lambda x: x[1], reverse=True)

        # Accumulate until threshold
        dominant = []
        cumulative = 0.0
        for intent, share in sorted_intents:
            dominant.append(intent)
            cumulative += share
            if cumulative >= threshold:
                break

        return dominant, dist_lower

    # ========== ACTION AXIS FOCUS LOGIC ==========

    def action_focus_from_rows(self, rows: pd.DataFrame, threshold: float = 0.80) -> Tuple[List[str], Dict[str, float]]:
        """
        Determine dominant action class(es) from a set of rows using threshold logic.
        Formerly: stream_focus_from_rows
        
        Returns:
            - List of dominant action classes (canonical case)
            - Distribution dict of all actions
        """
        if "action_axis" not in rows.columns or rows.empty:
            return [], {}

        counts = rows["action_axis"].value_counts()
        total = counts.sum()
        if total == 0:
            return [], {}

        dist = (counts / total).to_dict()

        # Sort by frequency descending
        sorted_actions = sorted(dist.items(), key=lambda x: x[1], reverse=True)

        # Accumulate until threshold
        dominant = []
        cumulative = 0.0
        for action, share in sorted_actions:
            dominant.append(action)
            cumulative += share
            if cumulative >= threshold:
                break

        return dominant, dist

    # ========== LAYER 2 COMPUTATION ==========

    def compute_layer2_summary(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate SEU-level metrics with THREE-AXIS framework.
        Computes Intent, Affect, and Action distributions per experience driver.
        """
        # Normalize axes first
        rdf = self._prepare_raw_for_snapshot(raw_df)
        
        required = {"experience_driver", "intent_axis", "action_axis"}
        missing = [c for c in required if c not in rdf.columns]
        if missing:
            raise ValueError(f"Missing required columns for Layer 2: {missing}")

        records = []
        
        for driver, grp in rdf.groupby("experience_driver"):
            driver = str(driver).strip()
            if not driver:
                continue

            size = len(grp)

            # INTENT AXIS distribution
            intent_dist = grp["intent_axis"].value_counts(normalize=True).to_dict()
            intent_dist_lower = {k.lower(): round(v, 4) for k, v in intent_dist.items()}

            # Determine intent focus
            intent_focus, _ = self.intent_focus_from_rows(grp, threshold=self.OU_CFG.get("action_threshold", 0.80))

            # ACTION AXIS distribution
            action_dist = grp["action_axis"].value_counts(normalize=True).to_dict()
            action_dist = {k: round(v, 4) for k, v in action_dist.items()}

            # AFFECT AXIS metrics (if present)
            affect_metrics = {}
            if "affect_axis" in grp.columns:
                affect_dist = grp["affect_axis"].value_counts(normalize=True).to_dict()
                affect_metrics["affect_distribution"] = {k: round(v, 4) for k, v in affect_dist.items()}
                
                # Compute affect pressure scores
                affect_primary = grp["affect_axis"].mode()[0] if not grp["affect_axis"].mode().empty else "Ambivalence"
                affect_metrics["affect_primary"] = affect_primary

            record = {
                "experience_driver": driver,
                "mention_count": size,
                "intent_axis_distribution": intent_dist_lower,
                "intent_axis_focus": intent_focus,
                "action_axis_distribution": action_dist,
                **affect_metrics,
            }

            records.append(record)

        return pd.DataFrame(records)

    # ========== BEHAVIORAL CLUSTERING ==========

    def cluster_behavior(
        self,
        rows: pd.DataFrame,
        driver: str,
        intent: str,
        action: str
    ) -> Tuple[pd.DataFrame, Dict, Dict, pd.DataFrame, Dict]:
        """
        Cluster behaviors within an (Experience Driver → Intent → Action) slice.
        Updated parameter names to reflect three-axis framework.
        
        Args:
            rows: DataFrame slice for this ED→Intent→Action combination
            driver: Experience driver label
            intent: Intent axis value (formerly feedback_type)
            action: Action axis value (formerly stream)
        
        Returns:
            - Clustered DataFrame with BCS groups
            - Full distribution dict
            - Cluster store (id → member rows)
            - DataFrame chunk for database
            - Full composites metadata
        """
        if rows.empty:
            return pd.DataFrame(), {}, {}, pd.DataFrame(), {}

        # Get signature fields
        sig_weights = self.OU_CFG.get("signature_weights", {})
        available_fields = [f for f in sig_weights.keys() if f in rows.columns]

        if not available_fields:
            if self.verbose:
                print(f"⚠️  No signature fields available for {driver} | {intent} | {action}")
            return pd.DataFrame(), {}, {}, pd.DataFrame(), {}

        # Build weighted signature
        signatures = []
        for _, row in rows.iterrows():
            parts = []
            for field in available_fields:
                val = row.get(field, "")
                if pd.notna(val) and str(val).strip():
                    weight = sig_weights.get(field, 1)
                    parts.extend([str(val).strip()] * weight)
            signatures.append(" ".join(parts) if parts else "")

        if not any(signatures):
            if self.verbose:
                print(f"⚠️  All signatures empty for {driver} | {intent} | {action}")
            return pd.DataFrame(), {}, {}, pd.DataFrame(), {}

        # Generate embeddings
        model = SentenceTransformer(self.OU_CFG["embedding_model"])
        embeddings = model.encode(signatures, convert_to_tensor=False, show_progress_bar=False)

        # Adaptive minimum cluster size
        adaptive_cfg = self.OU_CFG.get("adaptive_min_cluster", {})
        if adaptive_cfg.get("enabled", False):
            pct = adaptive_cfg.get("pct", 0.08)
            floor = adaptive_cfg.get("floor", 3)
            min_cluster_size = max(floor, int(len(rows) * pct))
        else:
            min_cluster_size = self.OU_CFG.get("min_cluster_size", 8)

        # Clustering with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)

        # Assign clusters
        rows = rows.copy()
        rows["bcs_cluster"] = cluster_labels
        rows["bcs_group_id"] = rows["bcs_cluster"].apply(
            lambda x: f"{driver[:20]}_{intent[:10]}_{action[:6]}_{x}" if x != -1 else f"{driver[:20]}_{intent[:10]}_{action[:6]}_singleton_{uuid4().hex[:8]}"
        )

        # Calculate cluster metadata
        cluster_store = {}
        full_composites = {}
        
        for gid, grp in rows.groupby("bcs_group_id"):
            cluster_size = len(grp)
            
            # Compute cohesion
            grp_indices = grp.index.tolist()
            grp_embeddings = embeddings[[i for i, idx in enumerate(rows.index) if idx in grp_indices]]
            
            if len(grp_embeddings) > 1:
                cohesion = float(np.mean(cosine_similarity(grp_embeddings)))
            else:
                cohesion = 1.0

            # Generate preview
            matters_vals = grp.get("matters", pd.Series([""] * len(grp)))
            matter_counts = Counter([str(m).strip() for m in matters_vals if pd.notna(m) and str(m).strip()])
            top_matters = [m for m, _ in matter_counts.most_common(3)]
            preview = " | ".join(top_matters) if top_matters else "No matters identified"

            # Store metadata
            full_composites[gid] = {
                "bcs_group_id": gid,
                "cluster_size": cluster_size,
                "bcs_share": round(cluster_size / len(rows), 4),
                "cluster_cohesion": round(cohesion, 4),
                "cluster_theme_preview": preview,
                "experience_driver": driver,
                "intent_axis": intent,
                "action_axis": action,
            }

            cluster_store[gid] = grp.to_dict("records")

        # Add cluster metadata to rows
        for gid in full_composites:
            mask = rows["bcs_group_id"] == gid
            rows.loc[mask, "cluster_cohesion"] = full_composites[gid]["cluster_cohesion"]
            rows.loc[mask, "cluster_theme_preview"] = full_composites[gid]["cluster_theme_preview"]

        # Distribution
        full_distribution = rows["bcs_group_id"].value_counts(normalize=True).to_dict()

        df_chunk = rows.copy()
        
        return rows, full_distribution, cluster_store, df_chunk, full_composites

    def _calculate_cluster_theme_distribution(self, clustered_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate distribution of cluster themes.
        """
        if clustered_df.empty or "cluster_theme_preview" not in clustered_df.columns:
            return {}
        
        counts = clustered_df.groupby("bcs_group_id")["cluster_theme_preview"].first().value_counts()
        total = counts.sum()
        
        if total == 0:
            return {}
        
        return {theme: round(count / total, 4) for theme, count in counts.items()}

    # ========== LAYER 3 DIAGNOSTICS ==========

    def compute_layer3_diagnostics(self, raw_df: pd.DataFrame, layer2_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate diagnostic layer showing Intent→Action relationships per SEU.
        Updates to three-axis framework.
        """
        rdf = self._prepare_raw_for_snapshot(raw_df)
        
        required = {"experience_driver", "intent_axis", "action_axis"}
        missing = [c for c in required if c not in rdf.columns]
        if missing:
            raise ValueError(f"Missing columns for Layer 3: {missing}")

        records = []
        
        for driver, grp in rdf.groupby("experience_driver"):
            driver = str(driver).strip()
            if not driver:
                continue

            # Get intent focus from Layer 2
            l2_row = layer2_df[layer2_df["experience_driver"] == driver]
            if l2_row.empty:
                continue

            intent_focus = self._safe_list(l2_row.iloc[0].get("intent_axis_focus", []))
            intent_dist = l2_row.iloc[0].get("intent_axis_distribution", {})

            if not intent_focus:
                continue

            # For each focused intent, show action breakdown
            for intent in intent_focus:
                intent_rows = grp[grp["intent_axis"] == intent]
                if intent_rows.empty:
                    continue

                action_dist = intent_rows["action_axis"].value_counts(normalize=True).to_dict()
                action_dist = {k: round(v, 4) for k, v in action_dist.items()}

                record = {
                    "experience_driver": driver,
                    "intent_axis": intent,
                    "intent_axis_audit_focus": intent_focus,
                    "intent_axis_distribution": intent_dist,
                    "action_axis_distribution": action_dist,
                    "mention_count": len(intent_rows),
                }

                records.append(record)

        return pd.DataFrame(records)

    # ========== GRANULAR SNAPSHOT ==========

    def compute_granular_details_snapshot(self, raw_df: pd.DataFrame, layer3_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build OU composites per (ED → Intent → Action) and persist clusters.
        Updated to three-axis framework.
        """
        if layer3_df is None or layer3_df.empty:
            raise ValueError("Layer 3 diagnostics must be supplied and non-empty.")

        must_cols = {"experience_driver", "intent_axis", "action_axis"}
        missing = [c for c in must_cols if c not in raw_df.columns]
        if missing:
            raise ValueError(f"Raw DF missing required columns: {missing}")

        # ✅ all normalization happens outside
        rdf = self._prepare_raw_for_snapshot(raw_df)

        thr = max(0.0, min(1.0, float(self.OU_CFG.get("action_threshold", 0.80))))

        all_df_chunks, all_full_composites, all_cluster_store = [], {}, {}
        records = []

        for _, hdr in layer3_df.iterrows():
            driver = str(hdr.get("experience_driver", "")).strip()
            if not driver:
                continue

            # ✅ layer3 already provides intent focus + dist
            intent_focus = self._safe_list(hdr.get("intent_axis_audit_focus", []))
            intent_dist = hdr.get("intent_axis_distribution", {}) or {}

            driver_rows = rdf[rdf["experience_driver"] == driver]
            if driver_rows.empty:
                continue

            for intent_key in intent_focus:
                intent_rows = driver_rows[driver_rows["intent_axis"] == intent_key]
                if intent_rows.empty:
                    continue

                # ✅ action dominance + distribution
                dominant_actions, action_distribution = self.action_focus_from_rows(intent_rows, threshold=thr)
                if not dominant_actions:
                    continue

                for action in dominant_actions:
                    action_rows = intent_rows[intent_rows["action_axis"] == action]
                    if action_rows.empty:
                        continue

                    clust_df, full_distribution, cluster_store, df_chunk, full_composites = self.cluster_behavior(
                        action_rows, 
                        driver=driver, 
                        intent=intent_key, 
                        action=action
                    )
                    cluster_theme_distribution = self._calculate_cluster_theme_distribution(clust_df)

                    if df_chunk is not None and not df_chunk.empty:
                        all_df_chunks.append(df_chunk)
                    all_cluster_store.update(cluster_store or {})
                    all_full_composites.update(full_composites or {})

                    if clust_df is not None and not clust_df.empty:
                        for gid, grp in clust_df.groupby("bcs_group_id"):
                            meta = (full_composites or {}).get(gid, {}) or {
                                "bcs_group_id": gid,
                                "cluster_size": int(len(grp)),
                                "bcs_share": round(len(grp) / max(len(clust_df), 1), 4),
                                "cluster_cohesion": float(grp.get("cluster_cohesion", pd.Series([0.0])).iloc[0]) if "cluster_cohesion" in grp.columns else 0.0,
                                "cluster_theme_preview": str(grp.get("cluster_theme_preview", pd.Series([""])).iloc[0]) if "cluster_theme_preview" in grp.columns else "",
                            }
                            records.append({
                                **meta,
                                "intent_axis_distribution": intent_dist,
                                "action_axis_distribution": action_distribution,
                                "cluster_theme_distribution": cluster_theme_distribution,
                            })

        if getattr(self, "verbose", False):
            print(f"\n📦 FINAL DEBUG SUMMARY")
            print(f"🔢 Total full_composites: {len(all_full_composites)}")
            print(f"🔢 Total cluster_store: {len(all_cluster_store)}")
            missing = [cid for cid in all_cluster_store if cid not in all_full_composites]
            print(f"❌ Missing composites for: {missing}")

        merged_df = pd.concat(all_df_chunks, ignore_index=True) if all_df_chunks else pd.DataFrame()
        if not merged_df.empty and all_cluster_store:
            self.create_cluster_database(
                df=merged_df,
                full_composites=all_full_composites,
                cluster_store=all_cluster_store,
                db_path="outputs/clusters.db"
            )

        return pd.DataFrame(records)

    def _apply_temporal_filter(self, details_df: pd.DataFrame, layer3_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply temporal intelligence filter to prioritize OUs.
        Updated to use affect_axis instead of emotion_primary.
        
        Args:
            details_df: Granular OU records from compute_granular_details_snapshot
            layer3_df: Layer 2 output with Emotion_Recency_Profile
        
        Returns:
            details_df with additional columns:
                - temporal_action: "ACTIVATE" or "DEFER"
                - temporal_confidence: 0-100
                - temporal_rationale: list of reason strings
        """
        from OU_prioritisation_V2 import TemporalIntelligenceFilter
        
        filter_engine = TemporalIntelligenceFilter()
        
        # Group OUs by experience_driver (each SEU gets filtered independently)
        filtered_rows = []
        
        for ed, ed_ous in details_df.groupby("experience_driver"):
            # Find matching SEU row
            seu_row = layer3_df[layer3_df["experience_driver"] == ed]
            
            if seu_row.empty:
                # No SEU match - pass through without filtering
                for _, ou_row in ed_ous.iterrows():
                    ou_row["temporal_action"] = "ACTIVATE"
                    ou_row["temporal_confidence"] = 50
                    ou_row["temporal_rationale"] = ["No SEU temporal data available"]
                    filtered_rows.append(ou_row)
                continue
            
            seu_dict = seu_row.iloc[0].to_dict()
            
            # Convert OU DataFrame rows to list of dicts
            ou_candidates = []
            for idx, ou_row in ed_ous.iterrows():
                ou_candidates.append({
                    "ou_id": ou_row.get("bcs_group_id"),
                    "ou_name": ou_row.get("cluster_theme_preview", "Unknown"),
                    "dominant_affect": ou_row.get("affect_axis"),  # Updated from dominant_emotion
                    "mention_count": ou_row.get("cluster_size", 1),
                    "_original_row": ou_row  # Keep reference to original
                })
            
            # Run filter
            result = filter_engine.filter_ous_by_temporal_intelligence(
                ou_candidates,
                seu_dict
            )
            
            # Merge results back into rows
            for ou in result["activate"]:
                row = ou["_original_row"].copy()
                decision = ou["temporal_decision"]
                row["temporal_action"] = "ACTIVATE"
                row["temporal_confidence"] = decision["confidence"]
                row["temporal_rationale"] = decision["rationale"]
                filtered_rows.append(row)
            
            for ou in result["defer"]:
                row = ou["_original_row"].copy()
                decision = ou["temporal_decision"]
                row["temporal_action"] = "DEFER"
                row["temporal_confidence"] = decision["confidence"]
                row["temporal_rationale"] = decision["rationale"]
                filtered_rows.append(row)
        
        # Reconstruct DataFrame
        filtered_df = pd.DataFrame(filtered_rows)
        
        # Sort: ACTIVATE first, then by confidence descending
        filtered_df["_sort_key"] = filtered_df["temporal_action"].map({"ACTIVATE": 0, "DEFER": 1})
        filtered_df = filtered_df.sort_values(
            by=["_sort_key", "temporal_confidence"],
            ascending=[True, False]
        ).drop(columns=["_sort_key"])
        
        if self.verbose:
            activated = (filtered_df["temporal_action"] == "ACTIVATE").sum()
            deferred = (filtered_df["temporal_action"] == "DEFER").sum()
            print(f"✅ Temporal filter: {activated} ACTIVATED, {deferred} DEFERRED")
        
        return filtered_df

    def create_cluster_database(
        self,
        df: pd.DataFrame,
        full_composites: Dict[str, Dict],
        cluster_store: Dict[str, List[Dict]],
        db_path: str = "outputs/clusters.db"
    ) -> None:
        """
        Persist cluster metadata and member details to SQLite.
        Updated column names to three-axis framework.
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables with updated schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_metadata (
                bcs_group_id TEXT PRIMARY KEY,
                experience_driver TEXT,
                intent_axis TEXT,
                action_axis TEXT,
                cluster_size INTEGER,
                bcs_share REAL,
                cluster_cohesion REAL,
                cluster_theme_preview TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bcs_group_id TEXT,
                experience_driver TEXT,
                intent_axis TEXT,
                action_axis TEXT,
                matters TEXT,
                context TEXT,
                semantic_customer_reality TEXT,
                FOREIGN KEY (bcs_group_id) REFERENCES cluster_metadata (bcs_group_id)
            )
        """)

        # Insert metadata
        for gid, meta in full_composites.items():
            cursor.execute("""
                INSERT OR REPLACE INTO cluster_metadata 
                (bcs_group_id, experience_driver, intent_axis, action_axis, 
                 cluster_size, bcs_share, cluster_cohesion, cluster_theme_preview)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                gid,
                meta.get("experience_driver", ""),
                meta.get("intent_axis", ""),
                meta.get("action_axis", ""),
                meta.get("cluster_size", 0),
                meta.get("bcs_share", 0.0),
                meta.get("cluster_cohesion", 0.0),
                meta.get("cluster_theme_preview", "")
            ))

        # Insert members
        for gid, members in cluster_store.items():
            for member in members:
                cursor.execute("""
                    INSERT INTO cluster_members
                    (bcs_group_id, experience_driver, intent_axis, action_axis,
                     matters, context, semantic_customer_reality)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    gid,
                    member.get("experience_driver", ""),
                    member.get("intent_axis", ""),
                    member.get("action_axis", ""),
                    member.get("matters", ""),
                    member.get("context", ""),
                    member.get("semantic_customer_reality", "")
                ))

        conn.commit()
        conn.close()

        if self.verbose:
            print(f"✅ Cluster database created: {db_path}")
            print(f"   - {len(full_composites)} clusters")
            print(f"   - {sum(len(members) for members in cluster_store.values())} total members")

    def run_full_pipeline(self) -> Dict[str, pd.DataFrame]:
        """
        Execute complete analysis pipeline with three-axis framework.
        
        Returns dict with:
            - layer2: SEU-level summary (Intent, Affect, Action distributions)
            - layer3: Intent→Action diagnostic view
            - details: Granular OU-level behavioral clusters
        """
        if self.verbose:
            print("\n" + "="*80)
            print("UNIVERSAL EXECUTION CONTRACT - THREE-AXIS ANALYSIS PIPELINE")
            print("="*80)
            print(f"Framework: Intent Axis | Affect Axis | Action Axis")
            print(f"Timeframe: {self.start_date} → {self.end_date}")
            print("="*80 + "\n")

        # Layer 2: SEU aggregation
        if self.verbose:
            print("🔄 Computing Layer 2: SEU Summary (Intent/Affect/Action distributions)...")
        self.layer2_df = self.compute_layer2_summary(self.raw_df)

        # Layer 3: Diagnostics
        if self.verbose:
            print("🔄 Computing Layer 3: Intent→Action diagnostics...")
        self.layer3_df = self.compute_layer3_diagnostics(self.raw_df, self.layer2_df)

        # Granular details
        if self.compute_granular and self.layer3_df is not None:
            if self.verbose:
                print("🔄 Computing Granular Layer: Behavioral clusters (OUs)...")
            self.details_df = self.compute_granular_details_snapshot(self.raw_df, self.layer3_df)
            
            # Apply temporal filtering if available
            if self.details_df is not None and not self.details_df.empty:
                if self.verbose:
                    print("🔄 Applying temporal intelligence filter...")
                self.details_df = self._apply_temporal_filter(self.details_df, self.layer2_df)

        if self.verbose:
            print("\n✅ Pipeline complete!")
            print(f"   Layer 2 (SEU): {len(self.layer2_df)} drivers")
            print(f"   Layer 3 (Diagnostics): {len(self.layer3_df)} intent slices")
            if self.details_df is not None:
                print(f"   Granular (OUs): {len(self.details_df)} behavioral clusters")

        return {
            "layer2": self.layer2_df,
            "layer3": self.layer3_df,
            "details": self.details_df
        }
