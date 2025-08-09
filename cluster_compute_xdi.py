from __future__ import annotations
    
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from rfi_policy import RFIPolicy, RFIPolicyConfig
    
import numpy as np
import sqlite3
import json
import os
import re
import uuid
import ast
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import acf
from typing import List, Dict, Any, Tuple
from collections import Counter
from uuid import uuid4     

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# === Flexible Date Range Analyzer ===
class FlexibleTimeframeAnalyzer:
    def __init__(self, input_path, output_dir="outputs", timeframe_days=75, compute_granular=True, verbose=True):
        """
        Flexible analyzer with plug-and-play timeframe logic and behavioral clustering setup.
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

        # Emotion scoring config (for ERI etc.)
        self.emotion_scores = {
            "Adoration": 3,
            "Appreciation": 1,
            "Ambivalence": 0,
            "Agitation": -1,
            "Anger": -3
        }

        # 👑 Signature weighting for behavioral clustering
        self.SIGNATURE_FIELDS = [
                "semantic_action_statement",       # 6x weight - THE GOLDMINE
                "matters",                         # 4x weight - BEHAVIORAL ESSENCE  
                "experience_driver",               # 3x weight - STRUCTURAL CONTEXT
                "opportunity_maximisation_stream", # 2x weight - STRATEGIC STREAM
                "context",                         # 2x weight - BEHAVIORAL BACKUP
                "customer_journey_stage"           # 1x weight - INTERACTION TIMING
                            ]

        # 🧠 Clustering parameters
        self.OU_CFG = {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "min_cluster_size": 8,
            "bcs_cumu_threshold": 0.80,
            "stream_threshold": 0.80,
            "skip_singletons": False
        }

        # Inside __init__ of FlexibleTimeframeAnalyzer
        self.model = SentenceTransformer(self.OU_CFG["embedding_model"])

        if self.verbose:
            self._print_initialization_summary()


    def _get_timeframe_suffix(self):
        if self.date_range_type == 'full':
            return "full"
        elif self.date_range_type == 'window':
            return f"{self.timeframe_days}d"
        else:
            start_str = self.start_date.strftime('%Y%m%d')
            end_str = self.end_date.strftime('%Y%m%d')
            return f"{start_str}_to_{end_str}"

    def _print_initialization_summary(self):
        print(f"⏱️ Analyzing {self.timeframe_days}-day window from {self.start_date} to {self.end_date}")
        print(f"📄 Input file: {self.input_path}")
        print(f"📁 Output directory: {self.output_dir}")

    def load_data(self):
        """Load and filter data based on the configured date range."""
        if not self.input_path or not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
            
        df = pd.read_csv(self.input_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        
        if self.date_range_type != 'full':
            df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]
            
        if self.verbose:
            print(f"📊 Loaded {len(df)} records for analysis")
            
        self.raw_df = df
        return df

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        if self.raw_df is None:
            self.load_data()
            
        # Instantiate Layer 2 with correct today_anchor and timeframe handling
        layer2 = Layer2Computer(
            df=self.raw_df,
            window_days=self.timeframe_days if self.date_range_type == 'window' else None,
            today_anchor=self.end_date,  # critical fix
            verbose=self.verbose
        )

        # Compute Layer 2 output
        self.layer2_df = layer2.compute()
       
        # Step 2: Layer 3 Analysis
        layer3 = Layer3Computer(
            layer2_df=self.layer2_df,
            raw_df=self.raw_df,
            timeframe_days=self.timeframe_days,
            today_anchor=self.end_date,   # ✅ pass max date as anchor
            verbose=self.verbose
        )

        self.layer3_df = layer3.compute()
        print(self.layer3_df.head())
        
        # Add time window label
        if self.date_range_type == 'full':
            window_label = "Full History"
        else:
            window_label = f"{self.start_date} to {self.end_date}"
        self.layer3_df["window"] = window_label
        
        # Step 3: Compute Emotional Focus
        focus_df = self.compute_emotional_focus()
        self.layer3_df = self.layer3_df.merge(focus_df, on="experience_driver", how="left")
        self.layer3_df["status"] = "Computed"

        # 🔽 NEW: Save output to CSV for inspection
        output_path = "outputs/layer3_output_debug.csv"
        os.makedirs("outputs", exist_ok=True)
        self.layer3_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ Layer 3 output saved to: {output_path}")
                
        # ✅ Save Layer 3 Phase 2 equivalent to longitudinal SQLite here
        self._save_longitudinal_sqlite()  # ✅ Your dedicated save function

         # Step 4: Granular Entity × Emotion × Opportunity Stream storytelling units
        if self.compute_granular:
            self.details_df = self.compute_granular_details_snapshot(self.raw_df, self.layer3_df)
            self._save_granular_sqlite()
            
        return self.layer3_df
    
    def _save_longitudinal_sqlite(self):
        """Save Layer 3 Phase 2 equivalent data (longitudinal summary) to SQLite."""
        if self.layer3_df is not None and not self.layer3_df.empty:
            try:
                # 🔧 Convert unsupported types (list/dict) to JSON strings
                for col in self.layer3_df.columns:
                    if self.layer3_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                        print(f"⚠️ Converting complex type in column '{col}' to JSON")
                        self.layer3_df[col] = self.layer3_df[col].apply(json.dumps)

                conn = sqlite3.connect(self.summary_sqlite_path)
                self.layer3_df.to_sql("sentientsignal_longitudinal", conn, if_exists="replace", index=False)
                conn.close()
                print(f"✅ Longitudinal summary saved to {self.summary_sqlite_path}")
            except Exception as e:
                print(f"❌ Error saving longitudinal summary: {e}")
        else:
            print("⚠️ No Layer 3 Phase 2 data available to save.")

    def _save_granular_sqlite(self):
        """Save full OU storytelling payloads (Layer 3 Phase 3) to SQLite."""
        if self.details_df is not None and not self.details_df.empty:
            try:
                # 🔧 Convert unsupported types (list/dict) to JSON strings
                for col in self.details_df.columns:
                    if self.details_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                        print(f"⚠️ Converting complex type in column '{col}' to JSON")
                        self.details_df[col] = self.details_df[col].apply(json.dumps)

                conn = sqlite3.connect(self.granular_sqlite_path)
                self.details_df.to_sql("entity_emotion_details", conn, if_exists="replace", index=False)
                conn.close()
                if self.verbose:
                    print(f"✅ Granular storytelling data saved to {self.granular_sqlite_path}")
            except Exception as e:
                print(f"❌ Error saving granular storytelling units: {e}")
        else:
            if self.verbose:
                print("⚠️ No granular storytelling units to save.")

    def compute_emotional_focus(self):
        """Compute dominant 5-group emotions and their distribution per experience_driver (80% threshold logic applied)."""
        df = self.raw_df
        valid_entities = set(self.layer3_df["experience_driver"].unique())

        # Canonical emotion group mapping
        canonical_map = {
            "adoration": "Adoration",
            "appreciation": "Appreciation",
            "ambivalence": "Ambivalence",
            "agitation": "Agitation",
            "anger": "Anger"
        }

        def get_dominant_emotions(group_df, threshold=0.8):
            mapped = (
                group_df["emotion_primary"]
                .dropna()
                .str.lower()
                .map(canonical_map)
                .dropna()
            )

            emotion_counts = mapped.value_counts()
            total = emotion_counts.sum()
            if total == 0:
                return pd.Series({
                    "emotional_audit_focus": [],
                    "emotion_distribution": {}
                })

            dominant_emotions = []
            cumulative = 0.0
            for emotion, count in emotion_counts.items():
                pct = count / total
                dominant_emotions.append(emotion)
                cumulative += pct
                if cumulative >= threshold:
                    break

            # Full distribution
            distribution = {k: round((v / total) * 100, 1) for k, v in emotion_counts.items()}

            return pd.Series({
                "emotional_audit_focus": dominant_emotions,
                "emotion_distribution": distribution
            })

        filtered_df = df[df["experience_driver"].isin(valid_entities)].copy()

        grouped = filtered_df.groupby("experience_driver").apply(get_dominant_emotions)

        # 🔧 Fix: Avoid inserting duplicate 'entity_type' column on reset_index
        if 'experience_driver' in grouped.index.names and 'experience_driver' in grouped.columns:
            grouped.index.name = None

        result = grouped.reset_index()

        return result

    """
    DecipherOS - Layer - 4 Behavioural Clustering Engine
    =================================================
    Implements the locked four-layer hierarchy:
        Experience Driver → Emotion (≥80%) → Opportunity Stream (≥80%) → Behavioural Cluster (≥80%)
    
    For every surviving Behavioural Cluster this module emits canonical Orchestration-Unit fields plus
    cluster metadata (share, size, cohesion). 
    
    Key design notes
    ----------------
    * Weighted signatures (MATTERS > CONTEXT/KEYWORDS > frame markers) drive dense, coherent clusters.
    * Adaptive algorithm choice: Agglomerative for small sets, HDBSCAN for scale.
    * 80% cumulative filters applied to both emotion and stream levels to prevent OU explosion.
    * Single-row combos are skipped entirely (no action required) - handled with a lightweight guard.
    """
       
    # ── Signature Builder Utilities ────────────────────────────────────────────────

    def _most(self, series: pd.Series) -> Any:
        """Return the mode of a Series or None if empty."""
        vals = series.dropna().tolist()
        return Counter(vals).most_common(1)[0][0] if vals else None

    def _build_signature(self, row: pd.Series) -> str:
        """
        🔥 SEMANTIC SIGNATURE BUILDER
        Optimized for maximum clustering accuracy by prioritizing behavioral essence
        over noise fields that create false similarities.
        """
        toks: List[str] = []
        
        for fld in self.SIGNATURE_FIELDS:
            val = row.get(fld)
            if pd.isna(val):
                continue
                
            # 🔧 Flatten list-like fields (esp. keywords)
            if isinstance(val, list):
                val = " ".join([str(v) for v in val])
            
            # 🎯 TIER 1: SEMANTIC GOLDMINES (Heavy Weight)
            if fld == "semantic_action_statement":
                toks += [str(val)] * 6  # PRIMARY DRIVER - captures both customer reality + strategic response
            elif fld == "matters":
                toks += [str(val)] * 4  # CORE BEHAVIORAL ESSENCE - the crux of the issue
                
            # 🎯 TIER 2: STRUCTURAL CONTEXT (Medium Weight)
            elif fld == "experience_driver":
                toks += [str(val)] * 3  # Distinguishes "Product Availability" vs "Mobile App Performance"
            elif fld == "opportunity_maximisation_stream":
                toks += [str(val)] * 2  # Fix/Optimize/Innovate/Amplify strategic context
                
            # 🎯 TIER 3: CONTEXTUAL SUPPORT (Light Weight)
            elif fld == "context":
                toks += [str(val)] * 2  # Behavioral backup details
            elif fld == "customer_journey_stage":
                toks.append(str(val))   # Interaction timing context
                
            else:
                # Fallback for any other fields (minimal weight)
                toks.append(str(val))
        
        return " ".join(toks)

    def get_emotional_focus_and_distribution(group_df: pd.DataFrame, *, emotion_col: str = "emotion_primary", threshold: float = 0.8) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
        """Return dominant emotions, percentage distribution, and normalised shares."""
        canonical_map = {
            "adoration": "Adoration",
            "appreciation": "Appreciation",
            "ambivalence": "Ambivalence",
            "agitation": "Agitation",
            "anger": "Anger",
        }
    
        mapped = (
            group_df[emotion_col]
            .dropna()
            .str.lower()
            .map(canonical_map)
            .dropna()
        )
    
        emotion_counts = mapped.value_counts()
        total = emotion_counts.sum()
        if total == 0:
            return [], {}, {}
    
        dominant_emotions: List[str] = []
        cumulative = 0.0
        for emotion, count in emotion_counts.items():
            pct = count / total
            dominant_emotions.append(emotion)
            cumulative += pct
            if cumulative >= threshold:
                break
    
        distribution = {k: round((v / total) * 100, 1) for k, v in emotion_counts.items()}
        normalised = {k: round(v / total, 3) for k, v in emotion_counts.items()}
        return dominant_emotions, distribution, normalised

    def apply_stream_threshold_and_distribution(self, stream_counts: pd.Series, *, threshold: float = 0.8) -> Tuple[List[str], Dict[str, float]]:
        """Return dominant streams (up to threshold) and full stream percentage breakdown."""
        total = stream_counts.sum()
        if total == 0:
            return [], {}

        dominant, cumulative = [], 0.0
        stream_distribution = {}

        for stream, count in stream_counts.items():
            pct = round(count / total, 4)
            stream_distribution[stream] = pct
            if cumulative < threshold:
                dominant.append(stream)
                cumulative += pct

        return dominant, stream_distribution

    def _distill_matters_label(self, matters_list: List[str]) -> str:
        """
        Return the most semantically central matters string from a cluster.
        """
        if not matters_list:
            return "No matters label available"

        model = SentenceTransformer(self.OU_CFG["embedding_model"])
        embeddings = model.encode(matters_list, convert_to_tensor=True, normalize_embeddings=True)

        centroid = embeddings.mean(dim=0, keepdim=True)
        sims = util.cos_sim(embeddings, centroid).squeeze(0)

        best_idx = sims.argmax().item()
        return matters_list[best_idx]
     
    def extract_batch_1_fields(self, cluster_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract Batch 1 fields (Deterministic Layer) for a given cluster.
        Uses direct inheritance for constant fields and _most() for high-confidence modes.
        """
        composite = {}

        # ── Group A: Cluster-Level Constants ──
        composite["experience_driver"] = cluster_df["experience_driver"].iloc[0]
        composite["emotion"] = cluster_df["emotion"].iloc[0]
        composite["opportunity_stream"] = cluster_df["opportunity_stream"].iloc[0]

        # ── Group B: Semi-Deterministic High-Confidence ──
        composite["feedback_type"] = self._most(cluster_df["feedback_type"]) or "Unknown"
        composite["theme"] = self._most(cluster_df["theme"]) or "Unknown"

        return composite
    
    def process_batch_2_fields(self, cluster_df: pd.DataFrame, batch_1_fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all Batch 2 semi-synthetic fields
        
        Args:
            cluster_df: DataFrame containing cluster rows
            batch_1_fields: Already processed deterministic fields from Batch 1
            
        Returns:
            Dictionary of processed Batch 2 fields
        """
        batch_2_composite = {}
        
        # ── SEMANTIC CORE FIELDS ──
        batch_2_composite['context'] = self._semantic_centroid_fusion(cluster_df['context'].tolist())
        batch_2_composite['keywords'] = self._dedupe_and_merge_keywords(cluster_df['keywords'].tolist())
        
        # ── OPERATIONAL DESCRIPTORS ──
        batch_2_composite['interaction_moment'] = self._semantic_mode(cluster_df['interaction_moment'].tolist())
        batch_2_composite['customer_journey_stage'] = self._semantic_mode(cluster_df['customer_journey_stage'].tolist())
        batch_2_composite['customer_journey'] = self._semantic_mode(cluster_df['customer_journey'].tolist())
        batch_2_composite['customer_effort_score'] = self._weighted_average_effort_score(cluster_df['customer_effort_score'].tolist())
        
        # ── STRUCTURAL IDENTIFIERS ──
        batch_2_composite['entity_name'] = self._extract_entity_name(cluster_df['entity_name'].tolist())
             
        return batch_2_composite
     
    def _semantic_centroid_fusion(self, context_list: List[str]) -> str:
        """
        Fuse multiple context strings into one representative context
        """
        if not context_list:
            return "No context available"
        
        # Clean contexts
        clean_contexts = [str(c).strip() for c in context_list if pd.notna(c) and str(c).strip()]
        if not clean_contexts:
            return "No context available"
        
        # If contexts are very similar, return the most frequent
        context_counts = Counter(clean_contexts)
        if len(context_counts) <= 3:  # Few unique contexts
            return context_counts.most_common(1)[0][0]
        
        # Find semantic centroid
        embeddings = self.model.encode(clean_contexts, normalize_embeddings=True)
        centroid = embeddings.mean(axis=0, keepdims=True)
        
        # Find most representative context
        similarities = cosine_similarity(embeddings, centroid).flatten()
        best_idx = np.argmax(similarities)
        return clean_contexts[best_idx]
    
    def _dedupe_and_merge_keywords(self, keywords_list: List[Any]) -> List[str]:
        """
        Deduplicate and merge keywords from multiple rows
        """
        all_keywords = []
        
        for kw in keywords_list:
            if pd.isna(kw):
                continue
            
            # Handle different keyword formats
            if isinstance(kw, list):
                all_keywords.extend([str(k).strip().lower() for k in kw])
            elif isinstance(kw, str):
                # Handle string representations of lists or comma-separated
                kw_clean = kw.strip()
                if kw_clean.startswith('[') and kw_clean.endswith(']'):
                    # Parse list-like string
                    kw_clean = kw_clean[1:-1].replace("'", "").replace('"', '')
                all_keywords.extend([k.strip().lower() for k in kw_clean.split(',')])
            else:
                all_keywords.append(str(kw).strip().lower())
        
        # Deduplicate and filter
        unique_keywords = list(set([kw for kw in all_keywords if kw and len(kw) > 1]))
        
        # Sort by frequency (most common first)
        keyword_counts = Counter(unique_keywords)
        return [kw for kw, count in keyword_counts.most_common()]
    
    def _semantic_mode(self, values_list: List[str]) -> str:
        """
        Get the most frequent value, with semantic validation for ties
        """
        if not values_list:
            return "Unknown"
        
        # Clean values
        clean_values = [str(v).strip() for v in values_list if pd.notna(v) and str(v).strip()]
        if not clean_values:
            return "Unknown"
        
        # Get frequency counts
        value_counts = Counter(clean_values)
        
        # If clear winner, return it
        if len(value_counts) == 1:
            return list(value_counts.keys())[0]
        
        most_common = value_counts.most_common()
        if most_common[0][1] > most_common[1][1]:
            return most_common[0][0]
        
        # Handle ties with semantic similarity
        tied_values = [val for val, count in most_common if count == most_common[0][1]]
        if len(tied_values) > 1:
            # Use semantic similarity to pick the most representative
            embeddings = self.model.encode(tied_values, normalize_embeddings=True)
            centroid = embeddings.mean(axis=0, keepdims=True)
            similarities = cosine_similarity(embeddings, centroid).flatten()
            best_idx = np.argmax(similarities)
            return tied_values[best_idx]
        
        return tied_values[0]
    
    def _weighted_average_effort_score(self, scores_list: List[Any]) -> int:
        """
        Calculate weighted average effort score, rounded to nearest integer
        """
        import statistics
        if not scores_list:
            return 4  # Default middle score
        
        # Clean and convert scores
        clean_scores = []
        for score in scores_list:
            if pd.notna(score):
                try:
                    clean_scores.append(float(score))
                except (ValueError, TypeError):
                    continue
        
        if not clean_scores:
            return 4
        
        # Calculate weighted average (simple mean for now)
        avg_score = statistics.mean(clean_scores)
        return round(avg_score)
    
    def _extract_entity_name(self, entity_names: List[str]) -> str:
        """
        Extract the most representative entity name
        """
        if not entity_names:
            return "Unknown Entity"
        
        # Clean entity names
        clean_names = [str(name).strip() for name in entity_names if pd.notna(name) and str(name).strip()]
        if not clean_names:
            return "Unknown Entity"
        
        # Return most frequent
        name_counts = Counter(clean_names)
        return name_counts.most_common(1)[0][0]

    
    def cluster_behavior(self, df: pd.DataFrame, driver: str, emotion: str, stream: str) -> Tuple[
    pd.DataFrame, List[Dict[str, Any]], Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Clusters a slice of rows into Behavioural Clusters.

        Returns:
            - filtered_df: rows from dominant clusters (≥80% cumulative share)
            - full_distribution: metadata dicts per cluster
            - cluster_store: full cluster DataFrames keyed by bcs_group_id
            - df: full annotated DataFrame
            - full_composites: cluster-level metadata keyed by bcs_group_id
        """
        from uuid import uuid4
       
        df = df.copy()
        df["signature"] = df.apply(self._build_signature, axis=1).str.lower()
        df = df.reset_index(drop=True)

        mod = SentenceTransformer(self.OU_CFG["embedding_model"])
        embeds = mod.encode(df["signature"].tolist(), normalize_embeddings=True, show_progress_bar=False)
        total_rows = len(df)

        # ── Adaptive Clustering ─────────────────────────────
        if total_rows < 2:
            labels = np.array([0] * total_rows)
        elif total_rows < 200:
            labels = AgglomerativeClustering(
                metric="cosine", 
                linkage="average", 
                distance_threshold=0.25, 
                n_clusters=None
            ).fit_predict(embeds)
        else:
            labels = hdbscan.HDBSCAN(
                metric="cosine",
                min_cluster_size=self.OU_CFG["min_cluster_size"],
                min_samples=2
            ).fit_predict(embeds)

        df["local_bcs_id"] = labels.astype(str)

        # 🔁 Post-fix HDBSCAN noise (-1) as unique singleton clusters
        noise_mask = df["local_bcs_id"] == "-1"
        df.loc[noise_mask, "local_bcs_id"] = df[noise_mask].apply(lambda _: str(uuid4()), axis=1)

        # ── Core Vars ───────────────────────────────────────
        prefix = f"{driver[:8]}_{emotion[:3]}_{stream[:3]}".lower()
        cluster_store: Dict[str, pd.DataFrame] = {}
        full_composites: Dict[str, Dict[str, Any]] = {}
        cluster_metadata: Dict[str, Dict[str, Any]] = {}

        # ── Per Cluster ─────────────────────────────────────
        df["bcs_group_id"] = None
        df["bcs_id"] = None
        df["bcs_share"] = None
        df["bcs_label"] = None
        df["cluster_cohesion"] = None
        
        for local_cid, grp in df.groupby("local_bcs_id"):
            # 🔒 Unique ID generation
            unique_part = uuid4().hex[:8]
            group_id = f"{prefix}_{unique_part}"
            row_ids = [uuid4().hex for _ in range(len(grp))]

            grp = grp.copy()
            grp["bcs_group_id"] = group_id
            grp["bcs_id"] = row_ids
            grp["bcs_share"] = len(grp) / total_rows

            # 🔍 Cohesion
            vecs = embeds[grp.index]
            if len(vecs) <= 1:
                cohesion = 1.0
            else:
                centroid = vecs.mean(axis=0, keepdims=True)
                sims = cosine_similarity(vecs, centroid).ravel()
                cohesion = float(sims.mean())

            # 🏷 Preview Label
            matters_list = grp["matters"].dropna().astype(str).tolist()
            raw_label = self._distill_matters_label(matters_list) or f"Cluster {group_id}"
            preview = (raw_label or "No preview available").strip().capitalize()
            truncated_preview = (preview[:77] + "…") if len(preview) > 80 else preview

            # 📦 Composite Construction
            first_row = grp.iloc[0]
            if len(grp) == 1:
                composite = {
                    "bcs_id": first_row["bcs_id"],
                    "bcs_group_id": group_id,
                    "cluster_size": 1,
                    "bcs_share": round(1 / total_rows, 4),
                    "cluster_cohesion": 1.0,
                    "cluster_theme_preview": truncated_preview,
                    **{k: first_row.get(k) for k in [
                        "customer_review", "experience_driver", "emotion", "opportunity_stream", "feedback_type",
                        "customer_journey", "customer_journey_stage", "interaction_moment",
                        "context", "keywords", "entity_name", "theme", "customer_effort_score",
                        "semantic_action_statement", "stream_justification", "matters", "behavioral_impact"
                    ]}
                }
            else:
                batch_1_fields = self.extract_batch_1_fields(grp)
                batch_2_fields = self.process_batch_2_fields(grp, batch_1_fields)
                composite = {
                    **batch_1_fields,
                    **batch_2_fields,
                    "bcs_id": first_row["bcs_id"],
                    "bcs_group_id": group_id,
                    "cluster_size": len(grp),
                    "bcs_share": round(len(grp) / total_rows, 4),
                    "cluster_cohesion": round(cohesion, 4),
                    "cluster_theme_preview": truncated_preview,
                    "semantic_action_statement": None,
                    "stream_justification": None,
                    "matters": None,
                    "behavioral_impact": None,
                    "comment_review": first_row.get("comment_review")
                }

            # 🧠 Store
            df.update(grp)
            cluster_store[group_id] = grp
            full_composites[group_id] = composite
            cluster_metadata[group_id] = {"label": truncated_preview, "cohesion": cohesion}

        # 🔁 Final Annotation
        df["bcs_label"] = df["bcs_group_id"].map(lambda gid: cluster_metadata.get(gid, {}).get("label"))
        df["cluster_cohesion"] = df["bcs_group_id"].map(lambda gid: cluster_metadata.get(gid, {}).get("cohesion"))

        # 🎯 Dominant Cluster Filter
        dominant_ids = []
        cumulative_share = 0.0
        cluster_order = df["bcs_group_id"].value_counts(normalize=True)
        for cid, share in cluster_order.items():
            dominant_ids.append(cid)
            cumulative_share += share
            if cumulative_share >= self.OU_CFG["bcs_cumu_threshold"]:
                break

        filtered_df = df[df["bcs_group_id"].isin(dominant_ids)].copy()

        return filtered_df, list(full_composites.values()), cluster_store, df, full_composites


    def create_cluster_database(self, df: pd.DataFrame, full_composites: Dict[str, Dict[str, Any]],
            cluster_store: Dict[str, pd.DataFrame], db_path: str = "clusters.db"):
        """
        Initializes the DB and saves each cluster (single or multi) with correct schema.
        Now includes bcs_group_id tracking to group cluster rows.
        """
        self.db_path = db_path
        self.init_database()

        for cid, grp in cluster_store.items():
            composite = full_composites.get(cid)
            if composite is None:
                print(f"⚠️ Skipping cluster {cid}: no composite data found.")
                continue

            # Attach group ID to each row before saving
            cluster_group_id = cid  # Use cluster_store key as group ID
            grp = grp.copy()
            grp["bcs_group_id"] = cluster_group_id

            if len(grp) == 1:
                self.save_single_cluster_to_db(grp, composite, cluster_group_id)
            else:
                self.save_multi_cluster_to_db(grp, composite, cluster_group_id)

        print("✅ All clusters saved to database.")


    def init_database(self):
        """Initialize the SQLite database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Drop table if exists (for fresh start)
        cursor.execute('DROP TABLE IF EXISTS clusters')
        
        # Create table schema with bcs_id as PRIMARY KEY
        cursor.execute('''
            CREATE TABLE clusters (
                bcs_id TEXT PRIMARY KEY,
                bcs_group_id TEXT,
                cluster_size INTEGER,
                bcs_share REAL,
                cluster_cohesion REAL,
                cluster_theme_preview TEXT,
                
                -- Batch 1 & 2 fields (cluster-level, replicated)
                customer_review TEXT,
                experience_driver TEXT,
                emotion TEXT,
                theme TEXT,
                opportunity_stream TEXT,
                feedback_type TEXT,
                customer_journey TEXT,
                customer_journey_stage TEXT,
                interaction_moment TEXT,
                context TEXT,
                keywords TEXT,
                entity_name TEXT,
                customer_effort_score REAL,
                
                -- Individual row fields (row-specific, preserved)
                semantic_action_statement TEXT,
                stream_justification TEXT,
                matters TEXT,
                behavioral_impact TEXT,
                       
                -- NEW: Injected by synthesis step
                problem_statement TEXT
                            )
        ''')
        
        # Now create index separately
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bcs_group_id ON clusters (bcs_group_id)')

        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")

    def save_single_cluster_to_db(self, grp: pd.DataFrame, composite: Dict[str, Any], cid: str):
        """Save single-row cluster to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        row = grp.iloc[0]
        bcs_id = row.get("bcs_id")
        bcs_group_id = row.get("bcs_group_id")

        # Handle keywords properly
        keywords_str = ""
        if isinstance(row.get('keywords'), list):
            keywords_str = ", ".join(str(k) for k in row.get('keywords', []))
        else:
            keywords_str = str(row.get('keywords', ''))

        # Insert the single row as-is
        cursor.execute('''
            INSERT OR REPLACE INTO clusters (
                bcs_id, bcs_group_id, cluster_size, bcs_share, cluster_cohesion, cluster_theme_preview,
                customer_review, experience_driver, emotion, theme, opportunity_stream, feedback_type,
                customer_journey, customer_journey_stage, interaction_moment, context,
                keywords, entity_name, customer_effort_score,
                semantic_action_statement, stream_justification, matters, behavioral_impact, problem_statement
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            bcs_id, bcs_group_id, 
            int(composite.get('cluster_size', 1)), 
            float(composite.get('bcs_share', 0.0)), 
            float(composite.get('cluster_cohesion', 0.0)), 
            composite.get('cluster_theme_preview', ''),
            row.get('customer_review'),
            row.get('experience_driver'), row.get('emotion'), row.get('theme'),
            row.get('opportunity_stream'), row.get('feedback_type'),
            row.get('customer_journey'), row.get('customer_journey_stage'),
            row.get('interaction_moment'), row.get('context'),
            keywords_str,
            row.get('entity_name'), 
            float(row.get('customer_effort_score', 0.0) or 0.0),
            row.get('semantic_action_statement'), row.get('stream_justification'),
            row.get('matters'), row.get('behavioral_impact'),
            row.get('problem_statement')  # 🆕 Inject new field
        ))

        conn.commit()
        conn.close()
        print(f"💾 Saved single-row cluster {cid}")

    def save_multi_cluster_to_db(self, grp: pd.DataFrame, composite: Dict[str, Any], cid: str):
        """Save multi-row cluster to database - replicate batch fields, preserve individual fields"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ✅ Safely extract replicated (batch) fields from the composite
        keywords_str = ""
        if isinstance(composite.get('keywords'), list):
            keywords_str = ", ".join(str(k) for k in composite.get('keywords', []))
        else:
            keywords_str = str(composite.get('keywords', ''))

        replicated_data = {
            'experience_driver': composite.get('experience_driver'),
            'emotion': composite.get('emotion'),
            'theme': composite.get('theme'),
            'opportunity_stream': composite.get('opportunity_stream'),
            'feedback_type': composite.get('feedback_type'),
            'customer_journey': composite.get('customer_journey'),
            'customer_journey_stage': composite.get('customer_journey_stage'),
            'interaction_moment': composite.get('interaction_moment'),
            'context': composite.get('context'),
            'keywords': keywords_str,
            'entity_name': composite.get('entity_name'),
            'customer_effort_score': float(composite.get('customer_effort_score', 0.0) or 0.0)
        }

        # 🔁 Insert each row in the cluster with shared + individual fields
        for _, row in grp.iterrows():
            bcs_id = row.get("bcs_id")
            bcs_group_id = row.get("bcs_group_id")
            cursor.execute('''
                INSERT OR REPLACE INTO clusters (
                    bcs_id, bcs_group_id, cluster_size, bcs_share, cluster_cohesion, cluster_theme_preview,
                    customer_review, experience_driver, emotion, theme, opportunity_stream, feedback_type,
                    customer_journey, customer_journey_stage, interaction_moment, context,
                    keywords, entity_name, customer_effort_score,
                    semantic_action_statement, stream_justification, matters, behavioral_impact, problem_statement
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                bcs_id, bcs_group_id,
                int(composite.get('cluster_size', 1)),
                float(composite.get('bcs_share', 0.0)),
                float(composite.get('cluster_cohesion', 0.0)),
                composite.get('cluster_theme_preview', ''),
                row.get('customer_review'),

                # ✅ Shared fields
                replicated_data['experience_driver'], replicated_data['emotion'],
                replicated_data['theme'], replicated_data['opportunity_stream'],
                replicated_data['feedback_type'], replicated_data['customer_journey'],
                replicated_data['customer_journey_stage'], replicated_data['interaction_moment'],
                replicated_data['context'], replicated_data['keywords'],
                replicated_data['entity_name'], replicated_data['customer_effort_score'],

                # ✅ Per-row fields
                row.get('semantic_action_statement'), row.get('stream_justification'),
                row.get('matters'), row.get('behavioral_impact'), row.get('problem_statement')
            ))

        conn.commit()
        conn.close()
        print(f"💾 Saved multi-row cluster {cid} with {len(grp)} rows")

    def compute_granular_details_snapshot(self, raw_df: pd.DataFrame, layer3_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute OU-level cluster summaries for Signal Capsule display only.
        This version excludes semantic fields (context, matters, etc.) which will be LLM-generated later.
        """
        
        all_df_chunks = []
        all_full_composites = {}
        all_cluster_store = {}

        if layer3_df is None:
            raise ValueError("Layer 3 diagnostics must be supplied.")

        records: List[Dict[str, Any]] = []

        for _, hdr in layer3_df.iterrows():
            driver = hdr["experience_driver"]
            emotion_focus = hdr.get("emotional_audit_focus", [])

            # 🔥 FIX: Convert string representation back to actual list
            if isinstance(emotion_focus, str):
                try:
                    emotion_focus = ast.literal_eval(emotion_focus)  # Convert string to list
                except (ValueError, SyntaxError):
                    emotion_focus = []  # Fallback to empty list if parsing fails

            emotion_dist = hdr.get("emotion_distribution", {})

            driver_rows = raw_df[raw_df["experience_driver"] == driver]

            for emotion in emotion_focus:
                emotion_rows = driver_rows[driver_rows["emotion_primary"].str.lower() == emotion.lower()]
                if emotion_rows.empty:
                    continue

                stream_counts = emotion_rows["opportunity_stream"].value_counts()
                dominant_streams, stream_distribution = self.apply_stream_threshold_and_distribution(
                    stream_counts, threshold=self.OU_CFG["stream_threshold"]
                )

                if not dominant_streams:
                    continue  # ✅ Skip if no dominant stream found

                for stream in dominant_streams:
                    stream_rows = emotion_rows[emotion_rows["opportunity_stream"] == stream]
                    if stream_rows.empty:
                        continue  # Skip if nothing to cluster

                    clust_df, full_distribution, cluster_store, df_chunk, full_composites = self.cluster_behavior(
                                                        stream_rows, driver=driver, emotion=emotion, stream=stream)


                    # Accumulate for final DB creation
                    all_df_chunks.append(df_chunk)
                    all_cluster_store.update(cluster_store)
                    all_full_composites.update(full_composites)

                    # Create lookup for quick access
                    meta_lookup = {meta["bcs_id"]: meta for meta in full_distribution}

                    for cid, grp in clust_df.groupby("bcs_id"):
                        meta = meta_lookup.get(cid, {})

                        # Inject cluster-local overrides
                        composite = {
                            **meta,
                            "emotion_distribution": emotion_dist,
                            "stream_distribution": stream_distribution,
                        }

                        records.append(composite)

        print(f"\n📦 FINAL DEBUG SUMMARY")
        print(f"🔢 Total full_composites: {len(all_full_composites)}")
        print(f"🔢 Total cluster_store: {len(all_cluster_store)}")
        missing = [cid for cid in all_cluster_store if cid not in all_full_composites]
        print(f"❌ Missing composites for: {missing}")
  
        # Create DB once using full data
        merged_df = pd.concat(all_df_chunks, ignore_index=True)
        self.create_cluster_database(
                df=merged_df,
                full_composites=all_full_composites,
                cluster_store=all_cluster_store,
                db_path="outputs/clusters.db"
            )

        return pd.DataFrame(records)
          
    def _infer_owner(self, df):
        """
        Infer the likely team owner based on customer journey stage and matters (root cause proxy).
        """
        stage = df["customer_journey_stage"].dropna().astype(str).str.lower().unique()
        matters = df["matters"].dropna().astype(str).str.lower().unique()

        weights = {
            "Product Team": 0,
            "Operations": 0,
            "Engineering": 0,
            "CX": 0,
            "Marketing": 0
        }

        # Weighting based on journey stage
        for val in stage:
            if any(x in val for x in ["checkout", "payment", "cart"]):
                weights["Product Team"] += 2
            if any(x in val for x in ["delivery", "order", "fulfillment"]):
                weights["Operations"] += 2
            if any(x in val for x in ["promotion", "pricing", "discount"]):
                weights["Marketing"] += 2
            if any(x in val for x in ["search", "discovery", "filter"]):
                weights["Product Team"] += 1

        # Weighting based on matters (root cause proxy)
        for val in matters:
            if any(x in val for x in ["system", "technical", "crash", "bug", "slow", "app"]):
                weights["Engineering"] += 2
            if any(x in val for x in ["communication", "confusion", "expectation", "instruction"]):
                weights["CX"] += 2
            if any(x in val for x in ["delivery", "driver", "delay", "slot", "incomplete"]):
                weights["Operations"] += 2

        top_team = max(weights, key=weights.get)
        return "CX / Ops" if weights[top_team] == 0 else top_team

  
# === Layer 2 Logic ===
class Layer2Computer:
    def __init__(self, df, window_days=None, today_anchor=None, verbose=False):
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"]).dt.date
        self.today = today_anchor if today_anchor else datetime.now().date()
        self.layer2_df = None
        self.verbose = verbose
        self.window_days = window_days
        self.timeframe_days = window_days

        self.emotion_scores = {
            "Adoration": 3,
            "Appreciation": 1,
            "Ambivalence": 0,
            "Agitation": -1,
            "Anger": -3
        }

        self.priority_matrix = self._define_priority_matrix()
        self.quadrant_purpose = self._define_quadrant_purpose()


    def _define_priority_matrix(self):
        return {
            ("Very Negative", "Emergency"): "P0",
            ("Very Negative", "Escalate"): "P1",
            ("Negative", "Emergency"): "P1",
            ("Very Negative", "Watch"): "P2",
            ("Negative", "Escalate"): "P2",
            ("Neutral", "Emergency"): "P2",
            ("Neutral", "Escalate"): "P2",
            ("Positive", "Emergency"): "P2",
            ("Positive", "Escalate"): "P2",
            ("Very Positive", "Emergency"): "P2",
            ("Negative", "Watch"): "P3",
            ("Neutral", "Watch"): "P3",
            ("Positive", "Watch"): "P3",
            ("Very Positive", "Escalate"): "P3",
            ("Very Positive", "Watch"): "P3",
            ("Negative", "Weak"): "P4",
            ("Very Negative", "Weak"): "P5",
            ("Very Negative", "Dormant"): "P5",
            ("Negative", "Dormant"): "P5",
            ("Neutral", "Weak"): "P5",
            ("Neutral", "Dormant"): "P5",
            ("Positive", "Weak"): "P5",
            ("Positive", "Dormant"): "P5",
            ("Very Positive", "Weak"): "P5",
            ("Very Positive", "Dormant"): "P5",
        }

    def _define_quadrant_purpose(self):
        return {
            ("Very Negative", "Emergency"): "Critical Crisis Response",
            ("Very Negative", "Escalate"): "Prevent Active Loyalty Churn",
            ("Very Negative", "Watch"): "Erosion Risk Early Watch",
            ("Very Negative", "Weak"): "Flag for Vulnerability",
            ("Very Negative", "Dormant"): "Archive for Pattern Watch",
            ("Negative", "Emergency"): "Urgent Loyalty Recovery",
            ("Negative", "Escalate"): "Contain Friction Emergence",
            ("Negative", "Watch"): "Tactical Corrections",
            ("Negative", "Weak"): "Investigate Minor Recurrence",
            ("Negative", "Dormant"): "Low Risk Monitoring",
            ("Neutral", "Emergency"): "Early Warning Activation",
            ("Neutral", "Escalate"): "Investigate Attrition Risks",
            ("Neutral", "Watch"): "Monitor Behavioral Shifts",
            ("Neutral", "Weak"): "Background Observation",
            ("Neutral", "Dormant"): "No Immediate Action",
            ("Positive", "Emergency"): "Expand Loyalty Influence",
            ("Positive", "Escalate"): "Loyalty Accelerant",
            ("Positive", "Watch"): "Deepen Retention Programs",
            ("Positive", "Weak"): "Gradual Improvements",
            ("Positive", "Dormant"): "Pulse Monitoring",
            ("Very Positive", "Emergency"): "Maximize Advocacy Surge",
            ("Very Positive", "Escalate"): "Scale Advocacy Programs",
            ("Very Positive", "Watch"): "Reignite Advocates",
            ("Very Positive", "Weak"): "Recognize Quiet Strength",
            ("Very Positive", "Dormant"): "Passive Monitoring"
        }

    def compute(self):
        grouped = self.df.groupby("experience_driver")
        results = []

        entity_counts = self.df["experience_driver"].value_counts()
        max_mentions = entity_counts.max()

        # Impact weights per priority tier
        impact_weights = {
            "P0": 1.00,
            "P1": 0.85,
            "P2": 0.70,
            "P3": 0.55,
            "P4": 0.40,
            "P5": 0.20
        }

        for entity, group in grouped:
            entry = {"experience_driver": entity}
            total_mentions = len(group)
            emotion_counts = group["emotion_primary"].value_counts().to_dict()

            eri_raw = sum(self.emotion_scores.get(emotion, 0) * count for emotion, count in emotion_counts.items())
            eri = eri_raw / total_mentions if total_mentions else 0
            eri_normalized = ((eri + 3) / 6) * 200 - 100

            loyalty_tier = self._map_loyalty_tier(eri_normalized)
            most_recent_date = group["date"].max()
            age_days = (self.today - most_recent_date).days
            R = 100 * np.exp(-age_days / self.timeframe_days)

            F = min(100, (np.log(total_mentions + 1) / np.log(max_mentions + 1)) * 100)
            RF = 0.6 * R + 0.4 * F

            rf_tier = self._map_rf_tier(RF)
            associated_names = sorted(group["entity_name"].dropna().unique().tolist())

            priority_status = self.priority_matrix.get((loyalty_tier, rf_tier), "Unmapped")
            impact_score = impact_weights.get(priority_status, 0)
            RFI = RF * impact_score

            entry.update({
                "ERI": round(eri_normalized, 2),
                "Loyalty_State": loyalty_tier,
                "Associated_Entity_Names": associated_names,
                "R": round(R, 2),
                "Most_Recent_Date": most_recent_date.strftime('%Y-%m-%d'),
                "F": round(F, 2),
                "No_of_Mentions": total_mentions,
                "RF": round(RF, 2),
                "RF_Urgency_Category": rf_tier,
                "ERI_RF_Quadrant": f"{loyalty_tier} x {rf_tier}",
                "Quadrant_Purpose": self.quadrant_purpose.get((loyalty_tier, rf_tier), "Unmapped Quadrant"),
                "Priority_Status": priority_status,
                "Impact_Score": round(impact_score, 2),
                "RFI": round(RFI, 2)
            })
            results.append(entry)

        self.layer2_df = pd.DataFrame(results)

         # 🔽 NEW: Save output to CSV for inspection
        output_path = "outputs/layer2_output_debug.csv"
        os.makedirs("outputs", exist_ok=True)
        self.layer2_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ Layer 2 output saved to: {output_path}")
        
        
        # 🔽 NEW: Optional RFI Policy Integration (AFTER layer2_df is fully built)
        # # Initialize RFI Policy with conservative defaults
        # rfi_config = RFIPolicyConfig(
        #     enable_rfi_tiebreak=True,   # Safe: only affects order when RF ties
        #     enable_rf_blend=False,      # Safe: keeps current RF behavior 
        #     alpha=0.80,
        #     tau=0.15
        # )
        # rfi_policy = RFIPolicy(rfi_config)

        # # Enrich with RFI features (adds RF_prime, OrderHint columns)
        # self.layer2_df = rfi_policy.enrich_with_rfp(self.layer2_df)

        # Optional: Get execution-ranked version (doesn't change original)
        # ranked_df = rfi_policy.rank_for_execution(self.layer2_df)
        # Save ranked version if needed
        # ranked_output_path = "outputs/layer2_ranked_debug.csv"
        # ranked_df.to_csv(ranked_output_path, index=False, encoding="utf-8")
        # print(f"✅ Layer 2 ranked output saved to: {ranked_output_path}")
     
        
        return self.layer2_df

    def _map_loyalty_tier(self, eri_normalized):
        if eri_normalized >= 80:
            return "Very Positive"
        elif eri_normalized >= 30:
            return "Positive"
        elif eri_normalized >= -10:
            return "Neutral"
        elif eri_normalized >= -50:
            return "Negative"
        else:
            return "Very Negative"

    def _map_rf_tier(self, RF):
        if RF >= 80:
            return "Emergency"
        elif RF >= 60:
            return "Escalate"
        elif RF >= 40:
            return "Watch"
        elif RF >= 20:
            return "Weak"
        else:
            return "Dormant"


# # === Quadrant Matrix Loader ===
# def get_quadrant_matrix():
#     """
#     Enhanced Quadrant Matrix with additional fields 
    
#     Fields: [Saturation_Tier, Momentum_Tier, Diagnostic_Label, Strategic_Narrative, 
#              Urgency_Code, Recommended_Owner, Action_Guidance, Momentum_Context, 
#              Saturation_Context, Trajectory_Story]
#     """
#     data = [
#         # Very High Saturation (0.90-1.00) - Champion/Peak Emotional Investment
#         ["🏆 Very High", "↑↑ 🚀 Strongly Rising", "Overload Alert", "Sentiment at ceiling, emotion surging — may cause burnout or backlash.", "🚨 Crisis", "CX/Comms", "Prepare relief interventions or redirection campaigns to prevent loyalty fatigue.", "Surging beyond sustainable peaks", "Already at maximum emotional investment", "Customers hitting dangerous loyalty overload while emotional pressure keeps building"],
#         ["🏆 Very High", "↑ 📈 Moderately Rising", "Plateau Pressure", "Near ceiling, momentum creeping — optimize and stabilize.", "⚠️ Risk", "CX/Product", "Reinforce strong areas without over-investing; monitor loyalty saturation.", "Creeping toward emotional limits", "Operating near peak satisfaction", "Strong loyalty base experiencing gentle upward pressure requiring careful management"],
#         ["🏆 Very High", "→ ➖ Stable", "Trust Plateau", "Maxed emotion, stable loyalty — monitor quietly.", "🔁 Watch", "CX/Insights", "Maintain service quality and prepare re-engagement initiatives if stagnation prolongs.", "Holding steady at emotional peaks", "Maxed out on emotional connection", "Customers at peak trust with stable emotional investment but vulnerable to stagnation"],
#         ["🏆 Very High", "↓ 📉 Moderately Falling", "Soft Fatigue", "Trust is waning — find freshness.", "⚠️ Risk", "Product/CX", "Introduce new features or messaging to reignite dormant advocates.", "Drifting down from emotional heights", "Losing ground from peak investment", "Previously passionate advocates showing signs of emotional fatigue and declining connection"],
#         ["🏆 Very High", "↓↓ 🧨 Strongly Falling", "Loyalty Collapse Risk", "Once loyal, now disenchanted — urgent rescue.", "🚨 Crisis", "CX Leadership", "Activate loyalty recovery programs and human-in-the-loop outreach.", "Plummeting from emotional summit", "Crashing down from maximum investment", "Champions rapidly becoming detractors with emotional trust collapsing from peak levels"],

#         # High Saturation (0.65-0.89) - Loyal/Strong Emotional Connection
#         ["✅ High", "↑↑ 🚀 Strongly Rising", "Optimization Zone", "Push higher carefully — strong base, rising emotion.", "🌱 Opportunity", "Marketing/Product", "Double down on what's working; deepen positive emotional signals.", "Accelerating from strong foundation", "Solid investment with room to grow", "Loyal customers gaining emotional momentum with clear headroom for deeper connection"],
#         ["✅ High", "↑ 📈 Moderately Rising", "Prime Expansion Zone", "Loyalty is forming, act decisively.", "🌱 Opportunity", "Marketing", "Scale reinforcement tactics and pre-loyalty rewards.", "Building steadily toward peak loyalty", "Strong base with expansion potential", "Customers transitioning into deeper loyalty with positive emotional trajectory"],
#         ["✅ High", "→ ➖ Stable", "Healthy Steady State", "Solid footing — nurture gradually.", "✅ Stable", "CX", "Continue nurturing but avoid unnecessary changes.", "Maintaining strong emotional stability", "Well-invested with sustainable levels", "Loyal customers maintaining steady positive connection without volatility"],
#         ["✅ High", "↓ 📉 Moderately Falling", "Cooling Off", "Risk of losing momentum — reignite.", "⚠️ Risk", "CX/Product", "Test new journeys or emotional engagement campaigns.", "Sliding from loyal connection", "Losing emotional investment gradually", "Previously loyal customers experiencing emotional drift requiring re-engagement"],
#         ["✅ High", "↓↓ 🧨 Strongly Falling", "Saturation Leakage", "Slipping from once strong — needs boost.", "🚨 Crisis", "CX/Ops", "Diagnose friction points and prevent emotional disengagement.", "Rapidly abandoning strong position", "Hemorrhaging established emotional value", "Loyal customers experiencing sharp emotional decline threatening established relationship"],

#         # Medium Saturation (0.45-0.64) - Neutral/Moderate Investment  
#         ["⚖️ Medium", "↑↑ 🚀 Strongly Rising", "Momentum Lift-Off", "Emotion awakening — scale now.", "🌱 Opportunity", "Marketing", "Capture momentum with smart CX nudges or rewards.", "Breaking through emotional resistance", "Building from moderate foundation", "Neutral customers experiencing emotional awakening with significant growth potential"],
#         ["⚖️ Medium", "↑ 📈 Moderately Rising", "Growth Window", "Emotion stabilizing and rising — support journey.", "🌱 Opportunity", "CX/Insights", "Spotlight rising themes; validate with wider audiences.", "Climbing toward positive territory", "Expanding from balanced starting point", "Customers showing steady emotional improvement with clear trajectory toward loyalty"],
#         ["⚖️ Medium", "→ ➖ Stable", "Balanced Neutral", "No urgency — observe.", "🔁 Watch", "CX", "No immediate action — continue observation.", "Holding in emotional equilibrium", "Balanced with no clear direction", "Customers maintaining neutral stance with stable but uninspiring emotional connection"],
#         ["⚖️ Medium", "↓ 📉 Moderately Falling", "Churn Warning", "Emotion stuck, direction unclear — diagnose early.", "⚠️ Risk", "CX/Analytics", "Run root cause analysis; test retention messaging.", "Drifting toward emotional disconnect", "Losing moderate investment slowly", "Neutral customers sliding toward negative territory requiring early intervention"],
#         ["⚖️ Medium", "↓↓ 🧨 Strongly Falling", "Indifference Trap", "Low velocity, no connection — emotional vacuum.", "🚨 Crisis", "CX Leadership", "Rebuild emotional relevance urgently; consider reboot strategies.", "Falling into emotional void", "Abandoning moderate connection rapidly", "Customers rapidly disengaging from neutral position toward complete indifference"],

#         # Low Saturation (0.25-0.44) - Vulnerable/Cracking Trust
#         ["⚠️ Low", "↑↑ 🚀 Strongly Rising", "Breakthrough Opportunity", "Momentum climbing out of emotional hole — catalyze.", "🌱 Opportunity", "CX/Marketing", "Celebrate early wins; encourage customer voice amplification.", "Surging upward from difficult position", "Minimal investment with huge upside", "Vulnerable customers experiencing emotional breakthrough with maximum improvement potential"],
#         ["⚠️ Low", "↑ 📈 Moderately Rising", "Recovery Surge", "Early signals of rebound — support.", "🌱 Opportunity", "CX", "Invest in emotional follow-up; reward vocal feedback.", "Climbing out of emotional deficit", "Building from low base steadily", "Previously frustrated customers showing recovery signals with room for significant growth"],
#         ["⚠️ Low", "→ ➖ Stable", "Friction State", "Low emotion, stagnant path — requires intervention.", "⚠️ Risk", "Product/Ops", "Audit for service or process gaps.", "Stuck in emotional limbo", "Trapped at low investment levels", "Customers maintaining negative connection without improvement or deterioration"],
#         ["⚠️ Low", "↓ 📉 Moderately Falling", "Danger Zone", "Downward pull + low loyalty — fix fast.", "🚨 Crisis", "CX/Ops", "Apply crisis workflows and cross-functional fixes.", "Sliding deeper into negativity", "Losing remaining emotional value", "Vulnerable customers declining further toward complete disconnection"],
#         ["⚠️ Low", "↓↓ 🧨 Strongly Falling", "Critical Stall", "Emotional damage deepening — act now.", "🚨 Crisis", "CX Leadership", "Initiate emotional damage control protocol.", "Plunging toward total disconnection", "Rapidly abandoning minimal investment", "Customers in emotional freefall from vulnerable position requiring immediate intervention"],

#         # Very Low Saturation (0.00-0.24) - At-Risk/Collapsed Trust
#         ["❌ Very Low", "↑↑ 🚀 Strongly Rising", "Signal Spike", "Warning volatility — intense rise from a bad place.", "⚠️ Risk", "CX", "Watch for false positives; investigate root cause of spike.", "Surging upward from emotional rock bottom", "Minimal emotional investment to lose", "Customers climbing out of despair but volatility suggests unstable foundation"],
#         ["❌ Very Low", "↑ 📈 Moderately Rising", "Erratic Revival", "Surprising movement, unstable still — handle carefully.", "⚠️ Risk", "CX", "Don't over-celebrate — check if emotion is anchored or episodic.", "Showing signs of life from low point", "Building from near-zero investment", "Previously disconnected customers demonstrating fragile recovery requiring careful nurturing"],
#         ["❌ Very Low", "→ ➖ Stable", "Dead Zone", "No movement, no emotion — emotional disengagement.", "🚨 Crisis", "CX", "Consider exit campaigns or silent churn save tactics.", "Flatlined at emotional rock bottom", "Zero emotional investment remaining", "Customers in complete emotional disconnection with no signs of recovery"],
#         ["❌ Very Low", "↓ 📉 Moderately Falling", "Decay Spiral", "All indicators down — abandon or overhaul.", "🚨 Crisis", "CX/Ops", "Run total overhaul diagnostics — emotional collapse imminent.", "Sinking deeper into emotional void", "Losing final remnants of connection", "Disconnected customers deteriorating further with total relationship breakdown imminent"],
#         ["❌ Very Low", "↓↓ 🧨 Strongly Falling", "Blackout State", "Customer trust lost — rebuild from scratch.", "🚨 Crisis", "Executive Team", "Relaunch brand experience — emotional trust annihilated.", "Plummeting deeper into emotional void", "Operating at rock bottom investment", "Customers have lost all trust and emotional connection is deteriorating further"]
#     ]

#     columns = [
#         "Saturation_Tier", "Momentum_Tier", "Diagnostic_Label", "Strategic_Narrative", 
#         "Urgency_Code", "Recommended_Owner", "Action_Guidance", "Momentum_Context", 
#         "Saturation_Context", "Trajectory_Story"
#     ]

#     return pd.DataFrame(data, columns=columns)

# # === Layer 3 Logic ===

# class Layer3Computer:
#     def __init__(self, layer2_df, raw_df, timeframe_days, today_anchor, verbose=False):
#         self.layer2_df = layer2_df.copy()
#         self.raw_df = raw_df.copy()
#         self.verbose = verbose
#         self.timeframe_days = timeframe_days
#         self.today = today_anchor  # ✅ anchor passed from Layer 2 logic
#         self.cutoff_date = self.today - timedelta(days=self.timeframe_days)

#         # Step 1: Filter raw_df to match the same date window
#         self.raw_df["date"] = pd.to_datetime(self.raw_df["date"]).dt.date
#         self.raw_df = self.raw_df[self.raw_df["date"] >= self.cutoff_date]

#         # Step 2: Map emotion scores
#         emotion_scores = {
#             "Adoration": 3,
#             "Appreciation": 1,
#             "Ambivalence": 0,
#             "Agitation": -1,
#             "Anger": -3
#         }
#         self.raw_df["emotion_score"] = self.raw_df["emotion_primary"].map(emotion_scores)

#         # Step 3: Initialize other configs
#         self.quadrant_matrix = get_quadrant_matrix()
#         self.layer3_df = None
#         self.pattern_lags = ["weekly", "monthly"]

#         if self.verbose:
#             print(f"📦 Layer 3 initialized using anchor date: {self.today}")
#             print(f"🪟 Timeframe: {self.cutoff_date} → {self.today}")
#             print(f"🧮 Feedback rows in window: {len(self.raw_df)}")
#             print(f"🔍 Layer 2 prioritized entities: {len(self.layer2_df)}")


#     def compute_normalized_eri(self, group):
#         raw_eri = group["emotion_score"].mean()
#         return ((raw_eri + 3) / 6) * 200 - 100

#     def _compute_trend_series(self, series) -> dict:
#         """
#         Trend percent = (end - start) / 200 * 100
#         → bounded to [-100, +100], where 100% means full-range ERI shift.
#         """
#         try:
#             if len(series) < 2:
#                 return {"symbol": "→", "trend_pct": 0.0}

#             start = float(series.iloc[0])
#             end   = float(series.iloc[-1])

#             pct = ((end - start) / 200.0) * 100.0   # scale-relative (% of full ERI range)
#             # Bound and round
#             pct = max(min(pct, 100.0), -100.0)
#             pct = round(pct, 2)

#             # Direction thresholds (feel free to tweak 5% → 3% if you want more sensitivity)
#             if pct > 5.0:
#                 symbol = "↑"
#             elif pct < -5.0:
#                 symbol = "↓"
#             else:
#                 symbol = "→"

#             return {"symbol": symbol, "trend_pct": pct}

#         except Exception as e:
#             if self.verbose:
#                 print(f"⚠️ Trend computation failed: {e}")
#             return {"symbol": "→", "trend_pct": 0.0}


#     def _compute_momentum_series(self, series) -> dict:
#         """
#         Momentum percent = (avg(second half) - avg(first half)) / 200 * 100
#         → bounded to [-100, +100], tiered by magnitude.
#         """
#         try:
#             n = len(series)
#             if n < 4:
#                 return {"symbol": "→", "delta": 0.0}

#             mid = n // 2
#             avg1 = float(series[:mid].mean())
#             avg2 = float(series[mid:].mean())

#             delta = ((avg2 - avg1) / 200.0) * 100.0   # scale-relative (% of full ERI range)
#             delta = max(min(delta, 100.0), -100.0)
#             delta = round(delta, 2)

#             # Tiers on the same % scale as trend (adjust thresholds as you like)
#             if   delta > 20.0: symbol = "↑↑"
#             elif delta > 5.0:  symbol = "↑"
#             elif delta < -20.0: symbol = "↓↓"
#             elif delta < -5.0:  symbol = "↓"
#             else:               symbol = "→"

#             return {"symbol": symbol, "delta": delta}

#         except Exception as e:
#             if self.verbose:
#                 print(f"⚠️ Momentum computation failed: {e}")
#             return {"symbol": "→", "delta": 0.0}

#     def _compute_volatility_series(self, series) -> dict:
#         try:
#             std = series.std()

#             if pd.isna(std):
#                 return {
#                     "tier": "✅ Stable",
#                     "score": 0.0
#                 }

#             if std <= 15:
#                 tier = "✅ Stable"
#             elif std <= 45:
#                 tier = "⚠ Fluctuating"
#             else:
#                 tier = "🔴 Highly Fluctuating"

#             return {
#                 "tier": tier,
#                 "score": round(std, 2)
#             }

#         except Exception as e:
#             if self.verbose:
#                 print(f"⚠️ Volatility computation failed: {e}")
#             return {
#                 "tier": "✅ Stable",
#                 "score": 0.0
#             }

#     def _compute_pattern_recognition(self, series, entity_data, lags):
#         lag_days = {"weekly": 7, "monthly": 30, "quarterly": 90}
#         # base result
#         result = {
#             "has_pattern": False,
#             "pattern_type": "None",
#             "pattern_strength": None,
#             "pattern_confidence": "None",
#             "pain_day": None,
#             "data_coverage_days": int((series.index.max() - series.index.min()).days + 1) if len(series) else 0,
#             "min_required_days": None,
#             "eri_by_day": None
#         }

#         # ensure continuity (daily)
#         full_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq='D')
#         series = series.reindex(full_index).ffill()

#         # flat or null series → exit
#         if series.std() == 0 or series.isnull().all():
#             if self.verbose:
#                 print("⚠️ Skipping pattern detection: Flat or null ERI time series")
#             return result

#         # coverage gating
#         series_days = (series.index.max() - series.index.min()).days + 1
#         valid_lags = [lag for lag in lags if series_days >= lag_days[lag] + 1]
#         if self.verbose:
#             print(f"🔍 Valid pattern lags for {series_days} days: {valid_lags}")

#         best_pattern, best_score = None, 0.0

#         for lag in valid_lags:
#             lag_val = lag_days[lag]
#             try:
#                 acf_vals = acf(series, nlags=lag_val, fft=True, missing='conservative')
#                 score = float(acf_vals[lag_val])

#                 if score >= 0.3 and score > best_score:
#                     confidence = "Strong" if score >= 0.6 else "Weak"
#                     pattern_type = lag.capitalize()

#                     best_pattern = {
#                         "has_pattern": True,
#                         "pattern_type": pattern_type,
#                         "pattern_strength": round(score, 3),
#                         "pattern_confidence": confidence,
#                         "pain_day": None,
#                         "data_coverage_days": int(series_days),
#                         "min_required_days": int(lag_val),
#                         "eri_by_day": None
#                     }

#                     if lag == "weekly":
#                         # normalized ERI per record: ((emotion_score + 3)/6)*200 - 100
#                         ed = entity_data.copy()
#                         ed["eri_norm"] = ((ed["emotion_score"] + 3.0) / 6.0) * 200.0 - 100.0
#                         weekday_means = (
#                             ed.assign(weekday=ed["date"].apply(lambda x: x.strftime("%A")))
#                             .groupby("weekday", as_index=True)["eri_norm"]
#                             .mean()
#                             .round(2)
#                         )
#                         if not weekday_means.empty:
#                             # pain day = lowest average ERI day
#                             best_pattern["pain_day"] = weekday_means.idxmin()
#                             # also expose the weekly ERI map for dashboards
#                             # order Mon→Sun if possible
#                             order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
#                             best_pattern["eri_by_day"] = {d: float(weekday_means.get(d)) if d in weekday_means.index else None for d in order}

#                     best_score = score

#             except Exception as e:
#                 if self.verbose:
#                     print(f"❌ ACF error for {lag} lag: {e}")

#         return best_pattern if best_pattern else result


#     def _compute_momentum_saturation_insight(self, eri_score, momentum_symbol) -> dict:
#         # --- 1) Inputs → tiers ---
#         saturation_index = (eri_score + 100) / 200.0  # 0..1
#         saturation_index = max(0.0, min(1.0, float(saturation_index)))

#         momentum_map = {
#             "↑↑": "↑↑ 🚀 Strongly Rising",
#             "↑":  "↑ 📈 Moderately Rising",
#             "→":  "→ ➖ Stable",
#             "↓":  "↓ 📉 Moderately Falling",
#             "↓↓": "↓↓ 🧨 Strongly Falling"
#         }
#         # Guard unknowns
#         momentum_symbol = momentum_symbol if momentum_symbol in momentum_map else "→"

#         if   saturation_index >= 0.90: sat_emoji, sat_clean = "🏆 Very High", "Very High"
#         elif saturation_index >= 0.65: sat_emoji, sat_clean = "✅ High", "High"
#         elif saturation_index >= 0.45: sat_emoji, sat_clean = "⚖️ Medium", "Medium"
#         elif saturation_index >= 0.25: sat_emoji, sat_clean = "⚠️ Low", "Low"
#         else:                          sat_emoji, sat_clean = "❌ Very Low", "Very Low"

#         mom_emoji = momentum_map[momentum_symbol]
#         # Clean momentum text after the emoji & space
#         mom_clean = mom_emoji.split(" ", 2)[2]  # "Strongly Rising" etc.

#         # --- 2) Lookup in matrix (source of truth) ---
#         qm = self.quadrant_matrix
#         quadrant_info = qm[(qm["Saturation_Tier"] == sat_emoji) & (qm["Momentum_Tier"] == mom_emoji)]

#         # --- 3) Confidence & borderline (optional richness, no behavior change) ---
#         # distance to nearest sat boundary (for hysteresis/readability)
#         sat_bounds = [0.0, 0.25, 0.45, 0.65, 0.90, 1.0]
#         nearest = min(sat_bounds, key=lambda b: abs(b - saturation_index))
#         sat_distance = abs(saturation_index - nearest)
#         borderline = sat_distance < 0.02  # within 2% of a boundary

#         mom_strength = {"↑↑": 1.0, "↑": 0.6, "→": 0.3, "↓": 0.6, "↓↓": 1.0}[momentum_symbol]
#         # simple 0..1 confidence: stronger momentum + farther from boundary → higher confidence
#         quadrant_confidence = round(0.5 * mom_strength + 0.5 * min(1.0, sat_distance / 0.20), 2)

#         matrix_key = f"{sat_clean}|{mom_clean}"  # handy for joins/debug

#         if not quadrant_info.empty:
#             q = quadrant_info.iloc[0]

#             payload = {
#                 "signal_classification": {
#                     "saturation_index": round(saturation_index, 2),
#                     "saturation_tier": sat_clean,
#                     "momentum_tier": mom_clean,
#                     "combined_quadrant": f"{mom_clean} Momentum × {sat_clean} Saturation"
#                 },
#                 "quadrant_interpretation": {
#                     "quadrant_label": q["Diagnostic_Label"],
#                     "urgency_level": q["Urgency_Code"],
#                     "interpretation": q["Strategic_Narrative"]
#                 },
#                 "tactical_insight": {
#                     "emotional_pulse": f"{q['Diagnostic_Label']} — {q['Momentum_Context']} while {q['Saturation_Context']}",
#                     "battle_status": f"{q['Urgency_Code']} {q['Diagnostic_Label']}",
#                     "strategic_reality": f"{q['Trajectory_Story']} — system recommends {q['Action_Guidance']}"
#                 },
#                 "actionable_strategy": {
#                     "momentum_context": q["Momentum_Context"],
#                     "saturation_context": q["Saturation_Context"],
#                     "trajectory_story": q["Trajectory_Story"],
#                     "action_guidance": q["Action_Guidance"],
#                     "recommended_owner": q["Recommended_Owner"]
#                 }
#             }

#         else:
#             # Fallback remains unchanged, but we’ll still add meta below
#             payload = {
#                 "signal_classification": {
#                     "saturation_index": round(saturation_index, 2),
#                     "saturation_tier": sat_clean,
#                     "momentum_tier": mom_clean,
#                     "combined_quadrant": "Unknown"
#                 },
#                 "quadrant_interpretation": {
#                     "quadrant_label": "Unknown",
#                     "urgency_level": "Unknown",
#                     "interpretation": "No data available for this quadrant"
#                 },
#                 "tactical_insight": {
#                     "emotional_pulse": "Unknown — No data available for emotional assessment",
#                     "battle_status": "⚠️ Risk Unknown",
#                     "strategic_reality": "No trajectory data available — system recommends data validation"
#                 },
#                 "actionable_strategy": {
#                     "momentum_context": "Unknown momentum pattern",
#                     "saturation_context": "Unknown saturation level",
#                     "trajectory_story": "No trajectory data available",
#                     "action_guidance": "Validate data sources and retry analysis",
#                     "recommended_owner": "Data Team"
#                 }
#             }

#         # --- 4) Non-breaking meta (adds value, keeps structure) ---
#         payload["meta"] = {
#             "matrix_key": matrix_key,
#             "saturation_tier_emoji": sat_emoji,
#             "momentum_tier_emoji": mom_emoji,
#             "quadrant_confidence": quadrant_confidence,  # 0..1 helper
#             "borderline": borderline
#         }
#         return payload

#     # === Layer 4 Logic ===
#     def _compute_signal_strength(self, trend: str, momentum: str, saturation_index: float) -> dict:
#         # Velocity Score
#         if trend in ("↑", "↓") and momentum == "↓↓":
#             velocity_score = 6
#             rationale = "Sharp trend shift with strong counter-momentum"
#         elif trend in ("↑", "↓") and momentum == "↑↑":
#             velocity_score = 5
#             rationale = "Rapid acceleration with positive momentum surge"
#         elif trend in ("↑", "↓") and momentum in ("↑", "↓"):
#             velocity_score = 4
#             rationale = "Moderate directional movement with matching momentum"
#         elif trend in ("↑", "↓") and momentum == "→":
#             velocity_score = 2
#             rationale = "Directional change but stable momentum"
#         elif trend == "→" and momentum in ("↑↑", "↓↓"):
#             velocity_score = 3
#             rationale = "Flat trend with sudden strong shift emerging"
#         elif trend == "→" and momentum in ("↑", "↓"):
#             velocity_score = 1
#             rationale = "Stable trend with mild fluctuation"
#         else:
#             velocity_score = 0
#             rationale = "No meaningful trend or momentum detected"

#         # Saturation Score
#         if saturation_index <= 0.20:
#             saturation_score = 4
#             saturation_rationale = "Very low emotional saturation — fresh pain or interest forming"
#         elif saturation_index <= 0.40:
#             saturation_score = 3
#             saturation_rationale = "Low saturation — likely early-stage signal"
#         elif saturation_index <= 0.60:
#             saturation_score = 2
#             saturation_rationale = "Medium saturation — emotionally stable signal"
#         elif saturation_index <= 0.80:
#             saturation_score = 1
#             saturation_rationale = "High saturation — signal is nearing emotional capacity"
#         else:
#             saturation_score = 0
#             saturation_rationale = "Very high saturation — emotional exhaustion or signal decay"

#         qssi = velocity_score + saturation_score

#         # QSSI Tier
#         if qssi >= 9:
#             qssi_tier = "💥 Critical Signal"
#             interpretation = "Extreme movement detected with low saturation — high potential volatility or opportunity"
#         elif qssi >= 6:
#             qssi_tier = "🔥 Strong Signal"
#             interpretation = "Substantial shift in emotional dynamics — active attention required"
#         elif qssi >= 4:
#             qssi_tier = "🌱 Emerging Signal"
#             interpretation = "Signal beginning to form — track evolution"
#         elif qssi >= 1:
#             qssi_tier = "🔁 Weak Signal"
#             interpretation = "Low signal strength — may resolve on its own"
#         else:
#             qssi_tier = "❌ No Signal"
#             interpretation = "Dormant — not actionable"

#         return {
#             "velocity_component": {
#                 "trend_symbol": trend,
#                 "momentum_symbol": momentum,
#                 "velocity_score": velocity_score,
#                 "velocity_rationale": rationale
#             },
#             "saturation_component": {
#                 "saturation_index": round(saturation_index, 2),
#                 "saturation_score": saturation_score,
#                 "saturation_rationale": saturation_rationale
#             },
#             "qssi_summary": {
#                 "qssi_score": qssi,
#                 "qssi_tier": qssi_tier,
#                 "qssi_interpretation": interpretation
#             }
#         }

#     def build_predictive_emotional_modeling(self, state_of_play, volatility, has_pattern, qssi_tier) -> dict:
#         """
#         Predictive Emotional Modeling (PEM)
#         - Backward compatible with your existing keys
#         - Adds: numeric probability, numeric confidence, horizon, rationale & feature vector
#         """

#         # ---- Pull upstream (state_of_play) ----
#         momentum_tier = state_of_play["momentum_tier"]      # "Strongly Rising" | "Moderately Rising" | "Stable" | "Moderately Falling" | "Strongly Falling"
#         saturation_tier = state_of_play["saturation_tier"]  # "Very High" | "High" | "Medium" | "Low" | "Very Low"
#         trajectory_story = state_of_play["trajectory_story"]
#         action_guidance = state_of_play["action_guidance"]
#         recommended_owner = state_of_play["recommended_owner"]

#         # ---- Lightweight mappings (explainable, no new inputs) ----
#         qssi_map = {
#             "💥 Critical Signal": 1.00,
#             "🔥 Strong Signal":   0.80,
#             "🌱 Emerging Signal": 0.50,
#             "🔁 Weak Signal":     0.20,
#             "❌ No Signal":       0.00
#         }
#         momentum_map = {
#             "Strongly Rising":    +1.00,
#             "Moderately Rising":  +0.60,
#             "Stable":              0.00,
#             "Moderately Falling": -0.60,
#             "Strongly Falling":   -1.00
#         }
#         # Headroom proxy from tier (higher = more room to escalate)
#         headroom_map = {
#             "Very High": 0.00,  # near ceiling
#             "High":      0.25,
#             "Medium":    0.50,
#             "Low":       0.75,
#             "Very Low":  1.00   # far from ceiling
#         }

#         qssi_strength = qssi_map.get(qssi_tier, 0.0)
#         momentum_score = momentum_map.get(momentum_tier, 0.0)
#         headroom = headroom_map.get(saturation_tier, 0.5)

#         # Normalize volatility into [0,1] using your tiers (<=15 stable, <=45 fluctuating, >45 high)
#         vol_norm = max(0.0, min(volatility / 45.0, 1.0))
#         stability = 1.0 - vol_norm

#         # ---- Two competing scores → escalation vs decay (bounded ≥0) ----
#         # Intuition:
#         # - Escalation loves: rising momentum, high QSSI, repeat pattern, headroom, stability
#         # - Decay loves: falling momentum, high QSSI (strong move, but down), low headroom, volatility
#         from math import fsum

#         esc_raw = (
#             0.50 * qssi_strength +
#             0.40 * max(0.0, momentum_score) +
#             0.30 * headroom +
#             0.20 * (1.0 if has_pattern else 0.0) +
#             0.20 * stability
#         )
#         dec_raw = (
#             0.50 * qssi_strength +
#             0.60 * max(0.0, -momentum_score) +
#             0.30 * (1.0 - headroom) +
#             0.20 * vol_norm
#         )
#         esc_raw = max(0.0, esc_raw)
#         dec_raw = max(0.0, dec_raw)

#         total = esc_raw + dec_raw
#         # Soft normalization to probabilities; if both are tiny, remain low
#         esc_prob = (esc_raw / total) if total > 1e-9 else 0.0
#         dec_prob = (dec_raw / total) if total > 1e-9 else 0.0

#         # ---- Final trajectory decision (keeps your original logic semantics) ----
#         # Preserve your rules first; then attach probability consistent with the chosen path.
#         signal_is_precursor = (
#             momentum_tier in ["Strongly Rising", "Moderately Rising"] and
#             qssi_tier in ["🔥 Strong Signal", "💥 Critical Signal"] and
#             has_pattern
#         )
#         signal_is_at_risk_of_decay = (
#             momentum_tier in ["Strongly Falling", "Moderately Falling"] and
#             volatility >= 15
#         )

#         if signal_is_precursor:
#             pem_trajectory = "Likely Escalation"
#             pem_probability = max(esc_prob, 0.65)  # ensure probability reflects the rule-based trigger
#             explanation = "Rising momentum + strong/critical signal + repeating pattern indicates intensification"
#             risk_class = "Opportunity"
#         elif signal_is_at_risk_of_decay:
#             pem_trajectory = "At Risk of Decay"
#             pem_probability = max(dec_prob, 0.65)
#             explanation = "Falling momentum + elevated volatility suggests emotional energy is fading or fracturing"
#             risk_class = "Risk"
#         else:
#             # Choose the higher model probability but label as stable/inconclusive if weak
#             if max(esc_prob, dec_prob) < 0.55:
#                 pem_trajectory = "Stable / Inconclusive"
#                 pem_probability = max(esc_prob, dec_prob)
#                 explanation = "No dominant predictive anchors; continue monitoring"
#                 risk_class = "Neutral"
#             else:
#                 # If model sees a stronger side, reflect it, but stay conservative in language
#                 if esc_prob > dec_prob:
#                     pem_trajectory = "Likely Escalation"
#                     pem_probability = esc_prob
#                     explanation = "Model indicates upward trajectory dominance (momentum/headroom/stability blend)"
#                     risk_class = "Opportunity"
#                 else:
#                     pem_trajectory = "At Risk of Decay"
#                     pem_probability = dec_prob
#                     explanation = "Model indicates downward trajectory dominance (momentum/volatility/ceiling pressure)"
#                     risk_class = "Risk"

#         # ---- Confidence model (numeric + tier) ----
#         # Keep your tiers, add a 0–1 score for dashboards
#         if has_pattern and volatility >= 10:
#             confidence_tier = "High"
#             confidence_score = 0.85
#         elif has_pattern or volatility >= 10:
#             confidence_tier = "Moderate"
#             confidence_score = 0.55
#         else:
#             confidence_tier = "Low"
#             confidence_score = 0.30

#         # ---- Extras: horizon + audit vector + version ----
#         horizon_days = getattr(self, "pem_horizon_days", 14)  # configurable at class level if you want
#         version = "PEM.v1.1"

#         feature_vector = {
#             "qssi_strength": round(qssi_strength, 3),
#             "momentum_score": round(momentum_score, 3),
#             "headroom": round(headroom, 3),
#             "volatility_norm": round(vol_norm, 3),
#             "stability": round(stability, 3),
#             "has_pattern": bool(has_pattern),
#             "esc_raw": round(esc_raw, 3),
#             "dec_raw": round(dec_raw, 3),
#             "esc_prob": round(esc_prob, 3),
#             "dec_prob": round(dec_prob, 3)
#         }

#         # ---- Backward-compatible payload (same blocks, richer fields) ----
#         return {
#             "trajectory_forecast": {
#                 "pem_trajectory": pem_trajectory,
#                 "pem_probability": round(float(pem_probability), 3),   # 0–1 probability
#                 "pem_confidence": confidence_tier,
#                 "confidence_score": round(confidence_score, 2),        # 0–1 score
#                 "risk_class": risk_class,                              # Opportunity | Risk | Neutral
#                 "horizon_days": int(horizon_days),
#                 "explanation": explanation,
#                 "version": version
#             },
#             "signal_diagnostics": {
#                 "has_repeating_pattern": bool(has_pattern),
#                 "volatility_score": round(float(volatility), 2),
#                 "momentum_tier": momentum_tier,
#                 "saturation_tier": saturation_tier,
#                 "qssi_tier": qssi_tier,
#                 # optional pass-through if available upstream:
#                 # "qssi_score": state_of_play.get("qssi_score")
#             },
#             "future_risk_profile": {
#                 "trajectory_story": trajectory_story,
#                 "action_guidance": action_guidance,
#                 "recommended_owner": recommended_owner
#             },
#             "audit": {
#                 "feature_vector": feature_vector
#             }
#         }

        
#     def compute(self):
#         results = []
#         skipped_entities = []

#         # Normalize early
#         self.raw_df["experience_driver"] = self.raw_df["experience_driver"].str.strip()

#         # Focus on P0–P3 entities from Layer 2
#         entities = self.layer2_df[self.layer2_df["Priority_Status"].isin(["P0", "P1", "P2", "P3"])]

#         for _, row in entities.iterrows():
#             experience_driver = str(row["experience_driver"]).strip()

#             # Meta fields
#             pclass = row["Priority_Status"]
#             associated_names = row.get("Associated_Entity_Names")
#             most_recent_mention = row.get("Most_Recent_Date")
#             no_of_mentions = row.get("No_of_Mentions")
#             rf_urgency_category = row.get("RF_Urgency_Category")
#             eri_rf_quadrant = row.get("ERI_RF_Quadrant")
#             emotion_perception_tier = row.get("Loyalty_State")

#             # Slice raw_df for this ED
#             data = self.raw_df[self.raw_df["experience_driver"] == experience_driver]

#             if data.empty:
#                 if self.verbose:
#                     print(f"❌ DROP [{experience_driver}] → Reason: No match found in raw_df")
#                 skipped_entities.append((experience_driver, "No raw data"))
#                 continue

#             # Build daily ERI series
#             daily_eri = (
#                 data.groupby("date")
#                 .apply(self.compute_normalized_eri)
#                 .sort_index()
#             )

#             if len(daily_eri) > 1:
#                 full_index = pd.date_range(start=daily_eri.index.min(), end=daily_eri.index.max(), freq="D")
#                 daily_eri = daily_eri.reindex(full_index).bfill().ffill()

#             if len(daily_eri) <= 1:
#                 if self.verbose:
#                     print(f"⏭️ DROP [{experience_driver}] → Only {len(daily_eri)} ERI points after reindexing (needs ≥2)")
#                 skipped_entities.append((experience_driver, f"Insufficient days: {len(daily_eri)}"))
#                 continue

#             # Build each diagnostic block
#             trend_result = self._compute_trend_series(daily_eri)
#             momentum_result = self._compute_momentum_series(daily_eri)
#             volatility_result = self._compute_volatility_series(daily_eri)
#             pattern_block = self._compute_pattern_recognition(daily_eri, data, self.pattern_lags)

#             trend_symbol = trend_result["symbol"]
#             trend_pct = trend_result["trend_pct"]

#             momentum_symbol = momentum_result["symbol"]
#             momentum_delta = momentum_result["delta"]

#             volatility_tier = volatility_result["tier"]
#             volatility_score = volatility_result["score"]

#             # Momentum × Saturation quadrant logic
#             quadrant_block = self._compute_momentum_saturation_insight(row["ERI"], momentum_symbol)

#             # Signal Strength Engine (uses quadrant's saturation index)
#             signal_strength_block = self._compute_signal_strength(trend_symbol, momentum_symbol, quadrant_block["signal_classification"]["saturation_index"])

#             # Predictive Emotional Modeling (PEM)
#             pem_block = self.build_predictive_emotional_modeling(
#             state_of_play={
#                 "momentum_tier": quadrant_block["signal_classification"]["momentum_tier"],
#                 "saturation_tier": quadrant_block["signal_classification"]["saturation_tier"],
#                 "trajectory_story": quadrant_block["actionable_strategy"]["trajectory_story"],
#                 "action_guidance": quadrant_block["actionable_strategy"]["action_guidance"],
#                 "recommended_owner": quadrant_block["actionable_strategy"]["recommended_owner"]
#             },
#             volatility=volatility_score,
#             has_pattern=pattern_block["has_pattern"],
#             qssi_tier=signal_strength_block["qssi_summary"]["qssi_tier"]
#         )

#             # Final output row
#             results.append({
#                 "experience_driver": experience_driver,
#                 "priority_class": pclass,
#                 "associated_entity_names": associated_names,
#                 "most_recent_mention": most_recent_mention,
#                 "no_of_mentions": no_of_mentions,
#                 "eri_score": round(row["ERI"], 2),
#                 "r_score": round(row["R"], 2),
#                 "f_score": round(row["F"], 2),
#                 "rf_score": round(row["RF"], 2),
#                 "rfi_score": round(row["RFI"], 2),
#                 "emotion_perception_tier": emotion_perception_tier,
#                 "rf_urgency_category": rf_urgency_category,
#                 "eri_rf_urgency_category": eri_rf_quadrant,

#                 # 🧠 XDI Phase 3 Modules (as blocks)
#                 "trend_block": {
#                     "trend_symbol": trend_symbol,
#                     "trend_pct": round(trend_pct, 2)
#                 },
#                 "momentum_block": {
#                     "momentum_symbol": momentum_symbol,
#                     "momentum_delta": round(momentum_delta, 2)
#                 },
#                 "volatility_block": {
#                     "volatility_tier": volatility_tier,
#                     "volatility_score": round(volatility_score, 2)
#                 },
#                 "pattern_block": {
#                     "has_pattern": pattern_block["has_pattern"],
#                     "pattern_type": pattern_block["pattern_type"] if pattern_block["has_pattern"] else "N/A",
#                     "pattern_strength": (round(pattern_block["pattern_strength"], 2)
#                                         if pattern_block["has_pattern"] and pattern_block["pattern_strength"] is not None else "N/A"),
#                     "pattern_confidence": pattern_block.get("pattern_confidence", "N/A"),
#                     "pain_day": pattern_block["pain_day"] if pattern_block["pattern_type"].lower() == "weekly" else "N/A",
#                     "data_coverage_days": pattern_block.get("data_coverage_days"),
#                     "min_required_days": pattern_block.get("min_required_days"),
#                     "eri_by_day": pattern_block.get("eri_by_day") if pattern_block["pattern_type"].lower() == "weekly" else None
#                 },
#                 "quadrant_block": quadrant_block,
#                 "signal_strength_block": signal_strength_block,
#                 "pem_block": pem_block
#             })

#         self.layer3_df = pd.DataFrame(results)

#         return self.layer3_df

# -*- coding: utf-8 -*-
# Layer 3 (Emotion-Driven Analytics Suite) + Layer 4 (QSSI) + PEM
# Requires: pandas as pd, numpy as np, statsmodels.tsa.stattools.acf

from datetime import timedelta
import pandas as pd
from statsmodels.tsa.stattools import acf


# === Quadrant Matrix Loader ===
def get_quadrant_matrix():
    """
    25-cell Saturation × Momentum matrix with full narratives.
    Columns:
      Saturation_Tier, Momentum_Tier, Diagnostic_Label, Strategic_Narrative,
      Urgency_Code, Recommended_Owner, Action_Guidance, Momentum_Context,
      Saturation_Context, Trajectory_Story
    """
    data = [
        # Very High
        ["🏆 Very High","↑↑ 🚀 Strongly Rising","Overload Alert","Sentiment at ceiling, emotion surging — may cause burnout or backlash.","🚨 Crisis","CX/Comms","Prepare relief interventions or redirection campaigns to prevent loyalty fatigue.","Surging beyond sustainable peaks","Already at maximum emotional investment","Customers hitting dangerous loyalty overload while emotional pressure keeps building"],
        ["🏆 Very High","↑ 📈 Moderately Rising","Plateau Pressure","Near ceiling, momentum creeping — optimize and stabilize.","⚠️ Risk","CX/Product","Reinforce strong areas without over-investing; monitor loyalty saturation.","Creeping toward emotional limits","Operating near peak satisfaction","Strong loyalty base experiencing gentle upward pressure requiring careful management"],
        ["🏆 Very High","→ ➖ Stable","Trust Plateau","Maxed emotion, stable loyalty — monitor quietly.","🔁 Watch","CX/Insights","Maintain service quality and prepare re-engagement initiatives if stagnation prolongs.","Holding steady at emotional peaks","Maxed out on emotional connection","Customers at peak trust with stable emotional investment but vulnerable to stagnation"],
        ["🏆 Very High","↓ 📉 Moderately Falling","Soft Fatigue","Trust is waning — find freshness.","⚠️ Risk","Product/CX","Introduce new features or messaging to reignite dormant advocates.","Drifting down from emotional heights","Losing ground from peak investment","Previously passionate advocates showing signs of emotional fatigue and declining connection"],
        ["🏆 Very High","↓↓ 🧨 Strongly Falling","Loyalty Collapse Risk","Once loyal, now disenchanted — urgent rescue.","🚨 Crisis","CX Leadership","Activate loyalty recovery programs and human-in-the-loop outreach.","Plummeting from emotional summit","Crashing down from maximum investment","Champions rapidly becoming detractors with emotional trust collapsing from peak levels"],
        # High
        ["✅ High","↑↑ 🚀 Strongly Rising","Optimization Zone","Push higher carefully — strong base, rising emotion.","🌱 Opportunity","Marketing/Product","Double down on what's working; deepen positive emotional signals.","Accelerating from strong foundation","Solid investment with room to grow","Loyal customers gaining emotional momentum with clear headroom for deeper connection"],
        ["✅ High","↑ 📈 Moderately Rising","Prime Expansion Zone","Loyalty is forming, act decisively.","🌱 Opportunity","Marketing","Scale reinforcement tactics and pre-loyalty rewards.","Building steadily toward peak loyalty","Strong base with expansion potential","Customers transitioning into deeper loyalty with positive emotional trajectory"],
        ["✅ High","→ ➖ Stable","Healthy Steady State","Solid footing — nurture gradually.","✅ Stable","CX","Continue nurturing but avoid unnecessary changes.","Maintaining strong emotional stability","Well-invested with sustainable levels","Loyal customers maintaining steady positive connection without volatility"],
        ["✅ High","↓ 📉 Moderately Falling","Cooling Off","Risk of losing momentum — reignite.","⚠️ Risk","CX/Product","Test new journeys or emotional engagement campaigns.","Sliding from loyal connection","Losing emotional investment gradually","Previously loyal customers experiencing emotional drift requiring re-engagement"],
        ["✅ High","↓↓ 🧨 Strongly Falling","Saturation Leakage","Slipping from once strong — needs boost.","🚨 Crisis","CX/Ops","Diagnose friction points and prevent emotional disengagement.","Rapidly abandoning strong position","Hemorrhaging established emotional value","Loyal customers experiencing sharp emotional decline threatening established relationship"],
        # Medium
        ["⚖️ Medium","↑↑ 🚀 Strongly Rising","Momentum Lift-Off","Emotion awakening — scale now.","🌱 Opportunity","Marketing","Capture momentum with smart CX nudges or rewards.","Breaking through emotional resistance","Building from moderate foundation","Neutral customers experiencing emotional awakening with significant growth potential"],
        ["⚖️ Medium","↑ 📈 Moderately Rising","Growth Window","Emotion stabilizing and rising — support journey.","🌱 Opportunity","CX/Insights","Spotlight rising themes; validate with wider audiences.","Climbing toward positive territory","Expanding from balanced starting point","Customers showing steady emotional improvement with clear trajectory toward loyalty"],
        ["⚖️ Medium","→ ➖ Stable","Balanced Neutral","No urgency — observe.","🔁 Watch","CX","No immediate action — continue observation.","Holding in emotional equilibrium","Balanced with no clear direction","Customers maintaining neutral stance with stable but uninspiring emotional connection"],
        ["⚖️ Medium","↓ 📉 Moderately Falling","Churn Warning","Emotion stuck, direction unclear — diagnose early.","⚠️ Risk","CX/Analytics","Run root cause analysis; test retention messaging.","Drifting toward emotional disconnect","Losing moderate investment slowly","Neutral customers sliding toward negative territory requiring early intervention"],
        ["⚖️ Medium","↓↓ 🧨 Strongly Falling","Indifference Trap","Low velocity, no connection — emotional vacuum.","🚨 Crisis","CX Leadership","Rebuild emotional relevance urgently; consider reboot strategies.","Falling into emotional void","Abandoning moderate connection rapidly","Customers rapidly disengaging from neutral position toward complete indifference"],
        # Low
        ["⚠️ Low","↑↑ 🚀 Strongly Rising","Breakthrough Opportunity","Momentum climbing out of emotional hole — catalyze.","🌱 Opportunity","CX/Marketing","Celebrate early wins; encourage customer voice amplification.","Surging upward from difficult position","Minimal investment with huge upside","Vulnerable customers experiencing emotional breakthrough with maximum improvement potential"],
        ["⚠️ Low","↑ 📈 Moderately Rising","Recovery Surge","Early signals of rebound — support.","🌱 Opportunity","CX","Invest in emotional follow-up; reward vocal feedback.","Climbing out of emotional deficit","Building from low base steadily","Previously frustrated customers showing recovery signals with room for significant growth"],
        ["⚠️ Low","→ ➖ Stable","Friction State","Low emotion, stagnant path — requires intervention.","⚠️ Risk","Product/Ops","Audit for service or process gaps.","Stuck in emotional limbo","Trapped at low investment levels","Customers maintaining negative connection without improvement or deterioration"],
        ["⚠️ Low","↓ 📉 Moderately Falling","Danger Zone","Downward pull + low loyalty — fix fast.","🚨 Crisis","CX/Ops","Apply crisis workflows and cross-functional fixes.","Sliding deeper into negativity","Losing remaining emotional value","Vulnerable customers declining further toward complete disconnection"],
        ["⚠️ Low","↓↓ 🧨 Strongly Falling","Critical Stall","Emotional damage deepening — act now.","🚨 Crisis","CX Leadership","Initiate emotional damage control protocol.","Plunging toward total disconnection","Rapidly abandoning minimal investment","Customers in emotional freefall from vulnerable position requiring immediate intervention"],
        # Very Low
        ["❌ Very Low","↑↑ 🚀 Strongly Rising","Signal Spike","Warning volatility — intense rise from a bad place.","⚠️ Risk","CX","Watch for false positives; investigate root cause of spike.","Surging upward from emotional rock bottom","Minimal emotional investment to lose","Customers climbing out of despair but volatility suggests unstable foundation"],
        ["❌ Very Low","↑ 📈 Moderately Rising","Erratic Revival","Surprising movement, unstable still — handle carefully.","⚠️ Risk","CX","Don't over-celebrate — check if emotion is anchored or episodic.","Showing signs of life from low point","Building from near-zero investment","Previously disconnected customers demonstrating fragile recovery requiring careful nurturing"],
        ["❌ Very Low","→ ➖ Stable","Dead Zone","No movement, no emotion — emotional disengagement.","🚨 Crisis","CX","Consider exit campaigns or silent churn save tactics.","Flatlined at emotional rock bottom","Zero emotional investment remaining","Customers in complete emotional disconnection with no signs of recovery"],
        ["❌ Very Low","↓ 📉 Moderately Falling","Decay Spiral","All indicators down — abandon or overhaul.","🚨 Crisis","CX/Ops","Run total overhaul diagnostics — emotional collapse imminent.","Sinking deeper into emotional void","Losing final remnants of connection","Disconnected customers deteriorating further with total relationship breakdown imminent"],
        ["❌ Very Low","↓↓ 🧨 Strongly Falling","Blackout State","Customer trust lost — rebuild from scratch.","🚨 Crisis","Executive Team","Relaunch brand experience — emotional trust annihilated.","Plummeting deeper into emotional void","Operating at rock bottom investment","Customers have lost all trust and emotional connection is deteriorating further"]
    ]
    cols = ["Saturation_Tier","Momentum_Tier","Diagnostic_Label","Strategic_Narrative",
            "Urgency_Code","Recommended_Owner","Action_Guidance","Momentum_Context",
            "Saturation_Context","Trajectory_Story"]
    return pd.DataFrame(data, columns=cols)


class Layer3Computer:
    def __init__(self, layer2_df, raw_df, timeframe_days, today_anchor, verbose=False):
        self.layer2_df = layer2_df.copy()
        self.raw_df = raw_df.copy()
        self.verbose = verbose
        self.timeframe_days = timeframe_days
        self.today = today_anchor
        self.cutoff_date = self.today - timedelta(days=self.timeframe_days)

        # 1) window filter
        self.raw_df["date"] = pd.to_datetime(self.raw_df["date"]).dt.date
        self.raw_df = self.raw_df[self.raw_df["date"] >= self.cutoff_date]

        # 2) emotion map
        emotion_scores = {"Adoration":3,"Appreciation":1,"Ambivalence":0,"Agitation":-1,"Anger":-3}
        self.raw_df["emotion_score"] = self.raw_df["emotion_primary"].map(emotion_scores)

        # 3) configs
        self.quadrant_matrix = get_quadrant_matrix()
        self.layer3_df = None
        self.pattern_lags = ["weekly","monthly"]  # enable "quarterly" when coverage ≥90 days

        if self.verbose:
            print(f"📦 Layer 3 @ {self.today} | Window: {self.cutoff_date} → {self.today}")
            print(f"🧮 Rows in window: {len(self.raw_df)} | L2 entities: {len(self.layer2_df)}")

    # === Core helpers ===
    def compute_normalized_eri(self, group):
        raw_eri = group["emotion_score"].mean()
        return ((raw_eri + 3) / 6) * 200 - 100

    def _compute_trend_series(self, series,
                          min_n=3,
                          smooth_min_n=7,
                          ewma_alpha=0.2,
                          pct_thresh=5.0,
                          snr_thresh=0.75) -> dict:
        """
        Window-agnostic trend:
        - Works for any n ≥ 3 (e.g., 7, 14, 28, 75 days)
        - Linear interpolate gaps; EWMA only if n ≥ smooth_min_n
        - Fit slope across window; convert to % of full ERI band (200 wide)
        - Tiny SNR guard scaled by sqrt(n)
        """
        try:
            s = pd.Series(series).astype(float)
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            n = len(s)
            if n < min_n:
                return {"symbol": "→", "trend_pct": 0.0}
    
            # de-gap gently; then (optionally) smooth
            s = s.interpolate(method="linear", limit_direction="both")
            if n >= smooth_min_n:
                s = s.ewm(alpha=ewma_alpha, adjust=False).mean()
    
            # quick flatness guard
            if s.max() - s.min() < 1e-9:
                return {"symbol": "→", "trend_pct": 0.0, "trend_snr": 0.0}
    
            # slope over the window (days 0..n-1)
            x = np.arange(n, dtype=float)
            slope = np.polyfit(x, s, 1)[0]           # ERI units/day
            delta = slope * (n - 1)                  # modeled change over the window
    
            # % of full ERI span (−100..+100 => 200)
            pct = (delta / 200.0) * 100.0
            pct = float(max(min(pct, 100.0), -100.0))
            pct_r = round(pct, 2)
    
            # SNR: day-to-day wiggle + length scaling
            noise = s.diff().std()
            if pd.isna(noise) or noise == 0.0:
                noise = 1.0
            snr = abs(delta) / (noise * max(np.sqrt(n), 1.0))
    
            if pct > pct_thresh and snr >= snr_thresh:
                sym = "↑"
            elif pct < -pct_thresh and snr >= snr_thresh:
                sym = "↓"
            else:
                sym = "→"
    
            return {"symbol": sym, "trend_pct": pct_r, "trend_snr": round(float(snr), 2)}
        except Exception:
            return {"symbol": "→", "trend_pct": 0.0}


    def _compute_momentum_series(self, series) -> dict:
        """
        Momentum = avg(second half) − avg(first half) on daily ERI
        - Linear interpolate gaps (avoids artificial persistence)
        - Optional EWMA smoothing (α=0.2) when n≥7
        - Light SNR guard on the two half-means
        - Same % scaling and thresholds (5%, 20%)
        """
        try:
            s = pd.Series(series).astype(float)
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            n = len(s)
            if n < 4:
                return {"symbol": "→", "delta": 0.0}
    
            # De-gap gently; then optional smoothing
            s = s.interpolate(method="linear", limit_direction="both")
            if n >= 7:
                s = s.ewm(alpha=0.2, adjust=False).mean()
    
            mid = n // 2
            first_half = s.iloc[:mid]
            second_half = s.iloc[mid:]
            if len(first_half) == 0 or len(second_half) == 0:
                return {"symbol": "→", "delta": 0.0}
    
            avg1 = float(first_half.mean())
            avg2 = float(second_half.mean())
            delta_raw = avg2 - avg1  # ERI units
    
            # Scale to % of full ERI band (−100..+100 => 200 wide)
            delta_pct = (delta_raw / 200.0) * 100.0
            delta_pct = float(max(min(delta_pct, 100.0), -100.0))
            delta_pct_r = round(delta_pct, 2)
    
            # Pooled standard error of the two half-means (guard tiny sizes)
            import math
            k1, k2 = max(len(first_half), 1), max(len(second_half), 1)
            sd1 = first_half.std() if k1 > 1 else 0.0
            sd2 = second_half.std() if k2 > 1 else 0.0
            se1 = sd1 / max(math.sqrt(k1), 1e-9)
            se2 = sd2 / max(math.sqrt(k2), 1e-9)
            pooled_se = math.sqrt(max(se1**2 + se2**2, 1e-12))
            snr = abs(delta_raw) / pooled_se if pooled_se > 0 else 0.0
    
            # Same thresholds, gated by light SNR
            if   delta_pct > 20.0 and snr >= 1.25: sym = "↑↑"
            elif delta_pct > 5.0  and snr >= 0.75: sym = "↑"
            elif delta_pct < -20.0 and snr >= 1.25: sym = "↓↓"
            elif delta_pct < -5.0  and snr >= 0.75: sym = "↓"
            else:                                   sym = "→"
    
            return {"symbol": sym, "delta": delta_pct_r}
        except Exception as e:
            if getattr(self, "verbose", False):
                print(f"⚠️ Momentum computation failed: {e}")
            return {"symbol": "→", "delta": 0.0}


    def _compute_volatility_series(self, series) -> dict:
        try:
            s = pd.Series(series).astype(float)
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            n = len(s)
            if n < 3:
                return {"tier": "✅ Stable", "score": 0.0, "score_adj": 0.0}
    
            # De-gap gently; then optional smoothing to tame spikes
            s = s.interpolate(method="linear", limit_direction="both")
            if n >= 7:
                s = s.ewm(alpha=0.2, adjust=False).mean()
    
            # Std as % of full ERI span (−100..+100 => 200)
            std = s.std()
            if pd.isna(std) or std == 0.0:
                return {"tier": "✅ Stable", "score": 0.0, "score_adj": 0.0}
    
            std_pct = (std / 200.0) * 100.0
            std_pct = round(float(std_pct), 2)
    
            # Length-normalized (for fair tiering across short vs long windows)
            score_adj = std_pct / max(np.sqrt(n), 1.0)
            score_adj = round(float(score_adj), 2)
    
            # Tiers on adjusted % (keeps labels fair across 7/14/28/75d windows)
            if score_adj <= 7.0:
                tier = "✅ Stable"
            elif score_adj <= 20.0:
                tier = "⚠ Fluctuating"
            else:
                tier = "🔴 Highly Fluctuating"
    
            return {"tier": tier, "score": std_pct, "score_adj": score_adj}
        except Exception as e:
            if getattr(self, "verbose", False):
                print(f"⚠️ Volatility computation failed: {e}")
            return {"tier": "✅ Stable", "score": 0.0, "score_adj": 0.0}


   def _compute_pattern_recognition(self, series, entity_data, lags):
        """
        Seasonal pattern detector (weekly / monthly / quarterly) for daily ERI.
        - Linear interpolate gaps (avoid ffill persistence)
        - Detrend + de-mean before ACF
        - ACF at seasonal lag with significance gate that scales with coverage
        - Weekly pain_day only if we have enough support per weekday
        """
        lag_days = {"weekly":7, "monthly":30, "quarterly":90}
        result = {
            "has_pattern": False,
            "pattern_type": "None",
            "pattern_strength": None,
            "pattern_confidence": "None",
            "pain_day": None,
            "data_coverage_days": int((series.index.max() - series.index.min()).days + 1) if len(series) else 0,
            "min_required_days": None,
            "eri_by_day": None
        }
    
        # 0) ensure continuous daily index
        if len(series) == 0:
            return result
        full_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq="D")
        s = series.reindex(full_index)
    
        # 1) de-gap gently, not ffill (avoid artificial persistence)
        s = s.interpolate(method="linear", limit_direction="both")
    
        # 2) detrend + center (avoid drift faking seasonality)
        # linear detrend
        try:
            x = np.arange(len(s), dtype=float)
            coef = np.polyfit(x, s.values.astype(float), 1)
            trend = coef[0]*x + coef[1]
            s_dt = pd.Series(s.values - trend, index=s.index)
        except Exception:
            s_dt = s.copy()
    
        # de-mean
        s_dt = s_dt - s_dt.mean()
    
        # quick flat/null guard
        if s_dt.std() == 0 or s_dt.isnull().all():
            if getattr(self, "verbose", False): print("⚠️ Skipping pattern: flat/null after detrend")
            return result
    
        # 3) coverage + valid lags
        series_days = (s_dt.index.max() - s_dt.index.min()).days + 1
        # require at least (lag + 1) and a modest extra buffer for stability
        valid_lags = [lag for lag in lags if series_days >= lag_days[lag] + 7]
        if getattr(self, "verbose", False):
            print(f"🔍 Valid pattern lags for {series_days} days: {valid_lags}")
    
        # 4) find strongest significant seasonal ACF
        best_pattern, best_score = None, 0.0
        for lag in valid_lags:
            L = lag_days[lag]
            try:
                # conservative acf; we only need value at lag L
                acf_vals = acf(s_dt, nlags=L, fft=True, missing='conservative')
                score = float(acf_vals[L])
    
                # significance gate ~ white-noise CI: ±1.96/sqrt(N)
                # use effective N = number of days
                N = len(s_dt)
                sig = 1.96 / max(np.sqrt(N), 1.0)
    
                # require both: passes CI AND above practical threshold (0.30)
                if (abs(score) >= sig) and (score >= 0.30) and (score > best_score):
                    # confidence tiers scale with strength above CI
                    if score >= 0.60:
                        confidence = "Strong"
                    elif score >= 0.45:
                        confidence = "Moderate"
                    else:
                        confidence = "Weak"
    
                    best_pattern = {
                        "has_pattern": True,
                        "pattern_type": lag.capitalize(),
                        "pattern_strength": round(score, 3),
                        "pattern_confidence": confidence,
                        "pain_day": None,
                        "data_coverage_days": int(series_days),
                        "min_required_days": int(L),
                        "eri_by_day": None
                    }
    
                    # 5) weekly extras: pain_day + eri_by_day only with support
                    if lag == "weekly":
                        ed = entity_data.copy()
                        # normalize ERI for the raw rows of this ED/day
                        ed["eri_norm"] = ((ed["emotion_score"] + 3.0)/6.0)*200.0 - 100.0
                        ed["weekday"] = ed["date"].apply(lambda x: x.strftime("%A"))
    
                        # require at least 2 observations per weekday to avoid flaky pain_day
                        counts = ed.groupby("weekday")["eri_norm"].size()
                        if (counts.min() if len(counts) >= 5 else 0) >= 2:
                            weekday_means = (
                                ed.groupby("weekday", as_index=True)["eri_norm"]
                                  .mean().round(2)
                            )
                            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                            if not weekday_means.empty:
                                # pick min on the ordered set for determinism
                                wmeans = {d: float(weekday_means.get(d)) if d in weekday_means.index else None for d in order}
                                # pain_day is the day with minimum mean ERI among days with data
                                avail = {d:v for d,v in wmeans.items() if v is not None}
                                if avail:
                                    best_pattern["pain_day"] = min(avail, key=avail.get)
                                    best_pattern["eri_by_day"] = wmeans
    
                    best_score = score
            except Exception as e:
                if getattr(self, "verbose", False): print(f"❌ ACF error for {lag}: {e}")
    
        return best_pattern if best_pattern else result

   def _compute_momentum_saturation_insight(self, eri_score, momentum_symbol) -> dict:
        """
        Map (saturation index from ERI, momentum symbol) -> 25-cell quadrant insight.
        - Tier-aware confidence (uses actual half-width of the current saturation tier)
        - Borderline flag when near a tier edge
        - Safe handling of NaN/inf ERI and unexpected momentum formatting
        """
        # 1) Saturation index + tiers
        try:
            sat_idx = (float(eri_score) + 100.0) / 200.0
        except Exception:
            sat_idx = 0.5  # neutral fallback
        if not np.isfinite(sat_idx):
            sat_idx = 0.5
        sat_idx = max(0.0, min(1.0, sat_idx))
    
        momentum_map = {
            "↑↑": "↑↑ 🚀 Strongly Rising",
            "↑":  "↑ 📈 Moderately Rising",
            "→":  "→ ➖ Stable",
            "↓":  "↓ 📉 Moderately Falling",
            "↓↓": "↓↓ 🧨 Strongly Falling"
        }
        momentum_symbol = momentum_symbol if momentum_symbol in momentum_map else "→"
    
        if   sat_idx >= 0.90: sat_emoji, sat_clean = "🏆 Very High", "Very High"
        elif sat_idx >= 0.65: sat_emoji, sat_clean = "✅ High", "High"
        elif sat_idx >= 0.45: sat_emoji, sat_clean = "⚖️ Medium", "Medium"
        elif sat_idx >= 0.25: sat_emoji, sat_clean = "⚠️ Low", "Low"
        else:                 sat_emoji, sat_clean = "❌ Very Low", "Very Low"
    
        mom_emoji = momentum_map[momentum_symbol]
        parts = mom_emoji.split(" ", 2)
        mom_clean = parts[2] if len(parts) == 3 else "Stable"
    
        # 2) Lookup in the 25-cell matrix
        qm = self.quadrant_matrix
        hit = qm[(qm["Saturation_Tier"] == sat_emoji) & (qm["Momentum_Tier"] == mom_emoji)]
    
        # 3) Confidence + borderline (tier-aware)
        sat_bounds = [0.0, 0.25, 0.45, 0.65, 0.90, 1.0]
        lower = max([b for b in sat_bounds if b <= sat_idx])
        upper = min([b for b in sat_bounds if b >= sat_idx])
        half_width = max((upper - lower) / 2.0, 1e-9)
        sat_distance = min(sat_idx - lower, upper - sat_idx)
        borderline = sat_distance < 0.02
    
        mom_strength = {"↑↑":1.0, "↑":0.6, "→":0.3, "↓":0.6, "↓↓":1.0}[momentum_symbol]
        quadrant_confidence = round(
            0.5 * mom_strength + 0.5 * min(1.0, sat_distance / half_width), 2
        )
    
        matrix_key = f"{sat_clean}|{mom_clean}"
    
        if not hit.empty:
            q = hit.iloc[0]
            payload = {
                "signal_classification": {
                    "saturation_index": round(sat_idx, 2),
                    "saturation_tier": sat_clean,
                    "momentum_tier": mom_clean,
                    "combined_quadrant": f"{mom_clean} Momentum × {sat_clean} Saturation"
                },
                "quadrant_interpretation": {
                    "quadrant_label": q["Diagnostic_Label"],
                    "urgency_level": q["Urgency_Code"],
                    "interpretation": q["Strategic_Narrative"]
                },
                "tactical_insight": {
                    "emotional_pulse": f"{q['Diagnostic_Label']} — {q['Momentum_Context']} while {q['Saturation_Context']}",
                    "battle_status": f"{q['Urgency_Code']} {q['Diagnostic_Label']}",
                    "strategic_reality": f"{q['Trajectory_Story']} — system recommends {q['Action_Guidance']}"
                },
                "actionable_strategy": {
                    "momentum_context": q["Momentum_Context"],
                    "saturation_context": q["Saturation_Context"],
                    "trajectory_story": q["Trajectory_Story"],
                    "action_guidance": q["Action_Guidance"],
                    "recommended_owner": q["Recommended_Owner"]
                }
            }
        else:
            payload = {
                "signal_classification": {
                    "saturation_index": round(sat_idx, 2),
                    "saturation_tier": sat_clean,
                    "momentum_tier": mom_clean,
                    "combined_quadrant": "Unknown"
                },
                "quadrant_interpretation": {
                    "quadrant_label": "Unknown",
                    "urgency_level": "Unknown",
                    "interpretation": "No data available for this quadrant"
                },
                "tactical_insight": {
                    "emotional_pulse": "Unknown — No data available for emotional assessment",
                    "battle_status": "⚠️ Risk Unknown",
                    "strategic_reality": "No trajectory data available — system recommends data validation"
                },
                "actionable_strategy": {
                    "momentum_context": "Unknown momentum pattern",
                    "saturation_context": "Unknown saturation level",
                    "trajectory_story": "No trajectory data available",
                    "action_guidance": "Validate data sources and retry analysis",
                    "recommended_owner": "Data Team"
                }
            }
    
        payload["meta"] = {
            "matrix_key": matrix_key,
            "saturation_tier_emoji": sat_emoji,
            "momentum_tier_emoji": mom_emoji,
            "quadrant_confidence": quadrant_confidence,
            "borderline": borderline
        }
        return payload

    # === Layer 4: Signal Strength (QSSI) ===
    def _compute_signal_strength(
        self,
        trend: str,
        momentum: str,
        saturation_index: float,
        trend_pct: float = None,          # optional: % of ERI band from trend module
        momentum_delta: float = None,     # optional: % of ERI band from momentum module
        n_days: int = None,               # optional: series length
        vol_norm: float = None            # optional: volatility normalized to [0,1]
    ) -> dict:
        # --- sanitize inputs ---
        sym_trend = trend if trend in ("↑", "→", "↓") else "→"
        sym_mom   = momentum if momentum in ("↑↑","↑","→","↓","↓↓") else "→"
        si = saturation_index
        try:
            si = float(si)
        except Exception:
            si = 0.5
        si = max(0.0, min(1.0, si))
    
        # --- velocity score (symbol-only base, same as your logic) ---
        if sym_trend in ("↑","↓") and sym_mom == "↓↓":
            velocity_score, rationale = 6, "Sharp trend shift with strong counter-momentum"
        elif sym_trend in ("↑","↓") and sym_mom == "↑↑":
            velocity_score, rationale = 5, "Rapid acceleration with positive momentum surge"
        elif sym_trend in ("↑","↓") and sym_mom in ("↑","↓"):
            velocity_score, rationale = 4, "Moderate directional movement with matching momentum"
        elif sym_trend in ("↑","↓") and sym_mom == "→":
            velocity_score, rationale = 2, "Directional change but stable momentum"
        elif sym_trend == "→" and sym_mom in ("↑↑","↓↓"):
            velocity_score, rationale = 3, "Flat trend with sudden strong shift emerging"
        elif sym_trend == "→" and sym_mom in ("↑","↓"):
            velocity_score, rationale = 1, "Stable trend with mild fluctuation"
        else:
            velocity_score, rationale = 0, "No meaningful trend or momentum detected"
    
        # --- optional micro-nudges (NO effect unless you pass values) ---
        bump = 0
        if trend_pct is not None:
            # +1 if the trend magnitude is clearly non-trivial
            if abs(float(trend_pct)) >= 10.0:
                bump += 1
        if momentum_delta is not None:
            # +1 if the momentum magnitude is clearly non-trivial
            if abs(float(momentum_delta)) >= 20.0:
                bump += 1
        # small conservatism for short/noisy windows, only if provided
        penalty = 0
        if n_days is not None and n_days < 10:
            penalty += 1
        if vol_norm is not None and float(vol_norm) > 0.8:
            penalty += 1
    
        velocity_score = max(0, min(6, velocity_score + bump - penalty))
    
        # --- saturation score (headroom) ---
        if   si <= 0.20:
            saturation_score, sat_r = 4, "Very low emotional saturation — fresh pain or interest forming"
        elif si <= 0.40:
            saturation_score, sat_r = 3, "Low saturation — likely early-stage signal"
        elif si <= 0.60:
            saturation_score, sat_r = 2, "Medium saturation — emotionally stable signal"
        elif si <= 0.80:
            saturation_score, sat_r = 1, "High saturation — signal is nearing emotional capacity"
        else:
            saturation_score, sat_r = 0, "Very high saturation — emotional exhaustion or signal decay"
    
        # --- composite ---
        qssi = int(velocity_score + saturation_score)  # stays in 0..10
    
        if   qssi >= 9:
            qssi_tier, interp = "💥 Critical Signal","Extreme movement detected with low saturation — high potential volatility or opportunity"
        elif qssi >= 6:
            qssi_tier, interp = "🔥 Strong Signal","Substantial shift in emotional dynamics — active attention required"
        elif qssi >= 4:
            qssi_tier, interp = "🌱 Emerging Signal","Signal beginning to form — track evolution"
        elif qssi >= 1:
            qssi_tier, interp = "🔁 Weak Signal","Low signal strength — may resolve on its own"
        else:
            qssi_tier, interp = "❌ No Signal","Dormant — not actionable"
    
        return {
            "velocity_component": {
                "trend_symbol": sym_trend,
                "momentum_symbol": sym_mom,
                "velocity_score": int(velocity_score),
                "velocity_rationale": rationale,
                # expose nudges only if they applied
                **({"nudges": {"bump": bump, "penalty": penalty}} if (bump or penalty) else {})
            },
            "saturation_component": {
                "saturation_index": round(si, 2),
                "saturation_score": int(saturation_score),
                "saturation_rationale": sat_r
            },
            "qssi_summary": {
                "qssi_score": int(qssi),
                "qssi_tier": qssi_tier,
                "qssi_interpretation": interp
            }
        }

   # === PEM (Predictive Emotional Modeling) — aligned to new modules ===
    def build_predictive_emotional_modeling(self, state_of_play, volatility, has_pattern, qssi_tier) -> dict:
        """
        Expect:
          - state_of_play: {
              momentum_tier: "Strongly Rising"|"Moderately Rising"|"Stable"|"Moderately Falling"|"Strongly Falling",
              saturation_tier: "Very High"|"High"|"Medium"|"Low"|"Very Low",
              trajectory_story, action_guidance, recommended_owner
            }
          - volatility: float  # **% of ERI band** (from volatility.score, not score_adj)
          - has_pattern: bool
          - qssi_tier: "💥 Critical Signal"|"🔥 Strong Signal"|"🌱 Emerging Signal"|"🔁 Weak Signal"|"❌ No Signal"
        """
    
        # --- unpack ---
        momentum_tier      = state_of_play["momentum_tier"]
        saturation_tier    = state_of_play["saturation_tier"]
        trajectory_story   = state_of_play["trajectory_story"]
        action_guidance    = state_of_play["action_guidance"]
        recommended_owner  = state_of_play["recommended_owner"]
    
        # --- maps (explainable + stable) ---
        qssi_map     = {"💥 Critical Signal":1.00, "🔥 Strong Signal":0.80, "🌱 Emerging Signal":0.50, "🔁 Weak Signal":0.20, "❌ No Signal":0.00}
        momentum_map = {"Strongly Rising":1.00, "Moderately Rising":0.60, "Stable":0.00, "Moderately Falling":-0.60, "Strongly Falling":-1.00}
        headroom_map = {"Very High":0.00, "High":0.25, "Medium":0.50, "Low":0.75, "Very Low":1.00}
    
        qssi_strength  = float(qssi_map.get(qssi_tier, 0.0))
        momentum_score = float(momentum_map.get(momentum_tier, 0.0))
        headroom       = float(headroom_map.get(saturation_tier, 0.5))
    
        # --- volatility normalization (now in % of ERI band) ---
        # old raw thresholds (10, 45) map to ≈5% and ≈22.5% of ERI band
        vol_pct  = max(0.0, float(volatility))
        vol_norm = min(vol_pct / 20.0, 1.0)   # ~20% considered "high"
        stability = 1.0 - vol_norm
    
        # --- directionalized QSSI + balanced components ---
        esc_raw = (
            0.35 * qssi_strength * max(0.0,  momentum_score) +  # QSSI pushes with direction
            0.40 * max(0.0,  momentum_score) +
            0.30 * headroom +
            0.20 * (1.0 if has_pattern else 0.0) +
            0.20 * stability
        )
        dec_raw = (
            0.35 * qssi_strength * max(0.0, -momentum_score) +  # QSSI pushes with direction
            0.60 * max(0.0, -momentum_score) +
            0.30 * (1.0 - headroom) +
            0.20 * vol_norm
        )
    
        esc_raw = max(0.0, esc_raw)
        dec_raw = max(0.0, dec_raw)
        total = esc_raw + dec_raw
        esc_prob = (esc_raw / total) if total > 1e-9 else 0.0
        dec_prob = (dec_raw / total) if total > 1e-9 else 0.0
    
        # --- rules-first overrides (aligned thresholds in % terms) ---
        signal_is_precursor = (
            momentum_tier in ["Strongly Rising", "Moderately Rising"] and
            qssi_tier in ["🔥 Strong Signal", "💥 Critical Signal"] and
            has_pattern
        )
        signal_is_at_risk_of_decay = (
            momentum_tier in ["Strongly Falling", "Moderately Falling"] and
            vol_pct >= 5.0   # ≈ old 10 units
        )
    
        if signal_is_precursor:
            pem_trajectory  = "Likely Escalation"
            pem_probability = max(esc_prob, 0.65)
            explanation     = "Rising momentum + strong/critical signal + repeating pattern indicates intensification"
            risk_class      = "Opportunity"
            rule_trigger, basis = "rising+strong_qssi+pattern", "esc_prob"
            counterfactual_pointer = "Flip if momentum ≤ 'Moderately Falling' or volatility > 22.5%"
        elif signal_is_at_risk_of_decay:
            pem_trajectory  = "At Risk of Decay"
            pem_probability = max(dec_prob, 0.65)
            explanation     = "Falling momentum + elevated volatility suggests emotional energy is fading or fracturing"
            risk_class      = "Risk"
            rule_trigger, basis = "falling+volatility", "dec_prob"
            counterfactual_pointer = "Flip if momentum ≥ 'Moderately Rising' and volatility < 5%"
        else:
            if max(esc_prob, dec_prob) < 0.55:
                pem_trajectory  = "Stable / Inconclusive"
                pem_probability = max(esc_prob, dec_prob)
                explanation     = "No dominant predictive anchors; continue monitoring"
                risk_class      = "Neutral"
            else:
                if esc_prob > dec_prob:
                    pem_trajectory, pem_probability = "Likely Escalation", esc_prob
                    explanation, risk_class = "Model indicates upward trajectory dominance (momentum/headroom/stability blend)", "Opportunity"
                else:
                    pem_trajectory, pem_probability = "At Risk of Decay", dec_prob
                    explanation, risk_class = "Model indicates downward trajectory dominance (momentum/volatility/ceiling pressure)", "Risk"
            rule_trigger = "model_choice"
            basis = "esc_prob" if esc_prob >= dec_prob else "dec_prob"
            counterfactual_pointer = "Flip if next horizon favors the opposite momentum tier"
    
        # --- confidence (aligned to % thresholds) ---
        if has_pattern and vol_pct >= 5.0:
            confidence_tier, confidence_score = "High", 0.85
        elif has_pattern or vol_pct >= 5.0:
            confidence_tier, confidence_score = "Moderate", 0.55
        else:
            confidence_tier, confidence_score = "Low", 0.30
    
        # --- elasticity (unchanged logic) ---
        elasticity_rating = (
            "High" if headroom >= 0.5 and abs(momentum_score) >= 0.6
            else "Moderate" if headroom >= 0.25
            else "Low"
        )
    
        horizon_days = int(getattr(self, "pem_horizon_days", 14))
        version = "PEM.v1.2"
    
        feature_vector = {
            "qssi_strength": round(qssi_strength, 3),
            "momentum_score": round(momentum_score, 3),
            "headroom": round(headroom, 3),
            "volatility_pct": round(vol_pct, 3),
            "volatility_norm": round(vol_norm, 3),
            "stability": round(stability, 3),
            "has_pattern": bool(has_pattern),
            "esc_raw": round(esc_raw, 3),
            "dec_raw": round(dec_raw, 3),
            "esc_prob": round(esc_prob, 3),
            "dec_prob": round(dec_prob, 3)
        }
    
        return {
            "trajectory_forecast": {
                "pem_trajectory": pem_trajectory,
                "pem_probability": round(float(pem_probability), 3),
                "pem_confidence": confidence_tier,
                "confidence_score": round(confidence_score, 2),
                "risk_class": risk_class,
                "horizon_days": horizon_days,
                "explanation": explanation,
                "version": version,
                "rule_trigger": rule_trigger,
                "basis": basis,
                "counterfactual_pointer": counterfactual_pointer
            },
            "signal_diagnostics": {
                "has_repeating_pattern": bool(has_pattern),
                "volatility_pct": round(vol_pct, 2),
                "momentum_tier": momentum_tier,
                "saturation_tier": saturation_tier,
                "qssi_tier": qssi_tier
            },
            "future_risk_profile": {
                "trajectory_story": trajectory_story,
                "action_guidance": action_guidance,
                "recommended_owner": recommended_owner
            },
            "audit": {
                "feature_vector": feature_vector,
                "elasticity_rating": elasticity_rating
            }
        }

   # === Main compute ===
    def compute(self):
        results, skipped = [], []
        self.raw_df["experience_driver"] = self.raw_df["experience_driver"].str.strip()
        entities = self.layer2_df[self.layer2_df["Priority_Status"].isin(["P0", "P1", "P2", "P3"])]
    
        for _, row in entities.iterrows():
            try:
                ed = str(row["experience_driver"]).strip()
                data = self.raw_df[self.raw_df["experience_driver"] == ed]
                if data.empty:
                    if self.verbose:
                        print(f"❌ DROP [{ed}] → no raw data")
                    skipped.append((ed, "No raw data"))
                    continue
    
                # daily ERI series
                daily_eri = (data.groupby("date").apply(self.compute_normalized_eri).sort_index())
                if len(daily_eri) > 1:
                    # full_idx = pd.date_range(start=daily_eri.index.min(), end=daily_eri.index.max(), freq="D")
                    # daily_eri = daily_eri.reindex(full_idx).bfill().ffill()
                    # USE YOUR ESTABLISHED WINDOW BOUNDARIES
                    full_idx = pd.date_range(start=self.cutoff_date, end=self.today, freq="D")
                    daily_eri = daily_eri.reindex(full_idx).bfill().ffill()
                if len(daily_eri) <= 1:
                    if self.verbose:
                        print(f"⏭️ DROP [{ed}] → insufficient days ({len(daily_eri)})")
                    skipped.append((ed, f"Insufficient days: {len(daily_eri)}"))
                    continue
    
                # modules
                trend = self._compute_trend_series(daily_eri)
                momentum = self._compute_momentum_series(daily_eri)
                volatility = self._compute_volatility_series(daily_eri)
                pattern_block = self._compute_pattern_recognition(
                    daily_eri, data, self.pattern_lags
                )
    
                trend_symbol, trend_pct = trend["symbol"], trend["trend_pct"]
                momentum_symbol, momentum_delta = momentum["symbol"], momentum["delta"]
    
                # volatility now in % of ERI band; keep adj if present
                volatility_tier = volatility["tier"]
                volatility_pct = float(volatility.get("score", 0.0))           # % of ERI band
                volatility_pct_adj = float(volatility.get("score_adj", volatility_pct))
                # normalized 0..1 for confidence/QSSI (≈20% considered "high")
                vol_norm = min(max(volatility_pct / 20.0, 0.0), 1.0)
    
                quadrant_block = self._compute_momentum_saturation_insight(
                    row["ERI"], momentum_symbol
                )
    
                # QSSI: pass new optional signals for tiny nudges (safe if unused)
                signal_strength_block = self._compute_signal_strength(
                    trend_symbol,
                    momentum_symbol,
                    quadrant_block["signal_classification"]["saturation_index"],
                    trend_pct=float(trend_pct),
                    momentum_delta=float(momentum_delta),
                    n_days=int(len(daily_eri)),
                    vol_norm=vol_norm
                )
    
                # PEM expects volatility **% of ERI band**
                pem_block = self.build_predictive_emotional_modeling(
                    state_of_play={
                        "momentum_tier": quadrant_block["signal_classification"]["momentum_tier"],
                        "saturation_tier": quadrant_block["signal_classification"]["saturation_tier"],
                        "trajectory_story": quadrant_block["actionable_strategy"]["trajectory_story"],
                        "action_guidance": quadrant_block["actionable_strategy"]["action_guidance"],
                        "recommended_owner": quadrant_block["actionable_strategy"]["recommended_owner"]
                    },
                    volatility=volatility_pct,                       # % of ERI band
                    has_pattern=pattern_block["has_pattern"],
                    qssi_tier=signal_strength_block["qssi_summary"]["qssi_tier"]
                )
    
                # provenance & horizons
                num_days_observed = int(len(daily_eri))
    
                # % days with any mentions — align index types first
                ds = data.groupby("date").size()
                ds.index = pd.to_datetime(ds.index)
                days_with_signal = ds.reindex(daily_eri.index, fill_value=0)
                pct_days_with_signal = round(
                    float((days_with_signal > 0).sum()) / num_days_observed, 3
                )
    
                # unified Layer-3 confidence (vol_norm uses new scaling)
                pattern_conf_norm = (
                    1.0 if pattern_block.get("pattern_confidence") == "Strong"
                    else 0.6 if pattern_block.get("pattern_confidence") == "Weak"
                    else 0.3
                )
                layer3_confidence_score = round(
                    min(1.0, max(0.0,
                        0.4 * quadrant_block["meta"]["quadrant_confidence"]
                        + 0.3 * (1.0 - vol_norm)
                        + 0.3 * pattern_conf_norm
                    )), 2
                )
    
                # storyline
                storyline = (
                    f"{ed} is in {quadrant_block['quadrant_interpretation']['quadrant_label']} with "
                    f"{quadrant_block['signal_classification']['momentum_tier']} momentum and "
                    f"{quadrant_block['signal_classification']['saturation_tier']} saturation. "
                    f"{quadrant_block['quadrant_interpretation']['interpretation']} "
                    f"Action: {quadrant_block['actionable_strategy']['action_guidance']} "
                    f"(Owner: {quadrant_block['actionable_strategy']['recommended_owner']})."
                )
    
                # safe helpers
                ptype = (pattern_block.get("pattern_type") or "").lower()
                has_weekly = ptype == "weekly"
                pattern_payload = {
                    "has_pattern": bool(pattern_block["has_pattern"]),
                    "pattern_type": pattern_block["pattern_type"] if pattern_block["has_pattern"] else None,
                    "pattern_strength": (
                        round(float(pattern_block["pattern_strength"]), 2)
                        if pattern_block["has_pattern"]
                        and pattern_block.get("pattern_strength") is not None
                        else None
                    ),
                    "pattern_confidence": pattern_block.get("pattern_confidence")
                    if pattern_block["has_pattern"] else None,
                    "pain_day": pattern_block["pain_day"] if has_weekly else None,
                    "data_coverage_days": pattern_block.get("data_coverage_days"),
                    "min_required_days": pattern_block.get("min_required_days"),
                    "eri_by_day": pattern_block.get("eri_by_day") if has_weekly else None
                }
    
                capsule_meta = {
                    "capsule_id": f"SC-{uuid4().hex[:12]}",
                    "generated_at": pd.Timestamp.utcnow().isoformat(),
                    "version": "XDI.v1",
                    "window_start_date": str(self.cutoff_date),
                    "window_end_date": str(self.today)
                }
    
                # build capsule
                results.append({
                    "experience_driver": ed,
                    "priority_class": row["Priority_Status"],
                    "associated_entity_names": row.get("Associated_Entity_Names"),
                    "most_recent_mention": row.get("Most_Recent_Date"),
                    "no_of_mentions": row.get("No_of_Mentions"),
                    "eri_score": round(row["ERI"], 2),
                    "r_score": round(row["R"], 2),
                    "f_score": round(row["F"], 2),
                    "rf_score": round(row["RF"], 2),
                    "rfi_score": round(row["RFI"], 2),
                    "emotion_perception_tier": row.get("Loyalty_State"),
                    "rf_urgency_category": row.get("RF_Urgency_Category"),
                    "eri_rf_urgency_category": row.get("ERI_RF_Quadrant"),
    
                    "trend_block": {"trend_symbol": trend_symbol, "trend_pct": round(trend_pct, 2)},
                    "momentum_block": {"momentum_symbol": momentum_symbol, "momentum_delta": round(momentum_delta, 2)},
                    "volatility_block": {
                        "volatility_tier": volatility_tier,
                        "volatility_score": round(volatility_pct, 2),        # % of ERI band (human-facing)
                        **({"volatility_score_adj": round(volatility_pct_adj, 2)} if "score_adj" in volatility else {})
                    },
                    "pattern_block": pattern_payload,
    
                    "momentum_saturation_insight": quadrant_block,
                    "signal_strength_index": signal_strength_block,
                    "predictive_emotion_forecast": pem_block,
    
                    "momentum_saturation_story": storyline,
                    "momentum_saturation_confidence": quadrant_block["meta"]["quadrant_confidence"],
                    "momentum_saturation_borderline_flag": quadrant_block["meta"]["borderline"],
                    "analytics_suite_confidence": layer3_confidence_score,
    
                    "provenance": {
                        "analysis_window_days": int(self.timeframe_days),
                        "total_days_observed": num_days_observed,
                        "signal_presence_pct": pct_days_with_signal,
                        "trend_analysis_days": num_days_observed,
                        "momentum_analysis_days": int(num_days_observed // 2),
                        "series_data_complete": True,
                        **capsule_meta
                    },
    
                    "pdca_hint": {
                        "momentum_saturation_label": quadrant_block["quadrant_interpretation"]["quadrant_label"],
                        "execution_urgency_level": quadrant_block["quadrant_interpretation"]["urgency_level"],
                        "suggested_action_owner": quadrant_block["actionable_strategy"]["recommended_owner"],
                        "preliminary_action_guidance": quadrant_block["actionable_strategy"]["action_guidance"]
                    }
                })
    
            except Exception as e:
                if self.verbose:
                    print(f"💥 FAIL [{ed}] → {e}")
                skipped.append((ed, str(e)))
                continue
    
        self.layer3_df = pd.DataFrame(results)
        self.skipped_entities = skipped
        return self.layer3_df

"""

`_compute_pattern_recognition()` function explained step-by-step

## **Purpose**
Detect whether an **ERI time series** (emotion score over time for an Experience Driver) shows a **repeating 
temporal pattern** — weekly, monthly, or quarterly — and, if so:

* Classify its type (Weekly/Monthly/Quarterly)
* Measure its strength & confidence
* Identify pain points (like worst-performing weekday)
* Return a structured diagnostic block


## **Inputs**

* **series** → Pandas Series, indexed by date, containing ERI values (normalized emotion score over time)
* **entity_data** → DataFrame with raw entity-level data (contains `emotion_score` and `date` columns)
* **lags** → List of period types to check, e.g., `["weekly", "monthly", "quarterly"]`


## **Step-by-Step Logic**

### 1️⃣ **Setup & Defaults**

```python
lag_days = {"weekly":7,"monthly":30,"quarterly":90}
```

Maps pattern types to the number of days in their cycle.
**Initial `result`** is a "no pattern" default — used if detection fails:

* `has_pattern` → False
* `pattern_type` → None
* `pattern_strength` → None
* `pain_day` → None
* Coverage metadata (`data_coverage_days`, `min_required_days`, `eri_by_day`)


### 2️⃣ **Ensure Continuity in the Time Series**

```python
full_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq='D')
series = series.reindex(full_index).ffill()
```

* Fills **gaps** in the time series (ensures every day is present)
* Forward-fills missing values to maintain continuity for autocorrelation

### 3️⃣ **Skip Flat or Empty Series**
If the series has **zero standard deviation** or is entirely null → no pattern possible, return defaults.


### 4️⃣ **Determine Which Lags Are Valid**

```python
valid_lags = [lag for lag in lags if series_days >= lag_days[lag] + 1]
```

* Only consider lags if there’s enough data coverage (e.g., you can’t check quarterly pattern if you have only 
60 days of data)
* `series_days` = total days of data coverage


### 5️⃣ **Search for the Strongest Pattern**

For each **valid lag**:
1. Compute **ACF** (Autocorrelation Function) for up to that lag length:

   ```python
   acf_vals = acf(series, nlags=L, fft=True, missing='conservative')
   ```
2. Look at correlation at **exact lag step** (`acf_vals[L]`).
3. If correlation score ≥ **0.3**, consider it a valid pattern:

   * `Strong` confidence if score ≥ **0.6**
   * Keep track of **best pattern found so far**.


### 6️⃣ **Special Handling for Weekly Patterns**
If the strongest pattern is **weekly**:

* Recompute **normalized ERI** for each day of week:

  ```python
  eri_norm = ((emotion_score + 3) / 6) * 200 - 100
  ```
* Group by weekday, find **average ERI** per day.
* Identify **worst performing weekday** (`pain_day`).
* Save per-day ERI distribution in `eri_by_day`.


### 7️⃣ **Return Result**
If **best pattern** was found → return it.
Otherwise → return "no pattern" default.

## **Key Output Fields**

* `has_pattern` → True/False
* `pattern_type` → Weekly/Monthly/Quarterly
* `pattern_strength` → Autocorrelation score (0–1)
* `pattern_confidence` → "Strong" or "Weak"
* `pain_day` → Worst-performing weekday (only for weekly)
* `eri_by_day` → Dict of weekday → average ERI (only for weekly)
* `data_coverage_days` → Total days of data available
* `min_required_days` → Days needed to detect pattern of that lag

## **Why This Matters in Decipher**
This module powers the **`has_pattern`** flag and related diagnostics in XDI Layer 3.
PEM uses `has_pattern` as a **predictive anchor** — repeating emotional rhythms indicate **either sustained opportunity or sustained risk** depending on the trend.

---------------------------------------------------------------------------------------------------------------------

`_compute_momentum_saturation_insight` step-by-step explanation:

## **Function Purpose**
This method combines **emotional saturation** (how “full” or “empty” the emotional tank is) with **momentum** 
(the direction and intensity of change) to classify a signal into a **strategic quadrant**.
It produces a **rich payload** describing the current state, quadrant meaning, tactical advice, and meta diagnostics.

## **Step-by-Step Breakdown**

### **1. Compute the Saturation Index**

```python
sat_idx = (eri_score + 100) / 200.0
sat_idx = max(0.0, min(1.0, float(sat_idx)))
```

* Converts ERI score (−100 to +100) into a **0.0–1.0** scale.
* Clamps to ensure values stay within range.
* **Meaning:** 0 = totally empty emotional energy, 1 = fully saturated emotional energy.

### **2. Map Momentum Symbols**

```python
momentum_map = {
    "↑↑": "↑↑ 🚀 Strongly Rising",
    "↑":  "↑ 📈 Moderately Rising",
    "→":  "→ ➖ Stable",
    "↓":  "↓ 📉 Moderately Falling",
    "↓↓": "↓↓ 🧨 Strongly Falling"
}
momentum_symbol = momentum_symbol if momentum_symbol in momentum_map else "→"
```

* Momentum comes in 5 discrete symbols.
* Each symbol has a **visual emoji** + **plain description**.
* If the input momentum symbol is invalid, it defaults to `"→"` (Stable).

### **3. Assign Saturation Tiers**

```python
if   sat_idx >= 0.90: sat_emoji, sat_clean = "🏆 Very High", "Very High"
elif sat_idx >= 0.65: sat_emoji, sat_clean = "✅ High", "High"
elif sat_idx >= 0.45: sat_emoji, sat_clean = "⚖️ Medium", "Medium"
elif sat_idx >= 0.25: sat_emoji, sat_clean = "⚠️ Low", "Low"
else:                 sat_emoji, sat_clean = "❌ Very Low", "Very Low"
```

* Emotional energy is categorized into **Very Low → Very High**.
* Emojis provide instant readability.

### **4. Lookup in the Quadrant Matrix**

```python
qm = self.quadrant_matrix
hit = qm[(qm["Saturation_Tier"] == sat_emoji) & (qm["Momentum_Tier"] == mom_emoji)]
```

* `quadrant_matrix` holds the **strategic meaning** for every Saturation × Momentum combination.
* A “hit” means we have a predefined interpretation and action for this emotional state.

### **5. Borderline Detection & Confidence**

```python
sat_bounds = [0.0, 0.25, 0.45, 0.65, 0.90, 1.0]
nearest = min(sat_bounds, key=lambda b: abs(b - sat_idx))
sat_distance = abs(sat_idx - nearest)
borderline = sat_distance < 0.02
mom_strength = {"↑↑":1.0,"↑":0.6,"→":0.3,"↓":0.6,"↓↓":1.0}[momentum_symbol]
quadrant_confidence = round(0.5 * mom_strength + 0.5 * min(1.0, sat_distance/0.20), 2)
```

* **Borderline:** True if we are within 2% of crossing into another saturation tier.
* **Quadrant Confidence:** Combines momentum strength with how close we are to a tier center.
* This is important for **forecast sensitivity** in PEM.


### **6. Build the Payload**

If quadrant data exists:

* **signal_classification** → Pure metrics (saturation, momentum, combined quadrant).
* **quadrant_interpretation** → Label, urgency code, and narrative from the quadrant matrix.
* **tactical_insight** → Emotional pulse, battle status, and strategic reality.
* **actionable_strategy** → Specific guidance, context, and suggested owner.

If quadrant data is missing:

* All fields return `"Unknown"` with instructions to validate data.

### **7. Add Meta Information**

```python
payload["meta"] = {
    "matrix_key": matrix_key,
    "saturation_tier_emoji": sat_emoji,
    "momentum_tier_emoji": mom_emoji,
    "quadrant_confidence": quadrant_confidence,
    "borderline": borderline
}
```

* `matrix_key` is a shorthand like `"High|Moderately Rising"`.
* Confidence score + borderline flag are preserved for decision-making transparency.


## **Key Strengths**

1. **Fully grounded in existing ERI, momentum, and quadrant data.**
2. **Borderline detection** ensures proactive alerts before a shift happens.
3. **Separation of metrics, meaning, and actions** — perfect for PDCA and PEM integration.
4. Produces an **execution-ready object** — no extra processing needed downstream.

--------------------------------------------------------------------------------------------------------------

 `_compute_signal_strength` (QSSI) step-by-step explanation:

# Function Purpose

Quantify **signal strength** by fusing:

* **Velocity** (trend × momentum) → how forcefully things are moving
* **Saturation** (headroom) → how much emotional capacity remains

Outputs a **QSSI score & tier** with human-readable rationales for each component.

# Step-by-Step

## 1) Velocity score (trend × momentum → 0..6)

```python
if trend in ("↑","↓") and momentum == "↓↓": velocity_score = 6
elif trend in ("↑","↓") and momentum == "↑↑": velocity_score = 5
elif trend in ("↑","↓") and momentum in ("↑","↓"): velocity_score = 4
elif trend in ("↑","↓") and momentum == "→": velocity_score = 2
elif trend == "→" and momentum in ("↑↑","↓↓"): velocity_score = 3
elif trend == "→" and momentum in ("↑","↓"): velocity_score = 1
else: velocity_score = 0
```

* Translates the **directional trajectory** (trend) and **recent force** (momentum) into a single **velocity dial**.
* Comes with a `velocity_rationale` string explaining the choice (e.g., “Sharp trend shift with strong counter-momentum”).

## 2) Saturation score (headroom → 0..4)

```python
si = clamp01(saturation_index)
if si <= .20: saturation_score = 4
elif si <= .40: 3
elif si <= .60: 2
elif si <= .80: 1
else: 0
```

* Lower saturation = **more headroom**, so **higher score** (more responsive/impactful).
* Provides `saturation_rationale` (e.g., “Very low emotional saturation — fresh pain or interest forming”).

## 3) QSSI calculation & tiering

```python
qssi = velocity_score + saturation_score  # 0..10
if qssi >= 9:  "💥 Critical Signal"
elif qssi >= 6:"🔥 Strong Signal"
elif qssi >= 4:"🌱 Emerging Signal"
elif qssi >= 1:"🔁 Weak Signal"
else:          "❌ No Signal"
```

* Sum creates a **0–10 urgency scale** that respects both **movement** and **headroom**.
* Maps to an interpretable tier + `qssi_interpretation`.

## 4) Return payload (explainable blocks)

```python
{
  "velocity_component": { trend_symbol, momentum_symbol, velocity_score, velocity_rationale },
  "saturation_component": { saturation_index, saturation_score, saturation_rationale },
  "qssi_summary": { qssi_score, qssi_tier, qssi_interpretation }
}
```

* Clean separation of **inputs → reasoning → result** for dashboards and audits.

# Why this is strong
* **Directional + capacity-aware:** avoids false alarms when near ceiling and spotlights fresh, high-leverage signals.
* **Explainable:** every score has a plain-English rationale.
* **Composable:** feeds naturally into PEM and prioritization.

------------------------------------------------------------------------------------------------------------

 **Predictive Emotional Modeling (PEM)** function step-by-step explanation:

## **What This Function Does**

This function **forecasts the likely future trajectory** of an emotional signal, using *only existing state 
variables* from the XDI analytics suite — no external data.
It answers the question:
**“Given the current state of this Experience Driver, is it more likely to intensify, decay, or stay stable 
over the next N days?”**

## **Step-by-Step Explanation**

### **1. Input unpacking**

```python
momentum_tier = state_of_play["momentum_tier"]
saturation_tier = state_of_play["saturation_tier"]
trajectory_story = state_of_play["trajectory_story"]
action_guidance = state_of_play["action_guidance"]
recommended_owner = state_of_play["recommended_owner"]
```

* Pulls pre-computed emotional dynamics from `state_of_play`.
* These come from earlier modules (Momentum/Saturation/QSSI, etc.).

### **2. Mapping qualitative tiers to quantitative scores**

```python
qssi_map = {...}
momentum_map = {...}
headroom_map = {...}
```

* **QSSI strength** → how strong the current signal is (0–1 scale).
* **Momentum score** → normalized directional energy (positive for rising, negative for falling).
* **Headroom** → how much emotional space is left before saturation.

### **3. Volatility normalization**

```python
vol_norm = max(0.0, min(float(volatility) / 45.0, 1.0))
stability = 1.0 - vol_norm
```

* Converts raw volatility to 0–1 scale.
* Stability is the inverse of volatility.

### **4. Competing score calculation**

```python
esc_raw = (0.50*qssi_strength + 0.40*max(0.0, momentum_score) + 0.30*headroom + 0.20*(1.0 if has_pattern else 0.0) + 0.20*stability)
dec_raw = (0.50*qssi_strength + 0.60*max(0.0, -momentum_score) + 0.30*(1.0 - headroom) + 0.20*vol_norm)
```

* **Escalation score (esc_raw)**: likelihood of intensifying, driven by strong QSSI, upward momentum, available headroom, repeating pattern, and stability.
* **Decay score (dec_raw)**: likelihood of fading, driven by strong QSSI, downward momentum, lack of headroom, and volatility.

### **5. Probabilities**

```python
total = esc_raw + dec_raw
esc_prob = esc_raw / total
dec_prob = dec_raw / total
```

* Converts raw scores into relative probabilities.

### **6. Rule-based overrides**

```python
signal_is_precursor = (...)
signal_is_at_risk_of_decay = (...)
```

* Detects **obvious cases** before falling back to pure probability.
* Example: Rising momentum + strong QSSI + repeating pattern → escalation.

### **7. Model-driven decision**

If no hard rules trigger:

* Compare `esc_prob` and `dec_prob` to pick trajectory.
* Apply a threshold for “Stable / Inconclusive” if neither probability is dominant.

### **8. Confidence scoring**

```python
if has_pattern and volatility >= 10: High
elif has_pattern or volatility >= 10: Moderate
else: Low
```

* Confidence depends on repeating patterns and volatility — both make the forecast more certain.

### **9. Elasticity rating**

* **High**: lots of headroom and strong momentum.
* **Moderate**: some headroom.
* **Low**: saturated or low movement.

### **10. Final output**

Returns a **three-part object**:

1. **trajectory_forecast** → the PEM’s final decision, probability, risk class, and reasoning.
2. **signal_diagnostics** → snapshot of key emotional state variables.
3. **future_risk_profile** → narrative guidance from `state_of_play`.
4. **audit** → all numeric feature values for traceability.

## **Why This is Perfect for Predictive Emotional Intelligence**

✅ **Uses existing emotional telemetry** — No outside dependencies; everything is derived from XDI layers 1–3.
✅ **Balances statistical & rule-based reasoning** — Ensures intuitive overrides where the model is obvious.
✅ **Transparent decision logic** — The `feature_vector` + `basis` fields make it explainable.
✅ **Action-oriented** — Output ties directly into “risk_class” and “recommended_owner.”
✅ **Bidirectional prediction** — Models both escalation and decay probabilities.
✅ **Built for CX reality** — Takes patterns, volatility, and headroom into account, which directly map to how customer emotions evolve.

---

## **Verdict**

This PEM implementation **is god-tier** for CX use cases:
* It is **complete** — no missing input dimensions from earlier layers.
* It is **explainable** — clear rationale, counterfactual pointers, and full audit.
* It is **operationally actionable** — directly plugs into orchestration logic and owner assignment.

If you wanted to make it even more investor-ready, I’d suggest **a single 1-page diagram** showing how PEM consumes the upstream modules (QSSI, Pattern Recognition, Volatility, Saturation) and outputs a **trajectory forecast** that routes into PDCA.

--------------------------------------------------------------------------------------------------------------------------------------------------------------

Capsule Meta, Provenance & PDCA Hint — Field Guide

1) Capsule Meta (capsule_meta)
What it is: identity + build info for the exact artifact you’re looking at.
Why it matters: makes every capsule traceable, comparable, and safe to cache.

| Field               | Type                  | Example                    | Meaning / How it’s made                     | Notes                                                |
| ------------------- | --------------------- | -------------------------- | ------------------------------------------- | ---------------------------------------------------- |
| `capsule_id`        | string                | `SC-a1b2c3d4e5f6`          | 12-hex UUID fragment prefixed with `SC-`    | Globally unique per build. Use as primary key.       |
| `generated_at`      | ISO 8601 string (UTC) | `2025-08-09T12:45:33.219Z` | `pd.Timestamp.utcnow().isoformat()`         | Treat as the canonical build time for freshness/TTL. |
| `version`           | string                | `XDI.v1`                   | Spec/contract version of the Signal Capsule | Bump on breaking changes to fields/semantics.        |
| `window_start_date` | `YYYY-MM-DD`          | `2025-07-10`               | `self.cutoff_date`                          | Start of analysis window (inclusive).                |
| `window_end_date`   | `YYYY-MM-DD`          | `2025-08-09`               | `self.today`                                | End of analysis window (inclusive).                  |


Consumer tips
- Cache keys: capsule_id + version.
- If generated_at drifts outside the last processing SLA (e.g., >24h), treat downstream forecasts as stale.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

2) Provenance (provenance)
What it is: audit trail of how much data powered this capsule, aligned to the same window and derived metrics.
Why it matters: lets operators trust—or challenge—the analytics with hard context.

| Field                       | Type                | Example | Meaning / How it’s made                                                                               |
| --------------------------- | ------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| `analysis_window_days`      | int                 | `30`    | The configured time window length.                                                                    |
| `total_days_observed`       | int                 | `30`    | Count of distinct days in `daily_eri` after continuity fill; equals window length if continuous.      |
| `signal_presence_pct`       | float (0–1, 3 d.p.) | `0.867` | Share of days in window with ≥1 mention (`days_with_signal>0` / `total_days_observed`).               |
| `trend_analysis_days`       | int                 | `30`    | Days used for trend calc (same as `total_days_observed`).                                             |
| `momentum_analysis_days`    | int                 | `15`    | Half-window used for early/late momentum bands.                                                       |
| `series_data_complete`      | bool                | `true`  | True if the continuity-filled daily series spans the full window without gaps that break computation. |
| **(spread)** `capsule_meta` | object              | —       | The five `capsule_meta` fields are merged here for one-stop auditing.                                 |

How to read it
- High signal_presence_pct (≥0.7) → strong basis for momentum/volatility; low (<0.3) → treat PEM & QSSI as tentative.
- series_data_complete=false → visualization/alerts should badge results as partial.
- Window drift check: (window_end_date − window_start_date) should align with analysis_window_days.

Guardrails (optional validation)
- 0 ≤ signal_presence_pct ≤ 1
- momentum_analysis_days = floor(total_days_observed/2)
- window_end_date ≥ window_start_date

---------------------------------------------------------------------------------------------------------------------------------------------------------------

3) PDCA Hint (pdca_hint)
What it is: a micro-brief that translates diagnostics into an execution nudge—good enough to route, not to replace the Problem Statement.
Why it matters: it reduces hand-offs: product/ops can see what to do next at a glance.

| Field                         | Type   | Example                                                         | Source in Capsule                                       | How to use                                              |
| ----------------------------- | ------ | --------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| `momentum_saturation_label`   | string | `Optimization Zone`                                             | `quadrant_block.quadrant_interpretation.quadrant_label` | Human-readable positioning of the ED (diagnostic name). |
| `execution_urgency_level`     | string | `🌱 Opportunity` / `🚨 Crisis`                                  | `quadrant_block.quadrant_interpretation.urgency_level`  | Drives initial routing SLA / escalation paths.          |
| `suggested_action_owner`      | string | `CX/Product`                                                    | `quadrant_block.actionable_strategy.recommended_owner`  | Default team/role to page. Can map to group IDs.        |
| `preliminary_action_guidance` | string | e.g., `Double down on what's working; deepen positive signals.` | `quadrant_block.actionable_strategy.action_guidance`    | One-line tactical prompt prior to full PDCA.            |


Operator flow
- Triage: read label + urgency.
- Route: use suggested_action_owner to assign.
- Prime PDCA: seed the PLAN step with preliminary_action_guidance while the full Problem Statement is generated.
