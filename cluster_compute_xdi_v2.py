from __future__ import annotations
    
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import pairwise_distances
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# from rfi_policy import RFIPolicy, RFIPolicyConfig
    
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
from statsmodels.tsa.stattools import acf

# --- helpers / imports you need somewhere top of file ---
import ast

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
                "opportunity_stream",              # 2x weight - STRATEGIC STREAM
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

    # ---------- cached model helper ----------
    def _ensure_st_model(self):
        if getattr(self, "_st_model", None) is None:
            model_name = getattr(self, "OU_CFG", {}).get("embedding_model", "all-MiniLM-L6-v2")
            self._st_model = SentenceTransformer(model_name)
        return self._st_model

    # ---------- emotion + stream distributions ----------
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

    def apply_stream_threshold_and_distribution(self, stream_counts: pd.Series, *, threshold: float = 0.8):
        total = stream_counts.sum()
        if total == 0:
            return [], {}
        dominant, cumulative = [], 0.0
        stream_distribution = {}
        for stream, count in stream_counts.items():
            pct = round(count / total, 4)
            stream_distribution[stream] = pct
            if cumulative >= threshold:
                continue
            dominant.append(stream)
            cumulative += pct
        return dominant, stream_distribution

    # --- simple mode helper used by extract_batch_1_fields ---
    def _most(self, series_or_values) -> str | None:
        """
        Return the most frequent non-empty string.
        Works with a pandas Series or any iterable.
        """
        try:
            vals = series_or_values.dropna().astype(str).str.strip().tolist()
        except AttributeError:
            vals = [str(v).strip() for v in series_or_values if pd.notna(v)]
        vals = [v for v in vals if v]
        if not vals:
            return None
        counts = Counter(vals)
        return counts.most_common(1)[0][0]

    # --- weighted signature builder used by cluster_behavior ---
    def _build_signature(self, row: pd.Series) -> str:
        """
        6x semantic_action_statement | 4x matters | 3x experience_driver |
        2x opportunity_stream | 2x context | 1x customer_journey_stage
        """
        def to_text(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            if isinstance(v, (list, tuple, set)):
                return " ".join(map(str, v))
            if isinstance(v, dict):
                return " ".join(f"{k}:{v}" for k, v in v.items())
            return str(v)

        def norm(s: str) -> str:
            return " ".join(str(s).lower().strip().split())

        fields = [
            ("semantic_action_statement", 6),
            ("matters", 4),
            ("experience_driver", 3),
            ("opportunity_stream", 2),   # only this exact column
            ("context", 2),
            ("customer_journey_stage", 1),
        ]

        parts: list[str] = []
        for key, w in fields:
            text = norm(to_text(row.get(key)))
            if text:
                parts.extend([text] * w)

        return " | ".join(parts)

    # ---------- semantic labelers ----------
    def _distill_matters_label(self, matters_list: List[str]) -> str:
        if not matters_list:
            return "No matters label available"
        model = self._ensure_st_model()
        embs = model.encode(matters_list, convert_to_tensor=True, normalize_embeddings=True)
        centroid = embs.mean(dim=0, keepdim=True)
        sims = util.cos_sim(embs, centroid).squeeze(0)
        best_idx = int(sims.argmax().item())
        return matters_list[best_idx] or "No matters label available"

    def extract_batch_1_fields(self, cluster_df: pd.DataFrame) -> Dict[str, Any]:
        composite = {}
        composite["experience_driver"] = cluster_df["experience_driver"].iloc[0]
        composite["emotion"] = cluster_df["emotion"].iloc[0]
        composite["opportunity_stream"] = cluster_df["opportunity_stream"].iloc[0]
        composite["feedback_type"] = self._most(cluster_df["feedback_type"]) or "Unknown"
        composite["theme"] = self._most(cluster_df["theme"]) or "Unknown"
        return composite

    def process_batch_2_fields(self, cluster_df: pd.DataFrame, batch_1_fields: Dict[str, Any]) -> Dict[str, Any]:
        batch_2 = {}
        batch_2['context'] = self._semantic_centroid_fusion(cluster_df['context'].tolist())
        batch_2['keywords'] = self._dedupe_and_merge_keywords(cluster_df['keywords'].tolist())
        batch_2['interaction_moment'] = self._semantic_mode(cluster_df['interaction_moment'].tolist())
        batch_2['customer_journey_stage'] = self._semantic_mode(cluster_df['customer_journey_stage'].tolist())
        batch_2['customer_journey'] = self._semantic_mode(cluster_df['customer_journey'].tolist())
        batch_2['customer_effort_score'] = self._weighted_average_effort_score(cluster_df['customer_effort_score'].tolist())
        batch_2['entity_name'] = self._extract_entity_name(cluster_df['entity_name'].tolist())
        return batch_2

    def _semantic_centroid_fusion(self, context_list: List[str]) -> str:
        if not context_list:
            return "No context available"
        clean = [str(c).strip() for c in context_list if pd.notna(c) and str(c).strip()]
        if not clean:
            return "No context available"
        counts = Counter(clean)
        if len(counts) <= 3:
            return counts.most_common(1)[0][0]
        model = self._ensure_st_model()
        embs = model.encode(clean, normalize_embeddings=True)
        centroid = embs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(embs, centroid).ravel()
        return clean[int(np.argmax(sims))]

    def _dedupe_and_merge_keywords(self, keywords_list: List[Any]) -> List[str]:
        all_kw = []
        for kw in keywords_list:
            if pd.isna(kw):
                continue
            if isinstance(kw, list):
                all_kw.extend([str(k).strip().lower() for k in kw])
            elif isinstance(kw, str):
                s = kw.strip()
                if s.startswith('[') and s.endswith(']'):
                    s = s[1:-1].replace("'", "").replace('"', '')
                all_kw.extend([k.strip().lower() for k in s.split(',') if k.strip()])
            else:
                all_kw.append(str(kw).strip().lower())
        uniq = list({k for k in all_kw if k and len(k) > 1})
        # optional: keep by frequency—requires counting original list; simple alpha sort:
        return sorted(uniq)

    def _semantic_mode(self, values_list: List[str]) -> str:
        if not values_list:
            return "Unknown"
        clean = [str(v).strip() for v in values_list if pd.notna(v) and str(v).strip()]
        if not clean:
            return "Unknown"
        counts = Counter(clean)
        if len(counts) == 1:
            return next(iter(counts))
        mc = counts.most_common()
        if mc[0][1] > mc[1][1]:
            return mc[0][0]
        tied = [v for v, c in mc if c == mc[0][1]]
        if len(tied) > 1:
            model = self._ensure_st_model()
            embs = model.encode(tied, normalize_embeddings=True)
            centroid = embs.mean(axis=0, keepdims=True)
            sims = cosine_similarity(embs, centroid).ravel()
            return tied[int(np.argmax(sims))]
        return tied[0]

    def _weighted_average_effort_score(self, scores_list: List[Any]) -> int:
        import statistics
        if not scores_list:
            return 4
        clean = []
        for s in scores_list:
            if pd.notna(s):
                try:
                    clean.append(float(s))
                except (ValueError, TypeError):
                    pass
        if not clean:
            return 4
        return int(round(statistics.mean(clean)))

    def _extract_entity_name(self, entity_names: List[str]) -> str:
        if not entity_names:
            return "Unknown Entity"
        clean = [str(n).strip() for n in entity_names if pd.notna(n) and str(n).strip()]
        if not clean:
            return "Unknown Entity"
        return Counter(clean).most_common(1)[0][0]

    # ---------- clustering ----------
    def cluster_behavior(self, df: pd.DataFrame, driver: str, emotion: str, stream: str):
        from uuid import uuid4
        df = df.copy()
        df["signature"] = df.apply(self._build_signature, axis=1).astype(str).str.lower()
        df = df.reset_index(drop=True)

        model = self._ensure_st_model()
        embeds = model.encode(df["signature"].tolist(), normalize_embeddings=True, show_progress_bar=False)
        total_rows = len(df)

        if total_rows < 2:
            labels = np.array([0] * total_rows)
        elif total_rows < 200:
            # sklearn version-safe cosine
            try:
                labels = AgglomerativeClustering(
                    metric="cosine", linkage="average",
                    distance_threshold=0.25, n_clusters=None
                ).fit_predict(embeds)
            except TypeError:
                dist = pairwise_distances(embeds, metric="cosine")
                labels = AgglomerativeClustering(
                    affinity="precomputed", linkage="average",
                    distance_threshold=0.25, n_clusters=None
                ).fit_predict(dist)
        else:
            labels = hdbscan.HDBSCAN(
                metric="cosine",
                min_cluster_size=self.OU_CFG["min_cluster_size"],
                min_samples=2
            ).fit_predict(embeds)

        df["local_bcs_id"] = labels.astype(str)

        # fix HDBSCAN noise to unique singletons
        noise_mask = df["local_bcs_id"] == "-1"
        if noise_mask.any():
            df.loc[noise_mask, "local_bcs_id"] = [str(uuid4()) for _ in range(int(noise_mask.sum()))]

        prefix = f"{driver[:8]}_{emotion[:3]}_{stream[:3]}".lower()
        cluster_store, full_composites, cluster_metadata = {}, {}, {}

        df["bcs_group_id"] = None
        df["bcs_id"] = None
        df["bcs_share"] = None
        df["bcs_label"] = None
        df["cluster_cohesion"] = None
        df["cluster_theme_preview"] = None

        for local_cid, grp in df.groupby("local_bcs_id"):
            unique_part = uuid4().hex[:8]
            group_id = f"{prefix}_{unique_part}"
            row_ids = [uuid4().hex for _ in range(len(grp))]

            grp = grp.copy()
            grp["bcs_group_id"] = group_id
            grp["bcs_id"] = row_ids
            grp["bcs_share"] = len(grp) / total_rows

            vecs = embeds[grp.index]
            if len(vecs) <= 1:
                cohesion = 1.0
            else:
                centroid = vecs.mean(axis=0, keepdims=True)
                sims = cosine_similarity(vecs, centroid).ravel()
                cohesion = float(sims.mean())

            matters_list = grp.get("matters", pd.Series(dtype=object)).dropna().astype(str).tolist()
            raw_label = self._distill_matters_label(matters_list) if len(matters_list) else f"Cluster {group_id}"
            preview = (raw_label or "No preview available").strip().capitalize()
            truncated_preview = (preview[:77] + "…") if len(preview) > 80 else preview

            # write preview to rows now (prevents KeyError later)
            grp["cluster_theme_preview"] = truncated_preview

            # unify customer_review (multi-row: centroid)
            if "customer_review" in grp.columns:
                customer_review_value = self._semantic_centroid_fusion(grp["customer_review"].astype(str).tolist())
            elif "comment_review" in grp.columns:
                customer_review_value = self._semantic_centroid_fusion(grp["comment_review"].astype(str).tolist())
            else:
                customer_review_value = None

            first_row = grp.iloc[0]
            if len(grp) == 1:
                composite = {
                    "bcs_id": first_row["bcs_id"],
                    "bcs_group_id": group_id,
                    "cluster_size": 1,
                    "bcs_share": round(1 / total_rows, 4),
                    "cluster_cohesion": 1.0,
                    "cluster_theme_preview": truncated_preview,
                    "customer_review": first_row.get("customer_review") or first_row.get("comment_review"),
                    **{k: first_row.get(k) for k in [
                        "experience_driver", "emotion", "opportunity_stream", "feedback_type",
                        "customer_journey", "customer_journey_stage", "interaction_moment",
                        "context", "keywords", "entity_name", "theme", "customer_effort_score",
                        "semantic_action_statement", "stream_justification", "matters", "behavioral_impact"
                    ]}
                }
            else:
                batch_1 = self.extract_batch_1_fields(grp)
                batch_2 = self.process_batch_2_fields(grp, batch_1)
                composite = {
                    **batch_1, **batch_2,
                    "bcs_id": first_row["bcs_id"],
                    "bcs_group_id": group_id,
                    "cluster_size": len(grp),
                    "bcs_share": round(len(grp) / total_rows, 4),
                    "cluster_cohesion": round(cohesion, 4),
                    "cluster_theme_preview": truncated_preview,
                    "customer_review": customer_review_value,
                    "semantic_action_statement": None,
                    "stream_justification": None,
                    "matters": None,
                    "behavioral_impact": None,
                }

            # store
            df.update(grp)
            cluster_store[group_id] = grp
            full_composites[group_id] = composite
            cluster_metadata[group_id] = {"label": truncated_preview, "cohesion": cohesion}

        # final annotation
        df["bcs_label"] = df["bcs_group_id"].map(lambda gid: cluster_metadata.get(gid, {}).get("label"))
        df["cluster_cohesion"] = df["bcs_group_id"].map(lambda gid: cluster_metadata.get(gid, {}).get("cohesion"))
        df["cluster_theme_preview"] = df["cluster_theme_preview"].fillna(
            df["bcs_group_id"].map(lambda gid: full_composites.get(gid, {}).get("cluster_theme_preview"))
        )

        # dominant cluster filter
        dominant_ids, cumulative_share = [], 0.0
        cluster_order = df["bcs_group_id"].value_counts(normalize=True)
        for cid, share in cluster_order.items():
            dominant_ids.append(cid)
            cumulative_share += share
            if cumulative_share >= self.OU_CFG["bcs_cumu_threshold"]:
                break
        filtered_df = df[df["bcs_group_id"].isin(dominant_ids)].copy()
        return filtered_df, list(full_composites.values()), cluster_store, df, full_composites

    # ---------- DB ----------
    def create_cluster_database(self, df: pd.DataFrame, full_composites: Dict[str, Dict[str, Any]],
                                cluster_store: Dict[str, pd.DataFrame], db_path: str = "clusters.db"):
        self.db_path = db_path
        self.init_database()
        for cid, grp in cluster_store.items():
            composite = full_composites.get(cid)
            if composite is None:
                print(f"⚠️ Skipping cluster {cid}: no composite data found.")
                continue
            grp = grp.copy()
            grp["bcs_group_id"] = cid
            if len(grp) == 1:
                self.save_single_cluster_to_db(grp, composite, cid)
            else:
                self.save_multi_cluster_to_db(grp, composite, cid)
        print("✅ All clusters saved to database.")

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('DROP TABLE IF EXISTS clusters')
        cur.execute('''
            CREATE TABLE clusters (
                bcs_id TEXT PRIMARY KEY,
                bcs_group_id TEXT,
                cluster_size INTEGER,
                bcs_share REAL,
                cluster_cohesion REAL,
                cluster_theme_preview TEXT,
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
                semantic_action_statement TEXT,
                stream_justification TEXT,
                matters TEXT,
                behavioral_impact TEXT,
                problem_statement TEXT
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_bcs_group_id ON clusters (bcs_group_id)')
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")

    def save_single_cluster_to_db(self, grp: pd.DataFrame, composite: Dict[str, Any], cid: str):
        conn = sqlite3.connect(self.db_path); cur = conn.cursor()
        row = grp.iloc[0]
        bcs_id = row.get("bcs_id"); bcs_group_id = row.get("bcs_group_id")
        # keywords
        kw = row.get('keywords')
        keywords_str = ", ".join(map(str, kw)) if isinstance(kw, list) else (str(kw) if kw is not None else "")
        cur.execute('''
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
            composite.get('customer_review') or row.get('customer_review') or row.get('comment_review'),
            row.get('experience_driver'), row.get('emotion'), row.get('theme'),
            row.get('opportunity_stream'), row.get('feedback_type'),
            row.get('customer_journey'), row.get('customer_journey_stage'),
            row.get('interaction_moment'), row.get('context'),
            keywords_str,
            row.get('entity_name'),
            float((row.get('customer_effort_score', 0.0) or composite.get('customer_effort_score', 0.0) or 0.0)),
            row.get('semantic_action_statement'), row.get('stream_justification'),
            row.get('matters'), row.get('behavioral_impact'),
            row.get('problem_statement') or ''
        ))
        conn.commit(); conn.close()
        print(f"💾 Saved single-row cluster {cid}")

    def save_multi_cluster_to_db(self, grp: pd.DataFrame, composite: Dict[str, Any], cid: str):
        conn = sqlite3.connect(self.db_path); cur = conn.cursor()
        # replicate composite fields
        kw = composite.get('keywords')
        keywords_str = ", ".join(map(str, kw)) if isinstance(kw, list) else (str(kw) if kw is not None else "")
        replicated = {
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
            'customer_effort_score': float(composite.get('customer_effort_score', 0.0) or 0.0),
            'customer_review': composite.get('customer_review')
        }
        for _, row in grp.iterrows():
            bcs_id = row.get("bcs_id"); bcs_group_id = row.get("bcs_group_id")
            cur.execute('''
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
                replicated['customer_review'] or row.get('customer_review') or row.get('comment_review'),
                replicated['experience_driver'], replicated['emotion'], replicated['theme'],
                replicated['opportunity_stream'], replicated['feedback_type'],
                replicated['customer_journey'], replicated['customer_journey_stage'], replicated['interaction_moment'],
                replicated['context'], replicated['keywords'], replicated['entity_name'],
                replicated['customer_effort_score'],
                row.get('semantic_action_statement'), row.get('stream_justification'),
                row.get('matters'), row.get('behavioral_impact'),
                row.get('problem_statement') or ''
            ))
        conn.commit(); conn.close()
        print(f"💾 Saved multi-row cluster {cid} with {len(grp)} rows")

    # ---------- theme distribution ----------
    def _calculate_cluster_theme_distribution(self, clust_df: pd.DataFrame) -> Dict[str, float]:
        if clust_df.empty:
            return {}
        col = (
            "cluster_theme_preview" if "cluster_theme_preview" in clust_df.columns
            else ("bcs_label" if "bcs_label" in clust_df.columns else None)
        )
        if not col:
            return {}
        counts = clust_df[col].value_counts()
        total = float(counts.sum() or 1)
        return {str(theme): round((cnt / total) * 100, 1) for theme, cnt in counts.items()}

    # ---------- snapshot ----------
    def compute_granular_details_snapshot(self, raw_df: pd.DataFrame, layer3_df: pd.DataFrame) -> pd.DataFrame:
        if layer3_df is None:
            raise ValueError("Layer 3 diagnostics must be supplied.")
        all_df_chunks, all_full_composites, all_cluster_store = [], {}, {}
        records: List[Dict[str, Any]] = []

        for _, hdr in layer3_df.iterrows():
            driver = hdr["experience_driver"]
            emotion_focus = hdr.get("emotional_audit_focus", [])
            if isinstance(emotion_focus, str):
                try:
                    emotion_focus = ast.literal_eval(emotion_focus)
                except (ValueError, SyntaxError):
                    emotion_focus = []
            emotion_dist = hdr.get("emotion_distribution", {}) or {}

            driver_rows = raw_df[raw_df["experience_driver"] == driver]
            for emotion in emotion_focus:
                emotion_rows = driver_rows[driver_rows["emotion_primary"].astype(str).str.lower() == str(emotion).lower()]
                if emotion_rows.empty:
                    continue

                stream_counts = emotion_rows["opportunity_stream"].value_counts()
                dominant_streams, stream_distribution = self.apply_stream_threshold_and_distribution(
                    stream_counts, threshold=self.OU_CFG["stream_threshold"]
                )
                if not dominant_streams:
                    continue

                for stream in dominant_streams:
                    stream_rows = emotion_rows[emotion_rows["opportunity_stream"] == stream]
                    if stream_rows.empty:
                        continue

                    clust_df, full_distribution, cluster_store, df_chunk, full_composites = self.cluster_behavior(
                        stream_rows, driver=driver, emotion=emotion, stream=stream
                    )
                    cluster_theme_distribution = self._calculate_cluster_theme_distribution(clust_df)

                    all_df_chunks.append(df_chunk)
                    all_cluster_store.update(cluster_store)
                    all_full_composites.update(full_composites)

                    for gid, grp in clust_df.groupby("bcs_group_id"):
                        meta = full_composites.get(gid, {})
                        composite = {
                            **meta,
                            "emotion_distribution": emotion_dist,
                            "stream_distribution": stream_distribution,
                            "cluster_theme_distribution": cluster_theme_distribution,
                        }
                        records.append(composite)

        print(f"\n📦 FINAL DEBUG SUMMARY")
        print(f"🔢 Total full_composites: {len(all_full_composites)}")
        print(f"🔢 Total cluster_store: {len(all_cluster_store)}")
        missing = [cid for cid in all_cluster_store if cid not in all_full_composites]
        print(f"❌ Missing composites for: {missing}")

        merged_df = pd.concat(all_df_chunks, ignore_index=True) if all_df_chunks else pd.DataFrame()
        if not merged_df.empty:
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
        self.cutoff_date = self.today - timedelta(days=self.timeframe_days - 1)
        
        # 1) window filter
        self.raw_df["date"] = pd.to_datetime(self.raw_df["date"], errors="coerce")
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
            if   delta_pct > 20.0 and snr >= 1.25: sym = "↑↑", label = "Strongly Rising", description = "Explosive upward movement" 
            elif delta_pct > 5.0  and snr >= 0.75: sym = "↑", label = "Moderately Rising", description = "Sustained growth"
            elif delta_pct < -20.0 and snr >= 1.25: sym = "↓↓", label = "Strongly Falling", description = "Sharp deterioration or regression"
            elif delta_pct < -5.0  and snr >= 0.75: sym = "↓", label = "Moderately Falling", description = "Beginning to cool or drop"
            else:                                   sym = "→", label = "Stable", description = "No major shift"
    
            return {"symbol": sym, "delta": delta_pct_r, "label": label, "description": description}
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
            "eri_by_day": None,
            "acf": None,
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
                        "eri_by_day": None,
                        "acf": {"lag_days": int(L), "score": round(score, 3)}
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

    # ===== Consistency Contracts (class methods) =====

    def saturation_contract(self):
        """
        Returns SAT_BINS:
        [(threshold_si, (emoji_tier, clean_tier), headroom_dict, qssi_score), ...]
        Ordered high→low thresholds.
        """
        return [
            (0.90, ("🏆 Very High","Very High"),
                {"tier":"🟢 Champion","eri_range":"+80 to +100",
                    "guidance":"Sustain / Amplify","interpretation":"Already won emotional trust"}, 0),
            (0.65, ("✅ High","High"),
                {"tier":"🟢 Loyal","eri_range":"+30 to +79",
                    "guidance":"Optimize","interpretation":"Still room for higher resonance"}, 1),
            (0.45, ("⚖️ Medium","Medium"),
                {"tier":"⚪ Neutral","eri_range":"-10 to +29",
                    "guidance":"Re-engage / Nudge","interpretation":"Flat or lightly positive"}, 2),
            (0.25, ("⚠️ Low","Low"),
                {"tier":"🟠 Vulnerable","eri_range":"-50 to -11",
                    "guidance":"Correct / Contain","interpretation":"Trust cracking"}, 3),
            (0.00, ("❌ Very Low","Very Low"),
                {"tier":"🔴 At-Risk","eri_range":"-100 to -51",
                    "guidance":"Crisis Repair","interpretation":"Loyalty collapsed"}, 4),
        ]

    def momentum_contract(self):
        """
        Momentum symbol → metadata.
        """
        return {
            "↑↑": {"label":"Strongly Rising","description":"Explosive upward movement","emoji_full":"↑↑ 🚀 Strongly Rising","strength":1.0},
            "↑":  {"label":"Moderately Rising","description":"Sustained growth","emoji_full":"↑ 📈 Moderately Rising","strength":0.6},
            "→":  {"label":"Stable","description":"No major shift","emoji_full":"→ ➖ Stable","strength":0.3},
            "↓":  {"label":"Moderately Falling","description":"Beginning to cool or drop","emoji_full":"↓ 📉 Moderately Falling","strength":0.6},
            "↓↓": {"label":"Strongly Falling","description":"Sharp deterioration or regression","emoji_full":"↓↓ 🧨 Strongly Falling","strength":1.0},
        }

    def sat_from_si(self, si):
        """
        Map saturation index in [0,1] → unified saturation info used across modules.
        Returns: {"emoji","clean","headroom","qssi_score","si"}
        """
        try:
            si = float(si)
        except Exception:
            si = 0.5
        si = max(0.0, min(1.0, si))
        for th, (emoji, clean), headroom, qssi_score in self.saturation_contract():
            if si >= th:
                return {"emoji": emoji, "clean": clean, "headroom": headroom,
                        "qssi_score": qssi_score, "si": si}
        # fallback (shouldn't hit)
        bins_ = self.saturation_contract()
        return {"emoji":"⚖️ Medium","clean":"Medium","headroom":bins_[2][2],"qssi_score":2,"si":si}

    def sat_from_eri(self, eri):
        """
        Convert ERI (−100..+100) → saturation index → same structure as sat_from_si.
        """
        try:
            si = (float(eri) + 100.0) / 200.0
        except Exception:
            si = 0.5
        return self.sat_from_si(si)

    def mom_details(self, symbol: str):
        """
        Get momentum metadata by symbol. Defaults to '→' if unknown.
        """
        mom = self.momentum_contract()
        return mom.get(symbol, mom["→"])

    def mom_symbol_from_label(self, label: str) -> str:
        """
        Reverse lookup: label → symbol (case-insensitive). Falls back to '→'.
        """
        label = (label or "").strip().lower()
        for sym, meta in self.momentum_contract().items():
            if meta["label"].lower() == label:
                return sym
        return "→"

    def saturation_tier_from_si(self, si: float) -> str:
        """
        Convenience: returns clean saturation tier name ('High','Medium',etc.).
        """
        return self.sat_from_si(si)["clean"]
   
    def _compute_momentum_saturation_insight(self, eri_score, momentum_symbol) -> dict:
        """
        Map (saturation from ERI, momentum symbol) → 25-cell quadrant insight,
        using the class contracts for consistency.
        """
        # 1) Canonical saturation via contract
        sat = self.sat_from_eri(eri_score)  # {'emoji','clean','headroom','qssi_score','si'}
        sat_emoji = sat["emoji"]            # e.g. "✅ High"
        sat_clean = sat["clean"]            # e.g. "High"
        si       = float(sat["si"])

        # 2) Canonical momentum via contract
        mom_meta = self.mom_details(momentum_symbol)   # safe default to '→'
        mom_emoji_full = mom_meta["emoji_full"]        # e.g. "↑ 📈 Moderately Rising"
        mom_clean      = mom_meta["label"]             # e.g. "Moderately Rising"
        mom_strength   = float(mom_meta["strength"])   # 1.0/0.6/0.3/...

        # 3) Look up 25-cell narrative
        qm  = self.quadrant_matrix
        hit = qm[(qm["Saturation_Tier"] == sat_emoji) & (qm["Momentum_Tier"] == mom_emoji_full)]

        # 4) Tier-aware confidence & borderline using contract bins
        # Build bounds from saturation_contract thresholds (ascending + 1.0 cap)
        bins = self.saturation_contract()  # [(th, (emoji,clean), headroom, qssi_score), ...] high→low
        bounds = sorted([th for th, *_ in bins] + [1.0])  # e.g. [0.0,0.25,0.45,0.65,0.90,1.0]
        lower = max(b for b in bounds if b <= si)
        upper = min(b for b in bounds if b >= si)
        half_width    = max((upper - lower) / 2.0, 1e-9)
        sat_distance  = min(si - lower, upper - si)
        borderline    = sat_distance < 0.02
        quadrant_conf = round(0.5 * mom_strength + 0.5 * min(1.0, sat_distance / half_width), 2)

        matrix_key = f"{sat_clean}|{mom_clean}"

        if not hit.empty:
            q = hit.iloc[0]
            payload = {
                "signal_classification": {
                    "saturation_index": round(si, 2),
                    "saturation_tier": sat_clean,
                    "loyalty_tier": sat["headroom"]["tier"],
                    "momentum_tier": mom_clean,
                    "combined_quadrant": f"{mom_clean} Momentum × {sat_clean} Saturation",
                },
                "headroom_alignment": sat["headroom"],
                "quadrant_interpretation": {
                    "quadrant_label": q["Diagnostic_Label"],
                    "urgency_level": q["Urgency_Code"],
                    "interpretation": q["Strategic_Narrative"],
                },
                "tactical_insight": {
                    "emotional_pulse": f"{q['Diagnostic_Label']}: {q['Momentum_Context']}; {q['Saturation_Context']}",
                    "battle_status": f"{q['Urgency_Code']} · {q['Diagnostic_Label']}",
                    "strategic_reality": f"{q['Trajectory_Story']}. Action: {q['Action_Guidance']}",
                },
                "actionable_strategy": {
                    "momentum_context": q["Momentum_Context"],
                    "saturation_context": q["Saturation_Context"],
                    "trajectory_story": q["Trajectory_Story"],
                    "action_guidance": q["Action_Guidance"],
                    "recommended_owner": q["Recommended_Owner"],
                    "headroom_guidance": sat["headroom"]["guidance"],
                },
            }
        else:
            # Keep headroom info even if the exact cell wasn’t found
            payload = {
                "signal_classification": {
                    "saturation_index": round(si, 2),
                    "saturation_tier": sat_clean,
                    "loyalty_tier": sat["headroom"]["tier"],
                    "momentum_tier": mom_clean,
                    "combined_quadrant": "Unknown",
                },
                "headroom_alignment": sat["headroom"],
                "quadrant_interpretation": {
                    "quadrant_label": "Unknown",
                    "urgency_level": "Unknown",
                    "interpretation": "No data available for this quadrant",
                },
                "tactical_insight": {
                    "emotional_pulse": "Unknown — No data available for emotional assessment",
                    "battle_status": "⚠️ Risk · Unknown",
                    "strategic_reality": "No trajectory data available — system recommends data validation",
                },
                "actionable_strategy": {
                    "momentum_context": "Unknown momentum pattern",
                    "saturation_context": "Unknown saturation level",
                    "trajectory_story": "No trajectory data available",
                    "action_guidance": "Validate data sources and retry analysis",
                    "recommended_owner": "Data Team",
                    "headroom_guidance": sat["headroom"]["guidance"],
                },
            }

        payload["meta"] = {
            "matrix_key": matrix_key,
            "saturation_tier_emoji": sat_emoji,
            "momentum_tier_emoji": mom_emoji_full,
            "quadrant_confidence": quadrant_conf,
            "borderline": borderline,
        }
        return payload

    # === Layer 4: Signal Strength (QSSI) — contract aligned ===
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
        try:
            si = float(saturation_index)
        except Exception:
            si = 0.5
        si = max(0.0, min(1.0, si))  # clamp

        # --- canonical contracts ---
        sat = self.sat_from_si(si)          # {'emoji','clean','headroom','qssi_score','si'}
        mom = self.mom_details(sym_mom)     # {'label','description','emoji_full','strength'}

        # --- velocity score (symbol-only base) ---
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

        # --- micro nudges (only if provided) ---
        bump = 0
        if trend_pct is not None and abs(float(trend_pct)) >= 10.0:
            bump += 1
        if momentum_delta is not None and abs(float(momentum_delta)) >= 20.0:
            bump += 1

        # light penalties for short/noisy windows (only if provided)
        penalty = 0
        if n_days is not None and int(n_days) < 10:
            penalty += 1
        if vol_norm is not None and float(vol_norm) > 0.8:
            penalty += 1

        velocity_score = max(0, min(6, velocity_score + bump - penalty))

        # --- headroom → saturation score (via contract) ---
        # sat['qssi_score'] is 0..4 aligned to your bins (Champion..At-Risk reversed)
        saturation_score = int(sat["qssi_score"])
        sat_r_map = {
            4: "At-Risk — loyalty collapsed; crisis repair",
            3: "Vulnerable — trust cracking; correct/contain",
            2: "Neutral — flat/lightly positive; re-engage/nudge",
            1: "Loyal — still room for resonance; optimize",
            0: "Champion — emotional trust won; sustain/amplify",
        }
        sat_r = sat_r_map.get(saturation_score, "Neutral — flat/lightly positive; re-engage/nudge")

        # --- composite QSSI (0..10) ---
        qssi = int(velocity_score + saturation_score)
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
                **({"nudges": {"bump": int(bump), "penalty": int(penalty)}} if (bump or penalty) else {})
            },
            "saturation_component": {
                "saturation_index": round(sat["si"], 2),
                "saturation_score": int(saturation_score),
                "saturation_rationale": sat_r,
                "loyalty_tier": sat["headroom"]["tier"],
                "headroom_guidance": sat["headroom"]["guidance"],
            },
            "qssi_summary": {
                "qssi_score": int(qssi),
                "qssi_tier": qssi_tier,
                "qssi_interpretation": interp
            },
            "audit": {
                "headroom": sat["headroom"],
                "saturation_tier_emoji": sat["emoji"],
                "momentum_label": mom["label"],
                **({"n_days": int(n_days)} if n_days is not None else {}),
                **({"vol_norm": float(vol_norm)} if vol_norm is not None else {}),
            }
        }

   # === PEM (Predictive Emotional Modeling) — aligned to new modules ===
    def build_predictive_emotional_modeling(self, state_of_play, volatility, has_pattern, qssi_tier) -> dict:
        """
        Inputs:
        state_of_play = {
            "momentum_tier": <label from contract, e.g. "Moderately Rising">,
            "saturation_tier": <"Very High"|"High"|"Medium"|"Low"|"Very Low">,
            "trajectory_story": str,
            "action_guidance": str,
            "recommended_owner": str
        }
        volatility: float  # % of ERI band (0..100)
        has_pattern: bool
        qssi_tier: "💥 Critical Signal"|"🔥 Strong Signal"|"🌱 Emerging Signal"|"🔁 Weak Signal"|"❌ No Signal"
        """
        # --- unpack & sanitize ---
        mom_label = (state_of_play.get("momentum_tier") or "Stable").strip()
        sat_tier  = (state_of_play.get("saturation_tier") or "Medium").strip()
        traj      = state_of_play.get("trajectory_story")
        guidance  = state_of_play.get("action_guidance")
        owner     = state_of_play.get("recommended_owner")

        try:
            vol_pct = max(0.0, float(volatility))
        except Exception:
            vol_pct = 0.0
        vol_norm = min(vol_pct / 20.0, 1.0)   # ~20% of ERI band ≈ "high"
        stability = 1.0 - vol_norm

        qssi_map = {
            "💥 Critical Signal":1.00,
            "🔥 Strong Signal":  0.80,
            "🌱 Emerging Signal":0.50,
            "🔁 Weak Signal":    0.20,
            "❌ No Signal":      0.00,
        }
        qssi_strength = float(qssi_map.get(qssi_tier, 0.0))

        # --- momentum via contract ---
        mom_symbol = self.mom_symbol_from_label(mom_label)              # "↑↑","↑","→","↓","↓↓"
        mom_meta   = self.mom_details(mom_symbol)                        # label/description/etc.
        dir_score  = {"↑↑":1.0, "↑":0.6, "→":0.0, "↓":-0.6, "↓↓":-1.0}[mom_symbol]  # signed

        # --- headroom via sat tier (contract-aligned bins) ---
        headroom_map = {"Very High":0.00, "High":0.25, "Medium":0.50, "Low":0.75, "Very Low":1.00}
        headroom = float(headroom_map.get(sat_tier, 0.50))

        # --- directionalized raw scores (same structure as before) ---
        esc_raw = (
            0.35 * qssi_strength * max(0.0,  dir_score) +
            0.40 * max(0.0,  dir_score) +
            0.30 * headroom +
            0.20 * (1.0 if has_pattern else 0.0) +
            0.20 * stability
        )
        dec_raw = (
            0.35 * qssi_strength * max(0.0, -dir_score) +
            0.60 * max(0.0, -dir_score) +
            0.30 * (1.0 - headroom) +
            0.20 * vol_norm
        )

        esc_raw = max(0.0, esc_raw)
        dec_raw = max(0.0, dec_raw)
        total   = esc_raw + dec_raw
        esc_prob = (esc_raw / total) if total > 1e-9 else 0.0
        dec_prob = (dec_raw / total) if total > 1e-9 else 0.0

        # --- rules-first overrides (aligned with % thresholds) ---
        signal_is_precursor = (dir_score > 0 and qssi_tier in {"🔥 Strong Signal","💥 Critical Signal"} and bool(has_pattern))
        signal_is_at_risk   = (dir_score < 0 and vol_pct >= 5.0)

        if signal_is_precursor:
            pem_trajectory  = "Likely Escalation"
            pem_probability = max(esc_prob, 0.65)
            risk_class      = "Opportunity"
            explanation     = "Rising momentum + strong/critical signal + repeating pattern indicates intensification"
            rule_trigger, basis = "rising+strong_qssi+pattern", "esc_prob"
            counterfactual_pointer = "Flip if momentum ≤ 'Moderately Falling' or volatility > 22.5%"
        elif signal_is_at_risk:
            pem_trajectory  = "At Risk of Decay"
            pem_probability = max(dec_prob, 0.65)
            risk_class      = "Risk"
            explanation     = "Falling momentum + elevated volatility suggests emotional energy is fading or fracturing"
            rule_trigger, basis = "falling+volatility", "dec_prob"
            counterfactual_pointer = "Flip if momentum ≥ 'Moderately Rising' and volatility < 5%"
        else:
            if max(esc_prob, dec_prob) < 0.55:
                pem_trajectory  = "Stable / Inconclusive"
                pem_probability = max(esc_prob, dec_prob)
                risk_class      = "Neutral"
                explanation     = "No dominant predictive anchors; continue monitoring"
            else:
                if esc_prob > dec_prob:
                    pem_trajectory, pem_probability = "Likely Escalation", esc_prob
                    risk_class, explanation = "Opportunity", "Upward trajectory dominance (momentum/headroom/stability blend)"
                else:
                    pem_trajectory, pem_probability = "At Risk of Decay", dec_prob
                    risk_class, explanation = "Risk", "Downward trajectory dominance (momentum/volatility/ceiling pressure)"
            rule_trigger = "model_choice"
            basis = "esc_prob" if esc_prob >= dec_prob else "dec_prob"
            counterfactual_pointer = "Flip if next horizon favors the opposite momentum tier"

        # --- confidence tiers (uses pattern + volatility % of ERI band) ---
        if has_pattern and vol_pct >= 5.0:
            confidence_tier, confidence_score = "High", 0.85
        elif has_pattern or vol_pct >= 5.0:
            confidence_tier, confidence_score = "Moderate", 0.55
        else:
            confidence_tier, confidence_score = "Low", 0.30

        # --- elasticity (unchanged, but contract-aligned headroom & momentum magnitude) ---
        elasticity_rating = (
            "High"     if headroom >= 0.5 and abs(dir_score) >= 0.6 else
            "Moderate" if headroom >= 0.25 else
            "Low"
        )

        horizon_days = int(getattr(self, "pem_horizon_days", 14))
        version = "PEM.v1.3"

        feature_vector = {
            "qssi_strength": round(qssi_strength, 3),
            "momentum_symbol": mom_symbol,
            "momentum_score": round(dir_score, 3),   # signed
            "headroom": round(headroom, 3),
            "volatility_pct": round(vol_pct, 3),
            "volatility_norm": round(vol_norm, 3),
            "stability": round(stability, 3),
            "has_pattern": bool(has_pattern),
            "esc_raw": round(esc_raw, 3),
            "dec_raw": round(dec_raw, 3),
            "esc_prob": round(esc_prob, 3),
            "dec_prob": round(dec_prob, 3),
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
                "momentum_tier": mom_label,
                "saturation_tier": sat_tier,
                "qssi_tier": qssi_tier
            },
            "future_risk_profile": {
                "trajectory_story": traj,
                "action_guidance": guidance,
                "recommended_owner": owner
            },
            "audit": {
                "feature_vector": feature_vector,
                "elasticity_rating": elasticity_rating,
                # quick consistency hints
                "consistency": {
                    "momentum_direction": "up" if dir_score > 0 else "down" if dir_score < 0 else "flat",
                    "pattern_supports_escalation": bool(has_pattern and dir_score > 0),
                    "volatility_pressure": "high" if vol_norm >= 0.8 else "moderate" if vol_norm >= 0.4 else "low",
                }
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
                # daily_eri = (data.groupby("date").apply(self.compute_normalized_eri).sort_index())
                daily_eri = (
                            data.groupby(data["date"].dt.normalize())
                                .apply(self.compute_normalized_eri)
                                .sort_index()
                        )
                
                if len(daily_eri) > 1:
                    # full_idx = pd.date_range(start=daily_eri.index.min(), end=daily_eri.index.max(), freq="D")
                    # daily_eri = daily_eri.reindex(full_idx).bfill().ffill()
                    # USE YOUR ESTABLISHED WINDOW BOUNDARIES

                    full_idx = pd.date_range(start=self.cutoff_date, 
                                             end = self.today - pd.Timedelta(days=1), 
                                             freq="D")
                    daily_eri = daily_eri.reindex(full_idx).bfill().ffill()
                
                series_data_complete = not daily_eri.isna().any()
                missing_days_pct = float(daily_eri.isna().mean()) if not series_data_complete else 0.0
                
                
                if len(daily_eri) <= 1:
                    if self.verbose:
                        print(f"⏭️ DROP [{ed}] → insufficient days ({len(daily_eri)})")
                    skipped.append((ed, f"Insufficient days: {len(daily_eri)}"))
                    continue
    
                # provenance & horizons - use this everywhere for counts/percentages
                analysis_window_days = self.timeframe_days
                num_days_observed = analysis_window_days  # keep the name if you reference it later

                # signal presence %
                # ds = data.groupby("date").size()
                ds = data.groupby(data["date"].dt.normalize()).size()
                ds.index = pd.to_datetime(ds.index)
                days_with_signal = ds.reindex(full_idx, fill_value=0)
                pct_days_with_signal = round(float((days_with_signal > 0).sum()) / analysis_window_days, 3)

                overall_rel = ("Low" if pct_days_with_signal < 0.10
                else "Moderate" if pct_days_with_signal < 0.40
                else "High")

                analysis_reliability = {
                    "level": overall_rel,
                    "signal_presence_pct": round(pct_days_with_signal, 3),
                    # "basis": ["coverage"]  # optional: add "short_window", "high_volatility" if you want later
                }

                # modules
                trend = self._compute_trend_series(daily_eri)
                momentum = self._compute_momentum_series(daily_eri)
                volatility = self._compute_volatility_series(daily_eri)
                pattern_block = self._compute_pattern_recognition(
                    daily_eri, data, self.pattern_lags
                )
    
                # trend_symbol, trend_pct, trend_snr = trend["symbol"], trend["trend_pct"], trend["trend_snr"]
                
                trend_symbol  = trend.get("symbol", "→")
                trend_pct     = trend.get("trend_pct", 0.0)
                trend_snr_opt = trend.get("trend_snr")  # may be None
                
                trend_block = {
                    "trend_symbol": trend_symbol,
                    "trend_pct": round(float(trend_pct), 2),
                    **({"trend_snr": round(float(trend_snr_opt), 2)} if trend_snr_opt is not None else {}),
                    "trend_reliability": analysis_reliability,
                    "n_days": num_days_observed
                }

                momentum_symbol, momentum_label, momentum_description, momentum_delta = momentum["symbol"], momentum["label"], momentum["description"], momentum["delta"]
                momentum_block = {
                    "momentum_symbol": momentum_symbol,
                    "momentum_label": momentum_label,
                    "momentum_description": momentum_description,
                    "momentum_delta": momentum_delta,
                    "momentum_reliability": analysis_reliability,
                    "n_days": num_days_observed
                }
                             
                
                # volatility now in % of ERI band; keep adj if present
                volatility_tier = volatility["tier"]
                volatility_pct = float(volatility.get("score", 0.0))           # % of ERI band
                volatility_pct_adj = float(volatility.get("score_adj", volatility_pct))
                
                volatility_block = {"volatility_tier": volatility_tier,
                                    "volatility_pct": round(volatility_pct,2),
                                    "volatility_pct_adj": round(volatility_pct_adj, 2),
                                    "n_days": num_days_observed
                                    }
                
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
                    n_days= num_days_observed,
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
    
                
                # unified Layer-3 confidence (vol_norm uses new scaling)
                # pattern_conf_norm = (
                #     1.0 if pattern_block.get("pattern_confidence") == "Strong"
                #     else 0.6 if pattern_block.get("pattern_confidence") == "Weak"
                #     else 0.3
                # )

                pc = pattern_block.get("pattern_confidence")
                pattern_conf_norm = (
                    1.0 if pc == "Strong"
                    else 0.6 if pc == "Moderate"
                    else 0.3 if pc == "Weak"
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
                    "eri_by_day": pattern_block.get("eri_by_day") if has_weekly else None,
                    "acf": pattern_block.get("acf"),  
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
    
                    "trend_block": trend_block,
                    "momentum_block": momentum_block,
                    "volatility_block": volatility_block,
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
                        "series_data_complete": series_data_complete,
                        "missing_days_pct": missing_days_pct,
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
