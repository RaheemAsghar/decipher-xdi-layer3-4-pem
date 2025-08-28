from __future__ import annotations
    
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
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

       # 👑 Signature weighting for behavioral clustering (explicit weights)       
        self.SIGNATURE_FIELDS = [
            ("semantic_action_statement", 6),   # THE GOLDMINE
            ("matters", 4),                     # BEHAVIORAL ESSENCE
            ("experience_driver", 3),           # STRUCTURAL CONTEXT
            ("opportunity_stream", 2),          # STRATEGIC STREAM
            ("context", 2),                     # BEHAVIORAL BACKUP
            ("customer_journey_stage", 1),      # INTERACTION TIMING
        ]

        # 🧠 Clustering parameters
        self.OU_CFG = {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # fully-qualified name
            "min_cluster_size": 8,
            "bcs_cumu_threshold": 0.80,
            "stream_threshold": 0.80,
            "skip_singletons": False,
        }

        # ✅ Do NOT instantiate a second model here.
        # Warm the cached model once so everything else uses the same instance.
        self._ensure_st_model()

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
            model_name = self.OU_CFG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            self._st_model = SentenceTransformer(model_name)
        return self._st_model

    def _encode_np(self, texts):
        """
        Numpy, L2-normalized embeddings for deterministic cosine similarity everywhere.
        """
        model = self._ensure_st_model()
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    # ---------- emotion + stream distributions ----------
    def compute_emotional_focus(self):
        """Compute dominant 5-group emotions and their distribution per experience_driver 
        (80% threshold logic applied)."""
        
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
        Build the weighted signature from self.SIGNATURE_FIELDS.
        Each field contributes 'weight' repeats of its normalized text.
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

        parts: list[str] = []
        for key, weight in self.SIGNATURE_FIELDS:
            text = norm(to_text(row.get(key)))
            if text:
                parts.extend([text] * int(weight))

        return " | ".join(parts)

    # === EMBEDDINGS (numpy, normalized) ===
    def _encode_np(self, texts):
        model = self._ensure_st_model()
        # numpy + L2 normalized → stable cosine everywhere
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    # === Centroid pick for short fields (deterministic) ===
    from sklearn.metrics.pairwise import cosine_similarity
    import re

    def _centroid_pick(self, texts):
        clean = [t for t in (texts or []) if isinstance(t, str) and t.strip()]
        if not clean:
            return None
        E = self._encode_np(clean)
        c = E.mean(axis=0, keepdims=True)
        sims = cosine_similarity(E, c).ravel()
        return clean[int(sims.argmax())]

    # === MMR extractive summary for paragraph fields (deterministic) ===
    def _mmr_summary(self, texts, top_k=4, diversity=0.7):
        
        top_k = max(1, int(top_k))
        diversity = max(0.0, min(1.0, float(diversity)))

        splitter = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
        sents = []
        for t in (texts or []):
            if not isinstance(t, str):
                continue
            t = re.sub(r'\s+', ' ', t.strip())
            sents.extend([s for s in splitter.split(t) if len(s.strip()) >= 30])

        if not sents:
            # fallback: truncated concat
            raw = " ".join([(t or "").strip() for t in (texts or []) if isinstance(t, str)])[:300]
            return raw or None

        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words='english')
        X = vec.fit_transform(sents).astype(float)
        centroid = X.mean(axis=0)
        rel = cosine_similarity(
                        np.asarray(X.toarray() if hasattr(X, "toarray") else X),
                        np.asarray(centroid.toarray() if hasattr(centroid, "toarray") else centroid)
                    ).ravel()

        selected, cand = [], list(range(len(sents)))
        while cand and len(selected) < top_k:
            if not selected:
                i = int(np.argmax(rel)); selected.append(i); cand.remove(i); continue
            sims = cosine_similarity(X[cand], X[selected]).max(axis=1)
            scores = diversity * rel[cand] - (1 - diversity) * sims
            j = int(np.argmax(scores)); sel = cand[j]; selected.append(sel); cand.remove(sel)
        selected.sort()
        return " ".join(sents[i] for i in selected)

         
    # ---------- semantic labelers ----------
    def _distill_matters_label(self, matters_list: List[str]) -> str:
        """
        Deterministically pick the representative Matters line closest to the centroid
        (numpy embeddings + cosine). No tensors/LLM.
        """
        if not matters_list:
            return "No matters label available"
        clean = [str(m).strip() for m in matters_list if pd.notna(m) and str(m).strip()]
        if not clean:
            return "No matters label available"
        return self._centroid_pick(clean) or "No matters label available"

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
        """
        Deterministic consolidation: if few unique, use mode; otherwise centroid-pick (numpy cosine).
        """
        if not context_list:
            return "No context available"
        clean = [str(c).strip() for c in context_list if pd.notna(c) and str(c).strip()]
        if not clean:
            return "No context available"
        counts = Counter(clean)
        if len(counts) <= 3:
            return counts.most_common(1)[0][0]
        return self._centroid_pick(clean) or "No context available"

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
        # keep by frequency (readability); filter junk
        counts = Counter([k for k in all_kw if k and len(k) > 1])
        return [k for k,_ in counts.most_common()]

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
        # tie-break via centroid-pick (semantic)
        tied = [v for v, c in mc if c == mc[0][1]]
        pick = self._centroid_pick(tied)
        return pick or tied[0]

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
        df = df.copy()

        df["signature"] = df.apply(self._build_signature, axis=1).astype(str).str.lower()
        df = df.reset_index(drop=True)

        # numpy embeddings (normalized)
        embeds = self._encode_np(df["signature"].tolist())
        total_rows = len(df)

        # --- clustering ---------------------------------------------------------
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

        # HDBSCAN noise → unique singletons
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

            # Label/preview from Matters (fallbacks to SAS/Context)
            matters_list = grp.get("matters", pd.Series(dtype=object)).dropna().astype(str).tolist()
            raw_label = self._distill_matters_label(matters_list) if len(matters_list) else None
            if not raw_label:
                sas_list = grp.get("semantic_action_statement", pd.Series(dtype=object)).dropna().astype(str).tolist()
                raw_label = self._centroid_pick(sas_list)
            if not raw_label:
                ctx_list = grp.get("context", pd.Series(dtype=object)).dropna().astype(str).tolist()
                raw_label = self._centroid_pick(ctx_list) or f"Cluster {group_id}"

            preview = (raw_label or "No preview available").strip().capitalize()
            truncated_preview = (preview[:77] + "…") if len(preview) > 80 else preview
            grp["cluster_theme_preview"] = truncated_preview  # write to rows now

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
                # batch fields (deterministic)
                batch_1 = self.extract_batch_1_fields(grp)
                batch_2 = self.process_batch_2_fields(grp, batch_1)

                # the 4 fields — filled deterministically (MMR → centroid fallback)
                sas_list = grp.get("semantic_action_statement", pd.Series(dtype=object)).dropna().astype(str).tolist()
                just_list = grp.get("stream_justification", pd.Series(dtype=object)).dropna().astype(str).tolist()
                mat_list  = grp.get("matters", pd.Series(dtype=object)).dropna().astype(str).tolist()
                beh_list  = grp.get("behavioral_impact", pd.Series(dtype=object)).dropna().astype(str).tolist()

                composite = {
                    **batch_1, **batch_2,
                    "bcs_id": first_row["bcs_id"],
                    "bcs_group_id": group_id,
                    "cluster_size": len(grp),
                    "bcs_share": round(len(grp) / total_rows, 4),
                    "cluster_cohesion": round(cohesion, 4),
                    "cluster_theme_preview": truncated_preview,
                    "customer_review": customer_review_value,

                    # ✅ NO LLM — deterministic consolidations:
                    "semantic_action_statement": self._mmr_summary(sas_list) or self._centroid_pick(sas_list),
                    "stream_justification":     self._centroid_pick(just_list),
                    "matters":                   self._mmr_summary(mat_list)  or self._centroid_pick(mat_list),
                    "behavioral_impact":         self._mmr_summary(beh_list)  or self._centroid_pick(beh_list),
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
        
        import os
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
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
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            # light, safe speed-ups
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("DROP TABLE IF EXISTS clusters")
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
            cur.execute('CREATE INDEX IF NOT EXISTS idx_ed_em_stream ON clusters (experience_driver, emotion, opportunity_stream)')
            conn.commit()
        print(f"✅ Database initialized: {self.db_path}")


    def save_single_cluster_to_db(self, grp: pd.DataFrame, composite: Dict[str, Any], cid: str):
    # normalize callable to avoid None → NoneType surprises
        def nz(v): return "" if v is None else v

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            row = grp.iloc[0]
            bcs_id = row.get("bcs_id"); bcs_group_id = row.get("bcs_group_id")

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
                nz(composite.get('cluster_theme_preview', '')),
                nz(composite.get('customer_review') or row.get('customer_review') or row.get('comment_review')),
                nz(row.get('experience_driver')), nz(row.get('emotion')), nz(row.get('theme')),
                nz(row.get('opportunity_stream')), nz(row.get('feedback_type')),
                nz(row.get('customer_journey')), nz(row.get('customer_journey_stage')),
                nz(row.get('interaction_moment')), nz(row.get('context')),
                nz(keywords_str),
                nz(row.get('entity_name')),
                float((row.get('customer_effort_score', 0.0) or composite.get('customer_effort_score', 0.0) or 0.0)),
                nz(row.get('semantic_action_statement')), nz(row.get('stream_justification')),
                nz(row.get('matters')), nz(row.get('behavioral_impact')),
                nz(row.get('problem_statement') or '')
            ))
            conn.commit()
        print(f"💾 Saved single-row cluster {cid}")


    def save_multi_cluster_to_db(self, grp: pd.DataFrame, composite: Dict[str, Any], cid: str):
        def nz(v): return "" if v is None else v

        kw = composite.get('keywords')
        keywords_str = ", ".join(map(str, kw)) if isinstance(kw, list) else (str(kw) if kw is not None else "")
        replicated = {
            'experience_driver': nz(composite.get('experience_driver')),
            'emotion': nz(composite.get('emotion')),
            'theme': nz(composite.get('theme')),
            'opportunity_stream': nz(composite.get('opportunity_stream')),
            'feedback_type': nz(composite.get('feedback_type')),
            'customer_journey': nz(composite.get('customer_journey')),
            'customer_journey_stage': nz(composite.get('customer_journey_stage')),
            'interaction_moment': nz(composite.get('interaction_moment')),
            'context': nz(composite.get('context')),
            'keywords': nz(keywords_str),
            'entity_name': nz(composite.get('entity_name')),
            'customer_effort_score': float(composite.get('customer_effort_score', 0.0) or 0.0),
            'customer_review': nz(composite.get('customer_review'))
        }

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
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
                    nz(composite.get('cluster_theme_preview', '')),
                    replicated['customer_review'] or nz(row.get('customer_review')) or nz(row.get('comment_review')),
                    replicated['experience_driver'], replicated['emotion'], replicated['theme'],
                    replicated['opportunity_stream'], replicated['feedback_type'],
                    replicated['customer_journey'], replicated['customer_journey_stage'], replicated['interaction_moment'],
                    replicated['context'], replicated['keywords'], replicated['entity_name'],
                    replicated['customer_effort_score'],
                    nz(row.get('semantic_action_statement')), nz(row.get('stream_justification')),
                    nz(row.get('matters')), nz(row.get('behavioral_impact')),
                    nz(row.get('problem_statement') or '')
                ))
            conn.commit()
        print(f"💾 Saved multi-row cluster {cid} with {len(grp)} rows")


    # ---------- theme distribution ----------
    def _calculate_cluster_theme_distribution(self, clust_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate percentage distribution of themes within a cluster DataFrame.
        Handles missing columns, NaNs, and empty strings robustly.
        """
        if clust_df.empty:
            return {}

        # Column preference order
        col = (
            "cluster_theme_preview" if "cluster_theme_preview" in clust_df.columns
            else ("bcs_label" if "bcs_label" in clust_df.columns else None)
        )
        if not col:
            return {}

        # Clean and filter valid values
        counts = (
            clust_df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .value_counts()
        )

        if counts.empty:
            return {}

        total = float(counts.sum() or 1)
        return {str(theme): round((cnt / total) * 100, 1) for theme, cnt in counts.items()}


    # ---------- snapshot ----------
    def compute_granular_details_snapshot(self, raw_df: pd.DataFrame, layer3_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build OU composites per (Experience Driver → Emotion → Stream) and persist clusters.
        Robust to sparse/messy inputs, fully deterministic, LLM-free.
        """
        if layer3_df is None or layer3_df.empty:
            raise ValueError("Layer 3 diagnostics must be supplied and non-empty.")

        # Validate essentials in raw_df
        must_cols = {"experience_driver", "emotion_primary", "opportunity_stream"}
        missing = [c for c in must_cols if c not in raw_df.columns]
        if missing:
            raise ValueError(f"Raw DF missing required columns: {missing}")

        # Normalize key columns once
        rdf = raw_df.copy()
        rdf["experience_driver"] = rdf["experience_driver"].astype(str).str.strip()
        rdf["emotion_primary"] = rdf["emotion_primary"].astype(str).str.strip().str.lower()
        rdf["opportunity_stream"] = rdf["opportunity_stream"].astype(str).str.strip()

        # Clamp stream threshold (defensive)
        thr = max(0.0, min(1.0, float(self.OU_CFG.get("stream_threshold", 0.80))))

        all_df_chunks: List[pd.DataFrame] = []
        all_full_composites: Dict[str, Dict[str, Any]] = {}
        all_cluster_store: Dict[str, pd.DataFrame] = {}
        records: List[Dict[str, Any]] = []

        # Iterate L3 headers
        for _, hdr in layer3_df.iterrows():
            driver = str(hdr.get("experience_driver", "")).strip()
            if not driver:
                continue

            # parse emotion focus (list)
            emotion_focus = hdr.get("emotional_audit_focus", [])
            if isinstance(emotion_focus, str):
                try:
                    emotion_focus = ast.literal_eval(emotion_focus)
                except (ValueError, SyntaxError):
                    emotion_focus = []
            if not isinstance(emotion_focus, (list, tuple, set)):
                emotion_focus = []

            # emotion distribution map (optional, for enrichment)
            emotion_dist = hdr.get("emotion_distribution", {}) or {}

            # rows for this driver
            driver_rows = rdf[rdf["experience_driver"] == driver]
            if driver_rows.empty:
                continue

            # per emotion in focus
            for emotion in emotion_focus:
                emo_key = str(emotion).strip().lower()
                if not emo_key:
                    continue
                emotion_rows = driver_rows[driver_rows["emotion_primary"] == emo_key]
                if emotion_rows.empty:
                    continue

                # streams for this emotion
                stream_series = (
                    emotion_rows["opportunity_stream"]
                    .dropna().astype(str).str.strip()
                    .replace("", np.nan).dropna()
                )
                if stream_series.empty:
                    continue

                stream_counts = stream_series.value_counts()
                dominant_streams, stream_distribution = self.apply_stream_threshold_and_distribution(
                    stream_counts, threshold=thr
                )
                if not dominant_streams:
                    continue

                # cluster for each dominant stream
                for stream in dominant_streams:
                    stream_rows = emotion_rows[emotion_rows["opportunity_stream"] == stream]
                    if stream_rows.empty:
                        continue

                    # Run clustering + composites
                    clust_df, full_distribution, cluster_store, df_chunk, full_composites = self.cluster_behavior(
                        stream_rows, driver=driver, emotion=emotion, stream=stream
                    )
                    cluster_theme_distribution = self._calculate_cluster_theme_distribution(clust_df)

                    if df_chunk is not None and not df_chunk.empty:
                        all_df_chunks.append(df_chunk)
                    all_cluster_store.update(cluster_store or {})
                    all_full_composites.update(full_composites or {})

                    # Build record per cluster group
                    if clust_df is not None and not clust_df.empty:
                        for gid, grp in clust_df.groupby("bcs_group_id"):
                            meta = (full_composites or {}).get(gid, {})
                            if not meta:
                                # build a minimal safe shell if ever missing
                                meta = {
                                    "bcs_group_id": gid,
                                    "cluster_size": int(len(grp)),
                                    "bcs_share": round(len(grp) / max(len(clust_df), 1), 4),
                                    "cluster_cohesion": float(grp.get("cluster_cohesion", pd.Series([0.0])).iloc[0])
                                    if "cluster_cohesion" in grp.columns else 0.0,
                                    "cluster_theme_preview": str(grp.get("cluster_theme_preview", pd.Series([""])).iloc[0])
                                    if "cluster_theme_preview" in grp.columns else "",
                                }
                            composite = {
                                **meta,
                                "emotion_distribution": emotion_dist,
                                "stream_distribution": stream_distribution,
                                "cluster_theme_distribution": cluster_theme_distribution,
                            }
                            records.append(composite)

        # Debug summary (only if verbose)
        if getattr(self, "verbose", False):
            print(f"\n📦 FINAL DEBUG SUMMARY")
            print(f"🔢 Total full_composites: {len(all_full_composites)}")
            print(f"🔢 Total cluster_store: {len(all_cluster_store)}")
            missing = [cid for cid in all_cluster_store if cid not in all_full_composites]
            print(f"❌ Missing composites for: {missing}")

        # Persist clusters DB if we have any chunks
        merged_df = pd.concat(all_df_chunks, ignore_index=True) if all_df_chunks else pd.DataFrame()
        if not merged_df.empty and all_cluster_store:
            self.create_cluster_database(
                df=merged_df,
                full_composites=all_full_composites,
                cluster_store=all_cluster_store,
                db_path="outputs/clusters.db"
            )
     
        # Return composites (one row per cluster group)
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
        self.timeframe_days = int(timeframe_days)

        # force today/cutoff as Python date
        self.today = pd.to_datetime(today_anchor).date()

        # fix off-by-one (gives 75 days when end = today - 1)
        self.cutoff_date = self.today - timedelta(days=self.timeframe_days)

        # coerce to date; drop bad rows
        self.raw_df["date"] = pd.to_datetime(self.raw_df["date"], errors="coerce").dt.date
        self.raw_df = self.raw_df.dropna(subset=["date"])

        # window filter (inclusive start; end handled in compute’s index)
        self.raw_df = self.raw_df[self.raw_df["date"] >= self.cutoff_date]

        # emotion map
        emotion_scores = {"Adoration":3,"Appreciation":1,"Ambivalence":0,"Agitation":-1,"Anger":-3}
        self.raw_df["emotion_score"] = self.raw_df["emotion_primary"].map(emotion_scores)

        self.quadrant_matrix = get_quadrant_matrix()
        self.layer3_df = None
        self.pattern_lags = ["weekly","monthly"]

        if self.verbose:
            print(f"📦 Layer 3 @ {self.today} | Window: {self.cutoff_date} → {self.today}")
            print(f"🧮 Rows in window: {len(self.raw_df)} | L2 entities: {len(self.layer2_df)}")

    # === Core helpers ===
    def compute_normalized_eri(self, group):
        raw_eri = group["emotion_score"].mean()
        return ((raw_eri + 3) / 6) * 200 - 100

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
            "↑↑": {"label":"Strongly Rising","description":"Explosive upward movement","emoji_full":"↑↑ 🚀 Strongly Rising","strength":1.0, "polarity":  1},
            "↑":  {"label":"Moderately Rising","description":"Sustained growth","emoji_full":"↑ 📈 Moderately Rising","strength":0.6, "polarity":  1},
            "→":  {"label":"Stable","description":"No major shift","emoji_full":"→ ➖ Stable","strength":0.3, "polarity":  0},
            "↓":  {"label":"Moderately Falling","description":"Beginning to cool or drop","emoji_full":"↓ 📉 Moderately Falling","strength":0.6, "polarity": -1},
            "↓↓": {"label":"Strongly Falling","description":"Sharp deterioration or regression","emoji_full":"↓↓ 🧨 Strongly Falling","strength":1.0, "polarity": -1},
        }

    def momentum_thresholds(self):
        """
        Percent-of-ERI-band thresholds & data-quality gates (white paper aligned).
        - Movement thresholds are % of the ERI band (−100..+100 => width 200).
        - presence_gate: classification short-circuit at very low coverage.
        - presence_full_credit: coverage where confidence scaling reaches 1.0.
        - snr_mod/snr_strong: SNR gates used in momentum & insight.
        """
        return {
            # movement thresholds
            "rise_mod": 5.0,
            "rise_strong": 20.0,
            "fall_mod": -5.0,
            "fall_strong": -20.0,
            "flat_band": 5.0,  # use as ±5% band around zero if needed

            # data-quality gates (match momentum calc)
            "snr_mod": 0.75,
            "snr_strong": 1.25,
            "presence_gate": 0.05,         # 5% -> force Stable if below this
            "presence_full_credit": 0.40,  # 40%+ -> full confidence credit
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
    
    def _compute_trend_series(self, series, *, pct_thresh: float = 5.0,    # % of 200-pt ERI band required for ↑/↓
    snr_thresh: float = 0.75    # tiny stability guard 
    ) -> dict:
        """
        Trend (final, compute()-aligned):
        Assumptions:
        - 'series' is a continuous daily ERI vector for the fixed analysis window
        - Eligibility (≥2 distinct days observed) already enforced in compute()
        - No presence gating or interpolation here

        Steps:
        1) Fit linear slope across the window (polyfit)
        2) Convert modeled change over window to % of ERI band (200 wide)
        3) Compute a light SNR (based on day-to-day wiggle and √n)
        4) Classify: ↑ if pct > +pct_thresh & SNR≥snr_thresh
                    ↓ if pct < -pct_thresh & SNR≥snr_thresh
                    → otherwise

        Returns:
        {"symbol": "↑|→|↓", "trend_pct": float, "trend_snr": float}
        """
        try:
            s = pd.Series(series).astype(float)
            n = int(len(s))

            # Flatness guard (very low dynamic range)
            if (s.max() - s.min()) < 1e-9:
                return {"symbol": "→", "trend_pct": 0.0, "trend_snr": 0.0}

            # Linear slope across window
            x = np.arange(n, dtype=float)
            slope = np.polyfit(x, s, 1)[0]          # ERI units per day
            delta = slope * (n - 1)                 # modeled change across the window

            # Scale to % of full ERI band (−100..+100 => width 200)
            pct = float(np.clip((delta / 200.0) * 100.0, -100.0, 100.0))
            pct_r = round(pct, 2)

            # Light SNR guard: day-to-day wiggle + length scaling
            noise = float(s.diff().std())
            if not np.isfinite(noise) or noise == 0.0:
                noise = 1.0
            snr = abs(delta) / (noise * max(np.sqrt(n), 1.0))
            snr_r = round(float(snr), 2)

            # Classification
            if pct >  pct_thresh and snr >= snr_thresh:
                sym = "↑"
            elif pct < -pct_thresh and snr >= snr_thresh:
                sym = "↓"
            else:
                sym = "→"

            return {"symbol": sym, "trend_pct": pct_r, "trend_snr": snr_r}

        except Exception:
            return {"symbol": "→", "trend_pct": 0.0, "trend_snr": 0.0}


    def _compute_momentum_series(self, series,) -> dict:
        """
        Momentum (final, compute()-aligned):
        - Assumes 'series' is continuous daily ERI for the fixed analysis window
        - Detects recent shift: avg(second half) − avg(first half)
        - Scales to % of the 200-pt ERI band
        - Uses helper thresholds (momentum_thresholds) and labels (mom_details)
        - Light SNR on half-means to avoid false positives

        Returns:
        {
            "symbol": "↑↑|↑|→|↓|↓↓",
            "delta": float,            # % of ERI band
            "label": str,
            "description": str,
            "snr": float,
            "n_days": int,
            "emoji_full": str,
            "strength": float,
            "polarity": int,
            "thresholds_used": {...}
        }
        """
        try:
            s = pd.Series(series).astype(float)
            n = int(len(s))
            meta_neutral = self.mom_details("→")

            # Require at least one point per half (compute() gives full window, so this will hold)
            mid = n // 2
            if mid < 1 or (n - mid) < 1:
                return {
                    "symbol": "→", "delta": 0.0, "label": meta_neutral["label"],
                    "description": meta_neutral["description"], "snr": 0.0, "n_days": n,
                    "emoji_full": meta_neutral.get("emoji_full"),
                    "strength": meta_neutral.get("strength"),
                    "polarity": meta_neutral.get("polarity"),
                    "thresholds_used": self.momentum_thresholds(),
                }

            # Thresholds
            t = self.momentum_thresholds()

            # Split halves (no interpolation/smoothing here)
            first_half, second_half = s.iloc[:mid], s.iloc[mid:]
            avg1, avg2 = float(first_half.mean()), float(second_half.mean())
            delta_raw = avg2 - avg1  # ERI units

            # Scale to % of ERI band (−100..+100 => width 200)
            delta_pct = float(np.clip((delta_raw / 200.0) * 100.0, -100.0, 100.0))
            delta_pct_r = round(delta_pct, 2)

            # Pooled SE for half-means → light SNR
            import math
            k1, k2 = max(len(first_half), 1), max(len(second_half), 1)
            sd1 = float(first_half.std()) if k1 > 1 else 0.0
            sd2 = float(second_half.std()) if k2 > 1 else 0.0
            se1 = sd1 / max(math.sqrt(k1), 1e-9)
            se2 = sd2 / max(math.sqrt(k2), 1e-9)
            pooled_se = math.sqrt(max(se1**2 + se2**2, 1e-12))
            snr = abs(delta_raw) / pooled_se if pooled_se > 0 else 0.0
            snr_r = round(float(snr), 2)

            # Symbol via helper thresholds
            if   (delta_pct > t["rise_strong"]) and (snr >= t["snr_strong"]): sym = "↑↑"
            elif (delta_pct > t["rise_mod"])    and (snr >= t["snr_mod"]):    sym = "↑"
            elif (delta_pct < t["fall_strong"]) and (snr >= t["snr_strong"]): sym = "↓↓"
            elif (delta_pct < t["fall_mod"])    and (snr >= t["snr_mod"]):    sym = "↓"
            else:                                                                sym = "→"

            meta = self.mom_details(sym)
            return {
                "symbol": sym,
                "delta": delta_pct_r,
                "label": meta["label"],
                "description": meta["description"],
                "snr": snr_r,
                "n_days": n,
                "emoji_full": meta.get("emoji_full"),
                "strength": meta.get("strength"),
                "polarity": meta.get("polarity"),
                "thresholds_used": {
                    "rise_mod": t["rise_mod"], "rise_strong": t["rise_strong"],
                    "fall_mod": t["fall_mod"], "fall_strong": t["fall_strong"],
                    "snr_mod": t["snr_mod"], "snr_strong": t["snr_strong"]
                },
            }
        except Exception:
            meta = self.mom_details("→")
            return {
                "symbol": "→", "delta": 0.0, "label": meta["label"], "description": meta["description"],
                "snr": 0.0, "n_days": 0, "emoji_full": meta.get("emoji_full"),
                "strength": meta.get("strength"), "polarity": meta.get("polarity"),
                "thresholds_used": self.momentum_thresholds(),
            }


    """

## **Trend × Momentum Grid**

| **Trend** ↓ / **Momentum** → | **Momentum ↑ (Improving)**                                                                                                           | **Momentum → (Stable)**                                                                                              | **Momentum ↓ (Worsening)**                                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Trend ↑ (Improving)**      | **Momentum Building** 🚀<br>Both long-term and recent signals positive.<br>👉 Amplify programs, celebrate advocacy, invest in scale. | **Plateau Watch** ⚖️<br>Long-term gains, but short-term flattening.<br>👉 Refresh engagement to prevent stagnation.  | *(Rare)* Long-term gains but short-term deterioration.<br>👉 Watch for micro-shocks, validate if anomaly or early erosion.     |
| **Trend → (Stable)**         | **Recovery Signal** 🌱<br>Flat overall, but recent upward push.<br>👉 Support and accelerate the bounce.                             | **True Stability** 🟢<br>Flat long-term and short-term.<br>👉 Monitor quietly, no major action.                      | **Early Erosion** ⚠️<br>Flat long-term, but recent dip.<br>👉 Early-warning: intervene before decline becomes entrenched.      |
| **Trend ↓ (Declining)**      | **Emerging Recovery** 🔄<br>Overall decline, but short-term improvement.<br>👉 Double down to reverse the slide.                     | **Structural Decline** 🔻<br>Steady long-term fall, no shift.<br>👉 Root-cause deep dive and corrective initiatives. | **Accelerating Collapse** 💥<br>Both long-term and recent negative.<br>👉 Crisis response: urgent retention and loyalty fixes. |

---

### **How to Read**

* **Rows (Trend):** Big-picture trajectory — improving, stable, or declining.
* **Columns (Momentum):** Recent shift — getting better, flat, or worse.
* **Cells:** Combined state + CX action cue.

👉 **Why it’s powerful:** This grid translates **numbers into moves**. Teams don’t just know “how customers feel,” they know **when to amplify, when to stabilize, when to fix, and when to mobilize crisis response.**
        
    """   
    
    def _compute_volatility_series(
    self,
    series,
    *,
    signal_presence_pct: float | None = None,   # e.g., 0.27 for 27% of days with signal
    series_data_complete: bool | None = None,   # pre-fill completeness from compute()
    detrend: bool = True,                       # remove linear trend before measuring volatility
    robust: bool = False                        # use MAD→σ for outlier resistance
) -> dict:
        """
        Volatility (final, compute()-aligned):
        Assumes 'series' is continuous daily ERI for the fixed analysis window.
        Measures dispersion of (detrended) ERI, normalizes by √n (fair across window lengths),
        and adjusts for sparse/incomplete coverage.

        Returns:
            {"tier": str, "score": float, "score_adj": float,
            "n_days": int, "method": str, "detrended": bool}
        """
        try:
            s = pd.Series(series).astype(float)
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            n  = int(len(s))

            if n < 2:
                return {"tier": "✅ Stable", "score": 0.0, "score_adj": 0.0,
                        "n_days": n, "method": "std", "detrended": False}

            # Flatness guard
            if (s.max() - s.min()) < 1e-12:
                return {"tier": "✅ Stable", "score": 0.0, "score_adj": 0.0,
                        "n_days": n, "method": "std", "detrended": False}

            # Detrend so slope doesn't inflate volatility
            if detrend and n >= 2:
                x = np.arange(n, dtype=float)
                coef = np.polyfit(x, s, 1)           # slope, intercept
                r = s - (coef[0] * x + coef[1])
            else:
                r = s

            # Dispersion: robust (MAD→σ) or standard deviation
            if robust:
                med = float(np.median(r))
                mad = float(np.median(np.abs(r - med)))
                sigma  = 1.4826 * mad
                method = "mad_sigma"
            else:
                sigma  = float(r.std(ddof=1))
                method = "std"

            if not np.isfinite(sigma) or sigma <= 0.0:
                return {"tier": "✅ Stable", "score": 0.0, "score_adj": 0.0,
                        "n_days": n, "method": method, "detrended": bool(detrend)}

            # Raw volatility as % of ERI band (−100..+100 => width 200)
            std_pct = float(np.clip((sigma / 200.0) * 100.0, 0.0, 100.0))
            std_pct_r = round(std_pct, 2)

            # Length normalization (apples-to-apples across n)
            score_len = std_pct / max(np.sqrt(n), 1.0)

            # Coverage-aware adjustment (reward fuller coverage; haircut for inferred series)
            presence = float(signal_presence_pct) if signal_presence_pct is not None else None
            coverage_factor = 1.0
            if presence is not None:
                # full credit at >= 40% of days with signal; floor at 0.2
                coverage_factor = max(0.2, min(1.0, presence / 0.40))
            if series_data_complete is False:
                coverage_factor *= 0.8

            score_adj = float(np.clip(score_len * coverage_factor, 0.0, 100.0))
            score_adj_r = round(score_adj, 2)

            # Tiering on adjusted score (simple and readable)
            if score_adj_r <= 7.0:
                tier = "✅ Stable"
            elif score_adj_r <= 20.0:
                tier = "⚠ Fluctuating"
            else:
                tier = "🔴 Highly Fluctuating"

            return {
                "tier": tier,
                "score": std_pct_r,       # raw % of ERI band
                "score_adj": score_adj_r, # normalized & coverage-adjusted
                "n_days": n,
                "method": method,
                "detrended": bool(detrend),
            }

        except Exception as e:
            if getattr(self, "verbose", False):
                print(f"⚠️ Volatility computation failed: {e}")
            return {"tier": "✅ Stable", "score": 0.0, "score_adj": 0.0,
                    "n_days": 0, "method": "std", "detrended": False}


    """

## 🔹 Trend x Momentum x Volatility Grid

| **Trend**     | **Momentum**    | **Volatility**        | **Interpretation**                                | **CX Instruction**                             |
| ------------- | --------------- | --------------------- | ------------------------------------------------- | ---------------------------------------------- |
| ↑ (Improving) | → (Stable)      | ✅ Stable              | Long-term gains holding steady and reliable       | Keep reinforcing; stable trust is compounding  |
| ↑ (Improving) | → (Stable)      | ⚠ Fluctuating         | Gains are real but customer mood wobbles          | Monitor; ensure uplift isn’t fragile           |
| ↑ (Improving) | → (Stable)      | 🔴 Highly Fluctuating | Signals point upward, but chaos under the surface | Don’t over-celebrate; stabilize before scaling |
| ↑ (Improving) | ↑ (Recent rise) | ✅ Stable              | Clear acceleration with consistent base           | Amplify — perfect moment to scale advocacy     |
| ↑ (Improving) | ↑ (Recent rise) | ⚠ Fluctuating         | Surge is happening, but customers uneven          | Seize positives, but reinforce weak spots      |
| ↑ (Improving) | ↑ (Recent rise) | 🔴 Highly Fluctuating | “Spike effect” — surge may collapse               | Treat as hype-cycle; confirm if sustainable    |
| → (Flat)      | ↓ (Recent dip)  | ✅ Stable              | Plateau with emerging warning                     | Investigate small cracks before they widen     |
| → (Flat)      | ↓ (Recent dip)  | ⚠ Fluctuating         | Customers unsettled, mixed signals                | Early intervention; sentiment at tipping point |
| → (Flat)      | ↓ (Recent dip)  | 🔴 Highly Fluctuating | Volatility + downturn = fragile loyalty           | Treat as high-risk; prepare crisis workflows   |
| ↓ (Declining) | ↑ (Recent rise) | ✅ Stable              | Recovery starting; reliable turnaround            | Support rebound with targeted initiatives      |
| ↓ (Declining) | ↑ (Recent rise) | ⚠ Fluctuating         | Customers showing rebound but unstable            | Encourage positives, shore up weaknesses       |
| ↓ (Declining) | ↑ (Recent rise) | 🔴 Highly Fluctuating | Recovery attempt is noisy and fragile             | Monitor tightly; avoid premature bets          |
| ↓ (Declining) | ↓ (Recent fall) | ✅ Stable              | Clear deterioration, stable pattern               | Act decisively — loyalists are slipping away   |
| ↓ (Declining) | ↓ (Recent fall) | ⚠ Fluctuating         | Decline underway, customers uneven                | Contain damage, look for segment splits        |
| ↓ (Declining) | ↓ (Recent fall) | 🔴 Highly Fluctuating | Emotional free-fall with chaos                    | Crisis protocol — stabilize or lose trust      |

---

### 🧩 How to read it:

* **Trend** = long-arc slope (better/worse/flat)
* **Momentum** = short-arc shift (is recent half different from the first half)
* **Volatility** = confidence lens (stable vs noisy vs chaotic)

👉 Together: you don’t just know what direction customers are moving, you know 
if it's real, recent, and reliable.

        
    """

    def _compute_pattern_recognition(
    self,
    series,                # continuous daily ERI (already window-aligned by compute())
    entity_data,           # raw rows for this ED within the window (for weekday stats only)
    lags,                  # e.g., ["weekly","monthly"]
    *,
    min_buffer_days: int = 7,
    epsilon_flat: float = 1e-9
) -> dict:
        """
        Pattern Recognition (lean, instructive):
        - Assumes `series` is already continuous daily ERI for the fixed window
        - Detrend + center, then ACF at weekly(7) / monthly(30) lags
        - Adaptive gate = CI + small delta; emits Candidate if CI cleared but gate not met
        - Weekly 'pain_day' only when coverage is adequate using observed (non-filled) data

        Returns:
        {
            "has_pattern": bool, "pattern_type": "Weekly|Monthly|None",
            "pattern_strength": float|None, "pattern_confidence": "Strong|Moderate|Weak|Candidate|None",
            "pain_day": "Monday..Sunday"|None, "data_coverage_days": int,
            "min_required_days": int|None, "eri_by_day": dict|None,
            "acf": {"lag_days": int, "score": float, "sig": float, "gate": float}|None
        }
        """
        import numpy as np
        import pandas as pd
        from statsmodels.tsa.stattools import acf as _acf

        lag_days = {"weekly": 7, "monthly": 30, "quarterly": 90}

        # empty/flat guards
        s = pd.Series(series).astype(float)
        n = int(len(s))
        if n == 0:
            return {"has_pattern": False, "pattern_type": "None", "pattern_strength": None,
                    "pattern_confidence": "None", "pain_day": None,
                    "data_coverage_days": 0, "min_required_days": None,
                    "eri_by_day": None, "acf": None}

        raw_std = float(np.nanstd(s.values))
        if not np.isfinite(raw_std) or raw_std == 0.0:
            return {"has_pattern": False, "pattern_type": "None", "pattern_strength": None,
                    "pattern_confidence": "None", "pain_day": None,
                    "data_coverage_days": n, "min_required_days": None,
                    "eri_by_day": None, "acf": None}

        # detrend + center
        try:
            x = np.arange(n, dtype=float)
            coef = np.polyfit(x, s.values, 1)
            s_dt = pd.Series(s.values - (coef[0]*x + coef[1]))
        except Exception:
            s_dt = s.copy()
        s_dt = s_dt - s_dt.mean()

        if (s_dt.std() < max(epsilon_flat, 0.02 * raw_std)) or s_dt.isnull().all():
            return {"has_pattern": False, "pattern_type": "None", "pattern_strength": None,
                    "pattern_confidence": "None", "pain_day": None,
                    "data_coverage_days": n, "min_required_days": None,
                    "eri_by_day": None, "acf": None}

        # valid lags by window length
        valid_lags = []
        for lag in lags:
            L = lag_days.get(lag)
            if L is not None and n >= (L + min_buffer_days):
                valid_lags.append(lag)

        if not valid_lags:
            return {"has_pattern": False, "pattern_type": "None", "pattern_strength": None,
                    "pattern_confidence": "None", "pain_day": None,
                    "data_coverage_days": n, "min_required_days": None,
                    "eri_by_day": None, "acf": None}

        # sensitivity (balanced defaults)
        # gate = CI + delta ; min_gate caps how low it can go
        delta, min_gate = 0.05, 0.18

        best_pattern, best_score = None, -1.0
        candidate = None

        for lag in valid_lags:
            L = lag_days[lag]
            try:
                acf_vals = _acf(s_dt, nlags=L, fft=True, missing='conservative')
                score = float(acf_vals[L])  # correlation at exact lag
                # white-noise 95% CI
                sig = 1.96 / max(np.sqrt(n), 1.0)
                gate = max(sig + delta, min_gate)

                # light harmonic support for weekly (14)
                if lag == "weekly" and n >= 14 + min_buffer_days:
                    try:
                        acf_vals2 = _acf(s_dt, nlags=14, fft=True, missing='conservative')
                        score_harm = float(acf_vals2[14])
                        if (score_harm >= gate) and (score_harm > score * 0.9):
                            score = max(score, score_harm)
                    except Exception:
                        pass

                # confirmed pattern
                if (score >= gate) and (score > best_score):
                    if score >= gate + 0.25:   conf = "Strong"
                    elif score >= gate + 0.12: conf = "Moderate"
                    else:                       conf = "Weak"

                    best_pattern = {
                        "has_pattern": True,
                        "pattern_type": lag.capitalize(),
                        "pattern_strength": round(score, 3),
                        "pattern_confidence": conf,
                        "pain_day": None,  # may fill below (weekly only)
                        "data_coverage_days": int(n),
                        "min_required_days": int(L),
                        "eri_by_day": None,
                        "acf": {"lag_days": int(L), "score": round(score, 3),
                                "sig": round(sig, 3), "gate": round(gate, 3)},
                    }
                    best_score = score

                # otherwise: first candidate (above CI but below practical gate)
                elif (score >= sig) and (candidate is None):
                    candidate = {
                        "has_pattern": False,
                        "pattern_type": lag.capitalize(),
                        "pattern_strength": round(score, 3),
                        "pattern_confidence": "Candidate",
                        "pattern_candidate": True,
                        "pattern_likelihood": float(min(0.59, 0.35 + max(0.0, score - sig) * 1.2)),
                        "why": "Cleared CI but below practical gate",
                        "data_coverage_days": int(n),
                        "min_required_days": int(L),
                        "acf": {"lag_days": int(L), "score": round(score, 3),
                                "sig": round(sig, 3), "gate": round(gate, 3)},
                    }

            except Exception as e:
                if getattr(self, "verbose", False):
                    print(f"❌ ACF error for {lag}: {e}")
                continue

        # Weekly pain-day (observed-only weekday means), only if confirmed weekly pattern
        if best_pattern and best_pattern["pattern_type"].lower() == "weekly":
            try:
                ed_obs = entity_data.copy()
                ed_obs["eri_norm"] = ((ed_obs["emotion_score"].astype(float) + 3.0) / 6.0) * 200.0 - 100.0
                ed_obs["date"] = pd.to_datetime(ed_obs["date"], errors="coerce").dt.date
                daily_obs = (ed_obs.groupby("date", as_index=True)["eri_norm"]
                                .mean()
                                .rename("eri_day")
                                .dropna())
                # weekday coverage check
                weekday_idx   = pd.Index(pd.to_datetime(daily_obs.index).weekday, name="weekday")
                daily_by_wd   = pd.Series(daily_obs.values, index=weekday_idx).groupby(level=0).agg(list)
                weekday_counts = daily_by_wd.apply(len)
                weekday_means  = daily_by_wd.apply(lambda lst: float(np.mean(lst)) if len(lst) else np.nan)

                # minimal coverage (balanced): ≥4 weekdays observed, at least 1 obs per counted weekday
                if (weekday_counts >= 1).sum() >= 4 and not weekday_means.empty:
                    ordered = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                    wd_means_named = {name: (float(round(weekday_means.get(i), 2)) if np.isfinite(weekday_means.get(i)) else None)
                                    for i, name in enumerate(ordered)}
                    avail = {d: v for d, v in wd_means_named.items() if v is not None}
                    if avail:
                        best_pattern["pain_day"]   = min(avail, key=avail.get)
                        best_pattern["eri_by_day"] = wd_means_named
            except Exception:
                pass

        if best_pattern:
            return best_pattern
        if candidate:
            # return early-warning state
            base = {"has_pattern": False, "pattern_type": "None", "pattern_strength": None,
                    "pattern_confidence": "None", "pain_day": None,
                    "data_coverage_days": int(n), "min_required_days": None,
                    "eri_by_day": None, "acf": None}
            base.update(candidate)
            return base

        return {"has_pattern": False, "pattern_type": "None", "pattern_strength": None,
                "pattern_confidence": "None", "pain_day": None,
                "data_coverage_days": int(n), "min_required_days": None,
                "eri_by_day": None, "acf": None}


    """

### **Why Pattern Recognition Completes the Quartet**

**The Four Pillars of Temporal Intelligence**

1. **Trend** → *“Which direction are we heading over time?”*
2. **Momentum** → *“Did something change recently from the baseline?”*
3. **Volatility** → *“How reliable or unpredictable is the signal?”*
4. **Pattern Recognition** → *“When does this occur, and can we anticipate it?”*

Together, they move us from **observation** to **prediction**.


### **Pattern Recognition’s Distinctive Contribution**

**1. Behavioral Prediction Power**

* **Pain Day Detection** → pinpoint recurring weak points (e.g., *“Mondays are consistently worst for refunds”*).
* **Seasonal Cycles** → detect weekly, monthly, or quarterly loops in customer frustration/delight.
* **Operational Timing** → tell teams *when* issues will most likely surface, enabling proactive resourcing.

**2. Statistical Sophistication**

* **Adaptive Gates** → thresholds scale with data coverage (avoids false positives).
* **Harmonic Validation** → weekly patterns double-checked with 14-day harmonics.
* **STL Fallback** → seasonal decomposition when coverage is sparse.
* **Candidate Mode** → flags emerging but not yet confirmed patterns.

**3. CX-Native Intelligence**

* **Coverage-aware** → respects real-world survey sparsity.
* **Detrending** → separates cycles from directional shifts.
* **Confidence Levels** → Strong / Moderate / Weak / Candidate → ensures clarity, not false certainty.


### **Why the Quartet Is Transformative**

* **Trend + Pattern** → *“Customer frustration spikes every Monday, and the overall baseline is deteriorating.”*
* **Momentum + Volatility** → *“Recent satisfaction surge is genuine and stable, not a random wobble.”*
* **All Four Together** → *“Every Q4 delivery complaints used to ease off, but this year momentum is rising with high volatility — act immediately.”*


### **The Leap for CX**

This quartet turns **Experience Drivers** into **temporal DNA** — a living code of *direction, shift, stability, and recurrence*.

* No longer just *“this happened”*.
* Now → *“this will happen, when, and with what confidence.”*

👉 That's why Decipher isn't just analytics. It's **CX physics** — codifying the laws of how customer emotion behaves over time.


### 🔹 Temporal Intelligence Grid (2x2)

|                                                             | **Event-Oriented** (Change & Timing)                                            | **Reliability-Oriented** (Consistency & Predictability)                            |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Directional** (Where are we heading?)                     | **Trend** 📈<br>Long-term slope: *Are we improving or declining overall?*       | **Volatility** 🌊<br>Signal stability: *Is this trajectory consistent or erratic?* |
| **Situational** (What just happened / when will it repeat?) | **Momentum** ⚡<br>Recent shift: *Has performance changed compared to baseline?* | **Pattern** 🔁<br>Recurring cycles: *When do issues reliably resurface?*           |

---

### 🧩 Why This Matters

* **Trend** gives the *macro direction*.
* **Momentum** catches *micro shifts*.
* **Volatility** tells you if either is *trustworthy*.
* **Pattern** reveals the *clock* — the *when* behind the *what*.

Together: **Direction + Change + Reliability + Timing = Predictive Emotional Infrastructure.**

    """

    def _compute_momentum_saturation_insight(
    self,
    eri_score,
    momentum_symbol,
    *,
    signal_presence_pct: float | None = None,
    vol_norm: float | None = None,
    momentum_snr: float | None = None,
    series_data_complete: bool | None = None,
    strict_matrix: bool = True   # set True to forbid narrative without a matrix hit
) -> dict:
        # --- Contracts & gates ---
        t = self.momentum_thresholds()
        presence_full = float(t.get("presence_full_credit", 0.40))
        snr_strong    = float(t.get("snr_strong", 1.25))

        # --- Input hygiene ---
        try:
            eri_score = max(-100.0, min(100.0, float(eri_score)))
        except Exception:
            eri_score = 0.0

        # Accept either a symbol ("↑") or label ("Moderately Rising")
        m_sym = (momentum_symbol or "→").strip()
        if len(m_sym) > 2:  # likely a label
            m_sym = self.mom_symbol_from_label(m_sym)

        # --- Saturation via contract ---
        sat  = self.sat_from_eri(eri_score)  # {'emoji','clean','headroom','qssi_score','si'}
        si   = float(sat["si"])
        sat_emoji = sat["emoji"]   # e.g., "✅ High"
        sat_clean = sat["clean"]   # e.g., "High"

        # --- Momentum via contract ---
        mom_meta       = self.mom_details(m_sym)  # safe default to '→'
        mom_emoji_full = mom_meta["emoji_full"]   # e.g., "↑ 📈 Moderately Rising"
        mom_clean      = mom_meta["label"]        # e.g., "Moderately Rising"
        mom_strength   = float(mom_meta.get("strength", 0.3))
        mom_polarity   = int(mom_meta.get("polarity", 0))

        # --- Quadrant lookup (matrix source of truth) ---
        qm = self.quadrant_matrix
        hit_df = qm[(qm["Saturation_Tier"] == sat_emoji) & (qm["Momentum_Tier"] == mom_emoji_full)]
        matrix_hit = not hit_df.empty

        # --- Saturation centrality / borderline ---
        bins   = self.saturation_contract()
        bounds = sorted([th for th, *_ in bins] + [1.0])
        lower  = max(b for b in bounds if b <= si)
        upper  = min(b for b in bounds if b >= si)
        half_w = max((upper - lower) / 2.0, 1e-9)
        sat_distance   = min(si - lower, upper - si)
        sat_centrality = float(min(1.0, sat_distance / half_w))
        borderline     = sat_centrality < min(0.15, 0.5 * half_w)

        # --- Confidence (simple, auditable) ---
        presence = None if signal_presence_pct is None else float(signal_presence_pct)
        presence_factor  = 1.0 if presence is None else max(0.3, min(1.0, presence / presence_full))
        stability_factor = 1.0 if vol_norm is None else max(0.0, min(1.0, 1.0 - float(vol_norm)))
        if momentum_snr is None:
            snr_factor = 1.0
        else:
            snr_val    = float(momentum_snr)
            snr_factor = max(0.5, min(1.0, snr_val / snr_strong))

        base_conf = (0.40 * mom_strength) + (0.35 * sat_centrality) + (0.25 * stability_factor)
        conf = base_conf * presence_factor * snr_factor
        if series_data_complete is False:
            conf *= 0.9
        quadrant_conf = round(float(max(0.0, min(conf, 1.0))), 2)

        combined_quadrant_key = f"{sat_emoji} × {mom_emoji_full}"
        matrix_key_human      = f"{sat_clean}|{mom_clean}"

        if matrix_hit:
            q = hit_df.iloc[0]
            payload = {
                "signal_classification": {
                    "saturation_index": round(si, 2),
                    "saturation_tier": sat_clean,
                    "loyalty_tier": sat["headroom"]["tier"],
                    "momentum_tier": mom_clean,
                    "combined_quadrant": q["Diagnostic_Label"],  # human-friendly anchor
                    "combined_quadrant_key": combined_quadrant_key
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
            if strict_matrix:
                return {
                    "error": "Quadrant matrix miss",
                    "inputs": {"sat_emoji": sat_emoji, "mom_emoji_full": mom_emoji_full},
                    "hint": "Check contracts and matrix integrity",
                }
            # lenient fallback (never crashes in prod)
            payload = {
                "signal_classification": {
                    "saturation_index": round(si, 2),
                    "saturation_tier": sat_clean,
                    "loyalty_tier": sat["headroom"]["tier"],
                    "momentum_tier": mom_clean,
                    "combined_quadrant": "Unknown",
                    "combined_quadrant_key": combined_quadrant_key
                },
                "headroom_alignment": sat["headroom"],
                "quadrant_interpretation": {
                    "quadrant_label": "Unknown",
                    "urgency_level": "Unknown",
                    "interpretation": "No matrix row found for this combination",
                },
                "tactical_insight": {
                    "emotional_pulse": "Unknown",
                    "battle_status": "⚠️ Risk · Unknown",
                    "strategic_reality": "Validate inputs or matrix integrity",
                },
                "actionable_strategy": {
                    "momentum_context": "Unknown",
                    "saturation_context": "Unknown",
                    "trajectory_story": "Unknown",
                    "action_guidance": "Validate data sources and retry analysis",
                    "recommended_owner": "Data Team",
                    "headroom_guidance": sat["headroom"]["guidance"],
                },
            }

        payload["meta"] = {
            "matrix_key": matrix_key_human,
            "matrix_hit": matrix_hit,
            "saturation_tier_emoji": sat_emoji,
            "momentum_tier_emoji": mom_emoji_full,
            "quadrant_confidence": quadrant_conf,
            "borderline": bool(borderline),
            "saturation_bin": {"lower": round(lower, 2), "upper": round(upper, 2)},
            "momentum_contract": {
                "label": mom_clean,
                "strength": round(mom_strength, 3),
                "polarity": mom_polarity,
                "emoji_full": mom_emoji_full,
            },
            "components": {
                "sat_centrality": round(sat_centrality, 3),
                "stability_factor": None if vol_norm is None else round(1.0 - float(vol_norm), 3),
                "presence_factor": None if presence is None else round(presence_factor, 3),
                "snr_factor": None if momentum_snr is None else round(snr_factor, 3),
                "gates_used": {
                    "presence_full_credit": presence_full,
                    "snr_strong": snr_strong
                }
            }
        }
        return payload

    """

    ### **Why Saturation x Momentum Is the Decipher Battleground**

    **The Longitudinal Quartet (Trend, Momentum, Volatility, Pattern)** gives us temporal 
    intelligence — *direction, change, stability, and timing*.

    But **Saturation x Momentum** is where those insights get *weaponized* into execution. 
    It maps emotional energy (saturation) against directional shifts (momentum) inside a 
    **25-cell quadrant matrix**, each cell carrying:

    * **Diagnostic Label** → what this state really means
    * **Urgency Code** → how fast to act (🚨 Crisis, ⚠ Risk, 🌱 Opportunity, etc.)
    * **Strategic Narrative** → the story of what's unfolding
    * **Action Guidance + Owner** → who should move, and how


    ### **What It Adds Beyond the Quartet**

    1. **Strategic Centrality**

    * **Saturation** = emotional *investment* (how deep trust or frustration runs)
    * **Momentum** = emotional *velocity* (whether that energy is rising, flat, or falling)
    * Together, they form the **operational compass**: not just *what's happening*, but 
    *what it means for loyalty*.

    2. **Quadrant Narratives**
    Each ED is positioned into a **living quadrant**:

    * *Critical Crisis Response*
    * *Maximize Advocacy Surge*
    * *Erosion Risk Early Watch*
    * …and 22 others.
        These aren't metrics — they're **battlefield roles**, instantly instructive to CX teams.

    3. **Confidence Layering**
    Confidence isn't a guess — it's **data-aware**:

    * Presence coverage (did this driver show up often enough?)
    * Volatility stability (is the signal steady or jittery?)
    * Momentum SNR (is the shift statistically reliable?)
        These scale quadrant assignments from *weak candidate* to *strongly confirmed*.


    ### **Why It's Transformative**

    * **Trend tells you direction.**
    * **Momentum tells you recent change.**
    * **Volatility tells you trustworthiness.**
    * **Pattern tells you recurrence.**

    👉 **Saturation x Momentum turns all of that into action.**

    It's the **decoding layer** where raw emotional telemetry becomes:

    * *“Prevent Active Loyalty Churn.”*
    * *“Scale Advocacy Programs.”*
    * *“Run total overhaul diagnostics — emotional collapse imminent.”*

    ---

    ### **The Leap**

    This module is what makes Decipher **operational infrastructure, not analytics**.

    It's not just *“the signal is declining.”*
    It's *“Loyalty collapse risk, Crisis protocol required, CX Leadership owner.”*

    That's why Saturation x Momentum is the **execution grammar of customer emotion** — 
    the **battlefield map** that no other CX system has.


    """

    # === Layer 4: Signal Strength (QSSI) — simple, non-directive ===
    def _compute_signal_strength(self, trend_symbol: str, momentum_symbol: str, saturation_index: float) -> dict:
        """
        QSSI - Quantified Signal Strength Index (minimal, contract-aligned)
        Score (0-10) = Velocity (trend × momentum, 0-6) + Saturation modifier (0-4 via sat_from_si)

        Tiers:
        9-10 → 💥 Critical   | Exceptionally strong signal
        6-8  → 🔥 Strong     | Clearly material signal
        4-5  → 🌱 Emerging   | Signal taking shape
        1-3  → 🔁 Weak       | Faint indications
        0    → ❌ No Signal  | No meaningful movement/direction
        """

        # sanitize
        trend = trend_symbol if trend_symbol in ("↑", "→", "↓") else "→"
        momentum = momentum_symbol if momentum_symbol in ("↑↑", "↑", "→", "↓", "↓↓") else "→"
        try:
            si = float(saturation_index)
            si = max(0.0, min(1.0, si))
        except Exception:
            si = 0.5

        # velocity (0–6)
        velocity_lookup = {
            ("↑", "↑↑"): (5, "Strong trend with explosive movement"),
            ("↓", "↑↑"): (5, "Strong trend with explosive movement"),
            ("↑", "↓↓"): (6, "Strong negative spiral — highest alert"),
            ("↓", "↓↓"): (6, "Strong negative spiral — highest alert"),
            ("↑", "↑"):  (4, "Moderate trend and movement"),
            ("↑", "↓"):  (4, "Moderate trend and movement"),
            ("↓", "↑"):  (4, "Moderate trend and movement"),
            ("↓", "↓"):  (4, "Moderate trend and movement"),
            ("↑", "→"):  (2, "Trend present, no momentum"),
            ("↓", "→"):  (2, "Trend present, no momentum"),
            ("→", "↑↑"): (3, "Flat trend but strong movement"),
            ("→", "↓↓"): (3, "Flat trend but strong movement"),
            ("→", "↑"):  (1, "Some movement, no trend"),
            ("→", "↓"):  (1, "Some movement, no trend"),
            ("→", "→"):  (0, "No meaningful trend or momentum detected"),
        }
        velocity_score, velocity_explanation = velocity_lookup.get(
            (trend, momentum), (0, "No meaningful trend or momentum detected")
        )

        # saturation modifier (0–4) via contract
        sat_info = self.sat_from_si(si)  # {"emoji","clean","headroom","qssi_score","si"}
        saturation_modifier = int(sat_info["qssi_score"])
        saturation_band = sat_info.get("clean", "Unknown")
        headroom = sat_info.get("headroom", {}) or {}
        saturation_interpretation = (
            headroom.get("interpretation") or headroom.get("guidance") or "Headroom mapping applied"
        )

        # score and neutral tiering
        qssi_score = int(velocity_score) + saturation_modifier
        if qssi_score >= 9:
            tier, tier_emoji = "Critical Signal", "💥"
            description = "Exceptionally strong signal (large movement within a saturated state)."
        elif qssi_score >= 6:
            tier, tier_emoji = "Strong Signal", "🔥"
            description = "Clearly material signal (meaningful movement/direction)."
        elif qssi_score >= 4:
            tier, tier_emoji = "Emerging Signal", "🌱"
            description = "Signal taking shape (moderate indications)."
        elif qssi_score >= 1:
            tier, tier_emoji = "Weak Signal", "🔁"
            description = "Faint indications; limited implications so far."
        else:
            tier, tier_emoji = "No Signal", "❌"
            description = "No meaningful movement or direction detected."

        # non-directive, plain-English meanings
        tier_meanings = {
            "Critical Signal":  "Exceptionally strong signal; stands out in reviews.",
            "Strong Signal":    "Clearly material; merits focused attention soon.",
            "Emerging Signal":  "Taking shape; monitor and seek corroboration.",
            "Weak Signal":      "Faint; keep on radar.",
            "No Signal":        "No meaningful movement or direction.",
        }

        return {
            "qssi_score": qssi_score,                          # 0–10
            "qssi_tier": f"{tier_emoji} {tier}",
            "qssi_description": description,                   # concise & neutral
            "qssi_interpretation": tier_meanings[tier],        # non-directive meaning
            "components": {
                "velocity_score": int(velocity_score),
                "velocity_explanation": velocity_explanation,
                "saturation_modifier": saturation_modifier,
                "saturation_band": saturation_band,
                "saturation_interpretation": saturation_interpretation,
            },
            "inputs": {
                "trend_symbol": trend,
                "momentum_symbol": momentum,
                "saturation_index": round(si, 3),
            },
            "urgency_answer": f"Urgency Level: {qssi_score}/10 - {description}",  # informational, not prescriptive
        }


    """

    ### **Business Question Alignment**

    The six modules now work as a **coherent diagnostic suite**, each answering a 
    **different operational question**. Together, they transform raw Experience Driver 
    signals into structured, actionable intelligence:

    * **Trend** → *“Are we improving or declining overall?”*
    Captures the long-arc slope of customer emotion, showing directional progress or erosion.

    * **Momentum** → *“Did something change recently?”*
    Detects short-term shifts versus baseline, highlighting sudden surges or dips.

    * **Volatility** → *“Can we trust this signal for planning?”*
    Measures stability versus noise, distinguishing reliable patterns from fragile fluctuations.

    * **Pattern Recognition** → *“When will this happen again?”*
    Identifies recurring cycles (weekly, monthly, quarterly) and potential “pain days.”

    * **Saturation x Momentum** → *“What's the strategic context?”*
    Places the signal inside a **25-cell quadrant map**, assigning diagnostic labels, 
    urgency codes, and strategic narratives that frame the bigger picture.

    * **QSSI (Quantified Signal Strength Index)** → *“How strong and urgent is this signal?”*
    Rolls the diagnostics into a simple **0-10 pointer score** with intuitive tiers — 
    Critical, Strong, Emerging, Weak, or No Signal.

    
    ### **Why This Matters**

    Each function is not an isolated calculation but a **lens**:

    * Trend and Momentum describe **direction and movement**.
    * Volatility and Pattern describe **stability and recurrence**.
    * Saturation x Momentum adds **strategic context**.
    * QSSI distills it into a **single strength pointer** for prioritization.

    👉 Together, they let CX teams move from *“What happened?”* → *“Why it matters?”* → 
    *“How strongly should we respond?”*

    """

   # === PEM (Predictive Emotional Modeling) — lean & intuitive ===

    def _normalize_vol_tier(self, volatility_tier: str) -> str:
        """
        Normalize volatility tier into one of:
        'Stable' | 'Fluctuating' | 'Highly Fluctuating'
        """
        vt = (volatility_tier or "").strip().lower()
        if "high" in vt:
            return "Highly Fluctuating"
        if "stable" in vt or "✅" in (volatility_tier or ""):
            return "Stable"
        if "fluctuat" in vt:
            return "Fluctuating"
        return "Fluctuating"


    def _pem_confidence(self, *, qssi: int, vol_tier: str, pattern_confidence: str | None) -> tuple[str, float]:
        """
        Simple, transparent confidence (Low/Moderate/High) with a 0–1 score.
        - QSSI gives the base.
        - Pattern confidence nudges up.
        - Volatility nudges down.
        """
        base = 0.30  # neutral base
        # QSSI contribution
        if qssi >= 8:   base += 0.35
        elif qssi >= 6: base += 0.25
        elif qssi >= 4: base += 0.15
        else:           base += 0.05

        # Pattern contribution
        pc = (pattern_confidence or "").strip().lower()
        if   pc == "strong":   base += 0.20
        elif pc == "moderate": base += 0.12
        elif pc == "weak":     base += 0.05

        # Volatility penalty
        vt = self._normalize_vol_tier(vol_tier)
        if   vt == "Highly Fluctuating": base -= 0.20
        elif vt == "Fluctuating":        base -= 0.10

        score = max(0.10, min(0.95, round(base, 2)))
        level = "High" if score >= 0.70 else "Moderate" if score >= 0.50 else "Low"
        return level, score


    def _pem_direction_label(self, trend_symbol: str, momentum_symbol: str) -> str:
        """
        Compact, intuitive direction descriptor combining trend & momentum.
        """
        tr, mo = trend_symbol, momentum_symbol
        if tr == "↑" and mo in ("↑↑", "↑"): return "improving"
        if tr == "↓" and mo in ("↓↓", "↓"): return "declining"
        if tr == "↑" and mo in ("→",):      return "improving (recent plateau)"
        if tr == "↓" and mo in ("→",):      return "declining (recent plateau)"
        if tr == "→" and mo in ("↑↑","↑"):  return "stabilizing upward"
        if tr == "→" and mo in ("↓↓","↓"):  return "stabilizing downward"
        if tr == "↓" and mo in ("↑↑","↑"):  return "recovering"
        if tr == "↑" and mo in ("↓↓","↓"):  return "cooling"
        return "stable"


    def build_predictive_emotional_modeling(
        self,
        *,
        trend_symbol: str,           # "↑", "→", "↓"
        momentum_symbol: str,        # "↑↑", "↑", "→", "↓", "↓↓"
        volatility_tier: str,        # "Stable" | "Fluctuating" | "Highly Fluctuating" (emojis allowed)
        volatility_score: float,     # adjusted volatility % (0..100) — for display only
        pattern_detected: bool,
        pattern_type: str | None,    # "Weekly" | "Monthly" | "Quarterly" | None
        pattern_confidence: str | None,   # "Strong" | "Moderate" | "Weak" | None
        pain_day: str | None,        # weekday if weekly pattern
        qssi_score: int,             # 0–10 from QSSI
        momentum_saturation_quadrant: dict | None,  # from _compute_momentum_saturation_insight
        horizon_days: int = 30
    ) -> dict:
        """
        PEM — forward-looking *pointer*, not prescription.
        Rules (simple):
        1) If signal is weak (QSSI < 4): no forecast; keep monitoring.
        2) If a pattern is detected (Strong/Moderate), report *when* it tends to recur.
        3) Else, describe likely trajectory qualitatively from Trend×Momentum.
        4) Always expose confidence and basis. Avoid deterministic dates.
        """
        # sanitize
        tr = trend_symbol if trend_symbol in ("↑","→","↓") else "→"
        mo = momentum_symbol if momentum_symbol in ("↑↑","↑","→","↓","↓↓") else "→"
        vt = self._normalize_vol_tier(volatility_tier)
        qssi = int(max(0, min(10, int(qssi_score or 0))))

        msq = momentum_saturation_quadrant or {}
        quad = (msq.get("quadrant_interpretation") or {}).get("quadrant_label", "Unknown")

        # Confidence (informational)
        conf_level, conf_score = self._pem_confidence(qssi=qssi, vol_tier=vt, pattern_confidence=pattern_confidence)

        # 1) Not enough signal for forecasting
        if qssi < 4:
            return {
                "forecast": {
                    "summary": "Insufficient signal for a forward pointer — keep monitoring.",
                    "basis": "Low signal strength",
                    "confidence_level": "None",
                },
                "watch": {
                    "focus": ["Trend direction", "Momentum shifts", "Volatility changes"],
                    "horizon_days": horizon_days
                },
                "signal_synthesis": {
                    "trend": tr, "momentum": mo, "volatility": vt,
                    "qssi": qssi, "pattern": {"detected": False}
                },
                "meta": {"quadrant": quad, "confidence_score": conf_score}
            }

        # 2) Pattern-led pointer
        if pattern_detected and (pattern_type or "").strip():
            pt = pattern_type.strip().capitalize()
            if pt == "Weekly":
                when_txt = f"on {pain_day}" if pain_day else "on its typical weekday"
                window_txt = "next week"
            elif pt == "Monthly":
                when_txt = "around the same month point"
                window_txt = "in the next month"
            elif pt == "Quarterly":
                when_txt = "around the same quarter phase"
                window_txt = "in the next quarter"
            else:
                when_txt, window_txt = "around its usual interval", "soon"

            summary = (
                f"Recurring pattern expected {window_txt} ({when_txt}); "
                f"current trajectory looks {self._pem_direction_label(tr, mo)}."
            )

            return {
                "forecast": {
                    "summary": summary,
                    "basis": f"{pt} pattern ({pattern_confidence or 'Unknown'} confidence)",
                    "confidence_level": conf_level
                },
                "watch": {
                    "windows": [pt],
                    "notes": ["Use pattern window as an early lookout; avoid over-committing to exact dates."],
                    "horizon_days": horizon_days
                },
                "signal_synthesis": {
                    "trend": tr, "momentum": mo, "volatility": vt,
                    "qssi": qssi,
                    "pattern": {"detected": True, "type": pt, "confidence": pattern_confidence, "pain_day": pain_day}
                },
                "meta": {"quadrant": quad, "confidence_score": conf_score}
            }

        # 3) Trajectory pointer (no strong pattern)
        direction = self._pem_direction_label(tr, mo)
        basis = "Directional trajectory from Trend × Momentum"
        if vt == "Highly Fluctuating":
            direction += " (unstable)"

        return {
            "forecast": {
                "summary": f"Over the next {horizon_days} days the signal is likely {direction}.",
                "basis": basis,
                "confidence_level": conf_level
            },
            "watch": {
                "focus": ["Momentum flips", "Volatility spikes", "Quadrant changes"],
                "horizon_days": horizon_days
            },
            "signal_synthesis": {
                "trend": tr, "momentum": mo,
                "volatility": vt, "volatility_score_pct": round(float(volatility_score or 0.0), 2),
                "qssi": qssi,
                "pattern": {"detected": False}
            },
            "meta": {"quadrant": quad, "confidence_score": conf_score}
        }


    def compute(self):
        results, skipped = [], []

        # Normalize ED keys
        self.raw_df["experience_driver"] = self.raw_df["experience_driver"].astype(str).str.strip()

        # L2-to-L3 eligibility: priority lens first
        entities = self.layer2_df[self.layer2_df["Priority_Status"].isin(["P0","P1","P2","P3"])]

        # ---- Build exact analysis window [start, end=today-1], length = timeframe_days ----
        end_date = self.today - timedelta(days=1)            # inclusive end
        start_date = self.cutoff_date                        # intended start
        expected_span = int(self.timeframe_days)
        actual_span = (end_date - start_date).days + 1
        if actual_span != expected_span:
            start_date = end_date - timedelta(days=expected_span - 1)

        full_idx = pd.date_range(start=start_date, end=end_date, freq="D").date
        full_idx = pd.Index(full_idx)                        # dtype: datetime.date
        analysis_window_days = len(full_idx)
        num_days_observed = analysis_window_days             # keep name for downstream payloads

        for _, row in entities.iterrows():
            try:
                ed = str(row["experience_driver"]).strip()

                # ---- Gate A: mentions (doctrine) ----
                if int(row.get("No_of_Mentions", 0)) < 2:
                    if self.verbose: print(f"⏭️ DROP [{ed}] → <2 mentions")
                    skipped.append((ed, "<2 mentions"))
                    continue

                # Subset raw windowed rows for this ED
                data = self.raw_df[self.raw_df["experience_driver"] == ed]
                if data.empty:
                    if self.verbose: print(f"❌ DROP [{ed}] → no raw data")
                    skipped.append((ed, "No raw data"))
                    continue

                # ---- Daily ERI per observed date (pure pre-fill) ----
                daily_eri_obs = (
                    data.groupby("date")
                        .apply(self.compute_normalized_eri)   # scalar per day
                        .sort_index()
                )

                # Align to full window BEFORE any fill → true coverage metrics
                daily_eri = daily_eri_obs.reindex(full_idx)

                # Coverage metrics (PRE-FILL)
                missing_days_pct = float(daily_eri.isna().mean())
                series_data_complete = not daily_eri.isna().any()   # pre-fill completeness flag
                observed_days = int(daily_eri.dropna().shape[0])

                # ---- Gate B: days (doctrine) ----
                if observed_days < 2:
                    if self.verbose: print(f"⏭️ DROP [{ed}] → insufficient days ({observed_days})")
                    skipped.append((ed, f"Insufficient days: {observed_days}"))
                    continue

                # Presence: % of days with ≥1 mention in-window (based on raw counts)
                ds = data.groupby("date").size().reindex(full_idx, fill_value=0)
                pct_days_with_signal = round(float((ds > 0).sum()) / analysis_window_days, 3)
                overall_rel = ("Low" if pct_days_with_signal < 0.10
                            else "Moderate" if pct_days_with_signal < 0.40
                            else "High")

                # ---- Interpolate for continuity (analysis track only) ----
                # Linear interpolation avoids step artifacts; confined to the window.
                daily_eri = daily_eri.interpolate(method="linear", limit_direction="both")

                # =========================
                # From here on, call modules with:
                #   - daily_eri (continuous analysis series)
                #   - signal_presence_pct = pct_days_with_signal
                #   - series_data_complete (pre-fill flag)
                # Keep your existing module calls below this line.
                # =========================

                # ---- modules (example placeholders; keep your existing calls) ----
                trend = self._compute_trend_series(
                    daily_eri,
                    signal_presence_pct=pct_days_with_signal,
                    min_presence_gate=0.05,   # honor doctrine presence gate
                    min_n=2                   # aligned with ≥2 days rule
                )

                momentum = self._compute_momentum_series(
                    daily_eri,
                    signal_presence_pct=pct_days_with_signal
                )

                volatility = self._compute_volatility_series(
                    daily_eri,
                    signal_presence_pct=pct_days_with_signal,
                    series_data_complete=series_data_complete
                )

                pattern_block = self._compute_pattern_recognition(
                    daily_eri, data, self.pattern_lags
                )

                # ---- Quadrant / QSSI / PEM (your existing code continues) ----
                momentum_symbol = momentum.get("symbol", "→")
                quadrant_block = self._compute_momentum_saturation_insight(
                    row["ERI"],
                    momentum_symbol,
                    signal_presence_pct=pct_days_with_signal,
                    vol_norm=min(max((volatility.get("score_adj", volatility.get("score", 0.0)) / 20.0), 0.0), 1.0),
                    momentum_snr=momentum.get("snr"),
                    series_data_complete=series_data_complete
                )

                qssi_block = self._compute_signal_strength(
                    trend.get("symbol", "→"),
                    momentum_symbol,
                    quadrant_block["signal_classification"]["saturation_index"],
                )

                pem_block = self.build_predictive_emotional_modeling(
                    trend_symbol=trend.get("symbol", "→"),
                    momentum_symbol=momentum_symbol,
                    volatility_tier=volatility.get("tier", "✅ Stable"),
                    volatility_score=float(volatility.get("score_adj", volatility.get("score", 0.0))),
                    pattern_detected=bool(pattern_block.get("has_pattern")),
                    pattern_type=pattern_block.get("pattern_type"),
                    pattern_confidence=pattern_block.get("pattern_confidence"),
                    pain_day=pattern_block.get("pain_day"),
                    qssi_score=int(qssi_block["qssi_score"]),
                    momentum_saturation_quadrant=quadrant_block,
                    horizon_days=30
                )

                # ---- story, confidence, provenance (keep your structure) ----
                trend_symbol = trend.get("symbol", "→")
                storyline = (
                    f"{ed} is in {quadrant_block['quadrant_interpretation']['quadrant_label']} with "
                    f"{quadrant_block['signal_classification']['momentum_tier']} momentum and "
                    f"{quadrant_block['signal_classification']['saturation_tier']} saturation. "
                    f"{quadrant_block['quadrant_interpretation']['interpretation']} "
                    f"Action: {quadrant_block['actionable_strategy']['action_guidance']} "
                    f"(Owner: {quadrant_block['actionable_strategy']['recommended_owner']})."
                )

                pc = pattern_block.get("pattern_confidence")
                pattern_conf_norm = (1.0 if pc == "Strong" else
                                    0.6 if pc == "Moderate" else
                                    0.3 if pc == "Weak" else 0.3)

                vol_norm = min(max((volatility.get("score_adj", volatility.get("score", 0.0)) / 20.0), 0.0), 1.0)
                layer3_confidence_score = round(
                    min(1.0, max(0.0,
                        0.4 * float(quadrant_block["meta"]["quadrant_confidence"]) +
                        0.3 * (1.0 - vol_norm) +
                        0.3 * pattern_conf_norm
                    )), 2
                )

                from uuid import uuid4
                capsule_meta = {
                    "capsule_id": f"SC-{uuid4().hex[:12]}",
                    "generated_at": pd.Timestamp.utcnow().isoformat(),
                    "version": "XDI.v1",
                    "window_start_date": str(start_date),
                    "window_end_date": str(end_date)
                }
                                
                results.append({
                    "experience_driver": ed,
                    "priority_class": row["Priority_Status"],
                    "associated_entity_names": row.get("Associated_Entity_Names"),
                    "most_recent_mention": row.get("Most_Recent_Date"),
                    "no_of_mentions": row.get("No_of_Mentions"),
                    "eri_score": round(float(row["ERI"]), 2),
                    "r_score": round(float(row["R"]), 2),
                    "f_score": round(float(row["F"]), 2),
                    "rf_score": round(float(row["RF"]), 2),
                    "rfi_score": round(float(row["RFI"]), 2),
                    "emotion_perception_tier": row.get("Loyalty_State"),
                    "rf_urgency_category": row.get("RF_Urgency_Category"),
                    "eri_rf_urgency_category": row.get("ERI_RF_Quadrant"),

                    "trend_block": trend,
                    "momentum_block": momentum,
                    "volatility_block": volatility,
                    "pattern_block": {
                        "has_pattern": bool(pattern_block.get("has_pattern")),
                        "pattern_type": pattern_block.get("pattern_type") if pattern_block.get("has_pattern") else None,
                        "pattern_strength": (
                            round(float(pattern_block["pattern_strength"]), 2)
                            if pattern_block.get("has_pattern") and pattern_block.get("pattern_strength") is not None else None
                        ),
                        "pattern_confidence": pattern_block.get("pattern_confidence") if pattern_block.get("has_pattern") else None,
                        "pain_day": pattern_block.get("pain_day") if (pattern_block.get("pattern_type") or "").lower() == "weekly" else None,
                        "data_coverage_days": pattern_block.get("data_coverage_days"),
                        "min_required_days": pattern_block.get("min_required_days"),
                        "eri_by_day": pattern_block.get("eri_by_day") if (pattern_block.get("pattern_type") or "").lower() == "weekly" else None,
                        "acf": pattern_block.get("acf"),
                    },

                    "momentum_saturation_insight": quadrant_block,
                    "signal_strength_index": qssi_block,
                    "predictive_emotion_forecast": pem_block,

                    "momentum_saturation_story": storyline,
                    "momentum_saturation_confidence": quadrant_block["meta"]["quadrant_confidence"],
                    "momentum_saturation_borderline_flag": quadrant_block["meta"]["borderline"],
                    "analytics_suite_confidence": layer3_confidence_score,

                    "provenance": {
                        "analysis_window_days": int(self.timeframe_days),
                        "observed_days_with_signal": observed_days,          # ← add
                        "signal_presence_pct": pct_days_with_signal,
                        "trend_analysis_days": analysis_window_days,
                        "momentum_analysis_days": analysis_window_days // 2,
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


"""

# The Seven-Function Complete Emotional Intelligence Architecture
## From Analytics to Execution Engine to Predictive Intelligence

### The Revolutionary Insight

Customer emotion follows temporal physics. It has momentum, volatility, patterns, and 
strategic positioning. The breakthrough comes from treating emotion not as static 
survey scores, but as dynamic infrastructure that can be measured, predicted, and 
operationally routed into actionable business intelligence.

### Complete Seven-Function Emotional Intelligence Architecture

The seven-function suite provides comprehensive temporal intelligence for Experience Drivers, 
progressing from measurement to strategic action to predictive foresight:

**MEASUREMENT LAYER (Functions 1-4):**

1. **Trend** - Where are we going?
   - Linear regression across the entire window
   - Reveals consistent directional movement
   - Answers: "Are we getting better or worse over time?"

2. **Momentum** - What changed recently?
   - Compares recent period vs historical baseline
   - Detects inflection points and regime changes
   - Answers: "Has our recent performance shifted from the baseline?"

3. **Volatility** - How predictable is this signal?
   - Measures consistency and reliability of emotional patterns
   - Coverage-aware adjustment for data quality
   - Answers: "Can we trust this signal for planning?"

4. **Pattern** - When does this happen cyclically?
   - Detects weekly, monthly, quarterly seasonal cycles
   - Identifies pain days and recurring emotional patterns
   - Answers: "When will this issue surface and why?"

**STRATEGIC LAYER (Functions 5-6):**

5. **Momentum × Saturation** - What should we do about it?
   - Strategic positioning matrix combining directional change with emotional ceiling proximity
   - 25-cell quadrant system with urgency codes and tactical guidance
   - Answers: "What specific action should we take right now?"

6. **QSSI (Quantified Signal Strength Index)** - How urgently should we act?
   - Combines velocity score (trend × momentum) with saturation modifier
   - Transparent 0-10 scale with clear tier classifications
   - Answers: "How should we prioritize this signal among competing demands?"

**PREDICTIVE LAYER (Function 7):**

7. **PEM (Predictive Emotional Modeling)** - What will happen next?
   - Synthesizes all six functions into forward-looking business intelligence
   - Provides intervention timing and expected impact assessment
   - Answers: "What will happen to this Experience Driver in the next 30 days, and when 
   should we intervene for maximum impact?"

### Why This Seven-Function Architecture is Revolutionary

**Transforms Experience Drivers from analytics into a complete business intelligence engine:**

- **From Description to Prescription to Prediction**: Not just "customer satisfaction is 
declining" but "customers at 72% saturation with strong downward momentum (QSSI 8) 
will likely continue declining over next 14 days unless immediate CX/Product intervention 
occurs by Tuesday"

- **From Static to Dynamic to Predictive**: Emotion becomes infrastructure with temporal 
intelligence that forecasts future states and optimal intervention timing

- **From Insights to Actions to Strategic Foresight**: Every analysis includes current 
urgency level, recommended owner, specific tactical guidance, AND predictive timeline for 
maximum impact

### The Three-Layer Intelligence Progression

**Layer 1 - Temporal Measurement**: Functions 1-4 provide comprehensive understanding of 
current emotional dynamics across time dimensions

**Layer 2 - Strategic Positioning**: Functions 5-6 translate measurements into actionable 
business priorities with clear ownership and urgency levels

**Layer 3 - Predictive Intelligence**: Function 7 synthesizes all measurements and 
strategy into forward-looking guidance that enables proactive rather than reactive CX management

### What the World Had vs What It Needed

**What Existed:**
- Standard statistical techniques (regression, correlation, standard deviation)
- Customer feedback data streams
- CX analytics platforms
- Reactive problem-solving approaches

**What Was Missing:**
- Conceptual framework connecting emotion to execution
- Temporal physics applied to customer emotion
- Unified contract system making insights reusable across enterprise
- Strategic positioning matrix for tactical decision-making
- Predictive capability for proactive intervention
- Integration of measurement, strategy, and foresight into unified system

**The Gap:**
CX teams had the math and data but lacked the systematic approach to convert human 
emotion into operationally routable, statistically rigorous, strategically actionable, 
and predictively intelligent infrastructure.

### The Complete Execution and Prediction Engine

This architecture doesn't just measure customer emotion or plan responses—it orchestrates 
the entire emotional intelligence lifecycle:

- **Experience Drivers** become atomic units of emotional orchestration
- **Temporal analysis** provides comprehensive understanding of emotional dynamics
- **Strategic positioning** delivers specific actions with ownership assignment
- **Priority scoring** ensures optimal resource allocation
- **Predictive intelligence** enables proactive intervention timing
- **Contract system** ensures consistency across all enterprise touchpoints

### Business Value Progression

**Traditional CX**: "Customer satisfaction dropped this month"
**Five-Function Suite**: "Delivery complaints show strong downward momentum requiring 
immediate Operations team intervention"
**Seven-Function Suite**: "Delivery complaints (QSSI 8) will likely resurface next 
Tuesday based on weekly pattern - deploy preventive resources by Friday for 60% impact reduction"

### The Result

Customer emotion becomes infrastructure that enterprises can systematically understand, 
predict, and act upon with the same rigor as financial or operational metrics. 
The seven-function architecture provides:

- **Complete temporal understanding** of emotional dynamics
- **Clear operational priorities** for resource allocation
- **Strategic positioning** for intervention approaches
- **Predictive timing** for maximum intervention impact
- **Unified framework** for enterprise-wide emotional intelligence

This transforms customer experience from a reactive analytics function into a 
predictive competitive advantage.

-------------------------------------------------------------------------------------------

OLD QSSI CODE

    def _compute_signal_strength(
        self,
        trend: str,
        momentum: str,
        saturation_index: float,
        trend_pct: float = None,          # optional: % of ERI band from trend module
        momentum_delta: float = None,     # optional: % of ERI band from momentum module
        n_days: int = None,               # optional: series length
        vol_norm: float = None,           # optional: volatility normalized to [0,1] (use adjusted vol!)
        *,
        # NEW (optional): data quality/context
        signal_presence_pct: float | None = None,  # e.g., 0.27 means 27% of days had signal
        momentum_snr: float | None = None,         # pass self._compute_momentum_series(...).get("snr")
        series_data_complete: bool | None = None   # from provenance
    ) -> dict:
        # --- sanitize inputs ---
        sym_trend = trend if trend in ("↑", "→", "↓") else "→"
        # validate momentum symbol against contract keys (no hardcoded set)
        sym_mom = momentum if momentum in self.momentum_contract().keys() else "→"
        try:
            si = float(saturation_index)
        except Exception:
            si = 0.5
        si = max(0.0, min(1.0, si))  # clamp

        # --- single-source thresholds & gates ---
        t = self.momentum_thresholds()
        snr_mod   = float(t.get("snr_mod", 0.75))
        snr_strong= float(t.get("snr_strong", 1.25))
        presence_full_credit = float(t.get("presence_full_credit", 0.40))
        # optional bump thresholds (safe defaults preserve current behavior)
        trend_bump_thresh = float(t.get("qssi_trend_bump", 10.0))
        mom_bump_thresh   = float(t.get("qssi_mom_bump", 20.0))

        # --- canonical contracts ---
        sat = self.sat_from_si(si)          # {'emoji','clean','headroom','qssi_score','si'}
        mom = self.mom_details(sym_mom)     # {'label','description','emoji_full','strength',...}
        mom_strength = float(mom.get("strength", 0.3))  # 1.0/0.6/0.3/...

        # --- velocity score (symbol-only base) ---
        if sym_trend in ("↑","↓") and sym_mom == "↓↓":
            velocity_base, rationale = 6, "Sharp trend shift with strong counter-momentum"
        elif sym_trend in ("↑","↓") and sym_mom == "↑↑":
            velocity_base, rationale = 5, "Rapid acceleration with positive momentum surge"
        elif sym_trend in ("↑","↓") and sym_mom in ("↑","↓"):
            velocity_base, rationale = 4, "Moderate directional movement with matching momentum"
        elif sym_trend in ("↑","↓") and sym_mom == "→":
            velocity_base, rationale = 2, "Directional change but stable momentum"
        elif sym_trend == "→" and sym_mom in ("↑↑","↓↓"):
            velocity_base, rationale = 3, "Flat trend with sudden strong shift emerging"
        elif sym_trend == "→" and sym_mom in ("↑","↓"):
            velocity_base, rationale = 1, "Stable trend with mild fluctuation"
        else:
            velocity_base, rationale = 0, "No meaningful trend or momentum detected"

        # --- micro nudges (magnitude-aware; thresholds from config) ---
        bump = 0
        if trend_pct is not None and abs(float(trend_pct)) >= trend_bump_thresh:
            bump += 1
        if momentum_delta is not None and abs(float(momentum_delta)) >= mom_bump_thresh:
            bump += 1

        # --- quality factors (presence, stability, SNR) ---
        # presence: full credit at >= presence_full_credit; floor to 0.3 to avoid zeroing
        if signal_presence_pct is None:
            presence_factor = 1.0
        else:
            presence = float(signal_presence_pct)
            presence_factor = max(0.3, min(1.0, presence / presence_full_credit))

        # stability: 1 - vol_norm (if provided)
        if vol_norm is None:
            stability_factor = 1.0
        else:
            stability_factor = max(0.0, min(1.0, 1.0 - float(vol_norm)))

        # momentum SNR: scale to 1.0 at strong gate; floor at 0.5
        if momentum_snr is None:
            snr_factor = 1.0
        else:
            snr = float(momentum_snr)
            snr_factor = max(0.5, min(1.0, snr / snr_strong))

        # length/volatility penalties (unchanged heuristics)
        penalty = 0
        if n_days is not None and int(n_days) < 10:
            penalty += 1
        if vol_norm is not None and float(vol_norm) > 0.8:
            penalty += 1

        # combine base + bumps/penalties (0..6), then scale by data-aware factors and momentum strength
        velocity_symbolic = max(0, min(6, velocity_base + bump - penalty))
        velocity_scaled = (
            velocity_symbolic
            * (0.7 + 0.3 * mom_strength)   # decisiveness boost
            * presence_factor
            * stability_factor
            * snr_factor
        )
        velocity_score = int(round(max(0.0, min(6.0, velocity_scaled))))  # 0..6

        # --- headroom → saturation score (via contract) ---
        saturation_score = int(sat["qssi_score"])  # 0..4 (Champion..At-Risk reversed)
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

        # dominant driver for explainability
        dominant_driver = "velocity" if velocity_score >= saturation_score else "headroom"

        return {
            "velocity_component": {
                "trend_symbol": sym_trend,
                "momentum_symbol": sym_mom,
                "velocity_score": int(velocity_score),
                "velocity_rationale": rationale,
                "nudges": {"bump": int(bump), "penalty": int(penalty)} if (bump or penalty) else None
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
                "qssi_interpretation": interp,
                "qssi_strength_norm": round(qssi / 10.0, 3),   # 0..1 for PEM consumption
                "dominant_driver": dominant_driver
            },
            "audit": {
                "headroom": sat["headroom"],
                "saturation_tier_emoji": sat["emoji"],
                "momentum_label": mom["label"],
                **({"n_days": int(n_days)} if n_days is not None else {}),
                **({"vol_norm": float(vol_norm)} if vol_norm is not None else {}),
                "factors": {
                    "presence_factor": None if signal_presence_pct is None else round(presence_factor, 3),
                    "stability_factor": None if vol_norm is None else round(stability_factor, 3),
                    "snr_factor": None if momentum_snr is None else round(snr_factor, 3),
                    "mom_strength": round(mom_strength, 3),
                    "velocity_symbolic": int(velocity_symbolic)
                },
                "thresholds_used": {
                    "snr_mod": snr_mod, "snr_strong": snr_strong,
                    "presence_full_credit": presence_full_credit,
                    "qssi_trend_bump": trend_bump_thresh,
                    "qssi_mom_bump": mom_bump_thresh,
                }
            }
        }


signal_strength_block = self._compute_signal_strength(
                    trend_symbol,
                    momentum_symbol,
                    quadrant_block["signal_classification"]["saturation_index"],
                    trend_pct=trend_pct,
                    momentum_delta=momentum_delta,
                    n_days=num_days_observed,
                    vol_norm=vol_norm,
                    signal_presence_pct=pct_days_with_signal,
                    momentum_snr=momentum_snr,
                    series_data_complete=series_data_complete
                )

-----------------------------------------------------------------------------------------

OLD PEM CODE
def build_predictive_emotional_modeling(
    self,
    state_of_play,
    volatility,                 # % of ERI band (0..100) — pass adjusted % when series incomplete
    has_pattern,
    qssi_tier,                  # can be echoed to output, not used for logic
    *,
    signal_presence_pct: float | None = None,   # e.g., 0.27
    momentum_symbol: str | None = None,         # "↑↑","↑","→","↓","↓↓"
    pattern_confidence: str | None = None,      # "Weak"|"Moderate"|"Strong"|None
    pattern_type: str | None = None,            # "Weekly"|"Monthly"|"Quarterly"|None
    pattern_strength: float | None = None,      # raw ACF strength (e.g., 0.29)
    pain_day: str | None = None                 # e.g., "Tuesday"
) -> dict:
"""
        
"""
        PEM (Predictive Emotional Modeling) — contract-respecting.
        - Momentum strength/polarity from momentum_contract() via mom_details()
        - Saturation headroom/qssi_score from saturation_contract() (by tier)
        - QSSI: consume numeric qssi_score from Layer 4 (NO tier mapping here)
"""
"""

        # --- config/gates (single source) ---
        t = self.momentum_thresholds()
        presence_full_credit = float(t.get("presence_full_credit", 0.40))

        # --- unpack state ---
        mom_label = (state_of_play.get("momentum_tier") or "Stable").strip()
        sat_tier_clean = (state_of_play.get("saturation_tier") or "Medium").strip()
        traj      = state_of_play.get("trajectory_story")
        guidance  = state_of_play.get("action_guidance")
        owner     = state_of_play.get("recommended_owner")

        try:
            vol_pct = max(0.0, float(volatility))
        except Exception:
            vol_pct = 0.0
        vol_norm   = min(vol_pct / 20.0, 1.0)   # 20% of ERI band ~ "high"
        stability  = 1.0 - vol_norm

        # --- QSSI strength (numeric only) ---
        qssi_score = (
            state_of_play.get("qssi_score",
                state_of_play.get("qssi_summary", {}).get("qssi_score"))
        )
        qssi_strength = float(qssi_score) / 10.0 if qssi_score is not None else 0.0  # 0..1

        # --- Momentum (helpers only) ---
        sym = momentum_symbol or self.mom_symbol_from_label(mom_label or "")
        meta = self.mom_details(sym)                 # falls back to '→' inside
        strength = float(meta.get("strength", 0.3))
        polarity = int(meta.get("polarity", 0))      # -1, 0, 1
        dir_score = polarity * strength              # ↑↑=+1.0, ↑=+0.6, →=0.0, ↓=-0.6, ↓↓=-1.0

        # --- Saturation via contract (derive headroom scalar from bin qssi_score) ---
        qssi_score_for_tier = None
        headroom_dict = None
        for th, (emoji, clean), headroom, qssi_sc in self.saturation_contract():
            if clean == sat_tier_clean:
                qssi_score_for_tier = int(qssi_sc)
                headroom_dict = headroom
                break
        if qssi_score_for_tier is None:
            tmp = self.sat_from_si(0.5)
            qssi_score_for_tier = int(tmp["qssi_score"])
            headroom_dict = tmp["headroom"]

        # numeric headroom scalar aligned to contract bins (0 Champion .. 4 At-Risk)
        headroom_scalar = qssi_score_for_tier / 4.0

        # --- Pattern strength (normalized 0..1) ---
        pat_conf = (pattern_confidence or "").lower()
        if pattern_strength is not None:
            ps = max(0.0, min(1.0, float(pattern_strength)))
            pat_strength = ps if ps > 0.6 else min(1.0, ps / 0.6)  # gentle lift for ACF-like inputs
        else:
            pat_strength = 1.0 if pat_conf == "strong" else 0.6 if pat_conf == "moderate" else 0.3 if pat_conf == "weak" else 0.0

        has_pat = bool(has_pattern)
        pat_type = (pattern_type or "").lower()

        # --- Directionalized raw scores (pattern-aware) ---
        esc_raw = (
            0.35 * qssi_strength * max(0.0,  dir_score) +
            0.35 * max(0.0,  dir_score) +
            0.30 * (1.0 - headroom_scalar) +  # more headroom => more upside
            0.25 * stability
        )
        dec_raw = (
            0.35 * qssi_strength * max(0.0, -dir_score) +
            0.55 * max(0.0, -dir_score) +
            0.30 * headroom_scalar +          # closer to At-Risk => more decay propensity
            0.20 * vol_norm
        )

        # pattern-direction fusion (gentle)
        if has_pat and pat_strength > 0:
            boost = 0.10 * pat_strength
            if dir_score > 0:
                esc_raw *= (1.0 + boost)
            elif dir_score < 0:
                dec_raw *= (1.0 + boost)
            else:
                esc_raw *= (1.0 + 0.03 * pat_strength)
                dec_raw *= (1.0 + 0.03 * pat_strength)

        esc_raw = max(0.0, esc_raw)
        dec_raw = max(0.0, dec_raw)
        total   = esc_raw + dec_raw
        esc_prob = (esc_raw / total) if total > 1e-9 else 0.0
        dec_prob = (dec_raw / total) if total > 1e-9 else 0.0

        # --- Coverage & stability caps (data-aware; no tier mapping) ---
        if signal_presence_pct is None:
            presence_factor = 1.0
        else:
            presence = float(signal_presence_pct)
            presence_factor = max(0.3, min(1.0, presence / presence_full_credit))

        # score-based cap (0.50..0.95 from qssi_score), then quality factors
        score_cap = 0.5 + ((float(qssi_score) / 10.0) * 0.45) if qssi_score is not None else 0.5
        momentum_factor = 0.85 if sym == "→" else 1.0
        quality_cap = score_cap * presence_factor * (0.8 + 0.2 * stability) * momentum_factor
        quality_cap = float(min(max(0.5, quality_cap), 0.97))

        if esc_prob >= dec_prob:
            esc_prob = min(esc_prob, quality_cap)
        else:
            dec_prob = min(dec_prob, quality_cap)

        # --- Pattern-aware horizon ---
        base_horizon = int(getattr(self, "pem_horizon_days", 14))
        if has_pat:
            if   pat_type == "weekly":    horizon_days = 14
            elif pat_type == "monthly":   horizon_days = 30
            elif pat_type == "quarterly": horizon_days = 90
            else:                         horizon_days = base_horizon
        else:
            horizon_days = base_horizon

        # --- Rules-first overrides (use numeric qssi_score, not tier strings) ---
        strong_qssi = (qssi_score is not None and qssi_score >= 6)   # ≥6 = Strong/Critical band
        signal_is_precursor = (dir_score > 0 and strong_qssi and has_pat)
        signal_is_at_risk   = (dir_score < 0 and (vol_pct >= 5.0 or has_pat))

        if signal_is_precursor:
            pem_trajectory  = "Likely Escalation"
            pem_probability = max(esc_prob, 0.65)
            risk_class      = "Opportunity"
            explanation     = "Rising momentum with repeating pattern indicates intensification in the next cycle"
            rule_trigger, basis = "rising+strong_qssi+pattern", "esc_prob"
            counterfactual_pointer = "Flip if momentum ≤ 'Moderately Falling' or volatility > 22.5%"
        elif signal_is_at_risk:
            pem_trajectory  = "Recurring Decay Risk" if has_pat else "At Risk of Decay"
            pem_probability = max(dec_prob, 0.65)
            risk_class      = "Risk"
            explanation     = (
                f"Falling momentum{', recurring ' + (pattern_type or '').lower() + ' dip' if has_pat else ''} "
                f"suggests emotional energy is fading; pre-empt before the next window"
            )
            rule_trigger, basis = ("falling+volatility" if vol_pct >= 5 else "falling+pattern"), "dec_prob"
            counterfactual_pointer = (
                f"Flip if momentum ≥ 'Moderately Rising' and volatility < 5%; "
                f"also remove trigger before next { (pattern_type or 'cycle').lower() }"
            )
        else:
            if max(esc_prob, dec_prob) < 0.55:
                pem_trajectory  = ("Recurring Risk – " + (pattern_type or "Cycle")) if has_pat else "Stable / Inconclusive"
                pem_probability = max(esc_prob, dec_prob)
                risk_class      = "Risk" if has_pat else "Neutral"
                if has_pat:
                    day_hint = f" (pain day: {pain_day})" if pain_day else ""
                    explanation = f"Detected a repeating { (pattern_type or '').lower() } signal{day_hint}; monitor and intervene ahead of the next window"
                else:
                    explanation = "No dominant predictive anchors; continue monitoring"
            else:
                if esc_prob > dec_prob:
                    pem_trajectory, pem_probability = "Likely Escalation", esc_prob
                    risk_class = "Opportunity"
                    explanation = ("Upward trajectory dominance; headroom and stability support uplift"
                                if sym == "→" else
                                "Upward trajectory dominance (momentum/headroom/stability blend)")
                else:
                    pem_trajectory, pem_probability = ("Recurring Decay Risk" if has_pat else "At Risk of Decay"), dec_prob
                    risk_class = "Risk"
                    explanation = ("Downward trajectory dominance; " +
                                ("recurring pattern amplifies risk" if has_pat else "volatility/ceiling pressure drive risk"))
            rule_trigger = "model_choice"
            basis = "esc_prob" if esc_prob >= dec_prob else "dec_prob"
            counterfactual_pointer = (
                f"Flip if next horizon favors the opposite momentum tier; "
                f"pre-empt before next { (pattern_type or 'cycle').lower() }" if has_pat else
                "Flip if next horizon favors the opposite momentum tier"
            )

        # --- Confidence tier (pattern & volatility & presence) ---
        if has_pat and vol_pct >= 5.0:
            confidence_tier, confidence_score = "High", 0.85
        elif has_pat or vol_pct >= 5.0:
            confidence_tier, confidence_score = "Moderate", 0.55
        else:
            confidence_tier, confidence_score = "Low", 0.30
        confidence_score = float(confidence_score * (0.8 + 0.2 * presence_factor))
        confidence_score = round(min(max(confidence_score, 0.2), 0.95), 2)

        # --- Elasticity (derive moderate threshold from contract, not 0.6) ---
        moderate_strength = float(self.mom_details("↑").get("strength", 0.6))
        base_elasticity = (
            "High"     if (1.0 - headroom_scalar) >= 0.5 and abs(dir_score) >= moderate_strength else
            "Moderate" if (1.0 - headroom_scalar) >= 0.25 else
            "Low"
        )
        elasticity_rating = "High" if (has_pat and pat_strength >= 0.6 and base_elasticity == "Moderate") else base_elasticity

        version = "PEM.v1.7"

        feature_vector = {
            "qssi_score": int(qssi_score) if qssi_score is not None else None,
            "qssi_strength": round(qssi_strength, 3),
            "momentum_symbol": sym,
            "momentum_score": round(dir_score, 3),
            "headroom_scalar": round(headroom_scalar, 3),
            "volatility_pct": round(vol_pct, 3),
            "volatility_norm": round(vol_norm, 3),
            "stability": round(stability, 3),
            "has_pattern": has_pat,
            "pattern_confidence": (pattern_confidence or None),
            "pattern_type": (pattern_type or None),
            "pattern_strength": round(float(pattern_strength), 3) if pattern_strength is not None else None,
            "esc_raw": round(esc_raw, 3),
            "dec_raw": round(dec_raw, 3),
            "esc_prob": round(esc_prob, 3),
            "dec_prob": round(dec_prob, 3),
            "gates_used": {"presence_full_credit": presence_full_credit},
        }

        return {
            "trajectory_forecast": {
                "pem_trajectory": pem_trajectory,
                "pem_probability": round(float(pem_probability), 3),
                "pem_confidence": confidence_tier,
                "confidence_score": confidence_score,
                "risk_class": risk_class,
                "horizon_days": int(horizon_days),
                "explanation": (
                    f"{explanation} "
                    f"{'(pain day: ' + pain_day + ')' if (has_pat and pain_day) else ''}"
                ).strip(),
                "version": version,
                "rule_trigger": rule_trigger,
                "basis": basis,
                "counterfactual_pointer": counterfactual_pointer
            },
            "signal_diagnostics": {
                "has_repeating_pattern": has_pat,
                "pattern_type": (pattern_type or None),
                "pattern_confidence": (pattern_confidence or None),
                "pattern_pain_day": (pain_day or None),
                "volatility_pct": round(vol_pct, 2),
                "momentum_tier": mom_label,
                "saturation_tier": sat_tier_clean,
                "qssi_tier": qssi_tier,           # echo only; not used for logic
                "headroom_contract": headroom_dict,
            },
            "future_risk_profile": {
                "trajectory_story": traj,
                "action_guidance": guidance,
                "recommended_owner": owner
            },
            "audit": {
                "feature_vector": feature_vector,
                "elasticity_rating": elasticity_rating,
                "consistency": {
                    "momentum_direction": "up" if dir_score > 0 else "down" if dir_score < 0 else "flat",
                    "pattern_supports_escalation": bool(has_pat and dir_score > 0),
                    "volatility_pressure": "high" if vol_norm >= 0.8 else "moderate" if vol_norm >= 0.4 else "low",
                }
            }
        }


pem_vol_pct = vol_for_norm_pct  # disciplined input to PEM
                pem_block = self.build_predictive_emotional_modeling(
                    state_of_play={
                        "momentum_tier": quadrant_block["signal_classification"]["momentum_tier"],
                        "saturation_tier": quadrant_block["signal_classification"]["saturation_tier"],
                        "trajectory_story": quadrant_block["actionable_strategy"]["trajectory_story"],
                        "action_guidance": quadrant_block["actionable_strategy"]["action_guidance"],
                        "recommended_owner": quadrant_block["actionable_strategy"]["recommended_owner"]
                    },
                    volatility=pem_vol_pct,
                    has_pattern=bool(pattern_block["has_pattern"]),
                    qssi_tier=signal_strength_block["qssi_summary"]["qssi_tier"],
                    signal_presence_pct=pct_days_with_signal,
                    momentum_symbol=momentum_symbol,
                    pattern_confidence=pattern_block.get("pattern_confidence")
                )
        
"""