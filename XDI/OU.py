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

        # 🗺️ Canonical feedback types (concise)
        self.FT_MAP = {
            "compliment": "Compliment",
            "complaint": "Complaint",
            "question": "Question",
            "suggestion": "Suggestion",
            "request": "Request",
            **{k: "Product Usage Insight" for k in ("usage insight", "product usage insight")},
            **{k: "Emerging Trends / Market Insight" for k in ("emerging trends", "market insight", "emerging trends / market insight")},
        }

        # Derived valid set (no need to maintain separately)
        self.VALID_FEEDBACK_TYPES = set(self.FT_MAP.values())
 
        self.STREAM_MAP = {
            "fix": "Fix",
            "optimize": "Optimize",
            "optimise": "Optimize",   # UK variant
            "amplify": "Amplify",
            "innovate": "Innovate",
        }
        
        # Helpful enums for validation/logging
        self.VALID_STREAMS = {"Fix", "Optimize", "Amplify", "Innovate"}
        
        # Emotion scoring config (for ERI etc.)
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
                "pct": 0.08,   # 8% of rows in this ED→Type→Stream slice
                "floor": 3     # never below 3
            },

            "bcs_cumu_threshold": 0.80,
            "stream_threshold": 0.80,
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
            "semantic_customer_reality","matters","stream_justification","behavioral_impact"
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
                "stream_justification": 1,
                "customer_journey": 1,
            },
            "fallback_K3": {
                "matters": 6,
                "context": 4,
                "semantic_customer_reality": 3,
                "interaction_moment": 3,
                "customer_journey_stage": 2,
                "behavioral_impact": 2,
                "stream_justification": 1,
                "customer_journey": 1,
            },
        }

        # ✅ Default choice (and set starting τ accordingly)
        self.OU_CFG["signature_config_name"] = "default_K1"  # echo in OU meta for reproducibility
        self.OU_CFG["signature_weights"] = self.SIGNATURE_LIBRARY[self.OU_CFG["signature_config_name"]]
        self.OU_CFG["bcs_distance_threshold"] = self.OU_CFG["signature_threshold_start"][self.OU_CFG["signature_config_name"]]

        # (optional but nice): make intent explicit — we only use the first half of SAS
        self.OU_CFG["semantic_statement_mode"] = "customer_reality_only"


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
        focus_df = self.compute_feedbacktype_focus()
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

    # ---------- feedback type + stream distributions ----------
    def _canon_ft(self, s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.lower().map(self.FT_MAP)

    def _canon_stream(self, s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.lower().map(self.STREAM_MAP)
  
    def compute_feedbacktype_focus(self) -> pd.DataFrame:
        """
        Per-ED feedback type focus:
        - count & sort FTs by mentions,
        - take cumulatively until >= 0.80,
        - also return full distribution (percent, 1 dp).
        Produces: feedbacktype_audit_focus, feedback_type_distribution.
        """
        if self.layer3_df is None or self.layer3_df.empty:
            raise ValueError("Layer 3 diagnostics must be supplied and non-empty.")

        threshold = float(getattr(self, "OU_CFG", {}).get("feedbacktype_threshold", 0.80))
        threshold = max(0.0, min(1.0, threshold))

        df = self.raw_df.copy()
        df["experience_driver"] = df["experience_driver"].astype(str).str.strip()
        valid_ed = set(self.layer3_df["experience_driver"].astype(str).str.strip().unique())
        df = df[df["experience_driver"].isin(valid_ed)]

        # canonicalize once, drop unmapped
        df["feedback_type"] = self._canon_ft(df["feedback_type"])
        df = df.dropna(subset=["feedback_type"])

        def _per_ed(g: pd.DataFrame) -> pd.Series:
            counts = (
                g["feedback_type"].value_counts()
                .sort_values(ascending=False)               # most mentions first
                .sort_index(kind="mergesort")               # stable tie-break by label
            )
            total = int(counts.sum())
            if total == 0:
                return pd.Series({
                    "feedbacktype_audit_focus": [],
                    "feedback_type_distribution": {}
                })

            dominant, cumulative = [], 0.0
            for ft, cnt in counts.items():
                pct = cnt / total
                dominant.append(ft)
                cumulative += pct
                if cumulative >= threshold:
                    break

            distribution = {ft: round((cnt / total) * 100, 1) for ft, cnt in counts.items()}
            return pd.Series({
                "feedbacktype_audit_focus": dominant,
                "feedback_type_distribution": distribution
            })

        out = (
            df.groupby("experience_driver", sort=True, as_index=True)
            .apply(_per_ed)
            .reset_index()
        )
        return out

    # --- weighted signature builder used by cluster_behavior ---
    def _build_signature(self, row: pd.Series, sep: str = " | ") -> str:
        # pull config once
        weights = (self.OU_CFG.get("signature_weights") or {})
        allowed = getattr(self, "ALLOWED_FOR_SIGNATURE", frozenset())

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

        # deterministic order: weight desc, then field name
        fields = sorted(
            ((k, int(w)) for k, w in weights.items() if int(w) > 0 and k in allowed),
            key=lambda kv: (-kv[1], kv[0])
        )

        parts = []
        for col, w in fields:
            text = norm(to_text(row.get(col, "")))
            if text:
                parts.extend([text] * int(w))
        return sep.join(parts)

    # === EMBEDDINGS (numpy, normalized) ===
    def _encode_np(self, texts):
        model = self._ensure_st_model()
        # numpy + L2 normalized → stable cosine everywhere
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def _centroid_pick(self, texts):
        clean = [t for t in (texts or []) if isinstance(t, str) and t.strip()]
        if not clean:
            return None
        E = self._encode_np(clean)  # assume L2-normalized embeddings
        c = E.mean(axis=0, keepdims=True)
        sims = cosine_similarity(E, c).ravel()
        return clean[int(sims.argmax())]

    def _mmr_summary(self, texts, top_k=4, diversity=0.7):
        top_k = max(1, int(top_k))
        diversity = max(0.0, min(1.0, float(diversity)))

        splitter = re.compile(r'(?<=[.!?])\s+(?=[A-Za-z0-9])')
        sents = []
        for t in (texts or []):
            if not isinstance(t, str):
                continue
            t = re.sub(r'\s+', ' ', t.strip())
            # collect long-ish sentences if available
            cand = [s for s in splitter.split(t) if s.strip()]
            long = [s for s in cand if len(s.strip()) >= 30]
            sents.extend(long if long else cand)

        if not sents:
            raw = " ".join([(t or "").strip() for t in (texts or []) if isinstance(t, str)])[:300]
            return raw or None

        # de-duplicate to stabilize selection
        sents = list(dict.fromkeys(sents))

        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=10000)
        X = vec.fit_transform(sents)
        centroid = centroid = np.asarray(X.mean(axis=0))

        # relevance to centroid
        rel = cosine_similarity(X, centroid).ravel()

        selected, cand_idx = [], list(range(len(sents)))
        while cand_idx and len(selected) < top_k:
            if not selected:
                i = int(rel.argmax()); selected.append(i); cand_idx.remove(i); continue
            sims = cosine_similarity(X[cand_idx], X[selected]).max(axis=1)
            scores = diversity * rel[cand_idx] - (1 - diversity) * sims
            j = int(scores.argmax()); sel = cand_idx[j]
            selected.append(sel); cand_idx.remove(sel)

        selected.sort()
        return " ".join(sents[i] for i in selected)

    def _distill_matters_label(self, matters_list):
        if not matters_list:
            return "No matters label available"
        clean = [str(m).strip() for m in matters_list if pd.notna(m) and str(m).strip()]
        if not clean:
            return "No matters label available"
        return self._centroid_pick(clean) or "No matters label available"

    def process_batch_2_fields(self, cluster_df: pd.DataFrame, batch_1_fields: Dict[str, Any]) -> Dict[str, Any]:
        batch_2: Dict[str, Any] = {}

        # Context (deterministic consolidation)
        batch_2["context"] = self._semantic_centroid_fusion(cluster_df.get("context", pd.Series(dtype=object)).tolist())

        # Keywords (normalize + dedupe)
        batch_2["keywords"] = self._dedupe_and_merge_keywords(cluster_df.get("keywords", pd.Series(dtype=object)).tolist())

        # Locator (pick ONE Journey/Stage/Moment combo)
        journey, stage, moment, _ = self._collapse_locator_composite(cluster_df)
        batch_2["customer_journey"] = journey
        batch_2["customer_journey_stage"] = stage
        batch_2["interaction_moment"] = moment

        # CES → use existing helper exactly
        batch_2["customer_effort_score"] = self._weighted_average_effort_score(
            cluster_df.get("customer_effort_score", pd.Series(dtype=float)).tolist()
        )

        # Entity name → use existing helper exactly
        batch_2["entity_name"] = self._extract_entity_name(
            cluster_df.get("entity_name", pd.Series(dtype=object)).tolist()
        )

        return batch_2

    def extract_batch_1_fields(self, cluster_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Batch-1 fixed fields for an OU cluster (subset is already filtered by ED → Feedback Type → Stream).
        - experience_driver / feedback_type / opportunity_stream: taken directly from the subset
        - theme: collapsed deterministically (mode → semantic tiebreak)
        - emotion: collapsed (primary + specific) + full distribution
        """
        composite: Dict[str, Any] = {}

        # fixed by construction for this subset
        composite["experience_driver"]  = str(cluster_df.get("experience_driver", pd.Series(["Unknown"])).iloc[0]).strip()
        composite["feedback_type"]      = str(cluster_df.get("feedback_type", pd.Series(["Unknown"])).iloc[0]).strip()
        composite["opportunity_stream"] = str(cluster_df.get("opportunity_stream", pd.Series(["Unknown"])).iloc[0]).strip()

        # theme collapse (mode → semantic tiebreak); fallback "Unknown"
        composite["theme"] = self._collapse_theme(cluster_df)

        # emotion collapse (primary + specific) + distribution
        try:
            prim, spec, dist = self.collapse_emotion(cluster_df)
        except Exception:
            prim, spec, dist = None, None, {}

        composite["emotion_primary"]      = (prim or "Unknown")
        composite["emotion_specific"]     = (spec or None)
        composite["emotion_distribution"] = dist

        return composite
   
    def _semantic_centroid_fusion(self, context_list: list[str]) -> str:
        """
        Deterministic consolidation:
        - If few unique values (≤3): use mode; if tied, semantic tiebreak via centroid.
        - Otherwise: centroid-pick (numpy cosine).
        """
        if not context_list:
            return "No context available"

        # normalize + filter empties
        clean = [str(c).strip() for c in context_list if pd.notna(c) and str(c).strip()]
        if not clean:
            return "No context available"

        vc = pd.Series(clean).value_counts()

        # few uniques → mode with semantic tiebreak on ties
        if len(vc) <= 3:
            top_count = vc.iloc[0]
            tied_vals = vc[vc == top_count].index.tolist()
            if len(tied_vals) == 1:
                return tied_vals[0]
            # tie → semantic pick (fallback to first)
            try:
                pick = self._centroid_pick(tied_vals)
            except Exception:
                pick = None
            return pick or tied_vals[0]

        # many uniques → semantic representative over all
        try:
            pick = self._centroid_pick(clean)
        except Exception:
            pick = None
        return pick or "No context available"
 
    def _collapse_theme(self, grp: pd.DataFrame) -> str:
        col = grp.get("theme", pd.Series(dtype=object))
        vals = [str(v).strip() for v in col if pd.notna(v) and str(v).strip()]
        if not vals:
            return "Unknown"

        vc = pd.Series(vals).value_counts()
        # clear winner
        if len(vc) == 1 or (len(vc) > 1 and vc.iloc[0] > vc.iloc[1]):
            return vc.index[0]

        # tie-break semantically on the tied labels (LLM-free, centroid pick)
        tied = vc[vc == vc.iloc[0]].index.tolist()
        if hasattr(self, "_centroid_pick"):
            pick = self._centroid_pick(tied)
            return pick or tied[0]
        return tied[0]
    
    def collapse_emotion(self, grp: pd.DataFrame):
        EMO_SEVERITY = {"anger": 5, "agitation": 4, "ambivalence": 3, "appreciation": 2, "adoration": 1}
        prim = grp.get("emotion_primary", pd.Series(dtype=object)).astype(str).str.lower().str.strip()
        if prim.empty:
            return None, None, {}
        dist = (prim.value_counts(normalize=True) * 100).round(1).to_dict()

        vc = prim.value_counts()
        top = vc.index[0]
        ties = vc[vc == vc.iloc[0]].index.tolist()
        if len(ties) > 1:
            ces = grp.get("customer_effort_score", pd.Series(dtype=float))
            ces_means = {e: float(ces[prim == e].mean()) for e in ties}
            top = max(ties, key=lambda e: (ces_means.get(e, -1), EMO_SEVERITY.get(e, 0)))

        spec = grp.loc[prim == top, "emotion_specific"].astype(str).str.strip()
        if spec.empty:
            return top, None, dist

        sv = spec.value_counts()
        mode = sv.index[0]
        if len(sv) > 1 and sv.iloc[0] == sv.iloc[1]:
            mode = self._centroid_pick(spec.tolist()) or mode
        return top, mode, dist
 
    def _dedupe_and_merge_keywords(self, keywords_list):
        all_kw = []
        for kw in keywords_list:
            # robust null guard
            if kw is None or (isinstance(kw, float) and pd.isna(kw)):
                continue

            # normalize into a sequence of tokens
            if isinstance(kw, (list, tuple, set)):
                seq = kw
            elif isinstance(kw, str):
                s = kw.strip()
                if s.startswith('[') and s.endswith(']'):
                    s = s[1:-1].replace("'", "").replace('"', '')
                seq = [k for k in s.split(',')]
            else:
                seq = [str(kw)]

            # clean & collect
            for k in seq:
                tok = str(k).strip().lower()
                if tok and len(tok) > 1:
                    all_kw.append(tok)

        counts = Counter(all_kw)
        return [k for k, _ in counts.most_common()]

    def _norm_whitespace(self, s: str) -> str:
        return " ".join(str(s or "").strip().split())

    def _collapse_locator_composite(self, grp: pd.DataFrame):
        """
        Pick ONE (journey, stage, moment) combo that actually appears in the cluster.
        No aliases, no vocab. Deterministic:
        1) build normalized composite keys
        2) frequency (mode)
        3) tie-break via _centroid_pick on the composite strings
        4) display using the most frequent original variants for the chosen key
        Returns: (journey, stage, moment, distribution_dict)
        """
        # pull raw columns (as strings)
        j_raw = grp.get("customer_journey",        pd.Series(dtype=object)).astype(str).tolist()
        s_raw = grp.get("customer_journey_stage",  pd.Series(dtype=object)).astype(str).tolist()
        m_raw = grp.get("interaction_moment",      pd.Series(dtype=object)).astype(str).tolist()

        # build (normalized) composite keys and keep a map to original variants
        key_counts: Counter[tuple[str,str,str]] = Counter()
        key_variants: dict[tuple[str,str,str], list[tuple[str,str,str]]] = {}

        for j, s, m in zip(j_raw, s_raw, m_raw):
            jn = self._norm_whitespace(j).lower()
            sn = self._norm_whitespace(s).lower()
            mn = self._norm_whitespace(m).lower()
            if not (jn or sn or mn):
                continue
            key = (jn, sn, mn)
            key_counts[key] += 1
            key_variants.setdefault(key, []).append((
                self._norm_whitespace(j),  # keep original casing for display
                self._norm_whitespace(s),
                self._norm_whitespace(m),
            ))

        if not key_counts:
            return "Unknown", "Unknown", "Unknown", {}

        # mode by frequency
        top_key, top_cnt = key_counts.most_common(1)[0]
        tied_keys = [k for k, c in key_counts.items() if c == top_cnt]

        if len(tied_keys) == 1:
            chosen_key = top_key
        else:
            # tie-break: semantic representative on the composite strings (no hardcoded maps)
            def key_to_string(k: tuple[str,str,str]) -> str:
                # use normalized for stable comparison; display form comes later
                return " | ".join([p for p in k if p])

            tie_strs = [key_to_string(k) for k in tied_keys]
            picked_str = self._centroid_pick(tie_strs) or tie_strs[0]
            # resolve back to key
            chosen_idx = tie_strs.index(picked_str)
            chosen_key = tied_keys[chosen_idx]

        # build % distribution for audit (by normalized key)
        total = sum(key_counts.values())
        dist = {" | ".join([p for p in k if p]): round(100.0 * c / total, 1)
                for k, c in key_counts.items()}

        # choose display variants for the chosen key: per-component MOST frequent original
        variants = key_variants[chosen_key]
        j_disp = Counter([v[0] for v in variants if v[0]]).most_common(1)
        s_disp = Counter([v[1] for v in variants if v[1]]).most_common(1)
        m_disp = Counter([v[2] for v in variants if v[2]]).most_common(1)

        journey  = j_disp[0][0] if j_disp else "Unknown"
        stage    = s_disp[0][0] if s_disp else "Unknown"
        moment   = m_disp[0][0] if m_disp else "Unknown"

        return journey, stage, moment, dist

    def _weighted_average_effort_score(self, scores_list: list[Any], *, 
                                   method: str = "mean", clamp: tuple[float,float] = (1,7)) -> int:
        import statistics
        lo, hi = clamp
        if not scores_list:
            return 4  # neutral default

        clean: list[float] = []
        for s in scores_list:
            if pd.isna(s):
                continue
            try:
                v = float(s)
                if lo is not None and v < lo: v = lo
                if hi is not None and v > hi: v = hi
                clean.append(v)
            except (ValueError, TypeError):
                continue

        if not clean:
            return 4

        if method == "median":
            agg = statistics.median(clean)
        elif method == "trimmed_mean":
            clean_sorted = sorted(clean)
            k = max(1, int(0.1 * len(clean_sorted)))  # trim 10% each side
            trimmed = clean_sorted[k:-k] if len(clean_sorted) > 2*k else clean_sorted
            agg = sum(trimmed) / len(trimmed)
        else:
            agg = statistics.mean(clean)

        return int(round(agg))

    def _extract_entity_name(self, entity_names: list[str], *, 
                         min_top_share: float = 0.6) -> str:
        if not entity_names:
            return "Unknown Entity"

        def norm(x: str) -> str:
            s = str(x).strip()
            if not s or s.lower() in {"n/a", "na", "none", "null", "unknown", "-","—"}:
                return ""
            return s  # keep original casing for readability

        clean = [norm(n) for n in entity_names if pd.notna(n)]
        clean = [n for n in clean if n]
        if not clean:
            return "Unknown Entity"

        counts = Counter(clean)
        top, top_cnt = counts.most_common(1)[0]
        share = top_cnt / sum(counts.values())

        return top if share >= min_top_share else "Multiple Entities"

    # --- simple helpers ---
    def _sentence_case(self, s: str) -> str:
        s = (s or "").strip()
        return s[:1].upper() + s[1:] if s else s

    def _mode_or_centroid(self, texts: list[str]) -> str | None:
        """Prefer the most frequent clean string; fallback to centroid picker if available."""
        arr = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        if not arr:
            return None
        vc = pd.Series(arr).value_counts()
        top = vc.index[0] if not vc.empty else None
        if top:
            return top
        if hasattr(self, "_centroid_pick"):
            return self._centroid_pick(arr)
        return arr[0]

    def _norm_ws(self, s: str) -> str:
        return " ".join(str(s or "").strip().split())
    
    # --- tail formatter (unchanged; uses chosen composite) ---
    def _format_location_tail(self, cj: str, cjs: str, im: str) -> str:
        """
        Render the Journey/Stage/Moment tail.
        Styles:
        - 'short' (default): J:/S:/M:
        - 'long' : Journey:/Stage:/Moment:
        - 'journey_stage' : Journey:/Stage:
        - 'stage_only' : Stage:
        - 'none' : no tail
        """
        style = (getattr(self, "OU_CFG", {}) or {}).get("preview_location_style", "short").lower()

        def _wrap(parts: list[str]) -> str:
            return f" ({' | '.join(parts)})" if parts else ""

        if style == "none":
            return ""

        if style == "long":
            parts = []
            if cj:  parts.append(f"Journey: {cj}")
            if cjs: parts.append(f"Stage: {cjs}")
            if im:  parts.append(f"Moment: {im}")
            return _wrap(parts)

        if style == "journey_stage":
            parts = []
            if cj:  parts.append(f"Journey: {cj}")
            if cjs: parts.append(f"Stage: {cjs}")
            return _wrap(parts)

        if style == "stage_only":
            return _wrap([f"Stage: {cjs}"] if cjs else [])

        # default -> 'short'
        parts = []
        if cj:  parts.append(f"J: {cj}")
        if cjs: parts.append(f"S: {cjs}")
        if im:  parts.append(f"M: {im}")
        return _wrap(parts)

    # --- preview builder (signature-aware) ---
    def _make_root_cause_preview(self, grp: pd.DataFrame) -> str:
        """
        Build a short root-cause statement driven by the current signature config.
        Priority is taken from OU_CFG["signature_weights"] among these candidate fields:
        - matters
        - semantic_customer_reality  (first half of SAS)
        - context
        - behavioral_impact
        - stream_justification
        Then append a locator tail (Journey/Stage/Moment) per your style.
        Deterministic. Trims to 80 chars (adds … if longer).
        """

        # ensure semantic_customer_reality is available if SAS exists (lightweight guard)
        if "semantic_customer_reality" not in grp.columns and "semantic_action_statement" in grp.columns:
            def _extract_scr(t: str) -> str:
                t = (t or "").replace("\r", "\n")
                if "SECTION 1" in t:
                    part = t.split("SECTION 2", 1)[0].split("SECTION 1", 1)[-1]
                    lines = [ln.strip() for ln in part.splitlines() if ln.strip()]
                    if lines and lines[0].upper().startswith("THE CUSTOMER REALITY"):
                        lines = lines[1:]
                    scr = " ".join(lines).strip()
                else:
                    scr = t.split("\n\n", 1)[0].strip()
                return re.sub(r"\s+", " ", scr)
            try:
                grp = grp.copy()
                grp["semantic_customer_reality"] = grp["semantic_action_statement"].astype(str).apply(_extract_scr)
            except Exception:
                # if anything goes wrong, we simply won't use SCR as a candidate
                pass

        # derive candidate field order from signature weights (no hard-coding)
        weights = (self.OU_CFG or {}).get("signature_weights", {})
        candidate_fields = ["matters", "semantic_customer_reality", "context", "behavioral_impact", "stream_justification"]
        ordered_fields = sorted(
            ((f, int(weights.get(f, 0))) for f in candidate_fields),
            key=lambda kv: (-kv[1], kv[0])
        )
        ordered_fields = [f for f, w in ordered_fields if w > 0]

        # pick the first available non-empty text using your mode→centroid fallback
        mech = ""
        for col in ordered_fields:
            if col in grp.columns:
                cand = self._mode_or_centroid(grp[col].astype(str).tolist())
                if isinstance(cand, str) and cand.strip():
                    mech = cand.strip().rstrip(" .")
                    break

        # fallback if none of the candidates produced text
        if not mech:
            mech = "No preview available"

        # choose one valid locator combo from the data (deterministic)
        journey, stage, moment, _ = self._collapse_locator_composite(grp)
        tail = self._format_location_tail(journey, stage, moment)

        # stitch + tidy
        preview = self._sentence_case((mech + tail).strip())
        return (preview[:80] + "…") if len(preview) > 80 else preview


    def cluster_behavior(self, df: pd.DataFrame, driver: str, feedback_type: str, stream: str):
        df = df.copy()

        # weighted signature (uses self.SIGNATURE_FIELDS driven by OU_CFG["signature_weights"])
        df["signature"] = df.apply(self._build_signature, axis=1).astype(str).str.lower()
        df = df.reset_index(drop=True)

        # numpy embeddings (L2-normalized)
        embeds = self._encode_np(df["signature"].tolist())
        total_rows = len(df)

        # --- adaptive min cluster size ---
        if self.OU_CFG.get("adaptive_min_cluster", {}).get("enabled", False):
            mcs = max(
                int(self.OU_CFG["adaptive_min_cluster"].get("floor", 3)),
                int(math.ceil(self.OU_CFG["adaptive_min_cluster"].get("pct", 0.08) * total_rows))
            )
        else:
            mcs = int(self.OU_CFG.get("min_cluster_size", 8))  # fallback

        # distance threshold knob
        start_thr = float(  # NEW: keep start value separately
            self.OU_CFG.get("bcs_distance_threshold",
                            self.OU_CFG.get("signature_threshold_start", {}).get(
                                self.OU_CFG.get("signature_config_name","default_K1"), 0.30))
        )
        dist_thr = float(start_thr)  # current τ
        step = float(self.OU_CFG.get("bcs_distance_threshold_step", 0.02))            # NEW
        lo, hi = self.OU_CFG.get("bcs_distance_threshold_bounds", (0.25, 0.35))       # NEW
        band_lo, band_hi = self.OU_CFG.get("cluster_band_target", (3, 8))             # NEW
        singleton_cap = float(self.OU_CFG.get("singleton_cap", 0.40))                 # NEW
        max_adj = int(self.OU_CFG.get("bcs_threshold_max_adjustments", 1))            # NEW
        skip_singletons = bool(self.OU_CFG.get("skip_singletons", False))

        # --- clustering ---------------------------------------------------------
        if total_rows == 0:
            df["local_bcs_id"] = np.array([], dtype=str)
            return df

        # Helper: run agglomerative once with given threshold    # NEW
        def _run_agglom(thr: float):
            try:
                # sklearn >= 1.2
                labels_ = AgglomerativeClustering(
                    metric="cosine",
                    linkage="average",
                    distance_threshold=float(thr),
                    n_clusters=None
                ).fit_predict(embeds)
            except TypeError:
                # older sklearn expects precomputed distance matrix
                dist = pairwise_distances(embeds, metric="cosine")
                labels_ = AgglomerativeClustering(
                    affinity="precomputed",
                    linkage="average",
                    distance_threshold=float(thr),
                    n_clusters=None
                ).fit_predict(dist)
            return labels_

        # Helper: compute #clusters and singleton rate on labels (treat HDBSCAN noise as singletons)  # NEW
        def _cluster_stats(labels_arr: np.ndarray):
            labels_arr = np.asarray(labels_arr)
            n = len(labels_arr)
            noise_count = int(np.sum(labels_arr == -1))
            non_noise = labels_arr[labels_arr != -1]
            cnt = Counter(non_noise)
            n_clusters = len(set(non_noise.tolist())) + noise_count
            singleton = sum(1 for c in cnt.values() if c == 1) + noise_count
            s_rate = (singleton / n) if n else 0.0
            return n_clusters, s_rate

        # First pass
        used_agglom = False
        if total_rows < 2:
            labels = np.array([0], dtype=int)

        elif total_rows < 200:
            used_agglom = True
            labels = _run_agglom(dist_thr)

            # --- enforce adaptive min cluster size post-hoc (Agglomerative has no mcs) ---
            counts = Counter(labels)
            small_cids = {cid for cid, sz in counts.items() if sz < mcs}
            if small_cids:
                if skip_singletons:
                    # mark tiny clusters as noise (-1)
                    for cid in small_cids:
                        labels[labels == cid] = -1
                else:
                    # keep them but tag for downstream handling
                    df["_is_microcluster"] = False
                    for cid in small_cids:
                        df.loc[np.where(labels == cid)[0], "_is_microcluster"] = True

            # --- one-bump controller: adjust τ once if out of band ---------------------  # NEW
            if max_adj > 0:
                n_clusters, s_rate = _cluster_stats(labels)
                new_thr = dist_thr
                if (n_clusters > band_hi) or (s_rate > singleton_cap):
                    new_thr = min(hi, dist_thr + step)
                elif (n_clusters < band_lo):
                    new_thr = max(lo, dist_thr - step)

                if float(new_thr) != float(dist_thr):
                    dist_thr = float(new_thr)
                    # re-run once with adjusted threshold
                    labels = _run_agglom(dist_thr)

                    # re-apply min cluster size handling
                    counts = Counter(labels)
                    small_cids = {cid for cid, sz in counts.items() if sz < mcs}
                    if small_cids:
                        if skip_singletons:
                            for cid in small_cids:
                                labels[labels == cid] = -1
                        else:
                            df["_is_microcluster"] = False
                            for cid in small_cids:
                                df.loc[np.where(labels == cid)[0], "_is_microcluster"] = True

        else:
            # HDBSCAN for larger sets (cosine metric) — here we can apply mcs directly
            labels = hdbscan.HDBSCAN(
                metric="cosine",
                min_cluster_size=mcs,
                min_samples=max(1, mcs // 2)
            ).fit_predict(embeds)

        # Persist controller info for reproducibility (attr; non-breaking)         # NEW
        try:
            self._last_cluster_control = {
                "bcs_distance_threshold_start": float(start_thr),
                "bcs_distance_threshold_used": float(dist_thr),
                "cluster_band_target": f"{band_lo}–{band_hi}",
                "singleton_cap": float(singleton_cap),
                "used_agglomerative": bool(used_agglom),
            }
        except Exception:
            pass

        df["local_bcs_id"] = labels.astype(str)

        # HDBSCAN noise → unique singletons
        noise_mask = df["local_bcs_id"] == "-1"
        if noise_mask.any():
            df.loc[noise_mask, "local_bcs_id"] = [str(uuid4()) for _ in range(int(noise_mask.sum()))]

        # ✅ prefix reflects ED + FeedbackType + Stream (no emotion here)
        prefix = f"{driver[:8]}_{str(feedback_type)[:3]}_{stream[:3]}".lower()
        cluster_store, full_composites, cluster_metadata = {}, {}, {}

        # init annotation columns
        df["bcs_group_id"] = None
        df["bcs_id"] = None
        df["bcs_share"] = None
        df["bcs_label"] = None
        df["cluster_cohesion"] = None
        df["cluster_theme_preview"] = None

        for _, grp in df.groupby("local_bcs_id"):
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

            # 🧩 Deterministic root-cause/preview (allowed fields only)
            preview = self._make_root_cause_preview(grp)
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
                emo_prim = (first_row.get("emotion_primary") or "").strip().lower() or None
                emo_spec = (first_row.get("emotion_specific") or "").strip() or None
                emo_dist = {emo_prim: 100.0} if emo_prim else {}

                composite = {
                    "bcs_id": first_row["bcs_id"],
                    "bcs_group_id": group_id,
                    "cluster_size": 1,
                    "bcs_share": round(1 / total_rows, 4),
                    "cluster_cohesion": 1.0,
                    "cluster_theme_preview": truncated_preview,
                    "customer_review": first_row.get("customer_review") or first_row.get("comment_review"),

                    # hierarchy fields (fixed by subset)
                    "experience_driver": first_row.get("experience_driver"),
                    "feedback_type": first_row.get("feedback_type"),
                    "opportunity_stream": first_row.get("opportunity_stream"),

                    # emotion (cannot be omitted)
                    "emotion_primary": emo_prim or "unknown",
                    "emotion_specific": emo_spec,
                    "emotion_distribution": emo_dist,

                    # remaining passthroughs
                    "customer_journey": first_row.get("customer_journey"),
                    "customer_journey_stage": first_row.get("customer_journey_stage"),
                    "interaction_moment": first_row.get("interaction_moment"),
                    "context": first_row.get("context"),
                    "keywords": first_row.get("keywords"),
                    "entity_name": first_row.get("entity_name"),
                    "theme": first_row.get("theme"),
                    "customer_effort_score": first_row.get("customer_effort_score"),
                    "semantic_action_statement": first_row.get("semantic_action_statement"),
                    "stream_justification": first_row.get("stream_justification"),
                    "matters": first_row.get("matters"),
                    "behavioral_impact": first_row.get("behavioral_impact"),
                }

            else:
                batch_1 = self.extract_batch_1_fields(grp)          # includes emotion collapse + theme
                batch_2 = self.process_batch_2_fields(grp, batch_1) # context/keywords/locator/CES/entity

                sas_list = grp.get("semantic_action_statement", pd.Series(dtype=object)).dropna().astype(str).tolist()
                just_list = grp.get("stream_justification",   pd.Series(dtype=object)).dropna().astype(str).tolist()
                mat_list  = grp.get("matters",                 pd.Series(dtype=object)).dropna().astype(str).tolist()
                beh_list  = grp.get("behavioral_impact",       pd.Series(dtype=object)).dropna().astype(str).tolist()

                composite = {
                    **batch_1, **batch_2,
                    "bcs_id": first_row["bcs_id"],
                    "bcs_group_id": group_id,
                    "cluster_size": len(grp),
                    "bcs_share": round(len(grp) / total_rows, 4),
                    "cluster_cohesion": round(cohesion, 4),
                    "cluster_theme_preview": truncated_preview,
                    "customer_review": customer_review_value,

                    "semantic_action_statement": self._mmr_summary(sas_list) or self._centroid_pick(sas_list),
                    "stream_justification":     self._centroid_pick(just_list),
                    "matters":                  self._mmr_summary(mat_list)  or self._centroid_pick(mat_list),
                    "behavioral_impact":        self._mmr_summary(beh_list)  or self._centroid_pick(beh_list),
                }

            # store + annotate
            df.update(grp)
            cluster_store[group_id] = grp
            full_composites[group_id] = composite
            cluster_metadata[group_id] = {"label": truncated_preview, "cohesion": cohesion}

        # final annotation on row-level df
        df["bcs_label"] = df["bcs_group_id"].map(lambda gid: cluster_metadata.get(gid, {}).get("label"))
        df["cluster_cohesion"] = df["bcs_group_id"].map(lambda gid: cluster_metadata.get(gid, {}).get("cohesion"))
        df["cluster_theme_preview"] = df["cluster_theme_preview"].fillna(
            df["bcs_group_id"].map(lambda gid: full_composites.get(gid, {}).get("cluster_theme_preview"))
        )

        # dominant cluster filter (cumulative share threshold)
        dominant_ids, cumulative_share = [], 0.0
        cluster_order = df["bcs_group_id"].value_counts(normalize=True)
        for cid, share in cluster_order.items():
            dominant_ids.append(cid)
            cumulative_share += share
            if cumulative_share >= self.OU_CFG["bcs_cumu_threshold"]:
                break

        filtered_df = df[df["bcs_group_id"].isin(dominant_ids)].copy()

        # also persist τ used back into OU_CFG for transparency (non-breaking)     # NEW
        self.OU_CFG["bcs_distance_threshold"] = float(dist_thr)

        return filtered_df, list(full_composites.values()), cluster_store, df, full_composites


    # ---------- DB (member-first, no collapsing) ----------

    def _nz(self,v):
        return "" if v is None else v

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
                    emotion_primary TEXT,
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
            cur.execute('CREATE INDEX IF NOT EXISTS idx_ed_em_stream ON clusters (experience_driver, emotion_primary, opportunity_stream)')
            conn.commit()
        print(f"✅ Database initialized: {self.db_path}")

    def create_cluster_database(self, df: pd.DataFrame, full_composites: Dict[str, Dict[str, Any]],
                                cluster_store: Dict[str, pd.DataFrame], db_path: str = "clusters.db"):
        import os
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.init_database()  # uses your existing schema

        # Always write member rows as-is (composite only for META fallback)
        for cid, grp in cluster_store.items():
            composite = full_composites.get(cid, {})
            self.save_cluster_members_as_is(grp, composite, cid)

        print("✅ All clusters saved to database (member rows preserved).")

    def save_cluster_members_as_is(self, grp: pd.DataFrame, composite: Dict[str, Any], cid: str):
        """
        Persist each row in grp exactly as-is (no collapsing), plus cluster-level META:
        cluster_size, bcs_share, cluster_cohesion, cluster_theme_preview (fallback).
        """
        cluster_size       = int(composite.get("cluster_size", len(grp)))
        bcs_share          = float(composite.get("bcs_share", len(grp) / max(len(grp), 1)))
        cluster_cohesion   = float(composite.get("cluster_cohesion", 1.0))
        meta_theme_preview = self._nz(composite.get("cluster_theme_preview", ""))

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()

            for _, row in grp.iterrows():
                # Prefer per-row preview computed in cluster_behavior()
                theme_preview = self._nz(row.get("cluster_theme_preview", meta_theme_preview))

                # keywords can be list or str
                kw = row.get("keywords")
                if isinstance(kw, list):
                    keywords_str = ", ".join(map(str, kw))
                else:
                    keywords_str = self._nz(kw if kw is not None else "")

                # emotion: keep existing row field; fallback to emotion_primary if needed
                emotion_value = row.get("emotion_primary")
                if emotion_value is None:
                    emotion_value = row.get("emotion_primary")

                cur.execute('''
                    INSERT OR REPLACE INTO clusters (
                        bcs_id, bcs_group_id, cluster_size, bcs_share, cluster_cohesion, cluster_theme_preview,
                        customer_review, experience_driver, emotion_primary, theme, opportunity_stream, feedback_type,
                        customer_journey, customer_journey_stage, interaction_moment, context,
                        keywords, entity_name, customer_effort_score,
                        semantic_action_statement, stream_justification, matters, behavioral_impact, problem_statement
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self._nz(row.get("bcs_id")),
                    self._nz(row.get("bcs_group_id") or cid),
                    cluster_size,
                    bcs_share,
                    cluster_cohesion,
                    theme_preview,

                    self._nz(row.get("customer_review") or row.get("comment_review")),
                    self._nz(row.get("experience_driver")),
                    self._nz(emotion_value),
                    self._nz(row.get("theme")),
                    self._nz(row.get("opportunity_stream")),
                    self._nz(row.get("feedback_type")),

                    self._nz(row.get("customer_journey")),
                    self._nz(row.get("customer_journey_stage")),
                    self._nz(row.get("interaction_moment")),
                    self._nz(row.get("context")),

                    self._nz(keywords_str),
                    self._nz(row.get("entity_name")),
                    float(row.get("customer_effort_score", 0.0) or 0.0),

                    self._nz(row.get("semantic_action_statement")),
                    self._nz(row.get("stream_justification")),
                    self._nz(row.get("matters")),
                    self._nz(row.get("behavioral_impact")),
                    self._nz(row.get("problem_statement") or "")
                ))

            conn.commit()
        print(f"💾 Saved cluster {cid} with {len(grp)} member row(s) (as-is).")

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

    def _prepare_raw_for_snapshot(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize once so snapshot doesn’t have to."""
        rdf = raw_df.copy()
        rdf["experience_driver"]  = rdf["experience_driver"].astype(str).str.strip()
        # FT lower-case (snapshot matches on lower)
        rdf["feedback_type"]      = rdf["feedback_type"].astype(str).str.strip().str.lower()
        # Canonicalize streams to title-case set via map
        rdf["opportunity_stream"] = self._canon_stream(rdf["opportunity_stream"])
        rdf = rdf.dropna(subset=["experience_driver", "feedback_type", "opportunity_stream"])
        return rdf

    def _safe_list(self, x) -> list:
        """Layer3 focus can be list or a stringified list; make it a list[str] lower-cased."""
        if isinstance(x, list): items = x
        elif isinstance(x, (tuple, set)): items = list(x)
        elif isinstance(x, str):
            try:
                import ast
                items = ast.literal_eval(x)
                if not isinstance(items, (list, tuple, set)): items = []
            except Exception:
                items = []
        else:
            items = []
        return [str(v).strip().lower() for v in items if str(v).strip()]

    def stream_focus_from_rows(self, ft_rows: pd.DataFrame, *, threshold: float) -> tuple[list, dict]:
        """
        Compute dominant streams (≥ threshold cumulative) and full distribution (%, 1dp)
        for the already-filtered (ED, FT) rows. Assumes streams are canonicalized.
        """
        counts = (
            ft_rows["opportunity_stream"]
            .dropna()
            .value_counts()
            .sort_values(ascending=False)
            .sort_index(kind="mergesort")
        )
        total = int(counts.sum())
        if total == 0:
            return [], {}
        dominant, cumulative = [], 0.0
        distribution = {}
        for stream, cnt in counts.items():
            frac = cnt / total
            distribution[stream] = round(frac * 100, 1)
            if cumulative < threshold:
                dominant.append(stream)
                cumulative += frac
        return dominant, distribution

    def compute_granular_details_snapshot(self, raw_df: pd.DataFrame, layer3_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build OU composites per (ED → FT → Stream) and persist clusters.
        Thin loop: assumes inputs are pre-normalized.
        """
        if layer3_df is None or layer3_df.empty:
            raise ValueError("Layer 3 diagnostics must be supplied and non-empty.")

        must_cols = {"experience_driver", "feedback_type", "opportunity_stream"}
        missing = [c for c in must_cols if c not in raw_df.columns]
        if missing:
            raise ValueError(f"Raw DF missing required columns: {missing}")

        # ✅ all normalization happens outside
        rdf = self._prepare_raw_for_snapshot(raw_df)

        thr = max(0.0, min(1.0, float(self.OU_CFG.get("stream_threshold", 0.80))))

        all_df_chunks, all_full_composites, all_cluster_store = [], {}, {}
        records = []

        for _, hdr in layer3_df.iterrows():
            driver = str(hdr.get("experience_driver", "")).strip()
            if not driver:
                continue

            # ✅ layer3 already provides FT focus + dist; we only standardize list shape & lower-case
            ft_focus = self._safe_list(hdr.get("feedbacktype_audit_focus", []))
            ft_dist  = hdr.get("feedback_type_distribution", {}) or {}

            driver_rows = rdf[rdf["experience_driver"] == driver]
            if driver_rows.empty:
                continue

            for ft_key in ft_focus:  # already lower-case
                ft_rows = driver_rows[driver_rows["feedback_type"] == ft_key]
                if ft_rows.empty:
                    continue

                # ✅ stream dominance + distribution handled in helper (streams already canonical)
                dominant_streams, stream_distribution = self.stream_focus_from_rows(ft_rows, threshold=thr)
                if not dominant_streams:
                    continue

                for stream in dominant_streams:
                    stream_rows = ft_rows[ft_rows["opportunity_stream"] == stream]
                    if stream_rows.empty:
                        continue

                    clust_df, full_distribution, cluster_store, df_chunk, full_composites = self.cluster_behavior(stream_rows, driver=driver, feedback_type=ft_key, stream=stream)
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
                                "feedback_type_distribution": ft_dist,
                                "stream_distribution": stream_distribution,
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