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
            "min_cluster_size": 8,  # fallback when adaptive is disabled
            "adaptive_min_cluster": {  # per-slice adaptive min cluster size
                "enabled": True,
                "pct": 0.08,   # 8% of rows in this ED→Type→Stream slice
                "floor": 3     # never below 3
            },
            "bcs_cumu_threshold": 0.80,
            "stream_threshold": 0.80,
            "skip_singletons": False,
            "preview_location_style": "short",
            "bcs_distance_threshold": 0.30  # tune within 0.25–0.35
        }

        # Signature weights + ordered fields
        self.OU_CFG.setdefault("signature_weights", {
            "matters": 6, "context": 4, "interaction_moment": 2,
            "customer_journey_stage": 1, "customer_journey": 1,
        })
        self.SIGNATURE_FIELDS = list(self.OU_CFG["signature_weights"].items())

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

    def _encode_np(self, texts):
        """
        Numpy, L2-normalized embeddings for deterministic cosine similarity everywhere.
        """
        model = self._ensure_st_model()
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

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

    # --- composite locator collapse (NO hardcoding) ---
    def _collapse_locator_composite(self, grp: pd.DataFrame):
        """
        Pick ONE (journey, stage, moment) combo that actually appears in the cluster.
        Steps: normalize → count → mode → semantic tiebreak on composite strings.
        Returns: (journey, stage, moment, distribution_dict)
        """
        j_raw = grp.get("customer_journey",       pd.Series(dtype=object)).astype(str).tolist()
        s_raw = grp.get("customer_journey_stage", pd.Series(dtype=object)).astype(str).tolist()
        m_raw = grp.get("interaction_moment",     pd.Series(dtype=object)).astype(str).tolist()

        key_counts: Counter[tuple[str,str,str]] = Counter()
        key_variants: dict[tuple[str,str,str], list[tuple[str,str,str]]] = {}

        for j, s, m in zip(j_raw, s_raw, m_raw):
            jn, sn, mn = self._norm_ws(j).lower(), self._norm_ws(s).lower(), self._norm_ws(m).lower()
            if not (jn or sn or mn):
                continue
            key = (jn, sn, mn)
            key_counts[key] += 1
            key_variants.setdefault(key, []).append((
                self._norm_ws(j), self._norm_ws(s), self._norm_ws(m)
            ))

        if not key_counts:
            return "Unknown", "Unknown", "Unknown", {}

        # mode, then semantic tiebreak if needed
        most = key_counts.most_common(1)[0][1]
        tied_keys = [k for k, c in key_counts.items() if c == most]
        if len(tied_keys) == 1:
            chosen_key = tied_keys[0]
        else:
            tie_strs = [" | ".join([p for p in k if p]) for k in tied_keys]
            picked = self._centroid_pick(tie_strs) if hasattr(self, "_centroid_pick") else tie_strs[0]
            chosen_key = tied_keys[tie_strs.index(picked)]

        total = sum(key_counts.values()) or 1
        dist = {" | ".join([p for p in k if p]): round(100.0 * c / total, 1)
                for k, c in key_counts.items()}

        # display values = most frequent original variants for the chosen key
        variants = key_variants[chosen_key]
        j_disp = Counter([v[0] for v in variants if v[0]]).most_common(1)
        s_disp = Counter([v[1] for v in variants if v[1]]).most_common(1)
        m_disp = Counter([v[2] for v in variants if v[2]]).most_common(1)

        journey = j_disp[0][0] if j_disp else "Unknown"
        stage   = s_disp[0][0] if s_disp else "Unknown"
        moment  = m_disp[0][0] if m_disp else "Unknown"

        return journey, stage, moment, dist

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

    # --- preview builder (now uses the composite) ---
    def _make_root_cause_preview(self, grp: pd.DataFrame) -> str:
        """
        Build a short root-cause statement using ONLY:
        (matters/context) + a REAL composite (Journey → Stage → Moment) chosen from the cluster.
        Deterministic, compact, OU-ready. Trims to 80 chars (adds … if longer).
        """
        # mechanism: prefer 'matters', fallback to 'context'
        mat = self._mode_or_centroid(grp.get("matters", pd.Series(dtype=object)).astype(str).tolist()) or ""
        ctx = self._mode_or_centroid(grp.get("context", pd.Series(dtype=object)).astype(str).tolist()) or ""
        mech = (mat or ctx).strip().rstrip(" .")

        # choose one valid locator combo from the data (no hardcoding)
        journey, stage, moment, _ = self._collapse_locator_composite(grp)
        tail = self._format_location_tail(journey, stage, moment)

        base = mech if mech else (journey or stage or moment or "No preview available")
        preview = self._sentence_case((base + tail).strip())

        return (preview[:80] + "…") if len(preview) > 80 else preview

    # ---------- clustering ----------
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

        # distance threshold knob (0.25–0.35 typical)
        dist_thr = float(self.OU_CFG.get("bcs_distance_threshold", 0.30))
        skip_singletons = bool(self.OU_CFG.get("skip_singletons", False))

        # --- clustering ---------------------------------------------------------
        if total_rows == 0:
            df["local_bcs_id"] = np.array([], dtype=str)
            return df

        if total_rows < 2:
            labels = np.array([0], dtype=int)

        elif total_rows < 200:
            # Agglomerative (cosine, average linkage) with distance_threshold from config
            try:
                # sklearn >= 1.2
                labels = AgglomerativeClustering(
                    metric="cosine",
                    linkage="average",
                    distance_threshold=dist_thr,
                    n_clusters=None
                ).fit_predict(embeds)
            except TypeError:
                # older sklearn expects precomputed distance matrix
                dist = pairwise_distances(embeds, metric="cosine")
                labels = AgglomerativeClustering(
                    affinity="precomputed",
                    linkage="average",
                    distance_threshold=dist_thr,
                    n_clusters=None
                ).fit_predict(dist)

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

        else:
            # HDBSCAN for larger sets (cosine metric) — here we can apply mcs directly
            labels = hdbscan.HDBSCAN(
                metric="cosine",
                min_cluster_size=mcs,
                min_samples=max(1, mcs // 2)
            ).fit_predict(embeds)

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
                # 🔒 single row → copy through, but include emotion fields & distribution
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
                # 🧮 multi-row → deterministic collapses
                batch_1 = self.extract_batch_1_fields(grp)        # includes emotion collapse + theme
                batch_2 = self.process_batch_2_fields(grp, batch_1)  # context/keywords/locator/CES/entity

                # the 4 “mega” fields — deterministic rollups (MMR → centroid fallback)
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

def get_trend_momentum_grid():
    """
    9-cell Trend × Momentum matrix with full narratives.
    Columns:
      Trend_Tier, Momentum_Tier, Diagnostic_Label, Strategic_Narrative,
      Action_Guidance
    """
    data = [
        # Trend ↑ Improving
        ["Trend ↑ (Improving)","Momentum ↑ (Improving)",
         "Momentum Building 🚀",
         "Both long-term and recent signals positive.",
         "Amplify programs, celebrate advocacy, invest in scale."],

        ["Trend ↑ (Improving)","Momentum → (Stable)",
         "Plateau Watch ⚖️",
         "Long-term gains, but short-term flattening.",
         "Refresh engagement to prevent stagnation."],

        ["Trend ↑ (Improving)","Momentum ↓ (Worsening)",
         "Micro-Shock (Rare)",
         "Long-term gains but short-term deterioration.",
         "Watch for micro-shocks, validate if anomaly or early erosion."],

        # Trend → Stable
        ["Trend → (Stable)","Momentum ↑ (Improving)",
         "Recovery Signal 🌱",
         "Flat overall, but recent upward push.",
         "Support and accelerate the bounce."],

        ["Trend → (Stable)","Momentum → (Stable)",
         "True Stability 🟢",
         "Flat long-term and short-term.",
         "Monitor quietly, no major action."],

        ["Trend → (Stable)","Momentum ↓ (Worsening)",
         "Early Erosion ⚠️",
         "Flat long-term, but recent dip.",
         "Early-warning: intervene before decline becomes entrenched."],

        # Trend ↓ Declining
        ["Trend ↓ (Declining)","Momentum ↑ (Improving)",
         "Emerging Recovery 🔄",
         "Overall decline, but short-term improvement.",
         "Double down to reverse the slide."],

        ["Trend ↓ (Declining)","Momentum → (Stable)",
         "Structural Decline 🔻",
         "Steady long-term fall, no shift.",
         "Root-cause deep dive and corrective initiatives."],

        ["Trend ↓ (Declining)","Momentum ↓ (Worsening)",
         "Accelerating Collapse 💥",
         "Both long-term and recent negative.",
         "Crisis response: urgent retention and loyalty fixes."]
    ]
    cols = ["Trend_Tier","Momentum_Tier","Diagnostic_Label",
            "Strategic_Narrative","Action_Guidance"]
    return pd.DataFrame(data, columns=cols)


def get_trend_momentum_volatility_grid():
    """
    15-cell Trend x Momentum x Volatility matrix.
    Columns:
      Trend_Tier, Momentum_Tier, Volatility_Tier, Interpretation, CX_Instruction
    """
    data = [
        # Trend ↑ Improving, Momentum → Stable
        ["↑ (Improving)","→ (Stable)","✅ Stable",
         "Long-term gains holding steady and reliable",
         "Keep reinforcing; stable trust is compounding"],

        ["↑ (Improving)","→ (Stable)","⚠ Fluctuating",
         "Gains are real but customer mood wobbles",
         "Monitor; ensure uplift isn’t fragile"],

        ["↑ (Improving)","→ (Stable)","🔴 Highly Fluctuating",
         "Signals point upward, but chaos under the surface",
         "Don’t over-celebrate; stabilize before scaling"],

        # Trend ↑ Improving, Momentum ↑ Recent rise
        ["↑ (Improving)","↑ (Recent rise)","✅ Stable",
         "Clear acceleration with consistent base",
         "Amplify — perfect moment to scale advocacy"],

        ["↑ (Improving)","↑ (Recent rise)","⚠ Fluctuating",
         "Surge is happening, but customers uneven",
         "Seize positives, but reinforce weak spots"],

        ["↑ (Improving)","↑ (Recent rise)","🔴 Highly Fluctuating",
         "“Spike effect” — surge may collapse",
         "Treat as hype-cycle; confirm if sustainable"],

        # Trend → Flat, Momentum ↓ Recent dip
        ["→ (Flat)","↓ (Recent dip)","✅ Stable",
         "Plateau with emerging warning",
         "Investigate small cracks before they widen"],

        ["→ (Flat)","↓ (Recent dip)","⚠ Fluctuating",
         "Customers unsettled, mixed signals",
         "Early intervention; sentiment at tipping point"],

        ["→ (Flat)","↓ (Recent dip)","🔴 Highly Fluctuating",
         "Volatility + downturn = fragile loyalty",
         "Treat as high-risk; prepare crisis workflows"],

        # Trend ↓ Declining, Momentum ↑ Recent rise
        ["↓ (Declining)","↑ (Recent rise)","✅ Stable",
         "Recovery starting; reliable turnaround",
         "Support rebound with targeted initiatives"],

        ["↓ (Declining)","↑ (Recent rise)","⚠ Fluctuating",
         "Customers showing rebound but unstable",
         "Encourage positives, shore up weaknesses"],

        ["↓ (Declining)","↑ (Recent rise)","🔴 Highly Fluctuating",
         "Recovery attempt is noisy and fragile",
         "Monitor tightly; avoid premature bets"],

        # Trend ↓ Declining, Momentum ↓ Recent fall
        ["↓ (Declining)","↓ (Recent fall)","✅ Stable",
         "Clear deterioration, stable pattern",
         "Act decisively — loyalists are slipping away"],

        ["↓ (Declining)","↓ (Recent fall)","⚠ Fluctuating",
         "Decline underway, customers uneven",
         "Contain damage, look for segment splits"],

        ["↓ (Declining)","↓ (Recent fall)","🔴 Highly Fluctuating",
         "Emotional free-fall with chaos",
         "Crisis protocol — stabilize or lose trust"],
    ]

    cols = ["Trend_Tier","Momentum_Tier","Volatility_Tier","Interpretation","CX_Instruction"]
    return pd.DataFrame(data, columns=cols)


def get_temporal_pattern_grid():
    """
    Temporal Pattern Action Grid
    Rows capture Pattern Type × Confidence with interpretation and CX instruction.
    Columns:
      Pattern_Type, Confidence_Level, Interpretation, CX_Instruction
    """
    data = [
        # Pain Day
        ["Pain Day", "✅ Strong",
         "Consistent spike in negative emotion on a specific day",
         "Proactively staff / resource to handle known pain-day load"],

        ["Pain Day", "⚠ Moderate",
         "Recurring issue but not always reliable",
         "Monitor closely, validate before committing resources"],

        ["Pain Day", "🔴 Weak / Candidate",
         "Emerging but unconfirmed spike",
         "Treat as early signal; don’t overfit; collect more cycles"],

        # Seasonal Cycle
        ["Seasonal Cycle", "✅ Strong",
         "Clear quarterly / monthly / weekly recurrence",
         "Plan capacity and customer comms in advance (e.g. holiday delivery surges)"],

        ["Seasonal Cycle", "⚠ Moderate",
         "Some recurrence, but irregular",
         "Hedge plans: prep but don’t over-invest"],

        ["Seasonal Cycle", "🔴 Weak / Candidate",
         "Hints of a cycle, not yet statistically reliable",
         "Flag as watchlist; continue data collection"],

        # Operational Timing
        ["Operational Timing", "✅ Strong",
         "Predictable spikes tied to process (e.g. refund backlog after weekend)",
         "Reconfigure operations to smooth flow, pre-empt customer frustration"],

        ["Operational Timing", "⚠ Moderate",
         "Timing link is visible but with exceptions",
         "Use targeted fixes; validate across more periods"],

        ["Operational Timing", "🔴 Weak / Candidate",
         "Possible operational link but noisy",
         "Tag for analyst review; don’t trigger enterprise-wide action yet"],
    ]

    cols = ["Pattern_Type", "Confidence_Level", "Interpretation", "CX_Instruction"]
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
        self.pattern_lags = ["weekly","monthly","quarterly"]

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
    
    def _compute_trend_series(self, series, eps=1e-6):
        """
        Direction-only TREND over window + audit percentages.
        - Gates (whitepaper): ↑ ≥ +5%, ↓ ≤ −5%, else →
        - dir_pct: direction-safe % = (now - prev) / max(|prev|, ε) * 100
        - pct_change: raw whitepaper % = (now - prev) / prev * 100, with band fallback when prev≈0
        """
        import numpy as np, pandas as pd

        s = pd.Series(series).astype(float).dropna()
        n = len(s)

        # thresholds (pull from contract if present)
        th = self.trend_thresholds() if hasattr(self, "trend_thresholds") else {"up_pct": 5.0, "down_pct": -5.0}
        up_pct, down_pct = float(th["up_pct"]), float(th["down_pct"])

        # not enough data → stable
        if n < 2 or not np.isfinite(s.iloc[0]) or not np.isfinite(s.iloc[-1]):
            return {
                "symbol": "→",
                "tier": "Trend → (Stable)",
                "dir_pct": 0.0,
                "pct_change": 0.0,
                "n_days": n,
                "method": "endpoint_delta",
                "thresholds_used": {"up_pct": up_pct, "down_pct": down_pct},
            }

        prev, now = float(s.iloc[0]), float(s.iloc[-1])
        delta = now - prev

        # raw whitepaper % (audit)
        if abs(prev) >= eps:
            raw_pct = (delta / prev) * 100.0
        else:
            raw_pct = (delta / 200.0) * 100.0  # ERI band fallback (−100..+100)

        # direction-safe % (classification)
        denom = abs(prev) if abs(prev) >= eps else 200.0
        dir_pct = (delta / denom) * 100.0

        # clamp & round
        pct_change = float(np.clip(round(raw_pct, 2), -100.0, 100.0))
        dir_pct_r  = float(np.clip(round(dir_pct, 2), -100.0, 100.0))

        # map to symbol using direction-safe %
        sym = "↑" if dir_pct_r >= up_pct else ("↓" if dir_pct_r <= down_pct else "→")
        tier = {"↑": "Trend ↑ (Improving)", "→": "Trend → (Stable)", "↓": "Trend ↓ (Declining)"}[sym]

        return {
            "symbol": sym,
            "tier": tier,
            "dir_pct": dir_pct_r,          # used for gating (direction-safe)
            "pct_change": pct_change,      # raw whitepaper % (audit)
            "n_days": n,
            "method": "endpoint_delta",
            "thresholds_used": {"up_pct": up_pct, "down_pct": down_pct},
        }

    def _compute_momentum_series(self, series, k=None) -> dict:
        """
        Momentum (whitepaper-simple, CX-friendly)
        - Compares mean(Last k days) vs mean(First k days).
        - If k is None: uses equal halves (k = floor(n/2)); ensures equal-length arcs.
        - Delta is scaled as % of the ERI band (200 pts).
        - Tiers (inclusive at the stable band edges):
            ↑↑  >= +20%
            ↑    >  +5% and < +20%
            →    between -5% and +5%  (inclusive)
            ↓    <  -5% and > -20%
            ↓↓  <= -20%
        """
        import numpy as np, pandas as pd

        s = pd.Series(series).astype(float)
        n = int(len(s))
        meta_neutral = self.mom_details("→")

        if n < 2:
            return {
                "symbol": "→",
                "delta": 0.0,
                "label": meta_neutral["label"],
                "description": meta_neutral["description"],
                "snr": None,
                "n_days": n,
                "emoji_full": meta_neutral.get("emoji_full"),
                "strength": meta_neutral.get("strength"),
                "polarity": meta_neutral.get("polarity"),
                "thresholds_used": self.momentum_thresholds(),
            }

        # choose equal-length arcs: first k vs last k
        if k is None:
            k = n // 2
        k = int(max(1, min(k, n // 2)))

        first_half  = s.iloc[:k]
        second_half = s.iloc[-k:]

        avg1 = float(np.nanmean(first_half))
        avg2 = float(np.nanmean(second_half))
        if not np.isfinite(avg1) or not np.isfinite(avg2):
            return {
                "symbol": "→",
                "delta": 0.0,
                "label": meta_neutral["label"],
                "description": meta_neutral["description"],
                "snr": None,
                "n_days": n,
                "emoji_full": meta_neutral.get("emoji_full"),
                "strength": meta_neutral.get("strength"),
                "polarity": meta_neutral.get("polarity"),
                "thresholds_used": self.momentum_thresholds(),
            }

        # scale to % of the 200-pt ERI band
        delta_raw   = avg2 - avg1
        delta_pct   = (delta_raw / 200.0) * 100.0
        delta_pct_r = float(np.clip(round(delta_pct, 2), -100.0, 100.0))

        t = self.momentum_thresholds()
        flat = float(t.get("flat_band", 5.0))
        r_mod, r_str = float(t["rise_mod"]), float(t["rise_strong"])
        f_mod, f_str = float(t["fall_mod"]), float(t["fall_strong"])

        # inclusive stable band per table (±5%)
        if -flat <= delta_pct_r <= flat:
            sym = "→"
        elif delta_pct_r >= r_str:
            sym = "↑↑"
        elif delta_pct_r > r_mod:
            sym = "↑"
        elif delta_pct_r <= f_str:
            sym = "↓↓"
        elif delta_pct_r < f_mod:
            sym = "↓"
        else:
            sym = "→"

        meta = self.mom_details(sym)
        return {
            "symbol": sym,
            "delta": delta_pct_r,       # % of band
            "label": meta["label"],
            "description": meta["description"],
            "snr": None,                # reserved; whitepaper-simple
            "n_days": n,
            "emoji_full": meta.get("emoji_full"),
            "strength": meta.get("strength"),
            "polarity": meta.get("polarity"),
            "thresholds_used": {
                "rise_mod": r_mod, "rise_strong": r_str,
                "fall_mod": f_mod, "fall_strong": f_str,
                "flat_band": flat
            },
        }


    """

## **Trend x Momentum Grid**

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

👉 **Why it's powerful:** This grid translates **numbers into moves**. Teams don't just know “how customers feel,” they know **when to amplify, when to stabilize, when to fix, and when to mobilize crisis response.**
        
    """   
    
    def _compute_volatility_series(self, series) -> dict:
        """
        Volatility (Whitepaper 3-Level Version)

        - Input: iterable/Series of ERI values in [-100, +100]; may contain None/NaN/±inf.
        - Uses ONLY observed points (drops NaN/None/±inf). No imputation. No detrending.
        - Requires n >= 2 to compute std. dev. (ddof=1).
        - Score = standard deviation of ERI over the window.
        - Tiers:
            0-15  -> ✅ Stable
            16-45 -> ⚠ Fluctuating
            46+   -> 🔴 Highly Fluctuating
        - Returns keys kept broadly compatible with prior usage.
        """
        import numpy as np

        # Coerce to float array and drop non-finite values
        arr = np.asarray(list(series), dtype=float)
        arr = arr[np.isfinite(arr)]
        n = int(arr.size)

        if n < 2:
            return {
                "tier": "Insufficient Data",
                "score": None,
                "score_adj": None,   # kept for compatibility; same as score when available
                "n_days": n,
                "method": "std",
                "detrended": False
            }

        # Plain standard deviation over observed ERI values (no detrend, no normalization)
        sigma = float(np.std(arr, ddof=1))
        score = round(sigma, 2)

        # Whitepaper tiers
        if score <= 15:
            tier = "✅ Stable"
        elif score <= 45:
            tier = "⚠ Fluctuating"
        else:
            tier = "🔴 Highly Fluctuating"

        return {
            "tier": tier,
            "score": score,
            "score_adj": score,  # identical in whitepaper mode (no extra adjustments)
            "n_days": n,
            "method": "std",
            "detrended": False
        }


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

    def _compute_pattern_series(self, series) -> dict:
        """
        Pattern Recognition (whitepaper) — always returns a full payload.
        If no cycle ≥ 0.30, we report 'No Pattern' with ACF scores filled.
        """
        import numpy as np
        import pandas as pd

        WEEKLY_MIN, MONTHLY_MIN, QUARTERLY_MIN = 8, 31, 91
        s = pd.Series(series).astype(float)
        n = int(len(s))

        def _analysis_period():
            if isinstance(s.index, pd.DatetimeIndex) and n > 0:
                return [str(s.index.min().date()), str(s.index.max().date())]
            return None

        def _eri_by_day_weekly():
            # Provide weekly breakdown if we have dates; else return a null-filled dict
            names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            if isinstance(s.index, pd.DatetimeIndex) and n >= WEEKLY_MIN:
                by_dow = s.groupby(s.index.dayofweek).mean(numeric_only=True)
                return {names[i]: (float(by_dow.get(i)) if i in by_dow.index else None) for i in range(7)}
            return {d: None for d in names}

        def _acf_at_lag(x: np.ndarray, lag: int) -> float:
            if x.size == 0 or lag >= x.size:
                return np.nan
            x = x.astype(float)
            x = x - np.nanmean(x)
            num = np.nansum(x[:-lag] * x[lag:])
            den = np.nansum(x * x)
            val = (num / den) if den > 0 else np.nan
            return float(np.clip(val, -1.0, 1.0))

        # If we can't even test the smallest official cycle, still return a filled payload
        if n < WEEKLY_MIN:
            return {
                "has_pattern": False,
                "pattern_type": "None",
                "pattern_strength": 0.0,
                "confidence": "Low",
                "confidence_tier": "🔴 Weak / Candidate",
                "pain_day": None,
                "eri_by_day": _eri_by_day_weekly(),
                "data_coverage_days": n,
                "min_required_days": {"Weekly": WEEKLY_MIN, "Monthly": MONTHLY_MIN, "Quarterly": QUARTERLY_MIN},
                "acf": {
                    "weekly":   {"lag_days": 7,  "score": None},
                    "monthly":  {"lag_days": 30, "score": None},
                    "quarterly":{"lag_days": 90, "score": None},
                    "chosen":   {"lag_days": None, "score": None}
                },
                "analysis_period": _analysis_period(),
                "status": "Insufficient Data",
                "reason": f"Need ≥{WEEKLY_MIN}d for Weekly tests."
            }

        # Compute ACFs (we have enough data for at least weekly)
        acf_w = _acf_at_lag(s.values, 7)   if n >= WEEKLY_MIN    else np.nan
        acf_m = _acf_at_lag(s.values, 30)  if n >= MONTHLY_MIN   else np.nan
        acf_q = _acf_at_lag(s.values, 90)  if n >= QUARTERLY_MIN else np.nan

        candidates = []
        if np.isfinite(acf_w): candidates.append(("Weekly",    abs(acf_w), 7,  acf_w))
        if np.isfinite(acf_m): candidates.append(("Monthly",   abs(acf_m), 30, acf_m))
        if np.isfinite(acf_q): candidates.append(("Quarterly", abs(acf_q), 90, acf_q))

        # If nothing is computable (e.g., all NaN), still return a filled 'No Pattern'
        if not candidates:
            best = ("None", 0.0, None, 0.0)
        else:
            # choose best |ACF|; tie-break prefers shorter cycle Weekly>Monthly>Quarterly
            rank = {"Weekly": 3, "Monthly": 2, "Quarterly": 1}
            candidates.sort(key=lambda t: (t[1], rank.get(t[0], 0)), reverse=True)
            best = candidates[0]

        pattern_type, strength_abs, lag_days, signed_acf = best
        strength = round(float(strength_abs), 3)

        # confidence per whitepaper thresholds
        if strength_abs >= 0.60:
            confidence, confidence_tier = "High", "✅ Strong"
        elif strength_abs >= 0.30:
            confidence, confidence_tier = "Medium", "⚠ Moderate"
        else:
            confidence, confidence_tier = "Low", "🔴 Weak / Candidate"

        has_pattern = bool(strength_abs >= 0.30)

        # only compute pain_day if we actually have a Weekly *and* a pattern; otherwise None
        pain_day = None
        if has_pattern and pattern_type == "Weekly" and isinstance(s.index, pd.DatetimeIndex):
            by_dow = s.groupby(s.index.dayofweek).mean(numeric_only=True)
            if by_dow.size > 0:
                names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                pain_day = names[int(by_dow.idxmin())]

        return {
            "has_pattern": has_pattern,                            # False when no discernible pattern
            "pattern_type": pattern_type if has_pattern else "None",
            "pattern_strength": strength if has_pattern else 0.0,  # keep 0.0 for 'No Pattern'
            "confidence": confidence if has_pattern else "Low",
            "confidence_tier": confidence_tier if has_pattern else "🔴 Weak / Candidate",
            "pain_day": pain_day if has_pattern else None,
            "eri_by_day": _eri_by_day_weekly(),                    # filled dict, even if 'None'
            "data_coverage_days": n,
            "min_required_days": {"Weekly": WEEKLY_MIN, "Monthly": MONTHLY_MIN, "Quarterly": QUARTERLY_MIN},
            "acf": {
                "weekly":   {"lag_days": 7,  "score": None if not np.isfinite(acf_w) else round(float(acf_w), 3)},
                "monthly":  {"lag_days": 30, "score": None if not np.isfinite(acf_m) else round(float(acf_m), 3)},
                "quarterly":{"lag_days": 90, "score": None if not np.isfinite(acf_q) else round(float(acf_q), 3)},
                # even if no pattern, show which cycle had the highest |ACF| (for transparency)
                "chosen":   {"lag_days": int(lag_days) if lag_days is not None else None,
                            "score": round(float(signed_acf), 3) if lag_days is not None else None}
            },
            "analysis_period": _analysis_period(),
            "status": "No Pattern" if not has_pattern else "OK",
            "reason": None if has_pattern else "All cycle scores below 0.30 (no discernible recurring pattern)."
        }



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
        sat_centrality = float(min(1.0, sat_distance / half_w))  # 0..1, center≈1, edge≈0
        borderline     = sat_centrality <= 0.05  # within 5% of a tier boundary

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
        QSSI (Quantified Signal Strength Index), whitepaper-exact.
        QSSI = Velocity(Trend x Momentum, 0-6) + Saturation Modifier (0-4 by SI bands).
        Tiers:
        9-10 💥 Critical | 6-8 🔥 Strong | 4-5 🌱 Emerging | 1-3 🔁 Weak | 0 ❌ No Signal
        """
        # --- sanitize inputs & accept labels ---
        trend_in = (trend_symbol or "→").strip()
        if trend_in in ("↑","→","↓"):
            trend = trend_in
        else:
            tl = trend_in.lower()
            trend = "↑" if tl.startswith("improv") else ("↓" if tl.startswith("declin") else "→")

        mom_in = (momentum_symbol or "→").strip()
        if mom_in in ("↑↑","↑","→","↓","↓↓"):
            momentum = mom_in
        else:
            # try label → symbol if provided (e.g., "Moderately Rising")
            momentum = self.mom_symbol_from_label(mom_in) if hasattr(self, "mom_symbol_from_label") else "→"

        try:
            si = float(saturation_index)
        except Exception:
            si = 0.5
        si = max(0.0, min(1.0, si))

        # --- Velocity score (0–6) per table ---
        velocity_lookup = {
            ("↑", "↑↑"): 5, ("↓", "↑↑"): 5,
            ("↑", "↓↓"): 6, ("↓", "↓↓"): 6,
            ("↑", "↑"): 4,  ("↑", "↓"): 4, ("↓", "↑"): 4, ("↓", "↓"): 4,
            ("↑", "→"): 2,  ("↓", "→"): 2,
            ("→", "↑↑"): 3, ("→", "↓↓"): 3,
            ("→", "↑"): 1,  ("→", "↓"): 1,
            ("→", "→"): 0,
        }
        velocity_score = int(velocity_lookup.get((trend, momentum), 0))

        # --- Saturation modifier (0–4) per QSSI bands (NOT the general saturation bins) ---
        if   si <= 0.20: saturation_modifier = 4
        elif si <= 0.40: saturation_modifier = 3
        elif si <= 0.60: saturation_modifier = 2
        elif si <= 0.80: saturation_modifier = 1
        else:            saturation_modifier = 0

        # (optional) pull human-readable tier from your saturation contract, but do not use its qssi_score
        sat_band = None
        if hasattr(self, "sat_from_si"):
            sat_info = self.sat_from_si(si)
            sat_band = sat_info.get("clean", None)

        # --- QSSI total & tiering ---
        qssi_score = int(velocity_score + saturation_modifier)
        if   qssi_score >= 9:  tier_emoji, tier, desc = "💥", "Critical Signal", "Signal is erupting/collapsing — act immediately"
        elif qssi_score >= 6:  tier_emoji, tier, desc = "🔥", "Strong Signal",   "Signal gaining strength — prioritize intervention"
        elif qssi_score >= 4:  tier_emoji, tier, desc = "🌱", "Emerging Signal", "Early signal — monitor or pre-activate"
        elif qssi_score >= 1:  tier_emoji, tier, desc = "🔁", "Weak Signal",     "Low movement — low urgency"
        else:                  tier_emoji, tier, desc = "❌", "No Signal",       "No trend, no motion, no headroom"

        return {
            "qssi_score": qssi_score,                          # 0–10
            "qssi_tier": f"{tier_emoji} {tier}",
            "qssi_description": desc,
            "components": {
                "velocity_score": velocity_score,
                "saturation_modifier": saturation_modifier,
                "saturation_index": round(si, 3),
                "saturation_band": sat_band,                   # informational only
            },
            "inputs": {
                "trend_symbol": trend,
                "momentum_symbol": momentum,
            }
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

    * **Volatility** → *“How predictable is this signal for planning?”*
    Measures stability versus noise, distinguishing reliable patterns from fragile fluctuations.

    * **Pattern Recognition** → *“When will this happen again?”*
    Identifies recurring cycles (weekly, monthly, quarterly) and potential “pain days.”

    * **Saturation x Momentum** → *“What's the strategic context?”*
    Places the signal inside a **25-cell quadrant map**, assigning diagnostic labels, 
    urgency codes, and strategic narratives that frame the bigger picture.

    * **QSSI (Quantified Signal Pointer)** → *“How strong and urgent is this signal?”*
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

    # === PEM (Predictive Emotional Modeling) — STRICT (no confidence scoring) ===
    # ---------- canonical label maps (no deductions) ----------
    _TM_TREND = {
        "↑": "Trend ↑ (Improving)",
        "→": "Trend → (Stable)",
        "↓": "Trend ↓ (Declining)",
    }
    _TM_MOMENTUM = {
        "↑↑": "Momentum ↑ (Improving)",
        "↑":  "Momentum ↑ (Improving)",
        "→":  "Momentum → (Stable)",
        "↓":  "Momentum ↓ (Worsening)",
        "↓↓": "Momentum ↓ (Worsening)",
    }

    _TMV_TREND = {
        "↑": "↑ (Improving)",
        "→": "→ (Flat)",
        "↓": "↓ (Declining)",
    }
    _TMV_MOMENTUM = {
        "↑↑": "↑ (Recent rise)",
        "↑":  "↑ (Recent rise)",
        "→":  "→ (Stable)",
        "↓":  "↓ (Recent dip)",
        "↓↓": "↓ (Recent dip)",
    }

    # ---------- normalization helpers ----------
    def _norm_vol_to_tmv(self, vol_tier: str) -> str:
        vt = (vol_tier or "").lower()
        if "high" in vt or "🔴" in vt:
            return "🔴 Highly Fluctuating"
        if "stable" in vt or "✅" in vt:
            return "✅ Stable"
        return "⚠ Fluctuating"

    def _norm_vol_simple(self, vol_tier: str) -> str:
        vt = (vol_tier or "").lower()
        if "high" in vt or "🔴" in vt:
            return "Highly Fluctuating"
        if "stable" in vt or "✅" in vt:
            return "Stable"
        return "Fluctuating"

    # ---------- QSSI tiering (display only) ----------
    def _qssi_tier(self, qssi: int) -> tuple[str, str]:
        q = int(max(0, min(10, qssi)))
        if q >= 9:  return ("💥 Critical", "critical")
        if q >= 6:  return ("🔥 Strong", "strong")
        if q >= 4:  return ("🌱 Emerging", "emerging")
        if q >= 1:  return ("🔁 Weak", "weak")
        return ("❌ No Signal", "none")

    # ---------- horizon (pattern windows only; else use provided fallback) ----------
    def _horizon_days(self, pattern_type: str | None, fallback_days: int) -> tuple[int, int]:
        """
        If a pattern exists, return fixed windows.
        Otherwise, DO NOT invent horizon — use the provided fallback.
        """
        pt = (pattern_type or "").lower()
        if pt == "weekly":    return (7, 14)
        if pt == "monthly":   return (21, 35)
        if pt == "quarterly": return (60, 100)
        return (fallback_days, fallback_days)

    # ---------- grid lookups (direct table reads; no deductions) ----------
    def _lookup_tm(self, trend_symbol: str, momentum_symbol: str) -> dict | None:
        df = get_trend_momentum_grid()
        t = self._TM_TREND.get(trend_symbol, self._TM_TREND["→"])
        m = self._TM_MOMENTUM.get(momentum_symbol, self._TM_MOMENTUM["→"])
        row = df[(df["Trend_Tier"] == t) & (df["Momentum_Tier"] == m)]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "label": r["Diagnostic_Label"],
            "narrative": r["Strategic_Narrative"],
            "guidance": r["Action_Guidance"],
        }

    def _lookup_tmv(self, trend_symbol: str, momentum_symbol: str, vol_tier: str) -> dict | None:
        df = get_trend_momentum_volatility_grid()
        t = self._TMV_TREND.get(trend_symbol, self._TMV_TREND["→"])
        m = self._TMV_MOMENTUM.get(momentum_symbol, self._TMV_MOMENTUM["→"])
        v = self._norm_vol_to_tmv(vol_tier)
        row = df[(df["Trend_Tier"] == t) & (df["Momentum_Tier"] == m) & (df["Volatility_Tier"] == v)]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "interpretation": r["Interpretation"],
            "instruction": r["CX_Instruction"],
        }

    def _lookup_pattern(self, pattern_type: str | None, pattern_confidence: str | None) -> dict | None:
        if not pattern_type or not pattern_confidence:
            return None
        df = get_temporal_pattern_grid()
        pc = (pattern_confidence or "").strip().lower()
        if "strong" in pc:
            conf = "✅ Strong"
        elif "moderate" in pc:
            conf = "⚠ Moderate"
        else:
            conf = "🔴 Weak / Candidate"
        pt = (pattern_type or "").strip().title()
        row = df[(df["Pattern_Type"] == pt) & (df["Confidence_Level"] == conf)]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "interpretation": r["Interpretation"],
            "instruction": r["CX_Instruction"],
            "conf_level": conf,
            "type": pt,
        }

    # ---------- plain direction label (exactly from symbols) ----------
    def _direction_label(self, tr: str, mo: str) -> str:
        if tr == "↑" and mo in ("↑↑", "↑"): return "improving"
        if tr == "↓" and mo in ("↓↓", "↓"): return "declining"
        if tr == "↑" and mo == "→":        return "improving (recent plateau)"
        if tr == "↓" and mo == "→":        return "declining (recent plateau)"
        if tr == "→" and mo in ("↑↑", "↑"): return "stabilizing upward"
        if tr == "→" and mo in ("↓↓", "↓"): return "stabilizing downward"
        if tr == "↓" and mo in ("↑↑", "↑"): return "recovering"
        if tr == "↑" and mo in ("↓↓", "↓"): return "cooling"
        return "stable"

    # ---------- PEM builder (STRICT — no confidence fields) ----------
    def build_predictive_emotional_modeling(
        self,
        *,
        trend_symbol: str,                 # "↑","→","↓"
        momentum_symbol: str,              # "↑↑","↑","→","↓","↓↓"
        volatility_tier: str,              # freeform; normalized
        volatility_score: float,           # pct (0..100) display
        pattern_detected: bool,
        pattern_type: str | None,          # "Weekly" | "Monthly" | "Quarterly" | "Operational Timing" | None
        pattern_confidence: str | None,    # "Strong" | "Moderate" | "Weak" | "Candidate" | None
        pain_day: str | None,              # weekday if weekly
        qssi_score: int,                   # 0..10 (gate + tier label only)
        momentum_saturation_quadrant: dict | None,  # from MS 25-cell (optional)
        saturation_index: float | None = None,      # optional (display only)
        horizon_days: int = 30                         # explicit fallback window
    ) -> dict:

        tr = trend_symbol if trend_symbol in ("↑", "→", "↓") else "→"
        mo = momentum_symbol if momentum_symbol in ("↑↑", "↑", "→", "↓", "↓↓") else "→"
        vt = self._norm_vol_simple(volatility_tier)
        qssi = int(max(0, min(10, int(qssi_score or 0))))

        # lookups (direct; no inference)
        tm  = self._lookup_tm(tr, mo) or {}
        tmv = self._lookup_tmv(tr, mo, volatility_tier) or {}
        msq = momentum_saturation_quadrant or {}
        msq_label = (msq.get("quadrant_interpretation") or {}).get("quadrant_label")

        grid_pattern_type = pattern_type
        if pattern_detected:
            # seasonal cycles in your grid are labeled "Seasonal Cycle"
            if (pattern_type or "").lower() in {"weekly", "monthly", "quarterly"}:
                grid_pattern_type = "Seasonal Cycle"
            # if it's weekly and we have a pain day, prefer "Pain Day"
            if (pattern_type or "").lower() == "weekly" and pain_day:
                grid_pattern_type = "Pain Day"

        pat = self._lookup_pattern(grid_pattern_type if pattern_detected else None, pattern_confidence)
       
        qssi_tier_emoji, _ = self._qssi_tier(qssi)
        h_min, h_max = self._horizon_days((pattern_type if pattern_detected else None), horizon_days)

        # weak signal → no forecast (gate by QSSI)
        if qssi < 4:
            return {
                "forecast": {
                    "summary": "Insufficient signal for a forward pointer — keep monitoring.",
                    "basis": "Low signal strength (QSSI < 4)",
                    "horizon_days": h_max
                },
                "watch": {
                    "focus": ["Trend direction", "Momentum shifts", "Volatility changes"],
                    "horizon_days": h_max
                },
                "signal_synthesis": {
                    "trend": tr, "momentum": mo, "volatility": vt,
                    "qssi": qssi, "qssi_tier": qssi_tier_emoji,
                    "pattern": {"detected": False}
                },
                "meta": {
                    "quadrant_TM": tm.get("label"),
                    "quadrant_TMV": tmv.get("interpretation"),
                    "quadrant_MS": msq_label,
                }
            }

        # pattern-led forecast (when available)
        if pat:
            dir_lbl = self._direction_label(tr, mo)

            # grid type from the Temporal Pattern Action Grid
            gt   = (pat.get("type") or "").strip().lower()      # "pain day" | "seasonal cycle" | "operational timing"
            # original detection type from the pattern module (for extra color)
            orig = (pattern_type or "").strip().lower()         # "weekly" | "monthly" | "quarterly" | "operational timing"

            when_txt = ""
            if gt == "pain day":
                when_txt = f" (likely on {pain_day})" if pain_day else " (recurring weekly pain day)"
            elif gt == "seasonal cycle":
                cyc = {"weekly": "weekly", "monthly": "monthly", "quarterly": "quarterly"}.get(orig)
                when_txt = f" ({cyc} recurrence)" if cyc else " (seasonal recurrence)"
            elif gt == "operational timing":
                when_txt = " (operational timing window)"

            summary = (
                f"{pat['interpretation']}{when_txt}. "
                f"Current trajectory is {dir_lbl}"
                + ("" if vt == "Stable" else f" and {vt.lower()}.")
            )
            basis = f"Pattern-led forecast: {pat['type']} / {pat['conf_level']}"
            if tm.get("label"): basis += f"; TM grid → {tm['label']}"
            if tmv.get("interpretation"): basis += f"; TMV grid → {tmv['interpretation']}"
            if msq_label: basis += f"; MS quadrant → {msq_label}"

            return {
                "forecast": {
                    "summary": summary,
                    "basis": basis,
                    "horizon_days": f"{h_min}-{h_max}"
                },
                "watch": {
                    "windows": [pat["type"]],
                    "notes": [pat["instruction"]],
                    "horizon_days": h_max
                },
                "signal_synthesis": {
                    "trend": tr,
                    "momentum": mo,
                    "volatility": vt,
                    "volatility_score_pct": round(float(volatility_score or 0.0), 2),
                    "qssi": qssi,
                    "qssi_tier": qssi_tier_emoji,
                    "saturation_index": None if saturation_index is None else round(float(saturation_index), 3),
                    "pattern": {
                        "detected": True,
                        "type": pat["type"],
                        "confidence": pat["conf_level"],
                        "pain_day": pain_day
                    }
                },
                "meta": {
                    "quadrant_TM": tm.get("label"),
                    "quadrant_TMV": tmv.get("interpretation"),
                    "quadrant_MS": msq_label,
                }
            }

        # trajectory-led forecast (no strong pattern)
        dir_lbl = self._direction_label(tr, mo)
        basis_parts = []
        if tm.get("label"):
            basis_parts.append(f"TM grid → {tm['label']}: {tm.get('narrative','')}".strip())
        if tmv.get("interpretation"):
            basis_parts.append(f"TMV grid → {tmv['interpretation']}")
        if msq_label:
            basis_parts.append(f"MS quadrant → {msq_label}")
        basis = "; ".join([p for p in basis_parts if p]) or "Directional synthesis from Trend×Momentum and Volatility tiering"

        summary = (
            f"Over the next {h_min}-{h_max} days the signal is likely {dir_lbl}"
            + ("" if vt == "Stable" else f" with {vt.lower()}")
            + "."
        )

        return {
            "forecast": {
                "summary": summary,
                "basis": basis,
                "horizon_days": f"{h_min}-{h_max}"
            },
            "watch": {
                "focus": ["Momentum flips", "Volatility spikes", "Quadrant changes"],
                "horizon_days": h_max
            },
            "signal_synthesis": {
                "trend": tr, "momentum": mo,
                "volatility": vt, "volatility_score_pct": round(float(volatility_score or 0.0), 2),
                "qssi": qssi, "qssi_tier": qssi_tier_emoji,
                "saturation_index": None if saturation_index is None else round(float(saturation_index), 3),
                "pattern": {"detected": False}
            },
            "meta": {
                "quadrant_TM": tm.get("label"),
                "quadrant_TMV": tmv.get("interpretation"),
                "quadrant_MS": msq_label,
            }
        }
   
    def compute(self):
        results, skipped = [], []

        # Normalize ED keys
        self.raw_df["experience_driver"] = (
            self.raw_df["experience_driver"].astype(str).str.strip()
        )

        # L2-to-L3 eligibility
        entities = self.layer2_df[self.layer2_df["Priority_Status"].isin(["P0","P1","P2","P3"])]

        # ---- Build exact analysis window [start, end=today-1], length = timeframe_days ----
        end_date = self.today - timedelta(days=1)            # inclusive end
        start_date = self.cutoff_date                        # intended start
        expected_span = int(self.timeframe_days)
        actual_span = (end_date - start_date).days + 1
        if actual_span != expected_span:
            start_date = end_date - timedelta(days=expected_span - 1)

        # Keep as DatetimeIndex so weekly pain-day works downstream
        full_idx = pd.date_range(start=start_date, end=end_date, freq="D")
        analysis_window_days = len(full_idx)

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
                # ensure DatetimeIndex
                daily_eri_obs.index = pd.to_datetime(daily_eri_obs.index)

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
                ds = data.groupby("date").size()
                ds.index = pd.to_datetime(ds.index)
                ds = ds.reindex(full_idx, fill_value=0)
                pct_days_with_signal = round(float((ds > 0).sum()) / analysis_window_days, 3)

                # ---- Interpolate for continuity (analysis track only) ----
                daily_eri = daily_eri.interpolate(method="linear", limit_direction="both")

                # =========================
                # L3 MODULES (the 7 diagnostics)
                # =========================

                # 1) Trend (whitepaper)
                trend = self._compute_trend_series(daily_eri)

                # 2) Momentum (whitepaper)
                momentum = self._compute_momentum_series(daily_eri)

                # 3) Volatility (whitepaper: std dev tiers)
                volatility = self._compute_volatility_series(daily_eri)
                vol_score = float(volatility.get("score_adj", volatility.get("score", 0.0)))

                # 4) Pattern Recognition (whitepaper: ACF 7/30/90; full payload even for "No Pattern")
                pattern_block = self._compute_pattern_series(daily_eri)

                # Current ERI (for SI/quadrant)
                eri_now = float(daily_eri.iloc[-1]) if np.isfinite(daily_eri.iloc[-1]) else 0.0

                # 5) Momentum × Saturation quadrant (matrix is source of truth)
                momentum_symbol = momentum.get("symbol", "→")
                quadrant_block = self._compute_momentum_saturation_insight(
                    eri_score=eri_now,
                    momentum_symbol=momentum_symbol,
                    # meta-only modifiers (do not affect matrix selection)
                    signal_presence_pct=pct_days_with_signal,
                    vol_norm=min(max(vol_score / 20.0, 0.0), 1.0),
                    momentum_snr=momentum.get("snr"),
                    series_data_complete=series_data_complete
                )

                # 6) QSSI (velocity + saturation modifier; 0–10)
                qssi_block = self._compute_signal_strength(
                    trend_symbol=trend.get("symbol", "→"),
                    momentum_symbol=momentum_symbol,
                    saturation_index=quadrant_block["signal_classification"]["saturation_index"],
                )

                # 7) PEM (STRICT; pattern grid mapping handled inside)
                pem_block = self.build_predictive_emotional_modeling(
                    trend_symbol=trend.get("symbol", "→"),
                    momentum_symbol=momentum_symbol,
                    volatility_tier=volatility.get("tier", "✅ Stable"),
                    volatility_score=vol_score,
                    pattern_detected=bool(pattern_block.get("has_pattern")),
                    pattern_type=pattern_block.get("pattern_type"),
                    pattern_confidence=pattern_block.get("confidence_tier"),
                    pain_day=pattern_block.get("pain_day"),
                    qssi_score=int(qssi_block["qssi_score"]),
                    momentum_saturation_quadrant=quadrant_block,
                    saturation_index=quadrant_block["signal_classification"]["saturation_index"],
                    horizon_days=30
                )

                # ---- Storyline (diagnostic-only; safe access) ----
                qi = quadrant_block.get("quadrant_interpretation", {}) or {}
                sc = quadrant_block.get("signal_classification", {}) or {}
                storyline = (
                    f"{ed} is in {qi.get('quadrant_label', 'Unknown')} with "
                    f"{sc.get('momentum_tier', '?')} momentum and "
                    f"{sc.get('saturation_tier', '?')} saturation. "
                    f"{qi.get('interpretation', '')}"
                ).strip()

                # Capsule meta for provenance
                capsule_meta = {
                    "capsule_id": f"SC-{uuid4().hex[:12]}",
                    "generated_at": pd.Timestamp.utcnow().isoformat(),
                    "capsule_type": "ED",
                    "capsule_version": "signal_capsule.v1.0",
                    "engine_version": "XDI.v1",
                    "window_start_date": str(start_date),
                    "window_end_date": str(end_date)
                }

                # ---- Assemble ED-centric Signal Capsule (7 modules + provenance) ----
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

                    # === L3 modules (the 7) ===
                    "trend_block": trend,
                    "momentum_block": momentum,
                    "volatility_block": volatility,
                    "pattern_block": pattern_block,
                    "momentum_saturation_insight": quadrant_block,     # (includes saturation/quadrant)
                    "signal_strength_index": qssi_block,               # QSSI
                    "predictive_emotion_forecast": pem_block,          # PEM

                    # Descriptive narrative + native module confidences
                    "momentum_saturation_story": storyline,
                    "momentum_saturation_confidence": quadrant_block.get("meta", {}).get("quadrant_confidence"),
                    "momentum_saturation_borderline_flag": quadrant_block.get("meta", {}).get("borderline"),

                    # Provenance
                    "provenance": {
                        "analysis_window_days": int(self.timeframe_days),
                        "observed_days_with_signal": observed_days,
                        "signal_presence_pct": pct_days_with_signal,
                        "trend_analysis_days": analysis_window_days,
                        "momentum_analysis_days": analysis_window_days // 2,
                        "series_data_complete": series_data_complete,
                        "missing_days_pct": missing_days_pct,
                        **capsule_meta
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
