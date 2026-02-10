
from __future__ import annotations

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import math
import sqlite3
import os
import re
import json
from datetime import timedelta
from typing import Dict, Any
from collections import Counter
from uuid import uuid4

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# =============================================================================
# OU Layer 2 V7 — FULL REPLICA (Behavioral Clustering) updated for v2 axes
# =============================================================================
# Field reality changes:
# - identity: (experience_driver, intent_axis, action_axis, affect_axis.affect_label)
# - context: context.text, context.keywords  -> context_text, context_keywords (virtual)
# - journey: patient_journey, journey_stage, interaction_moment
# - SAS: semantic_action_statement.section_1_patient_reality (+ section_2_strategic_response)
#
# Clustering mechanics preserved: weighted signature → embeddings → agglom/HDBSCAN
# + adaptive min-cluster + one-bump τ controller + centroid/MMR consolidation + DB.
# =============================================================================


class FlexibleTimeframeAnalyzer:
    def __init__(self, input_path, output_dir="outputs", timeframe_days=75, compute_granular=True, verbose=True):
        self.input_path = input_path
        self.output_dir = output_dir
        self.date_range_type = "window"
        self.timeframe_days = timeframe_days
        self.compute_granular = compute_granular
        self.verbose = verbose

        os.makedirs(self.output_dir, exist_ok=True)

        # Load input
        self.df = pd.read_csv(self.input_path)
        if "date" in self.df.columns:
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce").dt.date
            self.latest_data_date = self.df["date"].max()
            self.cutoff_date = self.latest_data_date - timedelta(days=self.timeframe_days)
            self.start_date = self.cutoff_date
            self.end_date = self.latest_data_date
            self.raw_df = self.df[self.df["date"].between(self.cutoff_date, self.latest_data_date)]
        else:
            self.latest_data_date = None
            self.cutoff_date = None
            self.start_date = None
            self.end_date = None
            self.raw_df = self.df.copy()

        # Holders (kept for compatibility with your pipeline)
        self.layer2_df = None
        self.layer3_df = None
        self.details_df = None

        # --- Clustering config (preserved) ---
        self.OU_CFG = {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "min_cluster_size": 8,
            "adaptive_min_cluster": {"enabled": True, "pct": 0.08, "floor": 3},
            "bcs_cumu_threshold": 0.80,
            "skip_singletons": False,
            "preview_location_style": "short",

            "signature_threshold_start": {"default_K1": 0.30, "fallback_K3": 0.32},
            "bcs_distance_threshold_bounds": (0.25, 0.35),
            "bcs_distance_threshold_step": 0.02,
            "bcs_threshold_max_adjustments": 1,
            "cluster_band_target": (3, 8),
            "singleton_cap": 0.40,
            "bcs_distance_threshold": 0.30,
        }

        # v2 signature fields
        self.ALLOWED_FOR_SIGNATURE = frozenset({
            "context_text",
            "context_keywords",
            "patient_journey",
            "journey_stage",
            "interaction_moment",
            "semantic_customer_reality",  # SAS section 1
            "matters",
            "action_axis_justification",
            "behavioral_impact",
        })

        # Signature weights (updated field names; same intent)
        self.SIGNATURE_LIBRARY = {
            "default_K1": {
                "matters": 6,
                "interaction_moment": 4,
                "journey_stage": 4,
                "context_text": 3,
                "semantic_customer_reality": 2,
                "behavioral_impact": 1,
                "action_axis_justification": 1,
                "patient_journey": 1,
                "context_keywords": 1,
            },
            "fallback_K3": {
                "matters": 6,
                "context_text": 4,
                "semantic_customer_reality": 3,
                "interaction_moment": 3,
                "journey_stage": 2,
                "behavioral_impact": 2,
                "action_axis_justification": 1,
                "patient_journey": 1,
                "context_keywords": 1,
            },
        }

        self.OU_CFG["signature_config_name"] = "default_K1"
        self.OU_CFG["signature_weights"] = self.SIGNATURE_LIBRARY[self.OU_CFG["signature_config_name"]]
        self.OU_CFG["bcs_distance_threshold"] = self.OU_CFG["signature_threshold_start"][self.OU_CFG["signature_config_name"]]

        self._st_model = None
        self._ensure_st_model()

    # ---------- cached model helper ----------
    def _ensure_st_model(self):
        if getattr(self, "_st_model", None) is None:
            model_name = self.OU_CFG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            self._st_model = SentenceTransformer(model_name)
        return self._st_model

    # ---------- v2 field extraction helpers ----------
    def _get_affect_label(self, row: pd.Series) -> str:
        if "affect_label" in row and pd.notna(row.get("affect_label")):
            return str(row.get("affect_label")).strip()
        ax = row.get("affect_axis")
        if isinstance(ax, dict):
            return str(ax.get("affect_label") or "").strip() or "Unknown"
        if isinstance(ax, str) and ax.strip():
            try:
                obj = json.loads(ax)
                if isinstance(obj, dict):
                    return str(obj.get("affect_label") or "").strip() or "Unknown"
            except Exception:
                pass
        return "Unknown"

    def _get_affect_confidence(self, row: pd.Series) -> str:
        ax = row.get("affect_axis")
        if isinstance(ax, dict):
            return str(ax.get("confidence_level") or "").strip()
        if isinstance(ax, str) and ax.strip():
            try:
                obj = json.loads(ax)
                if isinstance(obj, dict):
                    return str(obj.get("confidence_level") or "").strip()
            except Exception:
                pass
        if "affect_confidence_level" in row and pd.notna(row.get("affect_confidence_level")):
            return str(row.get("affect_confidence_level")).strip()
        return ""

    def _get_context_text(self, row: pd.Series) -> str:
        if "context_text" in row and pd.notna(row.get("context_text")):
            return str(row.get("context_text")).strip()
        ctx = row.get("context")
        if isinstance(ctx, dict):
            return str(ctx.get("text") or "").strip()
        if isinstance(ctx, str) and ctx.strip():
            try:
                obj = json.loads(ctx)
                if isinstance(obj, dict):
                    return str(obj.get("text") or "").strip()
            except Exception:
                pass
            return ctx.strip()
        return ""

    def _get_context_keywords(self, row: pd.Series):
        if "context_keywords" in row and row.get("context_keywords") is not None:
            return row.get("context_keywords")
        ctx = row.get("context")
        if isinstance(ctx, dict):
            return ctx.get("keywords") or []
        if isinstance(ctx, str) and ctx.strip():
            try:
                obj = json.loads(ctx)
                if isinstance(obj, dict):
                    return obj.get("keywords") or []
            except Exception:
                pass
        return []

    def _get_sas_section_1(self, row: pd.Series) -> str:
        if "semantic_customer_reality" in row and pd.notna(row.get("semantic_customer_reality")):
            return str(row.get("semantic_customer_reality")).strip()
        sas = row.get("semantic_action_statement")
        if isinstance(sas, dict):
            return str(sas.get("section_1_patient_reality") or "").strip()
        if isinstance(sas, str) and sas.strip():
            try:
                obj = json.loads(sas)
                if isinstance(obj, dict):
                    return str(obj.get("section_1_patient_reality") or "").strip()
            except Exception:
                pass
            return sas.split("\n\n", 1)[0].strip()
        return ""

    def _get_sas_section_2(self, row: pd.Series) -> str:
        sas = row.get("semantic_action_statement")
        if isinstance(sas, dict):
            return str(sas.get("section_2_strategic_response") or "").strip()
        if isinstance(sas, str) and sas.strip():
            try:
                obj = json.loads(sas)
                if isinstance(obj, dict):
                    return str(obj.get("section_2_strategic_response") or "").strip()
            except Exception:
                pass
        if "strategic_response" in row and pd.notna(row.get("strategic_response")):
            return str(row.get("strategic_response")).strip()
        return ""

    # ---------- signature builder ----------
    def _build_signature(self, row: pd.Series, sep: str = " | ") -> str:
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

        virtual = {
            "context_text": self._get_context_text(row),
            "context_keywords": self._get_context_keywords(row),
            "patient_journey": row.get("patient_journey") or row.get("customer_journey") or "",
            "journey_stage": row.get("journey_stage") or row.get("customer_journey_stage") or "",
            "interaction_moment": row.get("interaction_moment") or "",
            "semantic_customer_reality": self._get_sas_section_1(row),
            "matters": row.get("matters") or "",
            "action_axis_justification": row.get("action_axis_justification") or row.get("stream_justification") or "",
            "behavioral_impact": row.get("behavioral_impact") or "",
        }

        fields = sorted(
            ((k, int(w)) for k, w in weights.items() if int(w) > 0 and k in allowed),
            key=lambda kv: (-kv[1], kv[0])
        )

        parts = []
        for col, w in fields:
            text = norm(to_text(virtual.get(col, "")))
            if text:
                parts.extend([text] * int(w))
        return sep.join(parts)

    # === embeddings ===
    def _encode_np(self, texts):
        model = self._ensure_st_model()
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def _centroid_pick(self, texts):
        clean = [t for t in (texts or []) if isinstance(t, str) and t.strip()]
        if not clean:
            return None
        E = self._encode_np(clean)
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
            cand = [s for s in splitter.split(t) if s.strip()]
            long = [s for s in cand if len(s.strip()) >= 30]
            sents.extend(long if long else cand)

        if not sents:
            raw = " ".join([(t or "").strip() for t in (texts or []) if isinstance(t, str)])[:300]
            return raw or None

        sents = list(dict.fromkeys(sents))
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10000)
        X = vec.fit_transform(sents)
        centroid = np.asarray(X.mean(axis=0))
        rel = cosine_similarity(X, centroid).ravel()

        selected, cand_idx = [], list(range(len(sents)))
        while cand_idx and len(selected) < top_k:
            if not selected:
                i = int(rel.argmax())
                selected.append(i)
                cand_idx.remove(i)
                continue
            sims = cosine_similarity(X[cand_idx], X[selected]).max(axis=1)
            scores = diversity * rel[cand_idx] - (1 - diversity) * sims
            j = int(scores.argmax())
            sel = cand_idx[j]
            selected.append(sel)
            cand_idx.remove(sel)

        selected.sort()
        return " ".join(sents[i] for i in selected)

    # ---------- consolidation helpers ----------
    def _semantic_centroid_fusion(self, context_list: list[str]) -> str:
        if not context_list:
            return "No context available"
        clean = [str(c).strip() for c in context_list if pd.notna(c) and str(c).strip()]
        if not clean:
            return "No context available"

        vc = pd.Series(clean).value_counts()
        if len(vc) <= 3:
            top_count = vc.iloc[0]
            tied_vals = vc[vc == top_count].index.tolist()
            if len(tied_vals) == 1:
                return tied_vals[0]
            pick = self._centroid_pick(tied_vals)
            return pick or tied_vals[0]

        pick = self._centroid_pick(clean)
        return pick or "No context available"

    def _dedupe_and_merge_keywords(self, keywords_list):
        all_kw = []
        for kw in keywords_list:
            if kw is None or (isinstance(kw, float) and pd.isna(kw)):
                continue

            if isinstance(kw, (list, tuple, set)):
                seq = kw
            elif isinstance(kw, str):
                s = kw.strip()
                if s.startswith('[') and s.endswith(']'):
                    s = s[1:-1].replace("'", "").replace('"', '')
                seq = [k for k in s.split(',')]
            else:
                seq = [str(kw)]

            for k in seq:
                tok = str(k).strip().lower()
                if tok and len(tok) > 1:
                    all_kw.append(tok)

        counts = Counter(all_kw)
        return [k for k, _ in counts.most_common()]

    def _norm_whitespace(self, s: str) -> str:
        return " ".join(str(s or "").strip().split())

    def _collapse_locator_composite(self, grp: pd.DataFrame):
        j_raw = grp.get("patient_journey", pd.Series(dtype=object))
        if j_raw.empty and "customer_journey" in grp.columns:
            j_raw = grp.get("customer_journey", pd.Series(dtype=object))
        s_raw = grp.get("journey_stage", pd.Series(dtype=object))
        if s_raw.empty and "customer_journey_stage" in grp.columns:
            s_raw = grp.get("customer_journey_stage", pd.Series(dtype=object))
        m_raw = grp.get("interaction_moment", pd.Series(dtype=object))

        j_raw = j_raw.astype(str).tolist()
        s_raw = s_raw.astype(str).tolist()
        m_raw = m_raw.astype(str).tolist()

        key_counts: Counter[tuple[str, str, str]] = Counter()
        key_variants: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}

        for j, s, m in zip(j_raw, s_raw, m_raw):
            jn = self._norm_whitespace(j).lower()
            sn = self._norm_whitespace(s).lower()
            mn = self._norm_whitespace(m).lower()
            if not (jn or sn or mn):
                continue
            key = (jn, sn, mn)
            key_counts[key] += 1
            key_variants.setdefault(key, []).append((self._norm_whitespace(j), self._norm_whitespace(s), self._norm_whitespace(m)))

        if not key_counts:
            return "Unknown", "Unknown", "Unknown", {}

        top_key, top_cnt = key_counts.most_common(1)[0]
        tied_keys = [k for k, c in key_counts.items() if c == top_cnt]

        if len(tied_keys) == 1:
            chosen_key = top_key
        else:
            def key_to_string(k: tuple[str, str, str]) -> str:
                return " | ".join([p for p in k if p])

            tie_strs = [key_to_string(k) for k in tied_keys]
            picked_str = self._centroid_pick(tie_strs) or tie_strs[0]
            chosen_idx = tie_strs.index(picked_str)
            chosen_key = tied_keys[chosen_idx]

        total = sum(key_counts.values())
        dist = {" | ".join([p for p in k if p]): round(100.0 * c / total, 1) for k, c in key_counts.items()}

        variants = key_variants[chosen_key]
        j_disp = Counter([v[0] for v in variants if v[0]]).most_common(1)
        s_disp = Counter([v[1] for v in variants if v[1]]).most_common(1)
        m_disp = Counter([v[2] for v in variants if v[2]]).most_common(1)

        journey = j_disp[0][0] if j_disp else "Unknown"
        stage = s_disp[0][0] if s_disp else "Unknown"
        moment = m_disp[0][0] if m_disp else "Unknown"
        return journey, stage, moment, dist

    def _format_location_tail(self, cj: str, cjs: str, im: str) -> str:
        style = (getattr(self, "OU_CFG", {}) or {}).get("preview_location_style", "short").lower()

        def _wrap(parts: list[str]) -> str:
            return f" ({' | '.join(parts)})" if parts else ""

        if style == "none":
            return ""

        if style == "long":
            parts = []
            if cj: parts.append(f"Journey: {cj}")
            if cjs: parts.append(f"Stage: {cjs}")
            if im: parts.append(f"Moment: {im}")
            return _wrap(parts)

        if style == "journey_stage":
            parts = []
            if cj: parts.append(f"Journey: {cj}")
            if cjs: parts.append(f"Stage: {cjs}")
            return _wrap(parts)

        if style == "stage_only":
            return _wrap([f"Stage: {cjs}"] if cjs else [])

        parts = []
        if cj: parts.append(f"J: {cj}")
        if cjs: parts.append(f"S: {cjs}")
        if im: parts.append(f"M: {im}")
        return _wrap(parts)

    def _sentence_case(self, s: str) -> str:
        s = (s or "").strip()
        return s[:1].upper() + s[1:] if s else s

    def _mode_or_centroid(self, texts: list[str]) -> str | None:
        arr = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        if not arr:
            return None
        vc = pd.Series(arr).value_counts()
        top = vc.index[0] if not vc.empty else None
        if top:
            return top
        return self._centroid_pick(arr) or arr[0]

    def _make_root_cause_preview(self, grp: pd.DataFrame) -> str:
        weights = (self.OU_CFG or {}).get("signature_weights", {})
        candidate_fields = ["matters", "semantic_customer_reality", "context_text", "behavioral_impact", "action_axis_justification"]
        ordered_fields = sorted(((f, int(weights.get(f, 0))) for f in candidate_fields), key=lambda kv: (-kv[1], kv[0]))
        ordered_fields = [f for f, w in ordered_fields if w > 0]

        g = grp.copy()
        if "semantic_customer_reality" not in g.columns:
            g["semantic_customer_reality"] = g.apply(lambda r: self._get_sas_section_1(r), axis=1)
        if "context_text" not in g.columns:
            g["context_text"] = g.apply(lambda r: self._get_context_text(r), axis=1)

        mech = ""
        for col in ordered_fields:
            if col in g.columns:
                cand = self._mode_or_centroid(g[col].astype(str).tolist())
                if isinstance(cand, str) and cand.strip():
                    mech = cand.strip().rstrip(" .")
                    break

        if not mech:
            mech = "No preview available"

        journey, stage, moment, _ = self._collapse_locator_composite(g)
        tail = self._format_location_tail(journey, stage, moment)
        preview = self._sentence_case((mech + tail).strip())
        return (preview[:80] + "…") if len(preview) > 80 else preview

    # -------------------------------------------------------------------------
    # Core clustering
    # -------------------------------------------------------------------------
    def cluster_behavior(self, df: pd.DataFrame, driver: str, intent_axis: str, action_axis: str, affect_label: str):
        df = df.copy()

        # Virtual columns used by signature/preview
        df["context_text"] = df.apply(lambda r: self._get_context_text(r), axis=1)
        df["context_keywords"] = df.apply(lambda r: self._get_context_keywords(r), axis=1)
        df["semantic_customer_reality"] = df.apply(lambda r: self._get_sas_section_1(r), axis=1)

        df["signature"] = df.apply(self._build_signature, axis=1).astype(str).str.lower()
        df = df.reset_index(drop=True)

        embeds = self._encode_np(df["signature"].tolist())
        total_rows = len(df)

        if self.OU_CFG.get("adaptive_min_cluster", {}).get("enabled", False):
            mcs = max(
                int(self.OU_CFG["adaptive_min_cluster"].get("floor", 3)),
                int(math.ceil(self.OU_CFG["adaptive_min_cluster"].get("pct", 0.08) * total_rows))
            )
        else:
            mcs = int(self.OU_CFG.get("min_cluster_size", 8))

        start_thr = float(
            self.OU_CFG.get("bcs_distance_threshold",
                            self.OU_CFG.get("signature_threshold_start", {}).get(
                                self.OU_CFG.get("signature_config_name", "default_K1"), 0.30))
        )
        dist_thr = float(start_thr)
        step = float(self.OU_CFG.get("bcs_distance_threshold_step", 0.02))
        lo, hi = self.OU_CFG.get("bcs_distance_threshold_bounds", (0.25, 0.35))
        band_lo, band_hi = self.OU_CFG.get("cluster_band_target", (3, 8))
        singleton_cap = float(self.OU_CFG.get("singleton_cap", 0.40))
        max_adj = int(self.OU_CFG.get("bcs_threshold_max_adjustments", 1))
        skip_singletons = bool(self.OU_CFG.get("skip_singletons", False))

        if total_rows == 0:
            df["local_bcs_id"] = np.array([], dtype=str)
            return df, [], {}, df, {}

        def _run_agglom(thr: float):
            try:
                labels_ = AgglomerativeClustering(
                    metric="cosine",
                    linkage="average",
                    distance_threshold=float(thr),
                    n_clusters=None
                ).fit_predict(embeds)
            except TypeError:
                dist = pairwise_distances(embeds, metric="cosine")
                labels_ = AgglomerativeClustering(
                    affinity="precomputed",
                    linkage="average",
                    distance_threshold=float(thr),
                    n_clusters=None
                ).fit_predict(dist)
            return labels_

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

        used_agglom = False
        if total_rows < 2:
            labels = np.array([0], dtype=int)

        elif total_rows < 200:
            used_agglom = True
            labels = _run_agglom(dist_thr)

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

            if max_adj > 0:
                n_clusters, s_rate = _cluster_stats(labels)
                new_thr = dist_thr
                if (n_clusters > band_hi) or (s_rate > singleton_cap):
                    new_thr = min(hi, dist_thr + step)
                elif (n_clusters < band_lo):
                    new_thr = max(lo, dist_thr - step)

                if float(new_thr) != float(dist_thr):
                    dist_thr = float(new_thr)
                    labels = _run_agglom(dist_thr)

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
            labels = hdbscan.HDBSCAN(
                metric="cosine",
                min_cluster_size=mcs,
                min_samples=max(1, mcs // 2)
            ).fit_predict(embeds)

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

        noise_mask = df["local_bcs_id"] == "-1"
        if noise_mask.any():
            df.loc[noise_mask, "local_bcs_id"] = [str(uuid4()) for _ in range(int(noise_mask.sum()))]

        prefix = f"{driver[:8]}_{str(intent_axis)[:3]}_{str(action_axis)[:3]}_{str(affect_label)[:3]}".lower()
        cluster_store, full_composites, cluster_metadata = {}, {}, {}

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

            preview = self._make_root_cause_preview(grp)
            truncated_preview = (preview[:77] + "…") if len(preview) > 80 else preview
            grp["cluster_theme_preview"] = truncated_preview

            if "customer_review" in grp.columns:
                customer_review_value = self._semantic_centroid_fusion(grp["customer_review"].astype(str).tolist())
            elif "comment_review" in grp.columns:
                customer_review_value = self._semantic_centroid_fusion(grp["comment_review"].astype(str).tolist())
            else:
                customer_review_value = None

            first_row = grp.iloc[0]

            sas1_list = grp.get("semantic_customer_reality", pd.Series(dtype=object)).dropna().astype(str).tolist()
            sas2_list = grp.apply(lambda r: self._get_sas_section_2(r), axis=1).dropna().astype(str).tolist()

            just_list = grp.get("action_axis_justification", pd.Series(dtype=object)).dropna().astype(str).tolist()
            mat_list = grp.get("matters", pd.Series(dtype=object)).dropna().astype(str).tolist()
            beh_list = grp.get("behavioral_impact", pd.Series(dtype=object)).dropna().astype(str).tolist()

            composite = {
                "bcs_id": first_row["bcs_id"],
                "bcs_group_id": group_id,
                "cluster_size": int(len(grp)),
                "bcs_share": round(len(grp) / total_rows, 4),
                "cluster_cohesion": round(cohesion, 4),
                "cluster_theme_preview": truncated_preview,
                "customer_review": customer_review_value,

                "theme": first_row.get("theme", "Unknown"),
                "experience_driver": first_row.get("experience_driver"),
                "entity_name": first_row.get("entity_name"),
                "intent_axis": first_row.get("intent_axis", intent_axis),
                "action_axis": first_row.get("action_axis", action_axis),
                "affect_axis": {
                    "affect_label": self._get_affect_label(first_row),
                    "confidence_level": self._get_affect_confidence(first_row),
                },

                "journey": {
                    "patient_journey": first_row.get("patient_journey") or first_row.get("customer_journey"),
                    "journey_stage": first_row.get("journey_stage") or first_row.get("customer_journey_stage"),
                    "interaction_moment": first_row.get("interaction_moment"),
                },
                "context": {
                    "text": self._semantic_centroid_fusion(grp["context_text"].astype(str).tolist()),
                    "keywords": self._dedupe_and_merge_keywords(grp["context_keywords"].tolist()),
                },
                "customer_effort_score": float(first_row.get("customer_effort_score", 0.0) or 0.0),

                "matters": self._mmr_summary(mat_list) or self._centroid_pick(mat_list) or "",
                "semantic_action_statement": {
                    "section_1_patient_reality": self._mmr_summary(sas1_list) or self._centroid_pick(sas1_list) or "",
                    "section_2_strategic_response": self._mmr_summary(sas2_list) or self._centroid_pick(sas2_list) or "",
                },
                "action_axis_justification": self._centroid_pick(just_list) or "",
                "behavioral_impact": self._mmr_summary(beh_list) or self._centroid_pick(beh_list) or "",
            }

            df.update(grp)
            cluster_store[group_id] = grp
            full_composites[group_id] = composite
            cluster_metadata[group_id] = {"label": truncated_preview, "cohesion": cohesion}

        df["bcs_label"] = df["bcs_group_id"].map(lambda gid: cluster_metadata.get(gid, {}).get("label"))
        df["cluster_cohesion"] = df["bcs_group_id"].map(lambda gid: cluster_metadata.get(gid, {}).get("cohesion"))
        df["cluster_theme_preview"] = df["cluster_theme_preview"].fillna(
            df["bcs_group_id"].map(lambda gid: full_composites.get(gid, {}).get("cluster_theme_preview"))
        )

        dominant_ids, cumulative_share = [], 0.0
        cluster_order = df["bcs_group_id"].value_counts(normalize=True)
        for cid, share in cluster_order.items():
            dominant_ids.append(cid)
            cumulative_share += share
            if cumulative_share >= self.OU_CFG["bcs_cumu_threshold"]:
                break

        filtered_df = df[df["bcs_group_id"].isin(dominant_ids)].copy()
        self.OU_CFG["bcs_distance_threshold"] = float(dist_thr)

        return filtered_df, list(full_composites.values()), cluster_store, df, full_composites

    # -------------------------------------------------------------------------
    # DB persistence (member-first, as-is) — updated schema for v2 axes
    # -------------------------------------------------------------------------
    def _nz(self, v):
        return "" if v is None else v

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("DROP TABLE IF EXISTS clusters")
            cur.execute("""
                CREATE TABLE clusters (
                    bcs_id TEXT PRIMARY KEY,
                    bcs_group_id TEXT,
                    cluster_size INTEGER,
                    bcs_share REAL,
                    cluster_cohesion REAL,
                    cluster_theme_preview TEXT,
                    customer_review TEXT,

                    theme TEXT,
                    experience_driver TEXT,
                    entity_name TEXT,

                    intent_axis TEXT,
                    action_axis TEXT,
                    affect_label TEXT,
                    affect_confidence_level TEXT,

                    patient_journey TEXT,
                    journey_stage TEXT,
                    interaction_moment TEXT,

                    context_text TEXT,
                    context_keywords TEXT,

                    customer_effort_score REAL,

                    matters TEXT,
                    sas_section_1_patient_reality TEXT,
                    sas_section_2_strategic_response TEXT,
                    action_axis_justification TEXT,
                    behavioral_impact TEXT
                )
            """)
            cur.execute('CREATE INDEX IF NOT EXISTS idx_bcs_group_id ON clusters (bcs_group_id)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_ed_axes ON clusters (experience_driver, intent_axis, action_axis, affect_label)')
            conn.commit()
        print(f"✅ Database initialized: {self.db_path}")

    def create_cluster_database(self, df: pd.DataFrame, full_composites: Dict[str, Dict[str, Any]],
                                cluster_store: Dict[str, pd.DataFrame], db_path: str = "outputs/clusters_v2_axes.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.init_database()

        for cid, grp in cluster_store.items():
            composite = full_composites.get(cid, {})
            self.save_cluster_members_as_is(grp, composite, cid)

        print("✅ All clusters saved to database (member rows preserved).")

    def save_cluster_members_as_is(self, grp: pd.DataFrame, composite: Dict[str, Any], cid: str):
        cluster_size = int(composite.get("cluster_size", len(grp)))
        bcs_share = float(composite.get("bcs_share", len(grp) / max(len(grp), 1)))
        cluster_cohesion = float(composite.get("cluster_cohesion", 1.0))
        meta_theme_preview = self._nz(composite.get("cluster_theme_preview", ""))

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()

            for _, row in grp.iterrows():
                theme_preview = self._nz(row.get("cluster_theme_preview", meta_theme_preview))

                kw = row.get("context_keywords") if "context_keywords" in row else self._get_context_keywords(row)
                if isinstance(kw, list):
                    keywords_str = ", ".join(map(str, kw))
                else:
                    keywords_str = self._nz(kw if kw is not None else "")

                affect_label = self._get_affect_label(row)
                affect_conf = self._get_affect_confidence(row)

                sas1 = self._get_sas_section_1(row)
                sas2 = self._get_sas_section_2(row)

                cur.execute("""
                    INSERT OR REPLACE INTO clusters (
                        bcs_id, bcs_group_id, cluster_size, bcs_share, cluster_cohesion, cluster_theme_preview,
                        customer_review,
                        theme, experience_driver, entity_name,
                        intent_axis, action_axis, affect_label, affect_confidence_level,
                        patient_journey, journey_stage, interaction_moment,
                        context_text, context_keywords,
                        customer_effort_score,
                        matters, sas_section_1_patient_reality, sas_section_2_strategic_response,
                        action_axis_justification, behavioral_impact
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self._nz(row.get("bcs_id")),
                    self._nz(row.get("bcs_group_id") or cid),
                    cluster_size,
                    bcs_share,
                    cluster_cohesion,
                    theme_preview,

                    self._nz(row.get("customer_review") or row.get("comment_review")),

                    self._nz(row.get("theme")),
                    self._nz(row.get("experience_driver")),
                    self._nz(row.get("entity_name")),

                    self._nz(row.get("intent_axis")),
                    self._nz(row.get("action_axis")),
                    self._nz(affect_label),
                    self._nz(affect_conf),

                    self._nz(row.get("patient_journey") or row.get("customer_journey")),
                    self._nz(row.get("journey_stage") or row.get("customer_journey_stage")),
                    self._nz(row.get("interaction_moment")),

                    self._nz(row.get("context_text") if "context_text" in row else self._get_context_text(row)),
                    self._nz(keywords_str),

                    float(row.get("customer_effort_score", 0.0) or 0.0),

                    self._nz(row.get("matters")),
                    self._nz(sas1),
                    self._nz(sas2),
                    self._nz(row.get("action_axis_justification") or row.get("stream_justification")),
                    self._nz(row.get("behavioral_impact")),
                ))

            conn.commit()
        print(f"💾 Saved cluster {cid} with {len(grp)} member row(s) (as-is).")

    # -------------------------------------------------------------------------
    # Snapshot loop (v2 axes)
    # -------------------------------------------------------------------------
    def _prepare_raw_for_snapshot(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        rdf = raw_df.copy()
        rdf["experience_driver"] = rdf["experience_driver"].astype(str).str.strip()
        rdf["theme"] = rdf.get("theme", pd.Series(dtype=object)).astype(str).str.strip()
        rdf["intent_axis"] = rdf.get("intent_axis", pd.Series(dtype=object)).astype(str).str.strip()
        rdf["action_axis"] = rdf.get("action_axis", pd.Series(dtype=object)).astype(str).str.strip()
        return rdf.dropna(subset=["experience_driver", "intent_axis", "action_axis"])

    def compute_granular_details_snapshot_v2_axes(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        must_cols = {"experience_driver", "intent_axis", "action_axis"}
        missing = [c for c in must_cols if c not in raw_df.columns]
        if missing:
            raise ValueError(f"Raw DF missing required v2 axes columns: {missing}")

        rdf = self._prepare_raw_for_snapshot(raw_df)

        all_df_chunks, all_full_composites, all_cluster_store = [], {}, {}
        records = []

        for driver, driver_rows in rdf.groupby("experience_driver"):
            driver = str(driver).strip()
            if not driver or driver_rows.empty:
                continue

            driver_rows = driver_rows.copy()
            driver_rows["affect_label"] = driver_rows.apply(lambda r: self._get_affect_label(r), axis=1)

            for (intent_axis, action_axis, affect_label), slice_rows in driver_rows.groupby(["intent_axis", "action_axis", "affect_label"]):
                intent_axis = str(intent_axis).strip()
                action_axis = str(action_axis).strip()
                affect_label = str(affect_label).strip()

                if slice_rows.empty:
                    continue

                clust_df, _, cluster_store, df_chunk, full_composites = self.cluster_behavior(
                    slice_rows,
                    driver=driver,
                    intent_axis=intent_axis,
                    action_axis=action_axis,
                    affect_label=affect_label,
                )

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
                        records.append(meta)

        if getattr(self, "verbose", False):
            print(f"\n📦 FINAL DEBUG SUMMARY (v2 axes)")
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
                db_path=os.path.join(self.output_dir, "clusters_v2_axes.db"),
            )

        return pd.DataFrame(records)
