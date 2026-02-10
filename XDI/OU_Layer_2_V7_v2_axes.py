from __future__ import annotations

"""
OU Layer (Full Clustering) — Healthcare v2 Axes

This is a v2-schema aligned rewrite of OU_Layer_2_V7.py that preserves the SAME clustering logic
(signature-based semantic clustering; Agglomerative < 200 else HDBSCAN; adaptive min_cluster_size;
one-bump distance-threshold controller) while updating field reality end-to-end to:

- intent_axis: string
- affect_axis: { affect_label: string, confidence_level: string }
- action_axis: string
- journey: { patient_journey, journey_stage, interaction_moment }
- context: { text, keywords }
- orchestration_unit: { matters, semantic_action_statement{section_1_patient_reality, section_2_strategic_response}, ... }

Key invariant:
- SEU is state (computed elsewhere); this module births OUs (missions) from unpacked mentions.
- OU identity is: ED + intent_axis + action_axis + affect_label + behavioral cluster signature.

Expected input
--------------
A CSV (or DataFrame) with at minimum:
- theme
- experience_driver                 (canonical label: "Category → Subcategory")
- entity_name
- context_text                      (string)
- context_keywords                  (list-like str or JSON)
- intent_axis
- action_axis
- affect_label
- affect_confidence_level
- patient_journey
- journey_stage
- interaction_moment
- matters                           (string)
- sas_section_1_patient_reality     (string)
- sas_section_2_strategic_response  (string)
- action_axis_justification         (string)
- matters_extraction_source         (string)
- behavioral_impact                 (string)

If your upstream unpacking produces nested JSON, flatten it before calling.

Outputs
-------
A DataFrame of birthed OUs (behaviorally pure clusters) with v2-aligned OU payload fields plus
cluster metadata for auditability.
"""

import json
import math
import os
import re
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# -----------------------------
# Helpers
# -----------------------------

def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _norm_space(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip()) if s is not None else ""

def _safe_list(x: Any) -> List[str]:
    """Parse list-like values safely (JSON list, Python list string, comma string)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # JSON?
        try:
            j = json.loads(s)
            if isinstance(j, list):
                return [str(i) for i in j if str(i).strip()]
        except Exception:
            pass
        # Python list literal?
        try:
            import ast
            j = ast.literal_eval(s)
            if isinstance(j, list):
                return [str(i) for i in j if str(i).strip()]
        except Exception:
            pass
        # comma separated
        return [p.strip() for p in s.split(",") if p.strip()]
    return [str(x).strip()] if str(x).strip() else []


# -----------------------------
# Signature config (v2)
# -----------------------------

ALLOWED_FOR_SIGNATURE = frozenset({
    "intent_axis",
    "action_axis",
    "affect_label",
    "interaction_moment",
    "journey_stage",
    "patient_journey",
    "context_text",
    "context_keywords",
    "matters",
    "sas_section_1_patient_reality",
    "sas_section_2_strategic_response",
    "action_axis_justification",
    "matters_extraction_source",
    "behavioral_impact",
})

SIGNATURE_LIBRARY: Dict[str, Dict[str, int]] = {
    # default: mission purity (matters + SAS + axes)
    "default_K1": {
        "matters": 6,
        "sas_section_1_patient_reality": 4,
        "sas_section_2_strategic_response": 3,
        "interaction_moment": 4,
        "journey_stage": 3,
        "context_text": 3,
        "intent_axis": 3,
        "action_axis": 3,
        "affect_label": 2,
        "behavioral_impact": 2,
        "patient_journey": 1,
        "context_keywords": 1,
        "action_axis_justification": 1,
        "matters_extraction_source": 1,
    },
    # fallback: if SAS is missing, lean heavier on matters+context
    "fallback_K3": {
        "matters": 7,
        "context_text": 4,
        "interaction_moment": 4,
        "journey_stage": 3,
        "intent_axis": 3,
        "action_axis": 3,
        "affect_label": 2,
        "behavioral_impact": 2,
        "patient_journey": 1,
        "context_keywords": 1,
    },
}


@dataclass(frozen=True)
class OUConfig:
    signature_config_name: str = "default_K1"
    signature_threshold_start: float = 0.30
    signature_threshold_bump: float = 0.05
    cluster_band_target: Tuple[int, int] = (3, 8)
    singleton_cap: float = 0.40
    neighbors_k: int = 5
    min_cluster_size_floor: int = 5


# -----------------------------
# OU Birther (full clustering)
# -----------------------------

class OUBirtherV2Axes:
    def __init__(self, *, model_name: str = "all-MiniLM-L6-v2", config: Optional[OUConfig] = None):
        self.model_name = model_name
        self.cfg = config or OUConfig()
        self._model: Optional[SentenceTransformer] = None
        self._last_cluster_control: Dict[str, Any] = {}

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _build_signature(self, row: pd.Series) -> str:
        weights = SIGNATURE_LIBRARY.get(self.cfg.signature_config_name, SIGNATURE_LIBRARY["default_K1"])

        def to_text(col: str) -> str:
            if col == "context_keywords":
                return " ".join(_safe_list(row.get(col)))
            v = row.get(col, "")
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            return str(v)

        parts: List[str] = []
        ordered = sorted(((k, int(w)) for k, w in weights.items() if k in ALLOWED_FOR_SIGNATURE and int(w) > 0),
                         key=lambda kv: (-kv[1], kv[0]))
        for col, w in ordered:
            text = _norm_space(to_text(col)).lower()
            if text:
                parts.extend([f"{col}: {text}"] * int(w))
        return " | ".join(parts)

    def _encode(self, texts: List[str]) -> np.ndarray:
        model = self._ensure_model()
        return model.encode(texts, normalize_embeddings=True)

    def _adaptive_min_cluster_size(self, n: int) -> int:
        # simple adaptive policy: grows slowly with n, with a floor
        return max(self.cfg.min_cluster_size_floor, int(max(5, round(math.sqrt(max(n, 1))))))

    def _cluster_with_agglomerative(self, X: np.ndarray, distance_threshold: float) -> np.ndarray:
        # cosine distance = 1 - cosine similarity (embeddings are normalized)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        return clustering.fit_predict(X)

    def _cluster_with_hdbscan(self, X: np.ndarray, min_cluster_size: int) -> np.ndarray:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",          # embeddings normalized => euclidean relates to cosine
            cluster_selection_method="eom",
        )
        return clusterer.fit_predict(X)

    def _one_bump_controller(self, labels: np.ndarray, *, n: int, distance_threshold: float) -> float:
        """
        Keeps clustering in a healthy band by bumping distance_threshold once if needed.
        Mirrors the OU_Layer_2_V7 philosophy: deterministic, bounded adjustment.
        """
        # compute stats
        unique = [x for x in np.unique(labels) if x != -1]
        k = len(unique)
        singletons = np.sum(np.bincount(np.where(labels >= 0, labels, -1)[labels >= 0]) == 1) if np.any(labels >= 0) else 0
        singleton_ratio = (singletons / max(n, 1))

        self._last_cluster_control = {
            "n": n,
            "k": k,
            "singleton_ratio": float(round(singleton_ratio, 4)),
            "distance_threshold_start": distance_threshold,
            "distance_threshold_final": distance_threshold,
            "bumped": False,
        }

        lo, hi = self.cfg.cluster_band_target
        if n < 10:
            return distance_threshold

        needs_bump = (k < lo) or (k > hi) or (singleton_ratio > self.cfg.singleton_cap)
        if needs_bump:
            bumped = distance_threshold + self.cfg.signature_threshold_bump
            self._last_cluster_control["distance_threshold_final"] = bumped
            self._last_cluster_control["bumped"] = True
            return bumped

        return distance_threshold

    def cluster_behavior(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cluster a group of mentions into behaviorally pure clusters.

        df should already be scoped to a single (experience_driver, intent_axis, action_axis, affect_label) group.
        """
        if df.empty:
            return df.assign(bcs_label=-1, bcs_signature="", bcs_cluster_size=0, bcs_cluster_share=0.0)

        work = df.copy()
        work["bcs_signature"] = work.apply(self._build_signature, axis=1)

        sigs = work["bcs_signature"].astype(str).tolist()
        X = self._encode(sigs)

        n = len(work)
        dist0 = self.cfg.signature_threshold_start

        if n < 200:
            labels = self._cluster_with_agglomerative(X, distance_threshold=dist0)
            # controller bump once if needed
            dist1 = self._one_bump_controller(labels, n=n, distance_threshold=dist0)
            if dist1 != dist0:
                labels = self._cluster_with_agglomerative(X, distance_threshold=dist1)
        else:
            mcs = self._adaptive_min_cluster_size(n)
            labels = self._cluster_with_hdbscan(X, min_cluster_size=mcs)

        work["bcs_label"] = labels

        # cluster sizes / shares (excluding noise=-1 for size stats)
        counts = work["bcs_label"].value_counts(dropna=False).to_dict()
        work["bcs_cluster_size"] = work["bcs_label"].map(counts).fillna(0).astype(int)
        work["bcs_cluster_share"] = (work["bcs_cluster_size"] / max(n, 1)).round(4)

        return work

    def birth_ous(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full OU birthing:
        - group by (theme, experience_driver, intent_axis, action_axis, affect_label)
        - within each group, cluster_behavior()
        - for each cluster, build a v2-aligned OU payload + cluster metadata
        """
        required = [
            "theme", "experience_driver", "entity_name",
            "intent_axis", "action_axis", "affect_label", "affect_confidence_level",
            "patient_journey", "journey_stage", "interaction_moment",
            "context_text", "context_keywords",
            "matters", "sas_section_1_patient_reality", "sas_section_2_strategic_response",
            "action_axis_justification", "matters_extraction_source", "behavioral_impact",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for v2 OU birthing: {missing}")

        # normalize blanks
        work = df.copy()
        for c in required:
            if c == "context_keywords":
                work[c] = work[c].apply(_safe_list)
            else:
                work[c] = work[c].apply(_norm_space)

        group_cols = ["theme", "experience_driver", "intent_axis", "action_axis", "affect_label"]

        ou_rows: List[Dict[str, Any]] = []
        for keys, grp in work.groupby(group_cols, dropna=False, sort=True):
            theme, ed, intent, action, affect = keys
            clustered = self.cluster_behavior(grp)

            # build cluster-level composites (ignore noise clusters by default)
            for lbl, sub in clustered.groupby("bcs_label", sort=True):
                if int(lbl) == -1:
                    continue

                n = len(sub)
                share = float(round(n / max(len(clustered), 1), 4))

                # pick representative row by max cluster share then first (deterministic)
                rep = sub.sort_values(["bcs_cluster_share"], ascending=False).iloc[0]

                ou_id_seed = "||".join([
                    ed.lower(), intent.lower(), action.lower(), affect.lower(), rep["bcs_signature"].lower()
                ])
                ou_id = re.sub(r"[^a-z0-9]", "", __import__("hashlib").sha1(ou_id_seed.encode("utf-8")).hexdigest()[:18])

                ou_payload = {
                    "ou_id": ou_id,
                    "ou_name": f"{ed} — {intent} — {action} — {affect}",
                    "experience_driver": ed,
                    "entity_name": rep["entity_name"],
                    "theme": theme,
                    "intent_axis": intent,
                    "action_axis": action,
                    "affect_axis": {
                        "affect_label": affect,
                        "confidence_level": rep["affect_confidence_level"],
                    },
                    "interaction_moment": rep["interaction_moment"],
                    "context": rep["context_text"],
                    "matters": rep["matters"],
                    "semantic_action_statement": {
                        "section_1_patient_reality": rep["sas_section_1_patient_reality"],
                        "section_2_strategic_response": rep["sas_section_2_strategic_response"],
                    },
                    "action_axis_justification": rep["action_axis_justification"],
                    "matters_extraction_source": rep["matters_extraction_source"],
                    "behavioral_impact": rep["behavioral_impact"],
                }

                ou_rows.append({
                    **ou_payload,
                    # clustering metadata
                    "bcs_label": int(lbl),
                    "bcs_cluster_size": int(n),
                    "bcs_cluster_share": share,
                    "signature_config_name": self.cfg.signature_config_name,
                    "cluster_control": self._last_cluster_control,
                    "created_at_utc": _now_utc_iso(),
                })

        return pd.DataFrame(ou_rows)


# -----------------------------
# Convenience CLI-style runner
# -----------------------------

def run_ou_birthing_v2_axes(input_path: str, output_path: str) -> str:
    df = pd.read_csv(input_path)
    birther = OUBirtherV2Axes()
    out = birther.birth_ous(df)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path
