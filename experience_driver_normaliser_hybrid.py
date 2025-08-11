# ed_normalizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import json, os, re, hashlib, logging, pickle
from datetime import datetime
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:  # allow import to fail in environments without the lib
    SentenceTransformer = None
    st_util = None

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG (edit here as needed)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "embedding_model": "all-MiniLM-L6-v2",
    "registry_path": "ed_registry.v1.json",
    "proposed_queue_path": "proposed_ed_queue.jsonl",
    "embedding_cache_path": "embedding_cache.pkl",
    "tfidf_index_path": "tfidf_index.pkl",
    # Decision thresholds
    "theme_gate": 0.70,            # fuzzy theme min (0..1)
    "auto_accept": 0.86,           # final score band
    "soft_accept": 0.74,
    # Weights (hierarchical fuzzy)
    "weights_depth_4": [0.35, 0.25, 0.20, 0.20],  # Theme, Category, Subcat, Entity
    "weights_depth_3": [0.40, 0.35, 0.25],        # Theme, Category, Entity
    "weights_depth_2": [0.60, 0.40],              # Category, Entity
    # Blend fuzzy/semantic
    "blend_fuzzy": 0.55,
    "blend_semantic": 0.45,
    # Boosts / penalties
    "alias_boost": 0.06,
    "cue_boost": 0.02,
    "neg_penalty": 0.08,
    "journey_penalty": 0.03,
    "moment_penalty": 0.03,
    # Candidate pruning
    "tfidf_top_k": 50,
    # Adaptive cluster protections
    "min_cluster_size_bonus": 0.02,   # require +0.02 if cluster < 5
    "max_cluster_size_bonus": 0.01,   # require +0.01 if cluster > 20
    "small_cluster_lt": 5,
    "large_cluster_gt": 20,
}

SEP_PATTERN = re.compile(r"\s*(?:>|→|/|:|-)\s*")
PUNCT = re.compile(r"[^\w\s]")

# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class EDEntry:
    ed_id: str
    path: List[str]                          # ["Theme","Category","Subcategory","Entity"] (2..4 levels ok)
    aliases: List[str]
    negatives: List[str]
    cues_journey: List[str]
    cues_moments: List[str]
    description: str
    frozen: bool = True

class EDRegistry:
    def __init__(self, eds: Dict[str, EDEntry]):
        self.eds = eds
        # Theme strings present in registry (from path[0])
        self.themes: List[str] = sorted({e.path[0] for e in self.eds.values() if e.path})
        # Precompute a list for TF-IDF labels
        self._labels: List[str] = []
        self._label_ids: List[str] = []
        for e in self.eds.values():
            label = " > ".join(e.path)
            if e.aliases:
                label += " " + " ".join(e.aliases)
            self._labels.append(label)
            self._label_ids.append(e.ed_id)

    @classmethod
    def from_file(cls, path: str) -> "EDRegistry":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        eds = {}
        for ed_id, payload in raw.get("eds", {}).items():
            eds[ed_id] = EDEntry(
                ed_id=ed_id,
                path=payload.get("path", []),
                aliases=payload.get("aliases", []),
                negatives=payload.get("negatives", []),
                cues_journey=payload.get("cues", {}).get("journey", []),
                cues_moments=payload.get("cues", {}).get("moments", []),
                description=payload.get("description", ""),
                frozen=payload.get("frozen", True),
            )
        return cls(eds)

    def eds_in_theme(self, theme: str) -> List[EDEntry]:
        return [e for e in self.eds.values() if e.path and e.path[0].lower() == theme.lower()]

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def norm(s: Optional[str]) -> str:
    s = (s or "").lower().strip()
    s = PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s)

def split_path(s: str) -> List[str]:
    parts = [p for p in SEP_PATTERN.split(s) if p.strip()]
    return parts[:4]  # cap at 4

def rf_sim(a: str, b: str) -> float:
    return (fuzz.token_set_ratio(norm(a), norm(b)) or 0) / 100.0

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# ──────────────────────────────────────────────────────────────────────────────
# Embedding & centroid store
# ──────────────────────────────────────────────────────────────────────────────
class EmbeddingIndex:
    def __init__(self, model_name: str, cache_path: str):
        self.model_name = model_name
        self.cache_path = cache_path
        self.cache: Dict[str, np.ndarray] = self._load_cache()
        self.model = SentenceTransformer(model_name) if SentenceTransformer else None
        self.centroids: Dict[str, Tuple[np.ndarray, int]] = {}  # ed_id -> (centroid, count)

    def _load_cache(self) -> Dict[str, np.ndarray]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {}

    def save_cache(self):
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception:
            pass

    def embed(self, text: str) -> Optional[np.ndarray]:
        if not self.model:
            return None
        key = f"{self.model_name}:{sha1(text)}"
        if key in self.cache:
            return self.cache[key]
        emb = self.model.encode([text], convert_to_numpy=True)[0]
        self.cache[key] = emb
        return emb

    def cosine_to_centroid(self, ed_id: str, vec: np.ndarray) -> Optional[float]:
        if ed_id not in self.centroids:
            return None
        c, _ = self.centroids[ed_id]
        # safe cosine
        a = c / (np.linalg.norm(c) + 1e-12)
        b = vec / (np.linalg.norm(vec) + 1e-12)
        return float(np.dot(a, b))

    def update_centroid(self, ed_id: str, vec: np.ndarray):
        # incremental running mean
        if ed_id not in self.centroids:
            self.centroids[ed_id] = (vec.copy(), 1)
            return
        c, n = self.centroids[ed_id]
        new_c = (c * n + vec) / (n + 1)
        self.centroids[ed_id] = (new_c, n + 1)

# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF candidate pruner
# ──────────────────────────────────────────────────────────────────────────────
class TFIDFPruner:
    def __init__(self, labels: List[str], label_ids: List[str], index_path: str):
        self.labels = labels
        self.label_ids = label_ids
        self.index_path = index_path
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self._load_or_build()

    def _load_or_build(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    self.vectorizer, self.matrix = pickle.load(f)
                    return
            except Exception:
                pass
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", max_features=20000)
        self.matrix = self.vectorizer.fit_transform(self.labels)
        with open(self.index_path, "wb") as f:
            pickle.dump((self.vectorizer, self.matrix), f)

    def top_k(self, query: str, k: int) -> List[str]:
        if not self.vectorizer or self.matrix is None:
            return self.label_ids
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        idx = np.argsort(-sims)[:k]
        return [self.label_ids[i] for i in idx]

# ──────────────────────────────────────────────────────────────────────────────
# Normalizer
# ──────────────────────────────────────────────────────────────────────────────
class EDNormalizer:
    def __init__(self, cfg: Dict[str, Any] = None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.registry = EDRegistry.from_file(self.cfg["registry_path"])
        self.embed_index = EmbeddingIndex(self.cfg["embedding_model"], self.cfg["embedding_cache_path"])
        self.pruner = TFIDFPruner(self.registry._labels, self.registry._label_ids, self.cfg["tfidf_index_path"])
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ——— Signature (PDCA) ———
    @staticmethod
    def build_pdca_signature(row: Dict[str, Any]) -> str:
        toks: List[str] = []
        def add(field, times): 
            val = row.get(field)
            if val: toks.extend([str(val)] * times)
        add("semantic_action_statement", 8)
        add("behavioral_impact", 6)
        add("stream_justification", 4)
        add("customer_journey", 2)
        add("journey_stage", 1)
        return " ".join(toks).strip()

    # ——— Theme gate ———
    def _best_theme(self, candidate_theme: str) -> Optional[str]:
        best_theme = None
        best_sim = 0.0
        for theme in self.registry.themes:
            s = rf_sim(candidate_theme, theme)
            if s > best_sim:
                best_sim, best_theme = s, theme
        return best_theme if best_sim >= self.cfg["theme_gate"] else None

    # ——— Candidate pool ———
    def _candidates_for_theme(self, theme: str, candidate_label: str) -> List[EDEntry]:
        # prune with tfidf to speed; ensure union with alias hits
        pruned_ids = set(self.pruner.top_k(candidate_label, self.cfg["tfidf_top_k"]))
        pool = [e for e in self.registry.eds_in_theme(theme) if e.ed_id in pruned_ids]
        # ensure alias hits included
        last_part = split_path(candidate_label)[-1] if split_path(candidate_label) else candidate_label
        for e in self.registry.eds_in_theme(theme):
            if any(rf_sim(last_part, a) >= 0.90 for a in e.aliases):
                if e not in pool:
                    pool.append(e)
        return pool

    # ——— Hierarchical fuzzy score ———
    def _hier_score(self, parts: List[str], entry: EDEntry) -> Tuple[float, Dict[str, float]]:
        target = entry.path
        # right-align by depth overlap
        d = min(len(parts), len(target))
        cand = parts[-d:]
        targ = target[-d:]

        if d == 4:
            weights = self.cfg["weights_depth_4"]
            labels = ["theme", "category", "subcategory", "entity"]
        elif d == 3:
            weights = self.cfg["weights_depth_3"]
            labels = ["theme", "category", "entity"]
        elif d == 2:
            weights = self.cfg["weights_depth_2"]
            labels = ["category", "entity"]
        else:
            weights = [1.0]
            labels = ["theme"]

        # use trailing weights (align with depth d)
        weights = weights[-d:]

        sims = []
        per_level = {}
        for p, t, w, lab in zip(cand, targ, weights, labels[-d:]):
            s = rf_sim(p, t)
            sims.append(w * s)
            per_level[lab] = s
        return sum(sims), per_level

    # ——— Semantic score (to centroid) ———
    def _semantic_score(self, ed_id: str, signature: str) -> Optional[float]:
        if not signature or not self.embed_index.model:
            return None
        vec = self.embed_index.embed(signature)
        if vec is None:
            return None
        return self.embed_index.cosine_to_centroid(ed_id, vec)

    # ——— Boosts/penalties ———
    def _adjust(self, entry: EDEntry, parts: List[str], journey: Optional[str], moment: Optional[str]) -> Tuple[float, Dict[str, float]]:
        last = parts[-1] if parts else ""
        alias_hit = max([rf_sim(last, a) for a in entry.aliases], default=0.0) if entry.aliases else 0.0
        neg_hit = max([rf_sim(" ".join(parts), n) for n in entry.negatives], default=0.0) if entry.negatives else 0.0

        cue_boost = 0.0
        if journey and any(rf_sim(journey, j) >= 0.88 for j in entry.cues_journey):
            cue_boost += self.cfg["cue_boost"]
        if moment and any(rf_sim(moment, m) >= 0.88 for m in entry.cues_moments):
            cue_boost += self.cfg["cue_boost"]

        alias_boost = self.cfg["alias_boost"] if alias_hit >= 0.90 else 0.0
        neg_pen = self.cfg["neg_penalty"] if neg_hit >= 0.85 else 0.0

        return (alias_boost + cue_boost - neg_pen), {
            "alias_hit": round(alias_hit, 3),
            "neg_hit": round(neg_hit, 3),
            "alias_boost": alias_boost,
            "cue_boost": cue_boost,
            "neg_penalty": neg_pen
        }

    # ——— Final decision for one candidate ED ———
    def _score_entry(
        self,
        entry: EDEntry,
        parts: List[str],
        signature: str,
        journey: Optional[str],
        moment: Optional[str],
        cluster_size: int,
    ) -> Tuple[float, Dict[str, Any]]:
        s_hier, per_level = self._hier_score(parts, entry)
        s_sem = self._semantic_score(entry.ed_id, signature)
        s_sem = 0.0 if s_sem is None or np.isnan(s_sem) else float(s_sem)

        blend = self.cfg["blend_fuzzy"] * s_hier + self.cfg["blend_semantic"] * s_sem
        adj, dbg_adj = self._adjust(entry, parts, journey, moment)

        # adaptive protections
        req_bonus = 0.0
        if cluster_size < self.cfg["small_cluster_lt"]:
            req_bonus += self.cfg["min_cluster_size_bonus"]
        if cluster_size > self.cfg["large_cluster_gt"]:
            req_bonus += self.cfg["max_cluster_size_bonus"]

        final = max(0.0, min(1.0, blend + adj))
        return final, {
            "per_level": {k: round(v, 3) for k, v in per_level.items()},
            "s_hier": round(s_hier, 3),
            "s_sem": round(s_sem, 3),
            "blend_raw": round(blend, 3),
            "adjustments": dbg_adj,
            "req_bonus": req_bonus,
        }

    # ——— Public API: normalize one ED string + context ———
    def normalize(
        self,
        experience_driver_string: str,
        *,
        theme_hint: Optional[str] = None,
        problem_statement: Optional[str] = None,
        semantic_action_statement: Optional[str] = None,
        behavioral_impact: Optional[str] = None,
        stream_justification: Optional[str] = None,
        customer_journey: Optional[str] = None,
        journey_stage: Optional[str] = None,
        interaction_moment: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        ed_str = (experience_driver_string or "").strip()
        if not ed_str:
            return {"resolution": "propose_new", "reason": "empty_ed", "alternatives": []}

        parts = split_path(ed_str)
        if not parts:
            return {"resolution": "propose_new", "reason": "unparseable", "alternatives": []}

        # theme gate
        theme_guess = theme_hint or parts[0]
        theme = self._best_theme(theme_guess)
        if not theme:
            return {"resolution": "propose_new", "reason": "theme_gate_fail", "alternatives": []}

        # build label for pruning
        label_query = " ".join(parts)
        candidates = self._candidates_for_theme(theme, label_query)
        if not candidates:
            return {"resolution": "propose_new", "reason": "no_candidates_in_theme", "alternatives": []}

        # PDCA signature (semantic)
        row = {
            "semantic_action_statement": semantic_action_statement,
            "behavioral_impact": behavioral_impact,
            "stream_justification": stream_justification,
            "customer_journey": customer_journey,
            "journey_stage": journey_stage,
        }
        signature = self.build_pdca_signature(row)

        scored = []
        for e in candidates:
            # dummy sizes until you wire cluster sizes (can be 0)
            cluster_size = 10  # replace with real value from your IME/CSLI store if available
            score, dbg = self._score_entry(
                e, parts, signature, customer_journey, interaction_moment, cluster_size
            )
            scored.append((e, score, dbg))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_e, best_score, best_dbg = scored[0]
        alts = [{"ed_id": e.ed_id, "score": round(s, 3)} for (e, s, _) in scored[:top_k]]

        # apply adaptive bonus requirement (do not modify score; just change banding)
        required = 0.0
        required += best_dbg.get("req_bonus", 0.0)
        band = "propose_new"
        if best_score >= self.cfg["auto_accept"] + required:
            band = "auto_accept"
        elif best_score >= self.cfg["soft_accept"] + required:
            band = "soft_accept"

        result = {
            "canonical_ed_id": best_e.ed_id,
            "canonical_path": best_e.path,
            "confidence": round(best_score, 3),
            "resolution": band,
            "matched_by": ["fuzzy_hierarchical", "semantic_centroid" if signature else "fuzzy_only"],
            "alternatives": alts,
            "per_level": best_dbg["per_level"],
            "debug": {
                "s_hier": best_dbg["s_hier"],
                "s_sem": best_dbg["s_sem"],
                "blend_raw": best_dbg["blend_raw"],
                "adjustments": best_dbg["adjustments"],
                "required_bonus": best_dbg["req_bonus"],
                "theme_selected": theme,
            },
        }
        return result

    # ——— Batch helper (DataFrame-friendly) ———
    def normalize_rows(self, df, ed_col="experience_driver") -> Any:
        import pandas as pd
        # add columns
        out_cols = [
            "canonical_ed_id","canonical_ed_path","ed_confidence",
            "ed_resolution","ed_matched_by","ed_alternatives"
        ]
        for c in out_cols: 
            if c not in df.columns: df[c] = None

        for i, row in df.iterrows():
            res = self.normalize(
                str(row.get(ed_col, "")),
                theme_hint=None,
                problem_statement=row.get("problem_statement"),
                semantic_action_statement=row.get("semantic_action_statement"),
                behavioral_impact=row.get("behavioral_impact"),
                stream_justification=row.get("stream_justification"),
                customer_journey=row.get("customer_journey"),
                journey_stage=row.get("journey_stage"),
                interaction_moment=row.get("interaction_moment"),
            )
            df.at[i, "canonical_ed_id"] = res.get("canonical_ed_id")
            df.at[i, "canonical_ed_path"] = " > ".join(res.get("canonical_path") or [])
            df.at[i, "ed_confidence"] = res.get("confidence")
            df.at[i, "ed_resolution"] = res.get("resolution")
            df.at[i, "ed_matched_by"] = "|".join(res.get("matched_by", []))
            df.at[i, "ed_alternatives"] = json.dumps(res.get("alternatives", []))
        return df

    # ——— Commit and propose (separate from matching) ———
    def commit_assignment(self, decision: Dict[str, Any], pdca_signature: str):
        """
        Call this ONLY when decision['resolution'] == 'auto_accept'.
        Updates centroids (semantic) — no registry structure changes here.
        """
        if decision.get("resolution") != "auto_accept":
            return
        ed_id = decision["canonical_ed_id"]
        if not self.embed_index.model:
            return
        vec = self.embed_index.embed(pdca_signature)
        if vec is None:
            return
        self.embed_index.update_centroid(ed_id, vec)
        self.embed_index.save_cache()

    def propose_ed(self, candidate: Dict[str, Any]):
        """
        Append to proposals file (human/governance will later decide).
        """
        path = self.cfg["proposed_queue_path"]
        record = {
            "ts": datetime.utcnow().isoformat(),
            **candidate
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# Minimal CLI example
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    # Example usage with a tiny fake row (replace with your real data & registry)
    normalizer = EDNormalizer()

    row = {
        "experience_driver": "Payments > Card Management > Card Freeze Functionality > Block Card",
        "semantic_action_statement": "Customer tried to freeze the card urgently; feature failed to confirm.",
        "behavioral_impact": "Unauthorized transactions slipped through; trust erosion; financial risk.",
        "stream_justification": "Fix immediately: security-critical path must be reliable.",
        "customer_journey": "Security & Fraud Management",
        "journey_stage": "Card Freeze Attempt",
        "interaction_moment": "while attempting to freeze the card",
        "problem_statement": ""
    }

    res = normalizer.normalize(
        row["experience_driver"],
        semantic_action_statement=row["semantic_action_statement"],
        behavioral_impact=row["behavioral_impact"],
        stream_justification=row["stream_justification"],
        customer_journey=row["customer_journey"],
        journey_stage=row["journey_stage"],
        interaction_moment=row["interaction_moment"],
    )

    print(json.dumps(res, indent=2))

    # Commit example (only when auto_accept)
    if res["resolution"] == "auto_accept":
        sig = EDNormalizer.build_pdca_signature(row)
        normalizer.commit_assignment(res, sig)
