# ed_normalizer.py
# -*- coding: utf-8 -*-

import os, re, json, hashlib, logging, time, uuid, math
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ---------------------------
# Config (tweak as needed)
# ---------------------------
DEFAULT_CONFIG = {
    # Models / batching
    "embedding_model": "all-MiniLM-L6-v2",
    "embed_batch_size": 64,
    "show_embed_progress": False,
    "cache_embeddings": True,
    "cache_path": "cache/ed_embed_cache.pkl",

    # Paths
    "registry_path": "registry/ed_registry.json",
    "proposals_path": "registry/proposals.jsonl",

    # Thresholds
    "ml_accept": 0.86,            # cosine to accept an existing cluster
    "ml_strong": 0.90,            # very strong cosine
    "theme_gate_min": 0.75,       # (optional) if you add theme fuzzy later
    "fuzzy_cat_min": 0.86,        # category min
    "fuzzy_sub_min": 0.86,        # subcategory min
    "fuzzy_strong": 0.92,         # very strong fuzzy

    # Consensus
    "consensus_weight_ml": 0.7,
    "consensus_weight_fuzzy": 0.3,
    "consensus_min": 0.92,

    # Governance
    "auto_commit_new_on_empty_registry": True,  # day-1 bootstrap
    "auto_commit_new_when_strong": True,        # commit if ML>=0.92 OR (ML>=0.88 & fuzzy per-level>=0.94)
    "min_pdca_signature_length": 15,

    # Centroid / cluster constraints
    "max_cluster_size": 100,   # only affects confidence modifier (not hard stop)

    # 5Ws signature weights (must reflect your upstream)
    "signature_fields": {
        "what_reality": "semantic_action_statement.section_1_customer_reality",
        "what_matters": "matters",
        "what_context": "context",
        "when_moment": "interaction_moment",
        "where_driver": "experience_driver",
        "where_journey": "customer_journey",
        "where_stage": "customer_journey_stage",
        "why_justification": "stream_justification",
        "so_what_impact": "behavioral_impact",
    },
    "signature_weights": {
        "what_reality": 8,
        "what_matters": 6,
        "what_context": 2,
        "when_moment": 4,
        "where_driver": 3,
        "where_journey": 2,
        "where_stage": 1,
        "why_justification": 5,
        "so_what_impact": 7
    },

    # Logging
    "log_path": "logs/ed_normalizer.log",
    "log_level": "INFO",
}


# ---------------------------
# Utilities
# ---------------------------
ARROW = " → "
DELIMS_RE = re.compile(r"\s*(?:→|>|/|:|-)\s*")

def ensure_dirs(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def stable_ed_id(theme: str, canonical_label: str) -> str:
    key = f"{theme}::{canonical_label}".lower().strip()
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:10].upper()
    return f"ED-{h}"

def canonical_label_from_parts(category: str, subcategory: str) -> str:
    return f"{category.strip()}{ARROW}{subcategory.strip()}"

def parse_ed(raw: str) -> Tuple[str, str]:
    parts = [p.strip() for p in DELIMS_RE.split(raw, maxsplit=1)]
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------
# Registry (read/write with governance)
# ---------------------------
class EDRegistry:
    """
    Persistent store of ED clusters.
    Shape:
    {
      "version": "1.0",
      "created_at": "...",
      "updated_at": "...",
      "themes": { "<Theme>": {"ed_ids": [...] } },
      "clusters": {
        "ED-XXXX": {
          "theme": "...",
          "canonical_label": "Category → Subcategory",
          "category": "...",
          "subcategory": "...",
          "aliases": [...],
          "centroid": [float...],
          "count": int,
          "created_at": "...",
          "updated_at": "..."
        }
      }
    }
    """
    def __init__(self, path: str):
        self.path = path
        ensure_dirs(path)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {
                "version": "1.0",
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "themes": {},
                "clusters": {}
            }
            self.save()

    def save(self):
        self.data["updated_at"] = now_iso()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def is_empty(self) -> bool:
        return len(self.data["clusters"]) == 0

    def by_theme(self, theme: str) -> List[str]:
        return self.data["themes"].get(theme, {}).get("ed_ids", [])

    def get_cluster(self, ed_id: str) -> Optional[Dict]:
        return self.data["clusters"].get(ed_id)

    def list_theme_clusters(self, theme: str) -> List[Dict]:
        return [self.data["clusters"][eid] for eid in self.by_theme(theme)]

    # ---- governance-approved commits (safe updates) ----
    def commit_new_cluster(self, theme: str, category: str, subcategory: str,
                           centroid_vec: Optional[np.ndarray], alias: Optional[str] = None) -> str:
        canonical = canonical_label_from_parts(category, subcategory)
        ed_id = stable_ed_id(theme, canonical)

        if ed_id in self.data["clusters"]:
            # already exists; optionally add alias
            if alias and alias not in self.data["clusters"][ed_id]["aliases"]:
                self.data["clusters"][ed_id]["aliases"].append(alias)
            self.save()
            return ed_id

        self.data["clusters"][ed_id] = {
            "theme": theme,
            "canonical_label": canonical,
            "category": category,
            "subcategory": subcategory,
            "aliases": [alias] if alias else [],
            "centroid": (centroid_vec.tolist() if centroid_vec is not None else []),
            "count": 1 if centroid_vec is not None else 0,
            "created_at": now_iso(),
            "updated_at": now_iso()
        }
        self.data["themes"].setdefault(theme, {"ed_ids": []})
        if ed_id not in self.data["themes"][theme]["ed_ids"]:
            self.data["themes"][theme]["ed_ids"].append(ed_id)
        self.save()
        return ed_id

    def commit_alias(self, ed_id: str, alias: str):
        c = self.get_cluster(ed_id)
        if not c: return
        if alias not in c["aliases"]:
            c["aliases"].append(alias)
            c["updated_at"] = now_iso()
            self.save()

    def commit_observation(self, ed_id: str, vector: Optional[np.ndarray]):
        """
        Weighted centroid update; count += 1
        """
        c = self.get_cluster(ed_id)
        if not c or vector is None: return
        old = np.array(c["centroid"], dtype=float) if c["centroid"] else None
        if old is None or old.size == 0:
            c["centroid"] = vector.tolist()
            c["count"] = 1
        else:
            n = max(1, int(c.get("count", 1)))
            new_centroid = (old * n + vector) / (n + 1)
            c["centroid"] = new_centroid.tolist()
            c["count"] = n + 1
        c["updated_at"] = now_iso()
        self.save()


# ---------------------------
# Proposals (append-only)
# ---------------------------
class ProposalsSink:
    def __init__(self, path: str):
        self.path = path
        ensure_dirs(path)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                pass  # create empty file

    def append(self, obj: Dict):
        obj["_ts"] = now_iso()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------
# Embedding cache
# ---------------------------
class EmbeddingCache:
    def __init__(self, path: str, enabled: bool):
        self.path = path
        self.enabled = enabled
        ensure_dirs(path)
        if enabled and os.path.exists(path):
            try:
                self._store = pd.read_pickle(path)
            except Exception:
                self._store = {}
        else:
            self._store = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, vec: np.ndarray):
        self._store[key] = vec
        if self.enabled:
            pd.to_pickle(self._store, self.path)


# ---------------------------
# Normalizer (read-only matching, proposals out; optional safe auto-commit)
# ---------------------------
class EDNormalizer:
    def __init__(self, config: Dict = None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        ensure_dirs(self.cfg["log_path"])
        logging.basicConfig(
            level=getattr(logging, self.cfg["log_level"]),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.cfg["log_path"]), logging.StreamHandler()]
        )

        self.registry = EDRegistry(self.cfg["registry_path"])
        self.proposals = ProposalsSink(self.cfg["proposals_path"])

        self.model = None
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer(self.cfg["embedding_model"])
                logging.info(f"Loaded embedding model: {self.cfg['embedding_model']}")
            except Exception as e:
                logging.warning(f"Could not load embedding model ({e}). Running fuzzy-only.")
        else:
            logging.warning("sentence_transformers not available. Running fuzzy-only.")

        self.cache = EmbeddingCache(self.cfg["cache_path"], self.cfg["cache_embeddings"])

    # ----- Signature builder (5Ws) -----
    def build_signature(self, row: pd.Series) -> str:
        fields = self.cfg["signature_fields"]
        weights = self.cfg["signature_weights"]
        toks: List[str] = []

        def pick(key: str) -> str:
            col = fields.get(key)
            if not col: return ""
            val = row.get(col, "")
            return (str(val) if pd.notna(val) else "").strip()

        def add(key: str, times: int):
            v = pick(key)
            if v:
                toks.extend([v] * times)

        add("what_reality", weights["what_reality"])
        add("what_matters", weights["what_matters"])
        ctx = pick("what_context")
        if ctx:
            toks.extend([ctx[:100]] * weights["what_context"])  # short clause

        add("when_moment", weights["when_moment"])
        add("where_driver", weights["where_driver"])
        add("where_journey", weights["where_journey"])
        add("where_stage", weights["where_stage"])
        add("why_justification", weights["why_justification"])
        add("so_what_impact", weights["so_what_impact"])

        return " ".join(toks)

    # ----- Batch embed -----
    def embed_signatures(self, sigs: List[str]) -> List[Optional[np.ndarray]]:
        if not self.model:
            return [None] * len(sigs)

        out: List[Optional[np.ndarray]] = [None] * len(sigs)
        batch = self.cfg["embed_batch_size"]
        for i in range(0, len(sigs), batch):
            chunk = sigs[i:i+batch]
            keys = [hashlib.md5(s.encode("utf-8")).hexdigest() for s in chunk]
            to_compute_idx = []
            to_compute = []
            for j, (k, s) in enumerate(zip(keys, chunk)):
                vec = self.cache.get(k) if self.cache else None
                if vec is None:
                    to_compute_idx.append(j)
                    to_compute.append(s)
                else:
                    out[i+j] = vec

            if to_compute:
                arr = self.model.encode(
                    to_compute,
                    batch_size=min(batch, len(to_compute)),
                    show_progress_bar=self.cfg["show_embed_progress"],
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )
                for j_local, vec in enumerate(arr):
                    j = to_compute_idx[j_local]
                    out[i+j] = vec
                    if self.cache:
                        self.cache.set(keys[j], vec)

        return out

    # ----- Theme gate (exact for now; add fuzzy later if you need cross-theme) -----
    def theme_gate(self, theme: str, cluster_theme: str) -> bool:
        return (theme or "").strip().lower() == (cluster_theme or "").strip().lower()

    # ----- Fuzzy per-level against a candidate canonical label -----
    def fuzzy_per_level(self, raw_ed: str, canonical_label: str) -> Tuple[float, float]:
        raw_cat, raw_sub = parse_ed(raw_ed)
        can_cat, can_sub = parse_ed(canonical_label)
        cat_score = fuzz.token_set_ratio(raw_cat, can_cat) / 100.0 if raw_cat and can_cat else 0.0
        sub_score = fuzz.token_set_ratio(raw_sub, can_sub) / 100.0 if raw_sub and can_sub else 0.0
        return cat_score, sub_score

    # ----- ML match against theme clusters -----
    def ml_best_cluster(self, theme: str, vec: Optional[np.ndarray]) -> Tuple[Optional[str], float]:
        if vec is None:
            return None, 0.0
        clusters = self.registry.list_theme_clusters(theme)
        if not clusters:
            return None, 0.0
        best_id, best_cos = None, 0.0
        for c in clusters:
            centroid = np.array(c.get("centroid", []), dtype=float)
            if centroid.size == 0: 
                continue
            cos = cosine(vec, centroid)
            if cos > best_cos:
                best_cos, best_id = cos, stable_ed_id(theme, c["canonical_label"])
        return best_id, best_cos

    # ----- Decision logic for a single row -----
    def resolve_row(self, row: pd.Series, vec: Optional[np.ndarray]) -> Dict:
        cfg = self.cfg
        theme = (row.get("theme") or "").strip()
        raw_ed = (row.get("experience_driver") or "").strip()

        if not raw_ed:
            return {"resolution": "skip", "reason": "empty_ed"}

        # 1) ML candidate (by theme)
        ml_id, ml_cos = self.ml_best_cluster(theme, vec)
        ml_label = self.registry.get_cluster(ml_id)["canonical_label"] if ml_id else None

        # 2) Fuzzy per-level against ML candidate (if present)
        fuzzy_cat = fuzzy_sub = 0.0
        if ml_label:
            fuzzy_cat, fuzzy_sub = self.fuzzy_per_level(raw_ed, ml_label)

        # 3) Decide acceptance
        accepted_id = None
        method = ""
        consensus = ml_cos * cfg["consensus_weight_ml"] + ((fuzzy_cat+fuzzy_sub)/2.0) * cfg["consensus_weight_fuzzy"]

        if ml_id and ml_cos >= cfg["ml_accept"] and (fuzzy_cat >= cfg["fuzzy_cat_min"] and fuzzy_sub >= cfg["fuzzy_sub_min"]):
            accepted_id, method = ml_id, "ml+fuzzy_accept"
        elif ml_id and ml_cos >= cfg["ml_strong"]:
            accepted_id, method = ml_id, "ml_strong_accept"
        elif ml_id and consensus >= cfg["consensus_min"] and (fuzzy_cat >= cfg["fuzzy_cat_min"] and fuzzy_sub >= cfg["fuzzy_sub_min"]):
            accepted_id, method = ml_id, "consensus_accept"

        if accepted_id:
            return {
                "resolution": "attach_existing",
                "ed_id": accepted_id,
                "canonical_label": self.registry.get_cluster(accepted_id)["canonical_label"],
                "ml_cos": round(ml_cos, 4),
                "fuzzy_cat": round(fuzzy_cat, 4),
                "fuzzy_sub": round(fuzzy_sub, 4),
                "consensus": round(consensus, 4),
                "method": method
            }

        # 4) No acceptable match → propose new
        raw_cat, raw_sub = parse_ed(raw_ed)
        if not raw_sub:  # if only one token given, treat as sub under itself
            raw_sub = raw_cat

        canonical = canonical_label_from_parts(raw_cat, raw_sub)
        ed_id = stable_ed_id(theme, canonical)

        # auto-commit rules:
        will_autocommit = False
        if self.registry.is_empty() and self.cfg["auto_commit_new_on_empty_registry"]:
            will_autocommit = True
        elif self.cfg["auto_commit_new_when_strong"]:
            strong = (ml_cos >= 0.92) or (ml_cos >= 0.88 and min(fuzzy_cat, fuzzy_sub) >= 0.94)
            will_autocommit = bool(strong)

        # create proposal record
        proposal = {
            "type": "new_cluster",
            "proposed_ed_id": ed_id,
            "theme": theme,
            "canonical_label": canonical,
            "category": raw_cat,
            "subcategory": raw_sub,
            "alias_observed": raw_ed,
            "ml_cos": round(ml_cos, 4),
            "fuzzy_cat": round(fuzzy_cat, 4),
            "fuzzy_sub": round(fuzzy_sub, 4),
            "auto_commit": will_autocommit
        }
        self.proposals.append(proposal)

        if will_autocommit:
            self.registry.commit_new_cluster(theme, raw_cat, raw_sub, vec, alias=raw_ed)
            method = "auto_commit_new"
            return {
                "resolution": "new_committed",
                "ed_id": ed_id,
                "canonical_label": canonical,
                "ml_cos": round(ml_cos, 4),
                "fuzzy_cat": round(fuzzy_cat, 4),
                "fuzzy_sub": round(fuzzy_sub, 4),
                "method": method
            }

        return {
            "resolution": "proposed_new",
            "proposed_ed_id": ed_id,
            "canonical_label": canonical,
            "ml_cos": round(ml_cos, 4),
            "fuzzy_cat": round(fuzzy_cat, 4),
            "fuzzy_sub": round(fuzzy_sub, 4),
            "method": "propose"
        }

    # ----- Public entrypoint: normalize a dataframe -----
    def normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # build signatures (once)
        sigs: List[str] = []
        valid_sig_idx: List[int] = []
        for i, row in df.iterrows():
            s = self.build_signature(row)
            if len(s) >= self.cfg["min_pdca_signature_length"]:
                sigs.append(s); valid_sig_idx.append(i)
            else:
                sigs.append("")  # placeholder; we won't embed
                valid_sig_idx.append(i)

        # embeddings (batched)
        vecs = self.embed_signatures([s for s in sigs])

        # outputs
        out_cols = [
            "canonical_experience_driver", "ed_id",
            "normalization_confidence", "normalization_method",
            "ml_cos", "fuzzy_cat", "fuzzy_sub", "consensus", "resolution"
        ]
        for c in out_cols:
            df[c] = None

        # resolve each row
        for i, row in df.iterrows():
            vec = vecs[i] if i < len(vecs) else None
            res = self.resolve_row(row, vec)

            if res["resolution"] in ("attach_existing", "new_committed"):
                df.at[i, "canonical_experience_driver"] = res["canonical_label"]
                df.at[i, "ed_id"] = res.get("ed_id")
                df.at[i, "normalization_confidence"] = res.get("ml_cos")
                df.at[i, "normalization_method"] = res.get("method")
                df.at[i, "ml_cos"] = res.get("ml_cos")
                df.at[i, "fuzzy_cat"] = res.get("fuzzy_cat")
                df.at[i, "fuzzy_sub"] = res.get("fuzzy_sub")
                df.at[i, "consensus"] = res.get("consensus")
                df.at[i, "resolution"] = res["resolution"]

                # commit observation to centroid (only when we attached/committed)
                if res.get("ed_id") and vec is not None:
                    self.registry.commit_observation(res["ed_id"], vec)

            elif res["resolution"] == "proposed_new":
                df.at[i, "canonical_experience_driver"] = res["canonical_label"]
                df.at[i, "ed_id"] = res.get("proposed_ed_id")
                df.at[i, "normalization_confidence"] = res.get("ml_cos")
                df.at[i, "normalization_method"] = res.get("method")
                df.at[i, "ml_cos"] = res.get("ml_cos")
                df.at[i, "fuzzy_cat"] = res.get("fuzzy_cat")
                df.at[i, "fuzzy_sub"] = res.get("fuzzy_sub")
                df.at[i, "consensus"] = res.get("consensus")
                df.at[i, "resolution"] = "proposed_new"

            else:
                df.at[i, "resolution"] = res["resolution"]

        return df


# ---------------------------
# CLI-ish usage
# ---------------------------
def run_normalization(
    input_csv: str,
    output_dir: str = "outputs",
    config: Dict = None,
    domain_hint: str = "retail"  # reserved for future domain-tuned thresholds
):
    ensure_dirs(os.path.join(output_dir, "x.tmp"))
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    normalizer = EDNormalizer(cfg)
    df = pd.read_csv(input_csv)

    # sanity columns
    required_cols = ["theme", "experience_driver"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out_df = normalizer.normalize_df(df)

    base = os.path.splitext(os.path.basename(input_csv))[0]
    out_csv = os.path.join(output_dir, f"{base}_normalized.csv")
    out_df.to_csv(out_csv, index=False)

    # quick stats
    uniq_raw = df["experience_driver"].nunique()
    uniq_canon = out_df["canonical_experience_driver"].nunique()
    compression = (uniq_raw / uniq_canon) if uniq_canon else 1.0

    accepted = (out_df["resolution"] == "attach_existing").sum()
    committed = (out_df["resolution"] == "new_committed").sum()
    proposed = (out_df["resolution"] == "proposed_new").sum()

    print("\n=== ED Normalization Summary ===")
    print(f"Input rows:              {len(df):,}")
    print(f"Unique raw EDs:          {uniq_raw:,}")
    print(f"Unique canonical EDs:    {uniq_canon:,}")
    print(f"Compression ratio:       {compression:.2f}x")
    print(f"Attached existing:       {accepted:,}")
    print(f"New committed (auto):    {committed:,}")
    print(f"Proposed (needs review): {proposed:,}")
    print(f"Output CSV:              {out_csv}")
    print(f"Registry:                {cfg['registry_path']}")
    print(f"Proposals:               {cfg['proposals_path']}\n")

    return out_csv


if __name__ == "__main__":
    # Example:
    # python ed_normalizer.py
    INPUT = "data/decipher_retail_grocery_analytics_flattened.csv"
    if os.path.exists(INPUT):
        run_normalization(INPUT, output_dir="outputs")
    else:
        print("Place your CSV at data/decipher_retail_grocery_analytics_flattened.csv and rerun.")
