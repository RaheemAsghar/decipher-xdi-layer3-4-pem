# Enhanced EDNormalizer with critical improvements
# -*- coding: utf-8 -*-

import os, re, json, hashlib, logging, time, uuid, math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from textblob import TextBlob  # for spell correction
import Levenshtein  # for better string distance

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ---------------------------
# ENHANCED Config with new capabilities
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
    "synonyms_path": "registry/synonyms.json",
    "hierarchy_path": "registry/hierarchy.json",
    "validation_queue_path": "registry/validation_queue.jsonl",

    # Enhanced Thresholds
    "ml_accept": 0.86,
    "ml_strong": 0.90,
    "ml_perfect": 0.95,
    "fuzzy_cat_min": 0.90,
    "fuzzy_sub_min": 0.90,
    "fuzzy_strong": 0.95,
    "fuzzy_perfect": 0.98,
    
    # Semantic matching
    "semantic_threshold": 0.85,
    "cross_theme_penalty": 0.15,
    "enable_cross_theme": True,
    
    # Spell correction
    "enable_spell_correction": True,
    "spell_confidence_threshold": 0.80,
    
    # Temporal and confidence
    "temporal_decay_rate": 0.95,
    "min_confidence_threshold": 0.70,
    "uncertainty_band": [0.83, 0.88],
    
    # Consensus
    "consensus_weight_ml": 0.6,
    "consensus_weight_fuzzy": 0.25,
    "consensus_weight_semantic": 0.15,
    "consensus_min": 0.88,
    
    # Governance
    "auto_commit_new_on_empty_registry": True,
    "auto_commit_new_when_strong": True,
    "auto_commit_min_observations": 3,
    "require_human_validation_band": True,
    "min_pdca_signature_length": 15,

    # Grace and flexibility
    "grace_band": 0.02,
    "propose_only": False,
    
    # Quality checks
    "enable_anomaly_detection": True,
    "anomaly_threshold": 3.0,
    
    # 5Ws signature weights
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
# Utilities (from original)
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
# Registry (from original)
# ---------------------------
class EDRegistry:
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
        return [self.data["clusters"][eid] for eid in self.by_theme(theme) if eid in self.data["clusters"]]

    def commit_new_cluster(self, theme: str, category: str, subcategory: str,
                           centroid_vec: Optional[np.ndarray], alias: Optional[str] = None) -> str:
        canonical = canonical_label_from_parts(category, subcategory)
        ed_id = stable_ed_id(theme, canonical)

        if ed_id in self.data["clusters"]:
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
# Proposals (from original)
# ---------------------------
class ProposalsSink:
    def __init__(self, path: str):
        self.path = path
        ensure_dirs(path)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                pass

    def append(self, obj: Dict):
        obj["_ts"] = now_iso()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------------------
# Embedding cache (from original)
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
# Validation Queue for human review
# ---------------------------
class ValidationQueue:
    def __init__(self, path: str):
        self.path = path
        ensure_dirs(path)
    
    def add(self, item: Dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    def get_pending(self) -> List[Dict]:
        if not os.path.exists(self.path):
            return []
        items = []
        with open(self.path, "r") as f:
            for line in f:
                items.append(json.loads(line))
        return items

# ---------------------------
# ENHANCED Normalizer (9.5+ version)
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
        
        # NEW: Additional registries
        self.synonyms = self._load_synonyms()
        self.hierarchy = self._load_hierarchy()
        self.validation_queue = ValidationQueue(self.cfg["validation_queue_path"])
        self.observation_counts = defaultdict(int)

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

    def _load_synonyms(self) -> Dict[str, Set[str]]:
        path = self.cfg["synonyms_path"]
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                return {k: set(v) for k, v in data.items()}
        else:
            default_synonyms = {
                "payment": {"payment", "billing", "charge", "transaction", "checkout"},
                "delivery": {"delivery", "shipping", "shipment", "dispatch", "fulfillment"},
                "customer service": {"customer service", "support", "help desk", "assistance", "care"},
                "wait": {"wait", "delay", "hold", "queue", "pending"},
                "issue": {"issue", "problem", "error", "failure", "defect"},
                "long": {"long", "extended", "excessive", "prolonged", "lengthy"},
                "damaged": {"damaged", "broken", "defective", "faulty", "corrupted"},
                "missing": {"missing", "lost", "not found", "absent", "unavailable"},
            }
            ensure_dirs(path)
            with open(path, 'w') as f:
                json.dump({k: list(v) for k, v in default_synonyms.items()}, f, indent=2)
            return default_synonyms

    def _load_hierarchy(self) -> Dict[str, str]:
        path = self.cfg["hierarchy_path"]
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            default_hierarchy = {
                "Card declined": "Payment Issues",
                "Transaction failed": "Payment Issues",
                "Delayed shipment": "Delivery Issues",
                "Package lost": "Delivery Issues",
                "Long wait times": "Customer Service Issues",
                "Unhelpful agent": "Customer Service Issues",
            }
            ensure_dirs(path)
            with open(path, 'w') as f:
                json.dump(default_hierarchy, f, indent=2)
            return default_hierarchy

    def _spell_correct(self, text: str) -> Tuple[str, float]:
        if not self.cfg["enable_spell_correction"]:
            return text, 1.0
        
        try:
            blob = TextBlob(text)
            corrected = str(blob.correct())
            distance = Levenshtein.distance(text.lower(), corrected.lower())
            max_len = max(len(text), len(corrected))
            confidence = 1.0 - (distance / max_len) if max_len > 0 else 1.0
            
            if confidence >= self.cfg["spell_confidence_threshold"]:
                return corrected, confidence
        except:
            pass
        
        return text, 1.0

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if words1 == words2:
            return 1.0
        
        matches = 0
        total = max(len(words1), len(words2))
        
        for w1 in words1:
            for w2 in words2:
                if w1 == w2:
                    matches += 1
                    break
                for syn_group in self.synonyms.values():
                    if w1 in syn_group and w2 in syn_group:
                        matches += 0.9
                        break
        
        return matches / total if total > 0 else 0.0

    def _split_ed(self, raw_ed: str) -> Tuple[str, str]:
        s = (raw_ed or "").strip()
        s_corrected, spell_conf = self._spell_correct(s)
        if spell_conf >= self.cfg["spell_confidence_threshold"]:
            s = s_corrected
        
        s = re.sub(r'\s*[-–—→>/:]\s*', ' → ', s)
        
        if "→" not in s:
            return s, ""
        
        parts = s.split("→", 1)
        cat = parts[0].strip()
        sub = parts[1].strip() if len(parts) > 1 else ""
        
        cat_corrected, cat_conf = self._spell_correct(cat)
        sub_corrected, sub_conf = self._spell_correct(sub)
        
        if cat_conf >= self.cfg["spell_confidence_threshold"]:
            cat = cat_corrected
        if sub_conf >= self.cfg["spell_confidence_threshold"]:
            sub = sub_corrected
        
        return cat, sub

    def _update_fuzzy_registry(self, canonical: str, theme: str, canonical_cat: str, canonical_sub: str, raw_ed: str):
        ed_id = stable_ed_id(theme, canonical)
        existing = self.registry.get_cluster(ed_id)
        
        if existing:
            if raw_ed not in existing.get("aliases", []):
                self.registry.commit_alias(ed_id, raw_ed)
        
        return ed_id

    def _is_anomaly(self, vec: np.ndarray, theme_clusters: List[Dict]) -> bool:
        if not self.cfg["enable_anomaly_detection"] or len(theme_clusters) < 5:
            return False
        
        distances = []
        for cluster in theme_clusters:
            centroid = np.array(cluster.get("centroid", []), dtype=float)
            if centroid.size > 0:
                dist = 1.0 - cosine(vec, centroid)
                distances.append(dist)
        
        if not distances:
            return False
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        if std_dist == 0:
            return False
        
        min_dist = min(distances)
        z_score = (min_dist - mean_dist) / std_dist
        
        return z_score > self.cfg["anomaly_threshold"]

    def _fuzzy_component_match(self, theme: str, raw_ed: str) -> Tuple[str, float, str, Dict]:
        cat_min = self.cfg.get("fuzzy_cat_min", 0.90)
        sub_min = self.cfg.get("fuzzy_sub_min", 0.90)
        grace = self.cfg.get("grace_band", 0.02)
        
        cat_raw, sub_raw = self._split_ed(raw_ed)
        
        if not sub_raw:
            return raw_ed.strip(), 0.0, "no_structure", {"cat": 0.0, "sub": 0.0, "semantic": 0.0}
        
        theme_clusters = self.registry.list_theme_clusters(theme)
        
        parent_cat = self.hierarchy.get(sub_raw)
        if parent_cat and parent_cat != cat_raw:
            logging.info(f"Hierarchical suggestion: {cat_raw} → {sub_raw} should be under {parent_cat}")
        
        best_match = None
        best_score = 0.0
        best_method = "no_match"
        # FIX: Initialize these OUTSIDE the loop
        best_cat_score = 0.0
        best_sub_score = 0.0
        best_cat_semantic = 0.0
        best_sub_semantic = 0.0
        
        eds_in_theme = {}
        for cluster in theme_clusters:
            ed_id = stable_ed_id(cluster["theme"], cluster["canonical_label"])
            eds_in_theme[ed_id] = cluster
        
        cross_theme_candidates = {}
        if self.cfg["enable_cross_theme"]:
            for other_theme in self.registry.data["themes"]:
                if other_theme != theme:
                    for cluster in self.registry.list_theme_clusters(other_theme):
                        ed_id = stable_ed_id(cluster["theme"], cluster["canonical_label"])
                        cross_theme_candidates[ed_id] = cluster
        
        for cluster in eds_in_theme.values():
            if cluster["category"] == cat_raw and cluster["subcategory"] == sub_raw:
                return cluster["canonical_label"], 1.0, "exact_match", {"cat": 1.0, "sub": 1.0, "semantic": 1.0}
        
        all_candidates = list(eds_in_theme.values())
        
        for cluster in all_candidates:
            cat_fuzzy = fuzz.token_set_ratio(cat_raw, cluster["category"]) / 100.0
            cat_semantic = self._semantic_similarity(cat_raw, cluster["category"])
            cat_score = max(cat_fuzzy, cat_semantic * 0.95)
            
            sub_fuzzy = fuzz.token_set_ratio(sub_raw, cluster["subcategory"]) / 100.0
            sub_semantic = self._semantic_similarity(sub_raw, cluster["subcategory"])
            sub_score = max(sub_fuzzy, sub_semantic * 0.95)
            
            combined = (cat_score + sub_score) / 2.0
            
            if combined > best_score and cat_score >= (cat_min - grace) and sub_score >= (sub_min - grace):
                best_score = combined
                best_match = cluster
                best_method = "fuzzy_semantic"
                # FIX: Store the best scores
                best_cat_score = cat_score
                best_sub_score = sub_score
                best_cat_semantic = cat_semantic
                best_sub_semantic = sub_semantic
        
        if best_score < 0.85 and self.cfg["enable_cross_theme"]:
            for cluster in cross_theme_candidates.values():
                cat_fuzzy = fuzz.token_set_ratio(cat_raw, cluster["category"]) / 100.0
                cat_semantic = self._semantic_similarity(cat_raw, cluster["category"])
                cat_score = max(cat_fuzzy, cat_semantic * 0.95)
                
                sub_fuzzy = fuzz.token_set_ratio(sub_raw, cluster["subcategory"]) / 100.0
                sub_semantic = self._semantic_similarity(sub_raw, cluster["subcategory"])
                sub_score = max(sub_fuzzy, sub_semantic * 0.95)
                
                combined = ((cat_score + sub_score) / 2.0) * (1 - self.cfg["cross_theme_penalty"])
                
                if combined > best_score and combined >= 0.80:
                    best_score = combined
                    best_match = cluster
                    best_method = "cross_theme_match"
                    # FIX: Store the best scores for cross-theme too
                    best_cat_score = cat_score
                    best_sub_score = sub_score
                    best_cat_semantic = cat_semantic
                    best_sub_semantic = sub_semantic
                    logging.info(f"Cross-theme match: {raw_ed} → {cluster['canonical_label']} from theme {cluster['theme']}")
        
        if best_match and best_score >= (min(cat_min, sub_min) - grace):
            return best_match["canonical_label"], best_score, best_method, {
                "cat": best_cat_score,  # Now using the stored values
                "sub": best_sub_score, 
                "semantic": (best_cat_semantic + best_sub_semantic) / 2.0
            }
        
        canonical = f"{cat_raw} → {sub_raw}"
        return canonical, 0.0, "new_ed", {"cat": 0.0, "sub": 0.0, "semantic": 0.0}

    def ml_best_cluster(self, theme: str, vec: Optional[np.ndarray]) -> Tuple[Optional[str], float]:
        if vec is None:
            return None, 0.0
        
        clusters = self.registry.list_theme_clusters(theme)
        if not clusters:
            return None, 0.0
        
        if self._is_anomaly(vec, clusters):
            logging.warning(f"Anomaly detected in theme {theme}")
            
        best_id, best_cos = None, 0.0
        
        for c in clusters:
            centroid = np.array(c.get("centroid", []), dtype=float)
            if centroid.size == 0: 
                continue
            
            cos = cosine(vec, centroid)
            
            updated_str = c.get("updated_at", "")
            if updated_str:
                try:
                    updated = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                    days_old = (datetime.utcnow().replace(tzinfo=updated.tzinfo) - updated).days
                    months_old = days_old / 30.0
                    decay_factor = self.cfg["temporal_decay_rate"] ** months_old
                    cos *= decay_factor
                except:
                    pass
            
            if cos > best_cos:
                best_cos, best_id = cos, stable_ed_id(theme, c["canonical_label"])
        
        return best_id, best_cos

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
            toks.extend([ctx[:100]] * weights["what_context"])

        add("when_moment", weights["when_moment"])
        add("where_driver", weights["where_driver"])
        add("where_journey", weights["where_journey"])
        add("where_stage", weights["where_stage"])
        add("why_justification", weights["why_justification"])
        add("so_what_impact", weights["so_what_impact"])

        return " ".join(toks)

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

    def theme_gate(self, theme: str, cluster_theme: str) -> bool:
        return (theme or "").strip().lower() == (cluster_theme or "").strip().lower()

    def fuzzy_per_level(self, raw_ed: str, canonical_label: str) -> Tuple[float, float]:
        raw_cat, raw_sub = self._split_ed(raw_ed)
        can_cat, can_sub = self._split_ed(canonical_label)
        cat_score = fuzz.token_set_ratio(raw_cat, can_cat) / 100.0 if raw_cat and can_cat else 0.0
        sub_score = fuzz.token_set_ratio(raw_sub, can_sub) / 100.0 if raw_sub and can_sub else 0.0
        return cat_score, sub_score

    def fuzzy_match_ed(self, theme: str, raw_ed: str, domain: str = "general"):
        canonical, conf, method, part_scores = self._fuzzy_component_match(theme, raw_ed)
        return canonical, conf, method, part_scores

    def resolve_row(self, row: pd.Series, vec: Optional[np.ndarray]) -> Dict:
        cfg = self.cfg
        theme = (row.get("theme") or "").strip()
        raw_ed = (row.get("experience_driver") or "").strip()

        if not raw_ed:
            return {"resolution": "skip", "reason": "empty_ed"}

        ed_key = f"{theme}::{raw_ed}"
        self.observation_counts[ed_key] += 1

        ml_id, ml_cos = self.ml_best_cluster(theme, vec)
        ml_label = self.registry.get_cluster(ml_id)["canonical_label"] if ml_id else None

        fuzzy_canonical, fuzzy_conf, fuzzy_method, part_scores = self._fuzzy_component_match(theme, raw_ed)
        
        consensus = (
            ml_cos * cfg["consensus_weight_ml"] + 
            fuzzy_conf * cfg["consensus_weight_fuzzy"] +
            part_scores.get("semantic", 0) * cfg["consensus_weight_semantic"]
        )
        
        accepted_id = None
        method = ""
        confidence = 0.0
        
        if ml_cos >= cfg["ml_perfect"] and fuzzy_conf >= cfg["fuzzy_perfect"]:
            accepted_id = ml_id
            method = "perfect_match"
            confidence = min(ml_cos, fuzzy_conf)
        elif ml_id and ml_cos >= cfg["ml_strong"] and fuzzy_conf >= cfg["fuzzy_cat_min"]:
            accepted_id = ml_id
            method = "ml_strong_fuzzy_good"
            confidence = ml_cos
        elif consensus >= cfg["consensus_min"]:
            if ml_id:
                accepted_id = ml_id
                method = "consensus_accept"
                confidence = consensus
            elif fuzzy_method in ["fuzzy_semantic", "exact_match"]:
                ed_id = stable_ed_id(theme, fuzzy_canonical)
                if self.registry.get_cluster(ed_id):
                    accepted_id = ed_id
                    # CONTINUATION of resolve_row method
                    method = "fuzzy_semantic_accept"
                    confidence = fuzzy_conf
        
        # Check uncertainty band
        uncertainty_min, uncertainty_max = cfg["uncertainty_band"]
        if uncertainty_min <= consensus <= uncertainty_max and cfg["require_human_validation_band"]:
            self.validation_queue.add({
                "theme": theme,
                "raw_ed": raw_ed,
                "ml_candidate": ml_label,
                "ml_score": ml_cos,
                "fuzzy_candidate": fuzzy_canonical,
                "fuzzy_score": fuzzy_conf,
                "consensus": consensus,
                "timestamp": now_iso()
            })
            logging.info(f"Added to validation queue: {raw_ed} (consensus: {consensus:.3f})")

        if accepted_id:
            return {
                "resolution": "attach_existing",
                "ed_id": accepted_id,
                "canonical_label": self.registry.get_cluster(accepted_id)["canonical_label"],
                "ml_cos": round(ml_cos, 4),
                "fuzzy_conf": round(fuzzy_conf, 4),
                "semantic_score": round(part_scores.get("semantic", 0), 4),
                "consensus": round(consensus, 4),
                "confidence": round(confidence, 4),
                "method": method
            }

        # No acceptable match → create new
        cat_raw, sub_raw = self._split_ed(raw_ed)
        if not sub_raw:
            sub_raw = cat_raw

        canonical = canonical_label_from_parts(cat_raw, sub_raw)
        ed_id = stable_ed_id(theme, canonical)

        # Auto-commit rules
        will_autocommit = False
        observations = self.observation_counts.get(ed_key, 0)
        
        if self.registry.is_empty() and cfg["auto_commit_new_on_empty_registry"]:
            will_autocommit = True
        elif observations >= cfg["auto_commit_min_observations"]:
            will_autocommit = True
        elif cfg["auto_commit_new_when_strong"] and ml_cos >= 0.92:
            will_autocommit = True

        proposal = {
            "type": "new_cluster",
            "proposed_ed_id": ed_id,
            "theme": theme,
            "canonical_label": canonical,
            "category": cat_raw,
            "subcategory": sub_raw,
            "alias_observed": raw_ed,
            "ml_cos": round(ml_cos, 4),
            "fuzzy_conf": round(fuzzy_conf, 4),
            "semantic_score": round(part_scores.get("semantic", 0), 4),
            "consensus": round(consensus, 4),
            "observations": observations,
            "auto_commit": will_autocommit
        }
        self.proposals.append(proposal)

        if will_autocommit:
            self.registry.commit_new_cluster(theme, cat_raw, sub_raw, vec, alias=raw_ed)
            return {
                "resolution": "new_committed",
                "ed_id": ed_id,
                "canonical_label": canonical,
                "confidence": 1.0,
                "method": "auto_commit_new",
                "observations": observations
            }

        return {
            "resolution": "proposed_new",
            "proposed_ed_id": ed_id,
            "canonical_label": canonical,
            "confidence": 0.0,
            "method": "propose",
            "observations": observations
        }

    def normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point to normalize experience drivers in a dataframe.
        """
        # Build signatures
        sigs: List[str] = []
        valid_sig_idx: List[int] = []
        for i, row in df.iterrows():
            s = self.build_signature(row)
            if len(s) >= self.cfg["min_pdca_signature_length"]:
                sigs.append(s)
                valid_sig_idx.append(i)
            else:
                sigs.append("")
                valid_sig_idx.append(i)

        # Get embeddings
        vecs = self.embed_signatures(sigs)

        # Output columns
        out_cols = [
            "canonical_experience_driver", "ed_id",
            "normalization_confidence", "normalization_method",
            "ml_cos", "fuzzy_conf", "semantic_score", 
            "consensus", "resolution", "observations"
        ]
        for c in out_cols:
            df[c] = None

        # Process each row
        for i, row in df.iterrows():
            vec = vecs[i] if i < len(vecs) else None
            res = self.resolve_row(row, vec)

            if res["resolution"] in ("attach_existing", "new_committed"):
                df.at[i, "canonical_experience_driver"] = res.get("canonical_label")
                df.at[i, "ed_id"] = res.get("ed_id")
                df.at[i, "normalization_confidence"] = res.get("confidence", res.get("ml_cos", 0))
                df.at[i, "normalization_method"] = res.get("method")
                df.at[i, "ml_cos"] = res.get("ml_cos")
                df.at[i, "fuzzy_conf"] = res.get("fuzzy_conf")
                df.at[i, "semantic_score"] = res.get("semantic_score")
                df.at[i, "consensus"] = res.get("consensus")
                df.at[i, "resolution"] = res["resolution"]
                df.at[i, "observations"] = res.get("observations", 1)

                # Update centroid
                if res.get("ed_id") and vec is not None:
                    self.registry.commit_observation(res["ed_id"], vec)

            elif res["resolution"] == "proposed_new":
                df.at[i, "canonical_experience_driver"] = res.get("canonical_label")
                df.at[i, "ed_id"] = res.get("proposed_ed_id")
                df.at[i, "normalization_confidence"] = res.get("confidence", 0)
                df.at[i, "normalization_method"] = res.get("method")
                df.at[i, "ml_cos"] = res.get("ml_cos")
                df.at[i, "fuzzy_conf"] = res.get("fuzzy_conf")
                df.at[i, "semantic_score"] = res.get("semantic_score")
                df.at[i, "consensus"] = res.get("consensus")
                df.at[i, "resolution"] = "proposed_new"
                df.at[i, "observations"] = res.get("observations", 1)

            else:
                df.at[i, "resolution"] = res.get("resolution")

        # Summary logging
        total_rows = len(df)
        attached = (df["resolution"] == "attach_existing").sum()
        committed = (df["resolution"] == "new_committed").sum()
        proposed = (df["resolution"] == "proposed_new").sum()
        
        logging.info(f"Normalization complete: {attached} attached, {committed} committed, {proposed} proposed out of {total_rows} rows")
        
        pending_validations = len(self.validation_queue.get_pending())
        if pending_validations > 0:
            logging.info(f"{pending_validations} items pending human validation")

        return df


# ---------------------------
# CLI Usage (from original)
# ---------------------------
def run_normalization(
    input_csv: str,
    output_dir: str = "outputs",
    config: Dict = None,
    domain_hint: str = "retail"
):
    """
    Run the enhanced normalization on a CSV file.
    """
    ensure_dirs(os.path.join(output_dir, "x.tmp"))
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    normalizer = EDNormalizer(cfg)
    df = pd.read_csv(input_csv)

    # Check required columns
    required_cols = ["theme", "experience_driver"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Run normalization
    out_df = normalizer.normalize_df(df)

    # Save output
    base = os.path.splitext(os.path.basename(input_csv))[0]
    out_csv = os.path.join(output_dir, f"{base}_normalized.csv")
    out_df.to_csv(out_csv, index=False)

    # Print stats
    uniq_raw = df["experience_driver"].nunique()
    uniq_canon = out_df["canonical_experience_driver"].nunique()
    compression = (uniq_raw / uniq_canon) if uniq_canon else 1.0

    accepted = (out_df["resolution"] == "attach_existing").sum()
    committed = (out_df["resolution"] == "new_committed").sum()
    proposed = (out_df["resolution"] == "proposed_new").sum()

    print("\n=== ED Normalization Summary (Enhanced v9.5+) ===")
    print(f"Input rows:              {len(df):,}")
    print(f"Unique raw EDs:          {uniq_raw:,}")
    print(f"Unique canonical EDs:    {uniq_canon:,}")
    print(f"Compression ratio:       {compression:.2f}x")
    print(f"Attached existing:       {accepted:,}")
    print(f"New committed (auto):    {committed:,}")
    print(f"Proposed (needs review): {proposed:,}")
    print(f"Output CSV:              {out_csv}")
    print(f"Registry:                {cfg['registry_path']}")
    print(f"Proposals:               {cfg['proposals_path']}")
    print(f"Validation Queue:        {cfg['validation_queue_path']}\n")

    # Show validation items if any
    pending = normalizer.validation_queue.get_pending()
    if pending:
        print(f"\n⚠️  {len(pending)} items need human validation!")
        print("First 3 items:")
        for item in pending[:3]:
            print(f"  - {item['raw_ed']} (consensus: {item['consensus']:.3f})")

    return out_csv


if __name__ == "__main__":
    # Example usage
    INPUT = "data/decipher_retail_grocery_analytics_flattened.csv"
    if os.path.exists(INPUT):
        run_normalization(INPUT, output_dir="outputs")
    else:
        print("Place your CSV at data/decipher_retail_grocery_analytics_flattened.csv and rerun.")
        print("\nOr use the EDNormalizer class directly:")
        print("  normalizer = EDNormalizer()")
        print("  df_normalized = normalizer.normalize_df(your_dataframe)")