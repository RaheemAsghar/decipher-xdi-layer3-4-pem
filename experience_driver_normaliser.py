import sys
# Make stdout/stderr UTF-8 so logging can print → and ✓
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import os
import json
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timezone
import pickle
from collections import defaultdict
import logging
import re
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import threading
from functools import lru_cache

class GodTierCanonicalizer:
    """
    🚀 GOD-TIER Experience Driver Canonicalizer
    
    Production-grade semantic deduplication engine with:
    - Dual-gate safety enforcement 
    - Stable canonical IDs with audit trails
    - Human-in-the-loop proposal workflows
    - Blocklist/allowlist guardrails
    - ML-powered embedding similarity
    - Performance optimization & monitoring
    """
    
    def __init__(self, config_path="canonicalization_config.json", base_dir="registry"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.config = self.load_config(config_path)
        self.registry = self.load_registry()
                
        # ML Components
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words='english',
            lowercase=True
        )
        
        # Performance monitoring
        self.stats = {
            "total_processed": 0,
            "merges_approved": 0,
            "proposals_queued": 0,
            "blocklist_hits": 0,
            "errors": 0,
            "processing_time": 0
        }
        
        self.indexes = self.load_or_build_indexes()
        self.proposals_path = self.base_dir / "proposals.jsonl"
        self.audit_path = self.base_dir / "audit_trail.jsonl"

        self.setup_logging()
        self._lock = threading.Lock()  # Thread safety
        
        logging.info("🚀 God-Tier Canonicalizer initialized")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load enhanced configuration with safety rails"""
        default_config = {
            "domain_thresholds": {
                "financial_services": {"theme": 96, "category": 96, "subcategory": 95},
                "healthcare": {"theme": 96, "category": 95, "subcategory": 94},
                "legal": {"theme": 97, "category": 96, "subcategory": 95},
                "retail": {"theme": 92, "category": 88, "subcategory": 85},
                "ecommerce": {"theme": 90, "category": 87, "subcategory": 85},
                "support": {"theme": 85, "category": 83, "subcategory": 80},
                "social_media": {"theme": 82, "category": 80, "subcategory": 78},
                "general": {"theme": 90, "category": 88, "subcategory": 85}
            },
            
            # 🛡️ SAFETY RAILS
            "dual_gate_enforcement": True,
            "require_both_gates": True,
            "embedding_similarity_threshold": 0.85,
            "semantic_safety_margin": 0.05,
            
                     
            # 🎯 PERFORMANCE
            "batch_size": 1000,
            "cache_size": 10000,
            "rebuild_index_threshold": 100,
            "min_group_size": 2,
            "max_variants_per_ed": 50,
            
            # 📊 MONITORING
            "enable_audit_trail": True,
            "enable_performance_monitoring": True,
            "log_level": "INFO"
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # Deep merge configs
                self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _is_blocked_merge(self, ed1: str, ed2: str) -> bool:
        """No hard-coded blocks: always allow decision to be made by dual-gate/similarity."""
        return False

    def _is_allowed_merge(self, ed1: str, ed2: str) -> bool:
        """No hard-coded allows: no force-merge; rely on dual-gate/similarity only."""
        return False

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """Deep merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def load_registry(self) -> Dict[str, Any]:
        """Load registry with enhanced structure"""
        registry_path = self.base_dir / "canonical_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
                    # Ensure new structure fields exist
                    if "metadata" not in registry:
                        registry["metadata"] = {
                            "version": "2.0",
                            "created": datetime.now(timezone.utc).isoformat(),
                            "last_updated": datetime.now(timezone.utc).isoformat()
                        }
                    return registry
            except Exception as e:
                logging.error(f"Failed to load registry: {e}")
        
        # Create new registry with enhanced structure
        return {
            "metadata": {
                "version": "2.0",
                "created": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat()
            },
            "themes": {},
            "experience_drivers": {},
            "id_mappings": {},  # canonical_ed_id -> canonical_ed_text
            "frozen_eds":  [],
            "review_queue": []
        }
     
    def load_or_build_indexes(self) -> Dict[str, Any]:
        """Load or build performance indexes with embeddings"""
        index_path = self.base_dir / "god_tier_indexes.pkl"
        
        if index_path.exists():
            try:
                with open(index_path, 'rb') as f:
                    indexes = pickle.load(f)
                    # Validate index integrity
                    if self._validate_indexes(indexes):
                        logging.info("✅ Loaded existing indexes")
                        return indexes
                    else:
                        logging.warning("⚠️ Index validation failed, rebuilding...")
            except Exception as e:
                logging.error(f"Failed to load indexes: {e}")
        
        return self.build_indexes()
    
    def _validate_indexes(self, indexes: Dict) -> bool:
        """Validate index integrity"""
        required_keys = [
            "theme_lookup", "category_lookup", "subcategory_lookup", 
            "ed_vectors", "ed_list", "ed_embeddings", "last_rebuild"
        ]
        return all(key in indexes for key in required_keys)
    
    def build_indexes(self) -> Dict[str, Any]:
        """Build optimized lookup indexes with ML embeddings"""
        logging.info("🔨 Building God-Tier indexes...")
        
        indexes = {
            "theme_lookup": {},
            "category_lookup": defaultdict(list),
            "subcategory_lookup": defaultdict(list),
            "ed_vectors": None,
            "ed_embeddings": {},
            "ed_list": [],
            "id_to_text": {},  # canonical_ed_id -> text mapping
            "text_to_id": {},  # text -> canonical_ed_id mapping
            "last_rebuild": datetime.now(timezone.utc).isoformat(),
            "entry_count": 0,
            "build_version": "2.0"
        }
        
        # Build theme index with fuzzy lookup optimization
        for theme in self.registry["themes"].keys():
            normalized = self._normalize_text(theme)
            indexes["theme_lookup"][normalized] = theme
        
        # Build Experience Driver indexes with embeddings
        ed_texts = []
        for ed_key, ed_data in self.registry["experience_drivers"].items():
            ed_id = ed_data.get("canonical_ed_id")
            if ed_id:
                indexes["id_to_text"][ed_id] = ed_key
                indexes["text_to_id"][ed_key] = ed_id
            
            indexes["ed_list"].append(ed_key)
            ed_texts.append(ed_key)
            
            # Category lookup optimization
            category = ed_data.get("canonical_category", "")
            if category:
                normalized_cat = self._normalize_text(category)
                indexes["category_lookup"][normalized_cat].append(ed_key)
            
            # Subcategory lookup optimization
            subcategory = ed_data.get("canonical_subcategory", "")
            if subcategory:
                normalized_subcat = self._normalize_text(subcategory)
                indexes["subcategory_lookup"][normalized_subcat].append(ed_key)
        
        # Build ML vectors for semantic similarity
        if ed_texts:
            try:
                # TF-IDF vectors
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(ed_texts)
                indexes["ed_vectors"] = tfidf_matrix
                
                # Store individual embeddings for faster lookup
                for i, ed_text in enumerate(ed_texts):
                    indexes["ed_embeddings"][ed_text] = tfidf_matrix[i]
                
                indexes["entry_count"] = len(ed_texts)
                logging.info(f"✅ Built ML embeddings for {len(ed_texts)} Experience Drivers")
                
            except Exception as e:
                logging.error(f"Failed to build ML vectors: {e}")
        
        self.save_indexes(indexes)
        logging.info(f"🚀 God-Tier indexes built successfully")
        return indexes
    
    @lru_cache(maxsize=1000)
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text without mangling hyphens/apostrophes.
        Only unify arrow variants to the single glyph '→', collapse spaces,
        lowercase, and strip stray punctuation (but KEEP hyphen/apostrophe).
        """
        if not text:
            return ""

        s = str(text)

        # Unify common arrow spellings/glyphs to '→' (U+2192)
        s = (s.replace("->", "→")
            .replace("—>", "→")
            .replace("–>", "→")
            .replace("=>", "→")
            .replace("⇒", "→")
            .replace("➔", "→")
            .replace("➜", "→"))

        # Idempotent (if already arrow)
        s = s.replace("→", "→")

        # Basic canonicalization
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)

        # Allow letters/digits/underscore, spaces, hyphen, apostrophe, and '→'
        # (prevents accidental removal of meaningful '-' or "'")
        s = re.sub(r"[^\w\s\-→']", "", s)

        return s
  
    def _log_validation_error(self, error_type: str, raw_ed: str) -> None:
        """Log validation errors for monitoring"""
        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "validation_error",
            "error_type": error_type,
            "raw_ed": raw_ed
        }
        self._append_to_audit(error_entry)
    
    def get_domain_thresholds(self, domain: str = "general") -> Dict[str, int]:
        """Get domain-specific thresholds"""
        return self.config["domain_thresholds"].get(
            domain, 
            self.config["domain_thresholds"]["general"]
        )
    
    def fast_theme_match(self, raw_theme: str, domain: str = "general") -> Tuple[str, int]:
        """Optimized theme matching with safety checks"""
        if not raw_theme or pd.isna(raw_theme):
            return "Unknown", 0
        
        thresholds = self.get_domain_thresholds(domain)
        normalized = self._normalize_text(raw_theme)
        
        # O(1) exact match
        if normalized in self.indexes["theme_lookup"]:
            return self.indexes["theme_lookup"][normalized], 100
        
        # Fuzzy match with existing themes
        theme_choices = list(self.registry["themes"].keys())
        if not theme_choices:
            return self._create_new_theme(raw_theme), 100
        
        match = process.extractOne(
            raw_theme, 
            theme_choices, 
            scorer=fuzz.token_set_ratio
        )
        
        if match and match[1] >= thresholds["theme"]:
            return match[0], match[1]
        
        return self._create_new_theme(raw_theme), 100
    
   
    def _semantic_match(self, text1: str, text2: str, threshold: float = 0.9) -> bool:
        """Check semantic similarity between texts"""
        return fuzz.token_set_ratio(text1, text2) >= (threshold * 100)
    
    def _clean_experience_driver(self, raw_ed: str) -> Optional[str]:
        """Validate and normalize 'Category → Subcategory' format."""
        if not raw_ed or pd.isna(raw_ed):
            return None
        s = str(raw_ed).strip()
        # unify arrow variants only (mirror _normalize_text logic but without stripping chars)
        s = (s.replace("->","→").replace("—>","→").replace("–>","→")
            .replace("=>","→").replace("⇒","→").replace("➔","→").replace("➜","→"))
        if "→" not in s:
            self._log_validation_error("no_arrow", raw_ed)
            return None
        parts = [p.strip() for p in s.split("→")]
        if len(parts) != 2:
            self._log_validation_error("invalid_structure", raw_ed)
            return None
        category, subcategory = parts
        if not category or not subcategory:
            self._log_validation_error("empty_parts", raw_ed)
            return None
        return f"{category} → {subcategory}"
 
    def god_tier_ed_match(self, theme: str, raw_ed: str, domain: str = "general") -> Dict[str, Any]:
        """🚀 GOD-TIER Experience Driver matching with all safety rails"""
        
        # Clean and validate
        clean_ed = self._clean_experience_driver(raw_ed)
        if not clean_ed:
            return self._create_error_result(raw_ed, "invalid_format")
        
        thresholds = self.get_domain_thresholds(domain)
        category_raw, subcategory_raw = clean_ed.split("→")
        category_raw, subcategory_raw = category_raw.strip(), subcategory_raw.strip()
        
        # Find best matches
        cat_match, cat_score = self._enhanced_category_match(theme, category_raw, thresholds["category"])
        subcat_match, subcat_score = self._enhanced_subcategory_match(theme, cat_match, subcategory_raw, thresholds["subcategory"])
        
        proposed_canonical = f"{cat_match} → {subcat_match}"
        
        # 🛡️ DUAL-GATE ENFORCEMENT
        passes_category = cat_score >= thresholds["category"]
        passes_subcategory = subcat_score >= thresholds["subcategory"]
        dual_gate_passed = passes_category and passes_subcategory
        
        # 🚫 SAFETY CHECKS
        is_blocked = self._is_blocked_merge(clean_ed, proposed_canonical)
        is_allowed = self._is_allowed_merge(clean_ed, proposed_canonical)
        
        # --- FINAL DECISION (no allow/block lists) ---
        if dual_gate_passed:
            canonical_ed = proposed_canonical
            needs_review = False
            action = "merged"
            self.stats["merges_approved"] += 1
        else:
            canonical_ed = clean_ed
            needs_review = True
            action = "queued_for_review"
            self.stats["proposals_queued"] += 1
            self._queue_proposal({
                "theme": theme,
                "raw_ed": clean_ed,
                "raw_category": category_raw,
                "raw_subcategory": subcategory_raw,
                "suggested_canonical": proposed_canonical,
                "suggested_category": cat_match,
                "suggested_subcategory": subcat_match,
                "scores": {"category": cat_score, "subcategory": subcat_score},
                "gates": {
                    "category_passed": passes_category,
                    "subcategory_passed": passes_subcategory,
                    "dual_gate_passed": dual_gate_passed
                },
                "confidence": round((cat_score + subcat_score) / 2, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "domain": domain
            })
                
        # Update registry
        ed_id = self._update_ed_registry(canonical_ed, theme, cat_match, subcat_match, clean_ed, needs_review)
        
        return {
                "canonical_ed": canonical_ed,
                "canonical_ed_id": ed_id,
                "canonical_category": cat_match,
                "canonical_subcategory": subcat_match,
                "category_score": cat_score,
                "subcategory_score": subcat_score,
                "needs_review": needs_review,
                "action": action,
                "dual_gate_passed": dual_gate_passed,
                "safety_checks": {"is_blocked": False, "is_allowed": False}
            }
  
    def _create_error_result(self, raw_ed: str, error_type: str) -> Dict[str, Any]:
        """Create error result for invalid inputs"""
        self.stats["errors"] += 1
        return {
            "canonical_ed": raw_ed,
            "canonical_ed_id": None,
            "canonical_category": None,
            "canonical_subcategory": None,
            "category_score": 0,
            "subcategory_score": 0,
            "needs_review": True,
            "action": "error",
            "error_type": error_type,
            "dual_gate_passed": False,
            "safety_checks": {"is_blocked": False, "is_allowed": False}
        }
    
    def _enhanced_category_match(self, theme: str, category_raw: str, threshold: int) -> Tuple[str, int]:
        """Enhanced category matching with ML similarity"""
        normalized = self._normalize_text(category_raw)
        
        # Quick index lookup
        if normalized in self.indexes["category_lookup"]:
            candidates = self.indexes["category_lookup"][normalized]
            theme_filtered = [
                self.registry["experience_drivers"][ed]["canonical_category"]
                for ed in candidates
                if self.registry["experience_drivers"][ed]["theme"] == theme
            ]
            if theme_filtered:
                return theme_filtered[0], 100
        
        # Get existing categories for this theme
        existing_cats = list(set([
            ed["canonical_category"]
            for ed in self.registry["experience_drivers"].values()
            if ed["theme"] == theme and ed.get("canonical_category")
        ]))
        
        if not existing_cats:
            return category_raw, 100
        
        # Fuzzy matching
        match = process.extractOne(
            category_raw, 
            existing_cats, 
            scorer=fuzz.token_set_ratio
        )
        
        if match and match[1] >= threshold:
            return match[0], match[1]
        
        # ML semantic similarity backup
        similarity_score = self._calculate_semantic_similarity(category_raw, existing_cats)
        if similarity_score[1] >= self.config["embedding_similarity_threshold"]:
            # Convert cosine similarity to percentage
            return similarity_score[0], int(similarity_score[1] * 100)
        
        return category_raw, 100
    
    def _enhanced_subcategory_match(self, theme: str, category: str, subcategory_raw: str, threshold: int) -> Tuple[str, int]:
        """Enhanced subcategory matching with ML similarity"""
        normalized = self._normalize_text(subcategory_raw)
        
        # Quick index lookup
        if normalized in self.indexes["subcategory_lookup"]:
            candidates = self.indexes["subcategory_lookup"][normalized]
            filtered = [
                self.registry["experience_drivers"][ed]["canonical_subcategory"]
                for ed in candidates
                if (self.registry["experience_drivers"][ed]["theme"] == theme and
                    self.registry["experience_drivers"][ed]["canonical_category"] == category)
            ]
            if filtered:
                return filtered[0], 100
        
        # Get existing subcategories for this theme+category
        existing_subcats = list(set([
            ed["canonical_subcategory"]
            for ed in self.registry["experience_drivers"].values()
            if (ed["theme"] == theme and 
                ed["canonical_category"] == category and 
                ed.get("canonical_subcategory"))
        ]))
        
        if not existing_subcats:
            return subcategory_raw, 100
        
        # Fuzzy matching
        match = process.extractOne(
            subcategory_raw, 
            existing_subcats, 
            scorer=fuzz.token_set_ratio
        )
        
        if match and match[1] >= threshold:
            return match[0], match[1]
        
        # ML semantic similarity backup
        similarity_score = self._calculate_semantic_similarity(subcategory_raw, existing_subcats)
        if similarity_score[1] >= self.config["embedding_similarity_threshold"]:
            return similarity_score[0], int(similarity_score[1] * 100)
        
        return subcategory_raw, 100
    
    def _calculate_semantic_similarity(self, text: str, candidates: List[str]) -> Tuple[str, float]:
        """Calculate semantic similarity using TF-IDF embeddings"""
        if not candidates:
            return text, 0.0
        
        try:
            # Create a fresh vectorizer for this comparison
            # (We need to fit on the specific candidate set each time)
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=1000,
                stop_words='english',
                lowercase=True
            )
            
            # Fit and transform on candidates + input text
            all_texts = candidates + [text]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarities between input text (last) and candidates
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            return candidates[best_idx], best_similarity
            
        except Exception as e:
            logging.warning(f"Semantic similarity calculation failed: {e}")
            return text, 0.0
    
    def _queue_proposal(self, proposal: Dict[str, Any]) -> None:
        """Queue proposal for human review"""
        try:
            # assign a stable id and timestamp if missing
            proposal.setdefault("proposal_id", uuid.uuid4().hex[:12])
            proposal.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

            with open(self.proposals_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(proposal, ensure_ascii=False) + "\n")

            # in-memory queue
            self.registry["review_queue"].append(proposal)

        except Exception as e:
            logging.error(f"Failed to queue proposal: {e}")

    
    def _update_ed_registry(self, canonical_ed: str, theme: str, category: str, 
                           subcategory: str, raw_variant: str, needs_review: bool = False) -> str:
        """Update Experience Driver registry with stable IDs"""
        
        with self._lock:  # Thread safety
            eds = self.registry["experience_drivers"]
            
            if canonical_ed not in eds:
                # Create new ED entry with stable ID
                ed_id = f"ED-{uuid.uuid4().hex[:12]}"
                
                eds[canonical_ed] = {
                    "canonical_ed_id": ed_id,
                    "canonical_experience_driver": canonical_ed,
                    "canonical_category": category,
                    "canonical_subcategory": subcategory,
                    "theme": theme,
                    "raw_variants": [raw_variant],
                    "needs_review": needs_review,
                    "frozen": False,
                    "first_seen": datetime.now(timezone.utc).isoformat(),
                    "last_seen": datetime.now(timezone.utc).isoformat(),
                    "frequency": 1,
                    "version": 1
                }
                
                # Update ID mappings
                self.registry["id_mappings"][ed_id] = canonical_ed
                
                # Update theme's ED list
                if theme in self.registry["themes"]:
                    if canonical_ed not in self.registry["themes"][theme]["experience_drivers"]:
                        self.registry["themes"][theme]["experience_drivers"].append(canonical_ed)
                
                # Log creation
                self._append_to_audit({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "ed_created",
                    "canonical_ed": canonical_ed,
                    "canonical_ed_id": ed_id,
                    "theme": theme,
                    "raw_variant": raw_variant
                })
                
            else:
                # Update existing ED
                ed_entry = eds[canonical_ed]
                ed_id = ed_entry["canonical_ed_id"]
                
                if raw_variant not in ed_entry["raw_variants"]:
                    ed_entry["raw_variants"].append(raw_variant)
                    
                    # Limit variants to prevent bloat
                    max_variants = self.config.get("max_variants_per_ed", 50)
                    if len(ed_entry["raw_variants"]) > max_variants:
                        ed_entry["raw_variants"] = ed_entry["raw_variants"][-max_variants:]
                
                ed_entry["last_seen"] = datetime.now(timezone.utc).isoformat()
                ed_entry["frequency"] += 1
                
                if needs_review and not ed_entry.get("needs_review", False):
                    ed_entry["needs_review"] = needs_review
        
        return ed_id
    
    def _create_new_theme(self, raw_theme: str) -> str:
        """Create new theme entry with audit trail"""
        self.registry["themes"][raw_theme] = {
            "raw_variants": [raw_theme],
            "experience_drivers": [],
            "frozen": False,
            "created_date": datetime.now(timezone.utc).isoformat(),
            "version": 1
        }
        
        self._append_to_audit({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "theme_created",
            "theme": raw_theme
        })
        
        return raw_theme
    
    def _append_to_audit(self, entry: Dict[str, Any]) -> None:
        """Append entry to audit trail"""
        if not self.config.get("enable_audit_trail", True):
            return
            
        try:
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.error(f"Failed to write audit trail: {e}")
    
    def batch_process(self, df: pd.DataFrame, domain: str = "general",
                  batch_size: Optional[int] = None) -> pd.DataFrame:
        """🚀 GOD-TIER batch processing with comprehensive monitoring"""

        start_time = datetime.now()
        batch_size = batch_size or self.config["batch_size"]
        total_rows = len(df)
        if total_rows == 0:
            logging.info("ℹ️ Empty DataFrame, nothing to process.")
            return df.copy()

        # Validate input
        required_cols = {'theme', 'experience_driver'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Initialize result columns
        df = df.copy()

        result_columns = [
            "canonical_theme", "canonical_experience_driver", "canonical_ed_id",
            "canonical_category", "canonical_subcategory",
            "theme_match_score", "category_match_score", "subcategory_match_score",
            "needs_review", "action", "dual_gate_passed", "is_blocked", "is_allowed"
        ]
        str_cols   = ["canonical_theme","canonical_experience_driver","canonical_ed_id",
                    "canonical_category","canonical_subcategory","action"]
        bool_cols  = ["needs_review","dual_gate_passed","is_blocked","is_allowed"]
        int_cols   = ["theme_match_score","category_match_score","subcategory_match_score"]

        for c in str_cols:  df[c] = ""
        for c in bool_cols: df[c] = False
        for c in int_cols:  df[c] = 0

        # Ensure dtypes (optional but nice)
        df[bool_cols] = df[bool_cols].astype(bool)
        df[int_cols]  = df[int_cols].astype(int)

        logging.info(f"🚀 Processing {total_rows} rows in batches of {batch_size}")

        processed_count = 0
        error_count = 0

        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)

            # take a copy to avoid chained assignment ambiguity
            batch = df.iloc[start_idx:end_idx].copy()

            batch_errors = self._process_god_tier_batch(batch, domain, start_idx)
            error_count += batch_errors
            processed_count += (end_idx - start_idx)

            # ✅ write back results to main df
            df.loc[batch.index, result_columns] = batch[result_columns].values

            # Progress logging every ~5 batches
            if (start_idx // batch_size) % 5 == 0:
                progress = processed_count / total_rows * 100
                logging.info(f"⚡ Progress: {processed_count}/{total_rows} ({progress:.1f}%) - {error_count} errors")

        # Final statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats["total_processed"] += processed_count
        self.stats["processing_time"] += processing_time

        logging.info(f"✅ Batch processing complete: {processed_count} processed, "
                    f"{error_count} errors in {processing_time:.2f}s")

        return df
  
    def _process_god_tier_batch(self, batch: pd.DataFrame, domain: str, start_idx: int) -> int:
        """Process single batch with God-Tier safety and performance"""
        error_count = 0
        
        for i, row in batch.iterrows():
            try:
                raw_theme = str(row["theme"]).strip() if pd.notna(row["theme"]) else ""
                raw_ed = str(row["experience_driver"]).strip() if pd.notna(row["experience_driver"]) else ""
                
                if not raw_theme or not raw_ed:
                    logging.warning(f"⚠️ Empty theme or ED at row {i}")
                    error_count += 1
                    continue
                
                # Theme canonicalization
                canonical_theme, theme_score = self.fast_theme_match(raw_theme, domain)
                
                # Experience Driver canonicalization with all safety rails
                ed_result = self.god_tier_ed_match(canonical_theme, raw_ed, domain)
                
                # Update DataFrame with comprehensive results
                batch.at[i, "canonical_theme"] = canonical_theme
                batch.at[i, "canonical_experience_driver"] = ed_result["canonical_ed"]
                batch.at[i, "canonical_ed_id"] = ed_result["canonical_ed_id"] or ""
                batch.at[i, "canonical_category"] = ed_result["canonical_category"] or ""
                batch.at[i, "canonical_subcategory"] = ed_result["canonical_subcategory"] or ""
                batch.at[i, "theme_match_score"] = theme_score
                batch.at[i, "category_match_score"] = ed_result["category_score"]
                batch.at[i, "subcategory_match_score"] = ed_result["subcategory_score"]
                batch.at[i, "needs_review"] = ed_result["needs_review"]
                batch.at[i, "action"] = ed_result["action"]
                batch.at[i, "dual_gate_passed"] = ed_result["dual_gate_passed"]
                batch.at[i, "is_blocked"] = ed_result["safety_checks"]["is_blocked"]
                batch.at[i, "is_allowed"] = ed_result["safety_checks"]["is_allowed"]
                
            except Exception as e:
                logging.error(f"💥 Error processing row {i}: {e}")
                error_count += 1
                continue
        
        return error_count
    
    def save_indexes(self, indexes: Dict[str, Any]) -> None:
        """Save indexes with error handling and backup"""
        index_path = self.base_dir / "god_tier_indexes.pkl"
        backup_path = self.base_dir / "god_tier_indexes_backup.pkl"
        
        try:
            # Create backup of existing indexes
            if index_path.exists():
                import shutil
                shutil.copy2(index_path, backup_path)
            
            # Save new indexes
            with open(index_path, 'wb') as f:
                pickle.dump(indexes, f)
                
            logging.info("💾 Indexes saved successfully")
            
        except Exception as e:
            logging.error(f"💥 Failed to save indexes: {e}")
            # Restore backup if save failed
            if backup_path.exists():
                import shutil
                shutil.copy2(backup_path, index_path)
                logging.info("🔄 Restored indexes from backup")
    
    def save_registry(self) -> None:
        """Save registry with atomic writes and backup"""
        registry_path = self.base_dir / "canonical_registry.json"
        temp_path = self.base_dir / "canonical_registry_temp.json"
        backup_path = self.base_dir / "canonical_registry_backup.json"
        
        try:
            # Update metadata
            self.registry["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Create backup
            if registry_path.exists():
                import shutil
                shutil.copy2(registry_path, backup_path)
            
            # Atomic write: write to temp file first
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False, default=str)
            
            # Move temp file to final location
            temp_path.replace(registry_path)
            
            logging.info("💾 Registry saved successfully")
            
        except Exception as e:
            logging.error(f"💥 Failed to save registry: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
    
    def export_god_tier_report(self, output_path: str = "god_tier_canonicalization_report.csv") -> None:
        """Export comprehensive canonicalization report with analytics"""
        
        records = []
        for ed, data in self.registry["experience_drivers"].items():
            for variant in data["raw_variants"]:
                records.append({
                    "canonical_ed_id": data.get("canonical_ed_id"),
                    "raw_variant": variant,
                    "canonical_experience_driver": ed,
                    "theme": data.get("theme"),
                    "canonical_category": data.get("canonical_category"),
                    "canonical_subcategory": data.get("canonical_subcategory"),
                    "needs_review": data.get("needs_review", False),
                    "frozen": data.get("frozen", False),
                    "first_seen": data.get("first_seen"),
                    "last_seen": data.get("last_seen"),
                    "frequency": data.get("frequency", 0),
                    "variant_count": len(data["raw_variants"]),
                    "version": data.get("version", 1)
                })
        
        df = pd.DataFrame(records)
        
        # Enhanced analytics
        summary_stats = {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "total_canonical_eds": len(self.registry["experience_drivers"]),
            "total_variants": len(records),
            "compression_ratio": len(records) / len(self.registry["experience_drivers"]) if self.registry["experience_drivers"] else 0,
            "themes_count": len(self.registry["themes"]),
            "frozen_eds": len([ed for ed in self.registry["experience_drivers"].values() if ed.get("frozen", False)]),
            "pending_reviews": len([ed for ed in self.registry["experience_drivers"].values() if ed.get("needs_review", False)]),
            "processing_stats": self.stats.copy(),
            "config_summary": {
                "dual_gate_enforcement": self.config.get("dual_gate_enforcement", False),
                "blocklist_rules": len(self.config.get("blocklist_pairs", [])),
                "allowlist_rules": len(self.config.get("allowlist_pairs", []))
            }
        }
        
        # Save report
        df.to_csv(output_path, index=False)
        
        # Save detailed analytics
        analytics_path = output_path.replace('.csv', '_analytics.json')
        with open(analytics_path, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        # Generate proposals report if proposals exist
        # Generate proposals report if proposals exist
        if self.proposals_path.exists():
            proposals_df = self.export_proposals_report()
            proposals_path = output_path.replace('.csv', '_proposals.csv')
            # ✅ keep the proposal_id by resetting index to a column
            proposals_df.reset_index().to_csv(proposals_path, index=False)
            logging.info(f"📋 Proposals report saved: {proposals_path}")

        logging.info(f"📊 God-Tier report saved: {output_path}")
        logging.info(f"📈 Analytics saved: {analytics_path}")
    
    def export_proposals_report(self) -> pd.DataFrame:
        """Export human review proposals as DataFrame"""
        proposals = []
        if self.proposals_path.exists():
            try:
                with open(self.proposals_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            proposals.append(json.loads(line))
            except Exception as e:
                logging.error(f"Failed to read proposals: {e}")

        df = pd.DataFrame(proposals)
        if not df.empty and "proposal_id" in df.columns:
            df.set_index("proposal_id", inplace=True)
        return df

    def approve_proposal(self, proposal_id: str, action: str = "approve") -> bool:
        """Approve or reject a queued proposal"""
        try:
            proposals_df = self.export_proposals_report()
            if proposal_id in proposals_df.index:
                proposal = proposals_df.loc[proposal_id].to_dict()

                if action == "approve":
                    self._execute_proposal_merge(proposal)
                    logging.info(f"✅ Proposal {proposal_id} approved and merged")
                    return True

                if action == "reject":
                    self._mark_proposal_rejected(proposal)
                    logging.info(f"❌ Proposal {proposal_id} rejected")
                    return True

            logging.warning(f"Proposal id not found: {proposal_id}")
            return False

        except Exception as e:
            logging.error(f"Failed to process proposal {proposal_id}: {e}")
            return False
    
    def _execute_proposal_merge(self, proposal: Dict[str, Any]) -> None:
        """Execute an approved proposal merge"""
        # Update the registry to use the suggested canonical form
        theme = proposal["theme"]
        raw_ed = proposal["raw_ed"]
        suggested_canonical = proposal["suggested_canonical"]
        
        # Update existing entry or create new canonical mapping
        self._update_ed_registry(
            suggested_canonical, 
            theme, 
            proposal["suggested_category"], 
            proposal["suggested_subcategory"], 
            raw_ed, 
            needs_review=False
        )
        
        # Log the approval
        self._append_to_audit({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "proposal_approved",
            "raw_ed": raw_ed,
            "canonical_ed": suggested_canonical,
            "theme": theme
        })
    
    def _mark_proposal_rejected(self, proposal: Dict[str, Any]) -> None:
        """Mark a proposal as rejected"""
        self._append_to_audit({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "proposal_rejected",
            "raw_ed": proposal["raw_ed"],
            "suggested_canonical": proposal["suggested_canonical"],
            "theme": proposal["theme"]
        })
    
    def get_god_tier_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance and quality statistics"""
        total_variants = sum(len(ed["raw_variants"]) for ed in self.registry["experience_drivers"].values())
        
        quality_metrics = {
            "compression_efficiency": total_variants / len(self.registry["experience_drivers"]) if self.registry["experience_drivers"] else 0,
            "review_rate": self.stats["proposals_queued"] / self.stats["total_processed"] if self.stats["total_processed"] else 0,
            "approval_rate": self.stats["merges_approved"] / self.stats["total_processed"] if self.stats["total_processed"] else 0,
            "safety_hit_rate": self.stats["blocklist_hits"] / self.stats["total_processed"] if self.stats["total_processed"] else 0,
            "error_rate": self.stats["errors"] / self.stats["total_processed"] if self.stats["total_processed"] else 0
        }
        
        return {
            "registry_stats": {
                "total_themes": len(self.registry["themes"]),
                "total_experience_drivers": len(self.registry["experience_drivers"]),
                "total_variants": total_variants,
                "frozen_eds": len([ed for ed in self.registry["experience_drivers"].values() if ed.get("frozen", False)]),
                "pending_reviews": len([ed for ed in self.registry["experience_drivers"].values() if ed.get("needs_review", False)])
            },
            "processing_stats": self.stats.copy(),
            "quality_metrics": quality_metrics,
            "index_stats": {
                "entry_count": self.indexes.get("entry_count", 0),
                "last_rebuild": self.indexes.get("last_rebuild", "Never"),
                "build_version": self.indexes.get("build_version", "Unknown")
            },
            "safety_config": {
                "dual_gate_enforcement": self.config.get("dual_gate_enforcement", False),
                "blocklist_rules": len(self.config.get("blocklist_pairs", [])),
                "allowlist_rules": len(self.config.get("allowlist_pairs", [])),
                "embedding_threshold": self.config.get("embedding_similarity_threshold", 0.85)
            }
        }
    
    def setup_logging(self) -> None:
        """Setup enhanced logging with performance monitoring"""
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        
        # Create logs directory
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup handlers
        handlers = [
            logging.FileHandler(log_dir / "god_tier_canonicalizer.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
        
        for handler in handlers:
            handler.setFormatter(detailed_formatter)
            handler.setLevel(log_level)
        
        # Configure logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        logger.handlers.clear()
        for handler in handlers:
            logger.addHandler(handler)
        
        logging.info("🚀 God-Tier Canonicalizer logging initialized")


def main():
    """🚀 Main execution with God-Tier error handling and monitoring"""
    
    # Configuration
    input_file = "data/decipher_retail_grocery_analytics_flattened.csv"
    output_dir = "outputs"
    base_name = os.path.basename(input_file).replace(".csv", "")
    output_file = os.path.join(output_dir, f"{base_name}_god_tier_canonicalized.csv")
    report_file = os.path.join(output_dir, f"{base_name}_god_tier_report.csv")
    stats_file = os.path.join(output_dir, f"{base_name}_god_tier_stats.json")
    domain = "retail"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        start_time = datetime.now()
        
        # Initialize God-Tier engine
        print("🚀 Initializing God-Tier Canonicalizer...")
        engine = GodTierCanonicalizer()
        
        # Load and validate data
        print(f"📂 Loading data from: {input_file}")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        df = pd.read_csv(input_file)
        print(f"📊 Loaded {len(df)} rows")
        
        # Process with God-Tier safety
        print(f"⚡ Processing with domain: {domain}")
        processed_df = engine.batch_process(df, domain=domain)
        
        # Save results
        print("💾 Saving processed data...")
        processed_df.to_csv(output_file, index=False)
        engine.save_registry()
        
        # Generate comprehensive reports
        print("📊 Generating God-Tier reports...")
        engine.export_god_tier_report(report_file)
        
        # Calculate final statistics
        original_eds = df["experience_driver"].nunique()
        canonical_eds = processed_df["canonical_experience_driver"].nunique()
        compression_ratio = original_eds / canonical_eds if canonical_eds > 0 else 0
        
        needs_review = processed_df["needs_review"].sum()
        dual_gate_failures = (~processed_df["dual_gate_passed"]).sum()
        blocked_merges = processed_df["is_blocked"].sum()
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Get comprehensive stats
        god_tier_stats = engine.get_god_tier_stats()
        
        # Enhanced statistics report
        final_stats = {
            "processing_summary": {
                "original_unique_eds": int(original_eds),
                "canonical_unique_eds": int(canonical_eds),
                "compression_ratio": round(compression_ratio, 2),
                "total_records": len(df),
                "processing_time_seconds": round(processing_time, 2),
                "domain": domain
            },
            "safety_summary": {
                "records_needing_review": int(needs_review),
                "dual_gate_failures": int(dual_gate_failures),
                "blocked_merges": int(blocked_merges),
                "proposals_queued": god_tier_stats["processing_stats"]["proposals_queued"],
                "merges_approved": god_tier_stats["processing_stats"]["merges_approved"]
            },
            "god_tier_stats": god_tier_stats,
            "output_files": {
                "processed_data": output_file,
                "canonical_report": report_file,
                "stats_file": stats_file
            }
        }
        
        # Save comprehensive stats
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)
        
        # Print success summary
        print("\n" + "="*80)
        print("🏆 GOD-TIER CANONICALIZATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"📊 Original Experience Drivers:     {original_eds:,}")
        print(f"🎯 Canonical Experience Drivers:   {canonical_eds:,}")
        print(f"⚡ Compression Ratio:              {compression_ratio:.2f}x")
        print(f"⏱️  Processing Time:               {processing_time:.2f}s")
        print(f"🔍 Records Needing Review:         {needs_review:,}")
        print(f"🛡️ Dual-Gate Failures:            {dual_gate_failures:,}")
        print(f"🚫 Blocked Merges:                 {blocked_merges:,}")
        print(f"📋 Proposals Queued:               {god_tier_stats['processing_stats']['proposals_queued']:,}")
        print(f"✅ Merges Approved:                {god_tier_stats['processing_stats']['merges_approved']:,}")
        print(f"📁 Output Directory:               {output_dir}")
        print("="*80)
        
        if needs_review > 0:
            proposals_file = report_file.replace('.csv', '_proposals.csv')
            print(f"💡 Review proposals at: {proposals_file}")
        
        print("🚀 God-Tier Canonicalization Engine: MISSION ACCOMPLISHED")
        
    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        logging.error(f"Main processing failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)