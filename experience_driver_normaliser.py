# import sys
# # Make stdout/stderr UTF-8 so logging can print → and ✓
# if hasattr(sys.stdout, "reconfigure"):
#     sys.stdout.reconfigure(encoding="utf-8")
#     sys.stderr.reconfigure(encoding="utf-8")


# import pandas as pd
# import os
# import json
# import numpy as np
# from rapidfuzz import fuzz, process
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import pickle
# from collections import defaultdict
# import logging
# import re

# class HighPerformanceCanonicalizer:
#     def __init__(self, config_path="canonicalization_config.json"):
#         self.config = self.load_config(config_path)
#         self.registry = self.load_registry()
#         self.indexes = self.load_or_build_indexes()
#         self.vectorizer = TfidfVectorizer(
#             ngram_range=(1, 3),
#             max_features=10000,
#             stop_words='english'
#         )
#         self.setup_logging()
    
#     def load_config(self, config_path):
#         """Load domain-specific threshold configuration"""
#         default_config = {
#             "domain_thresholds": {
#                 "financial_services": {"theme": 95, "category": 95, "subcategory": 95},
#                 "healthcare": {"theme": 95, "category": 94, "subcategory": 93},
#                 "legal": {"theme": 96, "category": 95, "subcategory": 94},
#                 "retail": {"theme": 95, "category": 85, "subcategory": 80},
#                 "ecommerce": {"theme": 88, "category": 89, "subcategory": 90},
#                 "support": {"theme": 82, "category": 85, "subcategory": 87},
#                 "social_media": {"theme": 80, "category": 82, "subcategory": 85},
#                 "general": {"theme": 90, "category": 90, "subcategory": 90}
#             },
#             "batch_size": 1000,
#             "cache_size": 10000,
#             "rebuild_index_threshold": 100,  # Rebuild after N new entries
#             "semantic_similarity_threshold": 0.85,  # For TF-IDF backup
#             "min_group_size": 2  # Minimum items to form a canonical group
#         }
        
#         if os.path.exists(config_path):
#             with open(config_path, 'r') as f:
#                 user_config = json.load(f)
#                 default_config.update(user_config)
        
#         return default_config
    
#     def load_registry(self):
#         """Load existing registry or create new one"""
#         registry_path = "canonical_registry.json"
#         if os.path.exists(registry_path):
#             with open(registry_path, "r", encoding="utf-8") as f:
#                 return json.load(f)
#         return {"themes": {}, "experience_drivers": {}}
    
#     def load_or_build_indexes(self):
#         """Load pre-built indexes or create new ones"""
#         index_path = "performance_indexes.pkl"
#         if os.path.exists(index_path):
#             try:
#                 with open(index_path, 'rb') as f:
#                     return pickle.load(f)
#             except:
#                 logging.warning("Failed to load indexes, rebuilding...")
        
#         return self.build_indexes()
    
#     def build_indexes(self):
#         """Build optimized lookup indexes"""
#         logging.info("Building performance indexes...")
        
#         indexes = {
#             "theme_lookup": {},
#             "category_lookup": defaultdict(list),
#             "subcategory_lookup": defaultdict(list),
#             "ed_vectors": None,
#             "ed_list": [],
#             "last_rebuild": datetime.now().isoformat(),
#             "entry_count": 0
#         }
        
#         # Build theme index
#         for theme in self.registry["themes"].keys():
#             normalized = self._normalize_text(theme)
#             indexes["theme_lookup"][normalized] = theme
        
#         # Build Experience Driver indexes
#         ed_texts = []
#         for ed_key, ed_data in self.registry["experience_drivers"].items():
#             indexes["ed_list"].append(ed_key)
#             ed_texts.append(ed_key)
            
#             # Category lookup
#             category = ed_data.get("canonical_category", "")
#             if category:
#                 normalized_cat = self._normalize_text(category)
#                 indexes["category_lookup"][normalized_cat].append(ed_key)
            
#             # Subcategory lookup
#             subcategory = ed_data.get("canonical_subcategory", "")
#             if subcategory:
#                 normalized_subcat = self._normalize_text(subcategory)
#                 indexes["subcategory_lookup"][normalized_subcat].append(ed_key)
        
#         # Build vector representations for semantic similarity
#         if ed_texts:
#             try:
#                 tfidf_matrix = self.vectorizer.fit_transform(ed_texts)
#                 indexes["ed_vectors"] = tfidf_matrix
#                 indexes["entry_count"] = len(ed_texts)
#             except Exception as e:
#                 logging.warning(f"Failed to build vector index: {e}")
        
#         self.save_indexes(indexes)
#         logging.info(f"Built indexes for {len(ed_texts)} Experience Drivers")
#         return indexes
    
#     def _normalize_text(self, text):
#         """Normalize text for consistent lookups - IMPROVED"""
#         if not text:
#             return ""
        
#         # Handle different arrow formats first
#         text = re.sub(r"[→\-›>\u2192\u2190\u2194â†']", "→", text)

#         # Basic cleanup
#         text = text.lower().strip()
#         text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
#         text = re.sub(r'[^\w\s→]', '', text)  # Remove special chars except arrows
        
#         return text
    
#     def _clean_experience_driver(self, raw_ed):
#         """Clean and validate experience driver format"""
#         if not raw_ed or pd.isna(raw_ed):
#             return None
            
#         # Normalize arrow format
#         clean_ed = re.sub(r"[→\-›>\u2192\u2190\u2194â†']", "→", str(raw_ed).strip())
        
#         if '→' not in clean_ed:
#             logging.warning(f"Invalid ED format (no arrow): {raw_ed}")
#             return None
            
#         parts = clean_ed.split('→')
#         if len(parts) != 2:
#             logging.warning(f"Invalid ED format (multiple arrows): {raw_ed}")
#             return None
            
#         category, subcategory = [p.strip() for p in parts]
#         if not category or not subcategory:
#             logging.warning(f"Invalid ED format (empty parts): {raw_ed}")
#             return None
            
#         return f"{category} → {subcategory}"
    
#     def get_domain_thresholds(self, domain="general"):
#         """Get thresholds for specific domain"""
#         return self.config["domain_thresholds"].get(
#             domain, 
#             self.config["domain_thresholds"]["general"]
#         )
    
#     def fast_theme_match(self, raw_theme, domain="general"):
#         """Optimized theme matching using pre-built indexes"""
#         if not raw_theme or pd.isna(raw_theme):
#             return "Unknown", 0
            
#         thresholds = self.get_domain_thresholds(domain)
#         normalized = self._normalize_text(raw_theme)
        
#         # Exact match first (O(1))
#         if normalized in self.indexes["theme_lookup"]:
#             return self.indexes["theme_lookup"][normalized], 100
        
#         # Fuzzy match only if needed (O(n) but smaller n)
#         theme_choices = list(self.registry["themes"].keys())
#         if not theme_choices:
#             return self._create_new_theme(raw_theme), 100
        
#         match = process.extractOne(
#             raw_theme, 
#             theme_choices, 
#             scorer=fuzz.token_set_ratio
#         )
        
#         if match and match[1] >= thresholds["theme"]:
#             return match[0], match[1]
        
#         return self._create_new_theme(raw_theme), 100
    
#     def fast_ed_match(self, theme, raw_ed, domain="general"):
#         """High-performance Experience Driver matching - IMPROVED"""
#         # Clean and validate first
#         clean_ed = self._clean_experience_driver(raw_ed)
#         if not clean_ed:
#             return raw_ed, None, None, 0, 0
            
#         thresholds = self.get_domain_thresholds(domain)
#         category_raw, subcategory_raw = clean_ed.split("→")
#         category_raw = category_raw.strip()
#         subcategory_raw = subcategory_raw.strip()
        
#         # Quick category lookup using indexes
#         cat_match = self._fast_category_match(theme, category_raw, thresholds["category"])
#         subcat_match = self._fast_subcategory_match(theme, cat_match, subcategory_raw, thresholds["subcategory"])
        
#         canonical_ed = f"{cat_match} → {subcat_match}"
        
#         # Calculate confidence scores
#         cat_score = self._calculate_match_score(category_raw, cat_match)
#         subcat_score = self._calculate_match_score(subcategory_raw, subcat_match)
        
#         # Update or create registry entry
#         self._update_ed_registry(canonical_ed, theme, cat_match, subcat_match, clean_ed)
        
#         return canonical_ed, cat_match, subcat_match, cat_score, subcat_score
    
#     def _calculate_match_score(self, original, canonical):
#         """Calculate match confidence score"""
#         if original.lower() == canonical.lower():
#             return 100
#         return fuzz.token_set_ratio(original, canonical)
    
#     def _fast_category_match(self, theme, category_raw, threshold):
#         """Fast category matching using indexes - IMPROVED"""
#         normalized = self._normalize_text(category_raw)
        
#         # Quick lookup in category index
#         if normalized in self.indexes["category_lookup"]:
#             candidates = self.indexes["category_lookup"][normalized]
#             theme_filtered = [
#                 self.registry["experience_drivers"][ed]["canonical_category"]
#                 for ed in candidates
#                 if self.registry["experience_drivers"][ed]["theme"] == theme
#             ]
#             if theme_filtered:
#                 return theme_filtered[0]  # Return first match
        
#         # Semantic similarity fallback using TF-IDF
#         existing_cats = list(set([
#             ed["canonical_category"]
#             for ed in self.registry["experience_drivers"].values()
#             if ed["theme"] == theme
#         ]))
        
#         if existing_cats:
#             # Try fuzzy match first (faster)
#             match = process.extractOne(category_raw, existing_cats, scorer=fuzz.token_set_ratio)
#             if match and match[1] >= threshold:
#                 return match[0]
            
#             # TF-IDF semantic similarity as backup
#             try:
#                 vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
#                 all_texts = existing_cats + [category_raw]
#                 tfidf_matrix = vectorizer.fit_transform(all_texts)
#                 similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
                
#                 best_idx = np.argmax(similarities)
#                 if similarities[best_idx] >= self.config["semantic_similarity_threshold"]:
#                     return existing_cats[best_idx]
#             except:
#                 pass  # Fall back to original if TF-IDF fails
        
#         return category_raw
    
#     def _fast_subcategory_match(self, theme, category, subcategory_raw, threshold):
#         """Fast subcategory matching using indexes - IMPROVED"""
#         normalized = self._normalize_text(subcategory_raw)
        
#         # Quick lookup
#         if normalized in self.indexes["subcategory_lookup"]:
#             candidates = self.indexes["subcategory_lookup"][normalized]
#             filtered = [
#                 self.registry["experience_drivers"][ed]["canonical_subcategory"]
#                 for ed in candidates
#                 if (self.registry["experience_drivers"][ed]["theme"] == theme and
#                     self.registry["experience_drivers"][ed]["canonical_category"] == category)
#             ]
#             if filtered:
#                 return filtered[0]
        
#         # Semantic similarity for subcategories
#         existing_subcats = list(set([
#             ed["canonical_subcategory"]
#             for ed in self.registry["experience_drivers"].values()
#             if (ed["theme"] == theme and ed["canonical_category"] == category)
#         ]))
        
#         if existing_subcats:
#             # Fuzzy match first
#             match = process.extractOne(subcategory_raw, existing_subcats, scorer=fuzz.token_set_ratio)
#             if match and match[1] >= threshold:
#                 return match[0]
            
#             # TF-IDF backup
#             try:
#                 vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
#                 all_texts = existing_subcats + [subcategory_raw]
#                 tfidf_matrix = vectorizer.fit_transform(all_texts)
#                 similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
                
#                 best_idx = np.argmax(similarities)
#                 if similarities[best_idx] >= self.config["semantic_similarity_threshold"]:
#                     return existing_subcats[best_idx]
#             except:
#                 pass
        
#         return subcategory_raw
    
#     def batch_process(self, df, domain="general", batch_size=None):
#         """Process large datasets in optimized batches - IMPROVED ERROR HANDLING"""
#         batch_size = batch_size or self.config["batch_size"]
#         total_rows = len(df)
        
#         # Validate required columns
#         required_cols = {'theme', 'experience_driver'}
#         missing_cols = required_cols - set(df.columns)
#         if missing_cols:
#             raise ValueError(f"Missing required columns: {missing_cols}")
        
#         # Pre-allocate result columns
#         df = df.copy()
#         df["canonical_theme"] = ""
#         df["canonical_category"] = ""
#         df["canonical_subcategory"] = ""
#         df["canonical_experience_driver"] = ""
#         df["theme_match_score"] = 0
#         df["category_match_score"] = 0
#         df["subcategory_match_score"] = 0
#         df["needs_review"] = False
        
#         logging.info(f"Processing {total_rows} rows in batches of {batch_size}")
        
#         processed_count = 0
#         error_count = 0
        
#         for start_idx in range(0, total_rows, batch_size):
#             end_idx = min(start_idx + batch_size, total_rows)
#             batch = df.iloc[start_idx:end_idx]
            
#             batch_errors = self._process_batch(batch, domain, start_idx)
#             error_count += batch_errors
#             processed_count += (end_idx - start_idx)
            
#             if start_idx % (batch_size * 10) == 0:  # Log every 10 batches
#                 logging.info(f"Processed {processed_count}/{total_rows} rows ({processed_count/total_rows*100:.1f}%) - {error_count} errors")
        
#         logging.info(f"Batch processing complete: {processed_count} processed, {error_count} errors")
#         return df
    
#     def _process_batch(self, batch, domain, start_idx):
#         """Process a single batch efficiently - IMPROVED"""
#         error_count = 0
        
#         for i, row in batch.iterrows():
#             try:
#                 raw_theme = str(row["theme"]).strip() if pd.notna(row["theme"]) else ""
#                 raw_ed = str(row["experience_driver"]).strip() if pd.notna(row["experience_driver"]) else ""
                
#                 if not raw_theme or not raw_ed:
#                     logging.warning(f"Empty theme or ED at row {i}")
#                     error_count += 1
#                     continue
                
#                 canonical_theme, theme_score = self.fast_theme_match(raw_theme, domain)
#                 canonical_ed, canonical_cat, canonical_subcat, cat_score, subcat_score = self.fast_ed_match(canonical_theme, raw_ed, domain)
                
#                 # Determine if needs review
#                 min_threshold = self.get_domain_thresholds(domain)
#                 needs_review = (
#                     theme_score < min_threshold["theme"] or
#                     cat_score < min_threshold["category"] or
#                     subcat_score < min_threshold["subcategory"]
#                 )
                
#                 # Update DataFrame
#                 batch.at[i, "canonical_theme"] = canonical_theme
#                 batch.at[i, "canonical_category"] = canonical_cat or ""
#                 batch.at[i, "canonical_subcategory"] = canonical_subcat or ""
#                 batch.at[i, "canonical_experience_driver"] = canonical_ed
#                 batch.at[i, "theme_match_score"] = theme_score
#                 batch.at[i, "category_match_score"] = cat_score
#                 batch.at[i, "subcategory_match_score"] = subcat_score
#                 batch.at[i, "needs_review"] = needs_review
                
#             except Exception as e:
#                 logging.error(f"Error processing row {i}: {e}")
#                 error_count += 1
#                 continue
        
#         return error_count
    
#     def _create_new_theme(self, raw_theme):
#         """Create new theme entry"""
#         self.registry["themes"][raw_theme] = {
#             "raw_variants": [raw_theme],
#             "experience_drivers": [],
#             "frozen": False,
#             "created_date": datetime.today().strftime("%Y-%m-%d")
#         }
#         return raw_theme
    
#     def _update_ed_registry(self, canonical_ed, theme, cat_match, subcat_match, raw_variant):
#         """Update Experience Driver registry efficiently - IMPROVED"""
#         if canonical_ed not in self.registry["experience_drivers"]:
#             self.registry["experience_drivers"][canonical_ed] = {
#                 "canonical_experience_driver": canonical_ed,
#                 "canonical_category": cat_match,
#                 "canonical_subcategory": subcat_match,
#                 "theme": theme,
#                 "raw_variants": [raw_variant],
#                 "frozen": False,
#                 "first_seen": datetime.today().strftime("%Y-%m-%d"),
#                 "last_seen": datetime.today().strftime("%Y-%m-%d"),
#                 "frequency": 1
#             }
            
#             # Add to theme's ED list
#             if theme in self.registry["themes"]:
#                 if canonical_ed not in self.registry["themes"][theme]["experience_drivers"]:
#                     self.registry["themes"][theme]["experience_drivers"].append(canonical_ed)
#         else:
#             ed_entry = self.registry["experience_drivers"][canonical_ed]
#             if raw_variant not in ed_entry["raw_variants"]:
#                 ed_entry["raw_variants"].append(raw_variant)
#             ed_entry["last_seen"] = datetime.today().strftime("%Y-%m-%d")
#             ed_entry["frequency"] += 1
    
#     def save_indexes(self, indexes):
#         """Save indexes to disk for future use"""
#         try:
#             with open("performance_indexes.pkl", 'wb') as f:
#                 pickle.dump(indexes, f)
#         except Exception as e:
#             logging.error(f"Failed to save indexes: {e}")
    
#     def save_registry(self):
#         """Save updated registry"""
#         try:
#             with open("canonical_registry.json", "w", encoding="utf-8") as f:
#                 json.dump(self.registry, f, indent=2, ensure_ascii=False)
#         except Exception as e:
#             logging.error(f"Failed to save registry: {e}")
    
#     def setup_logging(self):
#         """Setup performance logging"""
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.FileHandler('canonicalization.log'),
#                 logging.StreamHandler()
#             ]
#         )
    
#     def export_canonical_report(self, output_path="canonicalization_report.csv"):
#         """Export detailed canonicalization report - IMPROVED"""
#         records = []
#         for ed, data in self.registry["experience_drivers"].items():
#             for variant in data["raw_variants"]:
#                 records.append({
#                     "raw_variant": variant,
#                     "canonical_experience_driver": ed,
#                     "theme": data.get("theme"),
#                     "canonical_category": data.get("canonical_category"),
#                     "canonical_subcategory": data.get("canonical_subcategory"),
#                     "needs_review": data.get("needs_review", False),
#                     "first_seen": data.get("first_seen"),
#                     "last_seen": data.get("last_seen"),
#                     "frozen": data.get("frozen"),
#                     "frequency": data.get("frequency", 0),
#                     "variant_count": len(data["raw_variants"])
#                 })
        
#         df = pd.DataFrame(records)
        
#         # Add summary stats
#         summary_stats = {
#             "total_canonical_eds": len(self.registry["experience_drivers"]),
#             "total_variants": len(records),
#             "compression_ratio": len(records) / len(self.registry["experience_drivers"]) if self.registry["experience_drivers"] else 0,
#             "themes_count": len(self.registry["themes"])
#         }
        
#         df.to_csv(output_path, index=False)
        
#         # Save summary separately
#         summary_path = output_path.replace('.csv', '_summary.json')
#         with open(summary_path, 'w') as f:
#             json.dump(summary_stats, f, indent=2)
            
#         logging.info(f"[✓] Canonicalization report saved: {output_path}")
#         logging.info(f"[✓] Summary stats saved: {summary_path}")
    
#     def get_performance_stats(self):
#         """Get performance statistics - ENHANCED"""
#         total_variants = sum(len(ed["raw_variants"]) for ed in self.registry["experience_drivers"].values())
        
#         return {
#             "total_themes": len(self.registry["themes"]),
#             "total_experience_drivers": len(self.registry["experience_drivers"]),
#             "total_variants": total_variants,
#             "compression_ratio": total_variants / len(self.registry["experience_drivers"]) if self.registry["experience_drivers"] else 0,
#             "index_entry_count": self.indexes.get("entry_count", 0),
#             "last_index_rebuild": self.indexes.get("last_rebuild", "Never"),
#             "frozen_eds": sum(1 for ed in self.registry["experience_drivers"].values() if ed.get("frozen", False))
#         }

# # Main execution remains the same but with better error handling
# def main():
#     """Main execution with improved error handling"""
#     # === INPUT & OUTPUT CONFIGURATION ===
#     input_file = "data/decipher_retail_grocery_analytics_flattened.csv"
#     output_dir = "outputs"
#     base_name = os.path.basename(input_file).replace(".csv", "")
#     output_file = os.path.join(output_dir, f"{base_name}_canonicalized.csv")
#     report_file = os.path.join(output_dir, f"{base_name}_canonical_map.csv")
#     stats_file = os.path.join(output_dir, f"{base_name}_explosion_stats.txt")
#     domain = "retail"

#     os.makedirs(output_dir, exist_ok=True)

#     try:
#         # === INIT & RUN ===
#         engine = HighPerformanceCanonicalizer()
#         print(f"[INFO] Loading data from: {input_file}")
        
#         if not os.path.exists(input_file):
#             print(f"[ERROR] Input file not found: {input_file}")
#             return
            
#         df = pd.read_csv(input_file)
#         print(f"[INFO] Loaded {len(df)} rows")
        
#         processed_df = engine.batch_process(df, domain=domain)

#         # === SAVE OUTPUT ===
#         processed_df.to_csv(output_file, index=False)
#         engine.save_registry()

#         # === GENERATE & SAVE REPORT ===
#         engine.export_canonical_report(report_file)

#         # === ENHANCED STATS ===
#         unique_raw = df["experience_driver"].nunique()
#         unique_canonical = processed_df["canonical_experience_driver"].nunique()
#         compression_ratio = unique_raw / unique_canonical if unique_canonical > 0 else 0

#         needs_review_count = processed_df[processed_df["needs_review"] == True].shape[0]
        
#         stats = engine.get_performance_stats()

#         with open(stats_file, 'w', encoding="utf-8") as f:
#             f.write("EXPERIENCE DRIVER CANONICALIZATION STATS\n")
#             f.write("=" * 50 + "\n\n")
#             f.write(f"Original unique experience drivers: {unique_raw}\n")
#             f.write(f"Canonical unique experience drivers: {unique_canonical}\n")
#             f.write(f"Compression ratio: {compression_ratio:.2f}x\n")
#             f.write(f"Total records processed: {len(df)}\n")
#             f.write(f"Mappings requiring review: {needs_review_count}\n")
#             f.write(f"Domain: {domain}\n")
#             f.write(f"Thresholds Used: {engine.get_domain_thresholds(domain)}\n")
#             f.write(f"Performance Stats: {stats}\n")

#         print("\n[✓] Canonicalization completed successfully")
#         print(f"Original Drivers:  {unique_raw}")
#         print(f"Canonical Drivers: {unique_canonical}")
#         print(f"Compression Ratio: {compression_ratio:.2f}x")
#         print(f"Needs Review:      {needs_review_count}")
#         print(f"Registry Stats:    {stats}")
#         print(f"Output directory:  {output_dir}\n")

#     except Exception as e:
#         print(f"[ERROR] Processing failed: {e}")
#         logging.error(f"Main processing failed: {e}")

# if __name__ == "__main__":
#     main()


import sys
# Ensure stdout/stderr can print UTF-8 symbols like → and ✓
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import re
import json
import pickle
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HighPerformanceCanonicalizer:
    """
    Experience Driver normaliser:
      - Respects the immutable triad: Theme → (Category → Subcategory) → Entity Name (entity not used here)
      - Strict Theme-scoped matching (no cross-theme merges)
      - Hybrid similarity: Fuzzy (token_set_ratio) first, then scoped TF-IDF backup
      - Conservative normalization (unify arrow variants to '→', collapse whitespace, lowercasing for scoring only)
      - Deterministic writeback and robust logging
      - Optional auto index rebuild when registry grows a lot
    """

    def __init__(self, config_path="canonicalization_config.json"):
        self.config = self._load_config(config_path)
        self.registry = self._load_registry()

        # Build TF-IDF vectorizer BEFORE building any indexes that may rely on it
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words="english",
            lowercase=True
        )

        self.indexes = self._load_or_build_indexes()
        self._setup_logging()

    # -----------------------------
    # Setup & Persistence
    # -----------------------------
    def _load_config(self, config_path):
        default = {
            "domain_thresholds": {
                # theme/category/subcategory review gates (RapidFuzz score 0–100)
                "financial_services": {"theme": 95, "category": 95, "subcategory": 95},
                "healthcare": {"theme": 95, "category": 94, "subcategory": 93},
                "legal": {"theme": 96, "category": 95, "subcategory": 94},
                "retail": {"theme": 95, "category": 85, "subcategory": 80},
                "ecommerce": {"theme": 88, "category": 89, "subcategory": 90},
                "support": {"theme": 82, "category": 85, "subcategory": 87},
                "social_media": {"theme": 80, "category": 82, "subcategory": 85},
                "general": {"theme": 90, "category": 90, "subcategory": 90}
            },
            "batch_size": 1000,
            "semantic_similarity_threshold": 0.86,   # TF-IDF cosine backup threshold (0–1)
            "rebuild_index_threshold": 200,          # Rebuild when N new EDs added
            "log_every_n_batches": 10                # Progress logging cadence
        }
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                try:
                    file_cfg = json.load(f)
                    default.update(file_cfg)
                except Exception as e:
                    print(f"[WARN] Failed to parse config, using defaults: {e}")
        return default

    def _load_registry(self):
        path = "canonical_registry.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except Exception as e:
                    print(f"[WARN] Failed to load registry, will start fresh: {e}")
        return {"themes": {}, "experience_drivers": {}}

    def _save_registry(self):
        try:
            with open("canonical_registry.json", "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save registry: {e}")

    def _load_or_build_indexes(self):
        pkl = "performance_indexes.pkl"
        if os.path.exists(pkl):
            try:
                with open(pkl, "rb") as f:
                    idx = pickle.load(f)
                return idx
            except Exception:
                logging.warning("Failed to load indexes, rebuilding…")
        return self._build_indexes()

    def _build_indexes(self):
        logging.info("Building indexes…")
        indexes = {
            "theme_lookup": {},           # normalized_theme -> canonical theme (string as entered)
            "category_lookup": defaultdict(list),    # norm_category -> [full_ED_key]
            "subcategory_lookup": defaultdict(list), # norm_subcat -> [full_ED_key]
            "ed_vectors": None,           # optional TF-IDF over ED strings
            "ed_list": [],                # list of ED strings in registry
            "last_rebuild": datetime.now().isoformat(),
            "entry_count": 0,
            "new_since_build": 0
        }

        # theme lookup
        for theme in self.registry["themes"].keys():
            indexes["theme_lookup"][self._normalize_text(theme)] = theme

        # ED indexes
        ed_texts = []
        for ed_key, ed_data in self.registry["experience_drivers"].items():
            indexes["ed_list"].append(ed_key)
            ed_texts.append(ed_key)

            cat = ed_data.get("canonical_category", "")
            sub = ed_data.get("canonical_subcategory", "")
            if cat:
                indexes["category_lookup"][self._normalize_text(cat)].append(ed_key)
            if sub:
                indexes["subcategory_lookup"][self._normalize_text(sub)].append(ed_key)

        if ed_texts:
            try:
                tfidf = self.vectorizer.fit_transform(ed_texts)
                indexes["ed_vectors"] = tfidf
                indexes["entry_count"] = len(ed_texts)
            except Exception as e:
                logging.warning(f"Failed TF-IDF build: {e}")

        logging.info(f"Built indexes for {len(ed_texts)} EDs")
        self._save_indexes(indexes)
        return indexes

    def _save_indexes(self, indexes=None):
        try:
            with open("performance_indexes.pkl", "wb") as f:
                pickle.dump(indexes or self.indexes, f)
        except Exception as e:
            logging.error(f"Failed to save indexes: {e}")

    def _maybe_rebuild_indexes(self):
        if self.indexes.get("new_since_build", 0) >= self.config["rebuild_index_threshold"]:
            logging.info("Rebuilding indexes (threshold reached)…")
            self.indexes = self._build_indexes()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("canonicalization.log"), logging.StreamHandler()]
        )

    # -----------------------------
    # Normalisation & Parsing
    # -----------------------------
    def _normalize_text(self, text: str) -> str:
        """Lowercase for similarity; collapse spaces; unify arrow variants to '→'."""
        if not isinstance(text, str):
            return ""
        # replace common arrow variants
        text = re.sub(r"(\-\>|›|»|->|→|➡|→)", "→", text)
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _clean_experience_driver(self, raw_ed: str):
        """Return 'Category → Subcategory' if valid; else None. Do NOT title-case (preserve human-authored casing)."""
        if not raw_ed or not isinstance(raw_ed, str):
            return None
        # unify arrows
        ed = re.sub(r"(\-\>|›|»|->|→|➡|→)", "→", raw_ed).strip()
        parts = [p.strip() for p in ed.split("→")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            logging.warning(f"Invalid ED format: {raw_ed}")
            return None
        return f"{parts[0]} → {parts[1]}"

    # -----------------------------
    # Matching (Theme-scoped)
    # -----------------------------
    def get_domain_thresholds(self, domain="general"):
        return self.config["domain_thresholds"].get(domain, self.config["domain_thresholds"]["general"])

    def _theme_match(self, raw_theme, domain="general"):
        """Exact by normalized theme first; otherwise add a new theme."""
        if not raw_theme:
            return "Unknown", 0
        thresholds = self.get_domain_thresholds(domain)
        key = self._normalize_text(raw_theme)
        if key in self.indexes["theme_lookup"]:
            return self.indexes["theme_lookup"][key], 100

        # Fallback fuzzy over existing themes (rare path)
        existing = list(self.registry["themes"].keys())
        if existing:
            name, score, _ = process.extractOne(raw_theme, existing, scorer=fuzz.token_set_ratio)
            if score >= thresholds["theme"]:
                return name, score

        # create new theme
        self._create_theme(raw_theme)
        self.indexes["theme_lookup"][key] = raw_theme
        return raw_theme, 100

    def _fast_category_match(self, theme, category_raw, threshold):
        # candidates: categories for this theme only
        existing_cats = list({
            v["canonical_category"]
            for v in self.registry["experience_drivers"].values()
            if v.get("theme") == theme and v.get("canonical_category")
        })
        if not existing_cats:
            return category_raw

        # Fuzzy first (fast)
        m = process.extractOne(category_raw, existing_cats, scorer=fuzz.token_set_ratio)
        if m and m[1] >= threshold:
            return m[0]

        # TF-IDF backup (scoped to theme's categories)
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            all_texts = existing_cats + [category_raw]
            mat = vec.fit_transform(all_texts)
            sims = cosine_similarity(mat[-1:], mat[:-1]).flatten()
            best = int(np.argmax(sims))
            if sims[best] >= self.config["semantic_similarity_threshold"]:
                return existing_cats[best]
        except Exception:
            pass

        return category_raw

    def _fast_subcategory_match(self, theme, category, subcategory_raw, threshold):
        existing_sub = list({
            v["canonical_subcategory"]
            for v in self.registry["experience_drivers"].values()
            if v.get("theme") == theme and v.get("canonical_category") == category and v.get("canonical_subcategory")
        })
        if not existing_sub:
            return subcategory_raw

        m = process.extractOne(subcategory_raw, existing_sub, scorer=fuzz.token_set_ratio)
        if m and m[1] >= threshold:
            return m[0]

        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            all_texts = existing_sub + [subcategory_raw]
            mat = vec.fit_transform(all_texts)
            sims = cosine_similarity(mat[-1:], mat[:-1]).flatten()
            best = int(np.argmax(sims))
            if sims[best] >= self.config["semantic_similarity_threshold"]:
                return existing_sub[best]
        except Exception:
            pass

        return subcategory_raw

    def _calc_match_score(self, a, b):
        if not a or not b:
            return 0
        if a.strip().lower() == b.strip().lower():
            return 100
        return fuzz.token_set_ratio(a, b)

    # -----------------------------
    # Registry Ops
    # -----------------------------
    def _create_theme(self, raw_theme):
        if raw_theme not in self.registry["themes"]:
            self.registry["themes"][raw_theme] = {
                "raw_variants": [raw_theme],
                "experience_drivers": [],
                "frozen": False,
                "created_date": datetime.today().strftime("%Y-%m-%d")
            }

    def _update_ed_registry(self, canonical_ed, theme, cat, subcat, raw_variant):
        eds = self.registry["experience_drivers"]
        if canonical_ed not in eds:
            eds[canonical_ed] = {
                "canonical_experience_driver": canonical_ed,
                "canonical_category": cat,
                "canonical_subcategory": subcat,
                "theme": theme,
                "raw_variants": [raw_variant],
                "frozen": False,
                "first_seen": datetime.today().strftime("%Y-%m-%d"),
                "last_seen": datetime.today().strftime("%Y-%m-%d"),
                "frequency": 1
            }
            # link to theme
            if theme in self.registry["themes"]:
                if canonical_ed not in self.registry["themes"][theme]["experience_drivers"]:
                    self.registry["themes"][theme]["experience_drivers"].append(canonical_ed)

            # track for potential index rebuild
            self.indexes["new_since_build"] = self.indexes.get("new_since_build", 0) + 1
        else:
            e = eds[canonical_ed]
            if raw_variant not in e["raw_variants"]:
                e["raw_variants"].append(raw_variant)
            e["last_seen"] = datetime.today().strftime("%Y-%m-%d")
            e["frequency"] += 1

    # -----------------------------
    # Public API
    # -----------------------------
    def batch_process(self, df, domain="general", batch_size=None):
        """Vectorised(ish) pass in batches; writes into a copy of df and returns it."""
        batch_size = batch_size or self.config["batch_size"]
        total = len(df)

        # Validate columns
        required = {"theme", "experience_driver"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        for col in [
            "canonical_theme", "canonical_category", "canonical_subcategory",
            "canonical_experience_driver", "theme_match_score",
            "category_match_score", "subcategory_match_score", "needs_review"
        ]:
            if col not in df.columns:
                df[col] = "" if "score" not in col and col != "needs_review" else (0 if "score" in col else False)

        thresholds = self.get_domain_thresholds(domain)
        log_stride = max(1, self.config.get("log_every_n_batches", 10))

        processed = 0
        errors = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = df.iloc[start:end]

            for i, row in batch.iterrows():
                try:
                    raw_theme = (row["theme"] or "").strip()
                    raw_ed = (row["experience_driver"] or "").strip()

                    if not raw_theme or not raw_ed:
                        logging.warning(f"Empty theme/ED at row {i}")
                        errors += 1
                        continue

                    canonical_theme, theme_score = self._theme_match(raw_theme, domain)

                    clean_ed = self._clean_experience_driver(raw_ed)
                    if not clean_ed:
                        # cannot parse ED, leave as-is but mark for review
                        batch.at[i, "canonical_theme"] = canonical_theme
                        batch.at[i, "canonical_experience_driver"] = raw_ed
                        batch.at[i, "needs_review"] = True
                        errors += 1
                        continue

                    cat_raw, sub_raw = [p.strip() for p in clean_ed.split("→")]
                    cat_match = self._fast_category_match(canonical_theme, cat_raw, thresholds["category"])
                    sub_match = self._fast_subcategory_match(canonical_theme, cat_match, sub_raw, thresholds["subcategory"])

                    canonical_ed = f"{cat_match} → {sub_match}"

                    cat_score = self._calc_match_score(cat_raw, cat_match)
                    sub_score = self._calc_match_score(sub_raw, sub_match)

                    # needs_review if any score below domain threshold
                    needs_review = (
                        theme_score < thresholds["theme"]
                        or cat_score < thresholds["category"]
                        or sub_score < thresholds["subcategory"]
                    )

                    # Write results
                    batch.at[i, "canonical_theme"] = canonical_theme
                    batch.at[i, "canonical_category"] = cat_match
                    batch.at[i, "canonical_subcategory"] = sub_match
                    batch.at[i, "canonical_experience_driver"] = canonical_ed
                    batch.at[i, "theme_match_score"] = theme_score
                    batch.at[i, "category_match_score"] = cat_score
                    batch.at[i, "subcategory_match_score"] = sub_score
                    batch.at[i, "needs_review"] = bool(needs_review)

                    # Update registry
                    self._update_ed_registry(canonical_ed, canonical_theme, cat_match, sub_match, clean_ed)

                except Exception as e:
                    logging.error(f"Row {i} error: {e}")
                    errors += 1
                    continue

            # write batch slice back
            df.iloc[start:end] = batch
            processed += (end - start)

            if (start // batch_size) % log_stride == 0:
                logging.info(f"Processed {processed}/{total} rows ({processed/total*100:.1f}%), errors={errors}")

            # optional index rebuild if many new entries were added
            self._maybe_rebuild_indexes()

        logging.info(f"Completed: {processed} processed, {errors} errors")
        return df

    def export_canonical_report(self, output_path="canonicalization_report.csv"):
        """Row-agnostic, registry-level mapping report + compact summary JSON."""
        rows = []
        for ed, data in self.registry["experience_drivers"].items():
            for variant in data.get("raw_variants", []):
                rows.append({
                    "raw_variant": variant,
                    "canonical_experience_driver": ed,
                    "theme": data.get("theme"),
                    "canonical_category": data.get("canonical_category"),
                    "canonical_subcategory": data.get("canonical_subcategory"),
                    "first_seen": data.get("first_seen"),
                    "last_seen": data.get("last_seen"),
                    "frozen": data.get("frozen"),
                    "frequency": data.get("frequency", 0),
                    "variant_count": len(data.get("raw_variants", []))
                })

        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")

        summary = {
            "total_canonical_eds": len(self.registry["experience_drivers"]),
            "total_variants": len(rows),
            "compression_ratio": (len(rows) / max(1, len(self.registry["experience_drivers"]))),
            "themes_count": len(self.registry["themes"]),
            "last_index_rebuild": self.indexes.get("last_rebuild")
        }
        with open(output_path.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logging.info(f"[✓] Report saved: {output_path}")

    def get_performance_stats(self):
        total_variants = sum(len(ed.get("raw_variants", [])) for ed in self.registry["experience_drivers"].values())
        return {
            "total_themes": len(self.registry["themes"]),
            "total_experience_drivers": len(self.registry["experience_drivers"]),
            "total_variants": total_variants,
            "compression_ratio": (total_variants / max(1, len(self.registry["experience_drivers"]))),
            "index_entry_count": self.indexes.get("entry_count", 0),
            "last_index_rebuild": self.indexes.get("last_rebuild", "Never"),
            "frozen_eds": sum(1 for ed in self.registry["experience_drivers"].values() if ed.get("frozen"))
        }


# -----------------------------
# CLI wrapper
# -----------------------------
def main():
    # === INPUT & OUTPUT CONFIGURATION ===
    input_file = "data/decipher_retail_grocery_analytics_flattened.csv"
    output_dir = "outputs"
    base_name = os.path.basename(input_file).replace(".csv", "")
    output_file = os.path.join(output_dir, f"{base_name}_canonicalized.csv")
    report_file = os.path.join(output_dir, f"{base_name}_canonical_map.csv")
    stats_file = os.path.join(output_dir, f"{base_name}_explosion_stats.txt")
    domain = "retail"

    os.makedirs(output_dir, exist_ok=True)

    try:
        engine = HighPerformanceCanonicalizer()
        print(f"[INFO] Loading data from: {input_file}")

        if not os.path.exists(input_file):
            print(f"[ERROR] Input file not found: {input_file}")
            return

        df = pd.read_csv(input_file)
        print(f"[INFO] Loaded {len(df)} rows")

        processed_df = engine.batch_process(df, domain=domain)
        processed_df.to_csv(output_file, index=False, encoding="utf-8")
        engine._save_registry()

        engine.export_canonical_report(report_file)

        unique_raw = df["experience_driver"].nunique()
        unique_canonical = processed_df["canonical_experience_driver"].nunique()
        compression_ratio = (unique_raw / unique_canonical) if unique_canonical else 0.0

        needs_review_count = processed_df["needs_review"].sum()
        stats = engine.get_performance_stats()

        with open(stats_file, "w", encoding="utf-8") as f:
            f.write("EXPERIENCE DRIVER CANONICALIZATION STATS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Original unique experience drivers: {unique_raw}\n")
            f.write(f"Canonical unique experience drivers: {unique_canonical}\n")
            f.write(f"Compression ratio: {compression_ratio:.2f}x\n")
            f.write(f"Total records processed: {len(df)}\n")
            f.write(f"Mappings requiring review: {int(needs_review_count)}\n")
            f.write(f"Domain: {domain}\n")
            f.write(f"Thresholds Used: {engine.get_domain_thresholds(domain)}\n")
            f.write(f"Performance Stats: {stats}\n")

        print("\n[✓] Canonicalization completed successfully")
        print(f"Original Drivers:  {unique_raw}")
        print(f"Canonical Drivers: {unique_canonical}")
        print(f"Compression Ratio: {compression_ratio:.2f}x")
        print(f"Needs Review:      {int(needs_review_count)}")
        print(f"Registry Stats:    {stats}")
        print(f"Output directory:  {output_dir}\n")

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        logging.error(f"Main processing failed: {e}")


if __name__ == "__main__":
    main()
