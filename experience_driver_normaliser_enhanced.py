import pandas as pd
import os
import json
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import pickle
from collections import defaultdict
import logging

class HighPerformanceCanonicalizer:
    def __init__(self, config_path="canonicalization_config.json"):
        self.config = self.load_config(config_path)
        self.registry = self.load_registry()
        self.indexes = self.load_or_build_indexes()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words='english'
        )
        self.setup_logging()
    
    def load_config(self, config_path):
        """Load domain-specific threshold configuration"""
        default_config = {
            "domain_thresholds": {
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
            "cache_size": 10000,
            "rebuild_index_threshold": 100  # Rebuild after N new entries
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_registry(self):
        """Load existing registry or create new one"""
        registry_path = "canonical_registry.json"
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"themes": {}, "experience_drivers": {}}
    
    def load_or_build_indexes(self):
        """Load pre-built indexes or create new ones"""
        index_path = "performance_indexes.pkl"
        if os.path.exists(index_path):
            try:
                with open(index_path, 'rb') as f:
                    return pickle.load(f)
            except:
                logging.warning("Failed to load indexes, rebuilding...")
        
        return self.build_indexes()
    
    def build_indexes(self):
        """Build optimized lookup indexes"""
        logging.info("Building performance indexes...")
        
        indexes = {
            "theme_lookup": {},
            "category_lookup": defaultdict(list),
            "subcategory_lookup": defaultdict(list),
            "ed_vectors": None,
            "ed_list": [],
            "last_rebuild": datetime.now().isoformat(),
            "entry_count": 0
        }
        
        # Build theme index
        for theme in self.registry["themes"].keys():
            normalized = self._normalize_text(theme)
            indexes["theme_lookup"][normalized] = theme
        
        # Build Experience Driver indexes
        ed_texts = []
        for ed_key, ed_data in self.registry["experience_drivers"].items():
            indexes["ed_list"].append(ed_key)
            ed_texts.append(ed_key)
            
            # Category lookup
            category = ed_data.get("canonical_category", "")
            if category:
                normalized_cat = self._normalize_text(category)
                indexes["category_lookup"][normalized_cat].append(ed_key)
            
            # Subcategory lookup
            subcategory = ed_data.get("canonical_subcategory", "")
            if subcategory:
                normalized_subcat = self._normalize_text(subcategory)
                indexes["subcategory_lookup"][normalized_subcat].append(ed_key)
        
        # Build vector representations for semantic similarity
        if ed_texts:
            try:
                tfidf_matrix = self.vectorizer.fit_transform(ed_texts)
                indexes["ed_vectors"] = tfidf_matrix
                indexes["entry_count"] = len(ed_texts)
            except Exception as e:
                logging.warning(f"Failed to build vector index: {e}")
        
        self.save_indexes(indexes)
        logging.info(f"Built indexes for {len(ed_texts)} Experience Drivers")
        return indexes
    
    def _normalize_text(self, text):
        """Normalize text for consistent lookups"""
        return text.lower().strip().replace("  ", " ")
    
    def get_domain_thresholds(self, domain="general"):
        """Get thresholds for specific domain"""
        return self.config["domain_thresholds"].get(
            domain, 
            self.config["domain_thresholds"]["general"]
        )
    
    def fast_theme_match(self, raw_theme, domain="general"):
        """Optimized theme matching using pre-built indexes"""
        thresholds = self.get_domain_thresholds(domain)
        normalized = self._normalize_text(raw_theme)
        
        # Exact match first (O(1))
        if normalized in self.indexes["theme_lookup"]:
            return self.indexes["theme_lookup"][normalized], 100
        
        # Fuzzy match only if needed (O(n) but smaller n)
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
    
    def fast_ed_match(self, theme, raw_ed, domain="general"):
        """High-performance Experience Driver matching"""
        if "→" not in raw_ed:
            return raw_ed.strip(), None, None, 0, 0
        
        thresholds = self.get_domain_thresholds(domain)
        category_raw, subcategory_raw = (x.strip() for x in raw_ed.split("→"))
        
        # Quick category lookup using indexes
        cat_match = self._fast_category_match(theme, category_raw, thresholds["category"])
        subcat_match = self._fast_subcategory_match(theme, cat_match, subcategory_raw, thresholds["subcategory"])
        
        canonical_ed = f"{cat_match} → {subcat_match}"
        
        # Update or create registry entry
        self._update_ed_registry(canonical_ed, theme, cat_match, subcat_match, f"{category_raw} → {subcategory_raw}")
        
        return canonical_ed, cat_match, subcat_match, 100, 100
    
    def _fast_category_match(self, theme, category_raw, threshold):
        """Fast category matching using indexes"""
        normalized = self._normalize_text(category_raw)
        
        # Quick lookup in category index
        if normalized in self.indexes["category_lookup"]:
            candidates = self.indexes["category_lookup"][normalized]
            theme_filtered = [
                self.registry["experience_drivers"][ed]["canonical_category"]
                for ed in candidates
                if self.registry["experience_drivers"][ed]["theme"] == theme
            ]
            if theme_filtered:
                return theme_filtered[0]  # Return first match
        
        # Fallback to fuzzy matching
        existing_cats = list(set([
            ed["canonical_category"]
            for ed in self.registry["experience_drivers"].values()
            if ed["theme"] == theme
        ]))
        
        if existing_cats:
            match = process.extractOne(category_raw, existing_cats, scorer=fuzz.token_set_ratio)
            if match and match[1] >= threshold:
                return match[0]
        
        return category_raw
    
    def _fast_subcategory_match(self, theme, category, subcategory_raw, threshold):
        """Fast subcategory matching using indexes"""
        normalized = self._normalize_text(subcategory_raw)
        
        # Quick lookup
        if normalized in self.indexes["subcategory_lookup"]:
            candidates = self.indexes["subcategory_lookup"][normalized]
            filtered = [
                self.registry["experience_drivers"][ed]["canonical_subcategory"]
                for ed in candidates
                if (self.registry["experience_drivers"][ed]["theme"] == theme and
                    self.registry["experience_drivers"][ed]["canonical_category"] == category)
            ]
            if filtered:
                return filtered[0]
        
        # Fallback to fuzzy matching
        existing_subcats = list(set([
            ed["canonical_subcategory"]
            for ed in self.registry["experience_drivers"].values()
            if (ed["theme"] == theme and ed["canonical_category"] == category)
        ]))
        
        if existing_subcats:
            match = process.extractOne(subcategory_raw, existing_subcats, scorer=fuzz.token_set_ratio)
            if match and match[1] >= threshold:
                return match[0]
        
        return subcategory_raw
    
    def batch_process(self, df, domain="general", batch_size=None):
        """Process large datasets in optimized batches"""
        batch_size = batch_size or self.config["batch_size"]
        total_rows = len(df)
        
        # Pre-allocate result columns
        df["canonical_theme"] = ""
        df["canonical_category"] = ""
        df["canonical_subcategory"] = ""
        df["canonical_experience_driver"] = ""
        df["theme_match_score"] = 0
        df["category_match_score"] = 0
        df["subcategory_match_score"] = 0
        
        logging.info(f"Processing {total_rows} rows in batches of {batch_size}")
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = df.iloc[start_idx:end_idx]
            
            self._process_batch(batch, domain, start_idx)
            
            if start_idx % (batch_size * 10) == 0:  # Log every 10 batches
                logging.info(f"Processed {end_idx}/{total_rows} rows ({end_idx/total_rows*100:.1f}%)")
        
        return df
    
    def _process_batch(self, batch, domain, start_idx):
        """Process a single batch efficiently"""
        for i, row in batch.iterrows():
            try:
                raw_theme = str(row["theme"]).strip()
                raw_ed = str(row["experience_driver"]).strip()
                
                canonical_theme, theme_score = self.fast_theme_match(raw_theme, domain)
                canonical_ed, canonical_cat, canonical_subcat, cat_score, subcat_score = self.fast_ed_match(canonical_theme, raw_ed, domain)
                
                # Update DataFrame
                batch.at[i, "canonical_theme"] = canonical_theme
                batch.at[i, "canonical_category"] = canonical_cat or ""
                batch.at[i, "canonical_subcategory"] = canonical_subcat or ""
                batch.at[i, "canonical_experience_driver"] = canonical_ed
                batch.at[i, "theme_match_score"] = theme_score
                batch.at[i, "category_match_score"] = cat_score
                batch.at[i, "subcategory_match_score"] = subcat_score
                
            except Exception as e:
                logging.error(f"Error processing row {i}: {e}")
                continue
    
    def _create_new_theme(self, raw_theme):
        """Create new theme entry"""
        self.registry["themes"][raw_theme] = {
            "raw_variants": [raw_theme],
            "experience_drivers": [],
            "frozen": False
        }
        return raw_theme
    
    def _update_ed_registry(self, canonical_ed, theme, cat_match, subcat_match, raw_variant):
        """Update Experience Driver registry efficiently"""
        if canonical_ed not in self.registry["experience_drivers"]:
            self.registry["experience_drivers"][canonical_ed] = {
                "canonical_experience_driver": canonical_ed,
                "canonical_category": cat_match,
                "canonical_subcategory": subcat_match,
                "theme": theme,
                "raw_variants": [raw_variant],
                "frozen": False,
                "first_seen": datetime.today().strftime("%Y-%m-%d"),
                "last_seen": datetime.today().strftime("%Y-%m-%d"),
                "frequency": 1
            }
            
            if canonical_ed not in self.registry["themes"][theme]["experience_drivers"]:
                self.registry["themes"][theme]["experience_drivers"].append(canonical_ed)
        else:
            ed_entry = self.registry["experience_drivers"][canonical_ed]
            if raw_variant not in ed_entry["raw_variants"]:
                ed_entry["raw_variants"].append(raw_variant)
            ed_entry["last_seen"] = datetime.today().strftime("%Y-%m-%d")
            ed_entry["frequency"] += 1
    
    def save_indexes(self, indexes):
        """Save indexes to disk for future use"""
        try:
            with open("performance_indexes.pkl", 'wb') as f:
                pickle.dump(indexes, f)
        except Exception as e:
            logging.error(f"Failed to save indexes: {e}")
    
    def save_registry(self):
        """Save updated registry"""
        with open("canonical_registry.json", "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
    
    def setup_logging(self):
        """Setup performance logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('canonicalization.log'),
                logging.StreamHandler()
            ]
        )
    
    def export_canonical_report(self, output_path="canonicalization_report.csv"):
        records = []
        for ed, data in self.registry["experience_drivers"].items():
            for variant in data["raw_variants"]:
                records.append({
                    "raw_variant": variant,
                    "canonical_experience_driver": ed,
                    "theme": data.get("theme"),
                    "canonical_category": data.get("canonical_category"),
                    "canonical_subcategory": data.get("canonical_subcategory"),
                    "needs_review": data.get("needs_review", False),
                    "first_seen": data.get("first_seen"),
                    "last_seen": data.get("last_seen"),
                    "frozen": data.get("frozen"),
                    "frequency": data.get("frequency", 0)
                })
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        logging.info(f"[✓] Canonicalization report saved: {output_path}") 
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            "total_themes": len(self.registry["themes"]),
            "total_experience_drivers": len(self.registry["experience_drivers"]),
            "index_entry_count": self.indexes.get("entry_count", 0),
            "last_index_rebuild": self.indexes.get("last_rebuild", "Never")
        }

# === USAGE EXAMPLE ===
def process_with_performance_engine(input_file, output_file, domain="general"):
    """Main processing function using high-performance engine"""
    
    # Initialize the engine
    engine = HighPerformanceCanonicalizer()
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Process with domain-specific settings
    processed_df = engine.batch_process(df, domain=domain)
    
    # Save results
    processed_df.to_csv(output_file, index=False)
    engine.save_registry()
    
    # Print performance stats
    stats = engine.get_performance_stats()
    print(f"✅ Processing complete!")
    print(f"📊 Performance Stats: {stats}")
    
    return processed_df

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

    # === INIT & RUN ===
    engine = HighPerformanceCanonicalizer()
    print(f"[INFO] Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    processed_df = engine.batch_process(df, domain=domain)

    # === SAVE OUTPUT ===
    processed_df.to_csv(output_file, index=False)
    engine.save_registry()

    # === GENERATE & SAVE REPORT ===
    engine.export_canonical_report(report_file)

    # === STATS ===
    unique_raw = df["experience_driver"].nunique()
    unique_canonical = processed_df["canonical_experience_driver"].nunique()
    compression_ratio = unique_raw / unique_canonical if unique_canonical > 0 else 0

    needs_review_count = processed_df[
        (processed_df["category_match_score"] < 90) |
        (processed_df["subcategory_match_score"] < 90)
    ].shape[0]

    with open(stats_file, 'w', encoding="utf-8") as f:
        f.write("EXPERIENCE DRIVER CANONICALIZATION STATS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Original unique experience drivers: {unique_raw}\n")
        f.write(f"Canonical unique experience drivers: {unique_canonical}\n")
        f.write(f"Compression ratio: {compression_ratio:.2f}x\n")
        f.write(f"Total records processed: {len(df)}\n")
        f.write(f"Mappings requiring review: {needs_review_count}\n")
        f.write(f"Domain Thresholds Used: {engine.get_domain_thresholds(domain)}\n")

    print("\n[✓] Canonicalization completed successfully")
    print(f"Original Drivers:  {unique_raw}")
    print(f"Canonical Drivers: {unique_canonical}")
    print(f"Compression Ratio: {compression_ratio:.2f}x")
    print(f"Needs Review:      {needs_review_count}")
    print(f"Output directory:  {output_dir}\n")

if __name__ == "__main__":
    main()
