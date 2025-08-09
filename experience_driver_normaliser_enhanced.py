import pandas as pd
import numpy as np
import os
import json
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import pickle
from datetime import datetime
from collections import defaultdict
from rapidfuzz import fuzz, process

class MLExperienceDriverNormalizer:
    """
    ML-powered Experience Driver Normalizer using semantic embeddings
    of PDCA problem statement fields for intelligent clustering
    """
    
    def __init__(self, config_path="ml_normalizer_config.json"):
        self.config = self.load_config(config_path)
        self.model = self.load_embedding_model()
        self.embedding_cache = self.load_embedding_cache()
        self.canonical_registry = self.load_canonical_registry()
        self.setup_logging()
        
    def load_config(self, config_path):
        """Load ML-specific configuration"""
        default_config = {
            "embedding_model": "all-MiniLM-L6-v2",  # Fast, good quality
            "similarity_threshold": 0.85,  # Cosine similarity threshold for clustering
            "clustering_threshold": 0.80,  # AgglomerativeClustering threshold
            "min_cluster_size": 2,  # Minimum items to form cluster
            "cache_embeddings": True,
            "batch_size": 100,
            "field_weights": {
                # Removed - now using signature builder with fixed weights
                "semantic_action_statement": 8,  # 8x weight (TIER 1: Problem essence)
                "behavioral_impact": 6,          # 6x weight (TIER 1: Consequence core)
                "stream_justification": 4,       # 4x weight (TIER 2: Strategic context)
                "customer_journey": 2,           # 2x weight (TIER 3: Journey context)
                "journey_stage": 1               # 1x weight (TIER 3: Journey timing)
            },
            "enable_fuzzy_fallback": True,  # Fallback to original fuzzy matching
            "fuzzy_threshold": 85
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_embedding_model(self):
        """Load sentence transformer model"""
        try:
            model = SentenceTransformer(self.config["embedding_model"])
            logging.info(f"✅ Loaded embedding model: {self.config['embedding_model']}")
            return model
        except Exception as e:
            logging.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def load_embedding_cache(self):
        """Load cached embeddings to avoid recomputation"""
        cache_path = "ml_embedding_cache.pkl"
        if os.path.exists(cache_path) and self.config["cache_embeddings"]:
            try:
                with open(cache_path, 'rb') as f:
                    cache = pickle.load(f)
                    logging.info(f"✅ Loaded embedding cache with {len(cache)} entries")
                    return cache
            except Exception as e:
                logging.warning(f"⚠️ Failed to load embedding cache: {e}")
        
        return {}
    
    def load_canonical_registry(self):
        """Load existing canonical registry or create new"""
        registry_path = "ml_canonical_registry.json"
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
                logging.info(f"✅ Loaded canonical registry with {len(registry.get('clusters', {}))} clusters")
                return registry
        
        return {
            "clusters": {},  # canonical_ed -> cluster info
            "embeddings_meta": {},  # metadata about embeddings
            "stats": {
                "total_processed": 0,
                "clusters_created": 0,
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def _build_pdca_signature(self, row: pd.Series) -> str:
        """
        🔥 PDCA SEMANTIC SIGNATURE BUILDER
        Optimized for maximum clustering accuracy by prioritizing problem essence
        over contextual noise that creates false similarities.
        """
        from typing import List
        
        toks: List[str] = []
        
        # 🎯 TIER 1: PROBLEM ESSENCE (Heavy Weight - captures core issue)
        semantic_action = row.get("semantic_action_statement")
        if pd.notna(semantic_action):
            toks += [str(semantic_action)] * 8  # PRIMARY DRIVER - customer reality + strategic response
        
        behavioral_impact = row.get("behavioral_impact") 
        if pd.notna(behavioral_impact):
            toks += [str(behavioral_impact)] * 6  # CONSEQUENCE CORE - what happens if ignored
        
        # 🎯 TIER 2: STRATEGIC CONTEXT (Medium Weight - shapes response type)
        stream_justification = row.get("stream_justification")
        if pd.notna(stream_justification):
            toks += [str(stream_justification)] * 4  # WHY this stream (Fix/Optimize/Innovate/Amplify)
        
        # 🎯 TIER 3: JOURNEY CONTEXT (Light Weight - situational placement)
        customer_journey = row.get("customer_journey")
        if pd.notna(customer_journey):
            toks += [str(customer_journey)] * 2  # WHERE in customer experience
        
        journey_stage = row.get("journey_stage")
        if pd.notna(journey_stage):
            toks.append(str(journey_stage))  # WHEN in journey flow
        
        return " ".join(toks)
    
    def get_embedding(self, text, cache_key=None):
        """
        Get embedding for text with caching
        """
        if cache_key and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            embedding = self.model.encode([text])[0]
            
            if cache_key and self.config["cache_embeddings"]:
                self.embedding_cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            logging.error(f"❌ Failed to create embedding: {e}")
            return None
    
    def find_similar_clusters(self, target_embedding, threshold=None):
        """
        Find existing clusters similar to target embedding
        """
        threshold = threshold or self.config["similarity_threshold"]
        
        if not self.canonical_registry["clusters"]:
            return []
        
        similarities = []
        
        for canonical_ed, cluster_info in self.canonical_registry["clusters"].items():
            if "centroid_embedding" not in cluster_info:
                continue
                
            centroid = np.array(cluster_info["centroid_embedding"])
            similarity = cosine_similarity([target_embedding], [centroid])[0][0]
            
            if similarity >= threshold:
                similarities.append({
                    "canonical_ed": canonical_ed,
                    "similarity": similarity,
                    "cluster_info": cluster_info
                })
        
        # Return top matches sorted by similarity
        return sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    
    def create_canonical_ed_name(self, experience_drivers):
        """
        Create canonical name from cluster of similar EDs
        """
        # Use the most frequent or first ED as base
        ed_counts = defaultdict(int)
        for ed in experience_drivers:
            ed_counts[ed] += 1
        
        # Get most common ED
        most_common = max(ed_counts.items(), key=lambda x: x[1])[0]
        
        # Clean and standardize
        canonical = most_common.strip()
        
        # TODO: Could add more sophisticated name generation logic here
        
        return canonical
    
    def update_cluster_registry(self, canonical_ed, members, centroid_embedding, pdca_signatures):
        """
        Update the canonical registry with new cluster
        """
        self.canonical_registry["clusters"][canonical_ed] = {
            "canonical_experience_driver": canonical_ed,
            "members": list(set(members)),  # Remove duplicates
            "member_count": len(set(members)),
            "centroid_embedding": centroid_embedding.tolist(),
            "created_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "sample_pdca_signatures": pdca_signatures[:3]  # Store first 3 for debugging
        }
        
        self.canonical_registry["stats"]["clusters_created"] += 1
        self.canonical_registry["stats"]["last_updated"] = datetime.now().isoformat()
    
    def ml_normalize_batch(self, df):
        """
        Main ML normalization function using semantic embeddings
        """
        logging.info(f"🚀 Starting ML normalization for {len(df)} rows")
        
        # Create PDCA signatures and embeddings
        df["pdca_signature"] = df.apply(self._build_pdca_signature, axis=1)
        
        # Filter out rows with empty PDCA signature
        valid_mask = df["pdca_signature"].str.len() > 10  # Minimum length check
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) == 0:
            logging.warning("⚠️ No valid PDCA signatures found")
            return df
        
        logging.info(f"📝 Created PDCA signatures for {len(df_valid)} valid rows")
        
        # Generate embeddings
        embeddings = []
        for i, row in df_valid.iterrows():
            cache_key = f"pdca_{hash(row['pdca_signature'])}"
            embedding = self.get_embedding(row["pdca_signature"], cache_key)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        logging.info(f"🎯 Generated {len(embeddings)} embeddings")
        
        # Initialize results
        df["canonical_experience_driver_ml"] = df["experience_driver"]
        df["ml_similarity_score"] = 0.0
        df["ml_cluster_id"] = ""
        df["ml_method"] = "original"
        
        # Process each valid row
        for idx, (df_idx, row) in enumerate(df_valid.iterrows()):
            try:
                current_embedding = embeddings[idx]
                experience_driver = row["experience_driver"]
                
                # Find similar existing clusters
                similar_clusters = self.find_similar_clusters(current_embedding)
                
                if similar_clusters:
                    # Assign to most similar cluster
                    best_match = similar_clusters[0]
                    df.at[df_idx, "canonical_experience_driver_ml"] = best_match["canonical_ed"]
                    df.at[df_idx, "ml_similarity_score"] = best_match["similarity"]
                    df.at[df_idx, "ml_cluster_id"] = best_match["canonical_ed"]
                    df.at[df_idx, "ml_method"] = "cluster_match"
                    
                    # Update cluster membership
                    cluster_info = self.canonical_registry["clusters"][best_match["canonical_ed"]]
                    if experience_driver not in cluster_info["members"]:
                        cluster_info["members"].append(experience_driver)
                        cluster_info["member_count"] = len(cluster_info["members"])
                        cluster_info["last_updated"] = datetime.now().isoformat()
                    
                else:
                    # Create new cluster
                    canonical_ed = self.create_canonical_ed_name([experience_driver])
                    
                    self.update_cluster_registry(
                        canonical_ed,
                        [experience_driver],
                        current_embedding,
                        [row["pdca_signature"]]
                    )
                    
                    df.at[df_idx, "canonical_experience_driver_ml"] = canonical_ed
                    df.at[df_idx, "ml_similarity_score"] = 1.0
                    df.at[df_idx, "ml_cluster_id"] = canonical_ed
                    df.at[df_idx, "ml_method"] = "new_cluster"
                
            except Exception as e:
                logging.error(f"❌ Error processing row {df_idx}: {e}")
                # Fallback to original ED
                continue
        
        # Update stats
        self.canonical_registry["stats"]["total_processed"] += len(df_valid)
        self.canonical_registry["stats"]["last_updated"] = datetime.now().isoformat()
        
        return df
    
    def fuzzy_fallback_normalize(self, df):
        """
        Fallback to fuzzy matching for rows without PDCA fields
        """
        if not self.config["enable_fuzzy_fallback"]:
            return df
        
        logging.info("🔄 Applying fuzzy fallback normalization")
        
        # Get rows that weren't processed by ML
        ml_unprocessed = df[df["ml_method"] == "original"]
        
        if len(ml_unprocessed) == 0:
            return df
        
        # Simple fuzzy matching on experience_driver
        experience_drivers = list(self.canonical_registry["clusters"].keys())
        
        for idx, row in ml_unprocessed.iterrows():
            ed = str(row["experience_driver"]).strip()
            
            if experience_drivers:
                match = process.extractOne(
                    ed, 
                    experience_drivers, 
                    scorer=fuzz.token_set_ratio
                )
                
                if match and match[1] >= self.config["fuzzy_threshold"]:
                    df.at[idx, "canonical_experience_driver_ml"] = match[0]
                    df.at[idx, "ml_similarity_score"] = match[1] / 100.0
                    df.at[idx, "ml_method"] = "fuzzy_fallback"
        
        return df
    
    def process_dataframe(self, df):
        """
        Main processing function
        """
        try:
            # ML-based normalization
            df_processed = self.ml_normalize_batch(df)
            
            # Fuzzy fallback for remaining
            df_final = self.fuzzy_fallback_normalize(df_processed)
            
            # Generate summary stats
            self.log_processing_stats(df_final)
            
            return df_final
            
        except Exception as e:
            logging.error(f"❌ Processing failed: {e}")
            raise
    
    def log_processing_stats(self, df):
        """
        Log processing statistics
        """
        stats = {
            "total_rows": len(df),
            "ml_clustered": len(df[df["ml_method"] == "cluster_match"]),
            "new_clusters": len(df[df["ml_method"] == "new_cluster"]),
            "fuzzy_fallback": len(df[df["ml_method"] == "fuzzy_fallback"]),
            "unchanged": len(df[df["ml_method"] == "original"]),
            "unique_original_eds": df["experience_driver"].nunique(),
            "unique_canonical_eds": df["canonical_experience_driver_ml"].nunique()
        }
        
        compression_ratio = stats["unique_original_eds"] / stats["unique_canonical_eds"] if stats["unique_canonical_eds"] > 0 else 1
        
        logging.info("📊 ML Processing Stats:")
        logging.info(f"  Total rows: {stats['total_rows']}")
        logging.info(f"  ML clustered: {stats['ml_clustered']}")
        logging.info(f"  New clusters: {stats['new_clusters']}")
        logging.info(f"  Fuzzy fallback: {stats['fuzzy_fallback']}")
        logging.info(f"  Unchanged: {stats['unchanged']}")
        logging.info(f"  Compression ratio: {compression_ratio:.2f}x")
        logging.info(f"  Original EDs: {stats['unique_original_eds']} → Canonical: {stats['unique_canonical_eds']}")
    
    def save_cache_and_registry(self):
        """
        Save embeddings cache and canonical registry
        """
        try:
            # Save embedding cache
            if self.config["cache_embeddings"]:
                with open("ml_embedding_cache.pkl", 'wb') as f:
                    pickle.dump(self.embedding_cache, f)
                logging.info(f"💾 Saved embedding cache with {len(self.embedding_cache)} entries")
            
            # Save canonical registry
            with open("ml_canonical_registry.json", "w", encoding="utf-8") as f:
                json.dump(self.canonical_registry, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 Saved canonical registry with {len(self.canonical_registry['clusters'])} clusters")
            
        except Exception as e:
            logging.error(f"❌ Failed to save cache/registry: {e}")
    
    def export_cluster_report(self, output_path="ml_cluster_report.csv"):
        """
        Export detailed cluster analysis report
        """
        records = []
        
        for canonical_ed, cluster_info in self.canonical_registry["clusters"].items():
            for member in cluster_info["members"]:
                records.append({
                    "member_experience_driver": member,
                    "canonical_experience_driver": canonical_ed,
                    "cluster_size": cluster_info["member_count"],
                    "created_date": cluster_info["created_date"],
                    "last_updated": cluster_info["last_updated"]
                })
        
        if records:
            df_report = pd.DataFrame(records)
            df_report.to_csv(output_path, index=False)
            logging.info(f"📋 Cluster report exported: {output_path}")
        else:
            logging.warning("⚠️ No clusters to export")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ml_normalizer.log'),
                logging.StreamHandler()
            ]
        )

# === USAGE EXAMPLE ===
def main():
    """
    ML Experience Driver Normalizer - Production Main Function
    """
    
    # === INPUT & OUTPUT CONFIGURATION ===
    input_file = "data/decipher_retail_grocery_analytics_flattened.csv"
    output_dir = "outputs"
    base_name = os.path.basename(input_file).replace(".csv", "")
    output_file = os.path.join(output_dir, f"{base_name}_ml_normalized.csv")
    cluster_report_file = os.path.join(output_dir, f"{base_name}_ml_cluster_report.csv")
    stats_file = os.path.join(output_dir, f"{base_name}_ml_normalization_stats.txt")
    domain = "retail"

    os.makedirs(output_dir, exist_ok=True)

    # === INIT & RUN ===
    normalizer = MLExperienceDriverNormalizer()
    print(f"[INFO] Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"[INFO] Processing {len(df)} rows with ML normalization...")
    processed_df = normalizer.process_dataframe(df)

    # === SAVE OUTPUT ===
    processed_df.to_csv(output_file, index=False)
    normalizer.save_cache_and_registry()

    # === GENERATE & SAVE CLUSTER REPORT ===
    normalizer.export_cluster_report(cluster_report_file)

    # === STATS CALCULATION ===
    unique_raw = df["experience_driver"].nunique()
    unique_canonical = processed_df["canonical_experience_driver_ml"].nunique()
    compression_ratio = unique_raw / unique_canonical if unique_canonical > 0 else 1

    # ML-specific stats
    ml_stats = {
        "total_rows": len(processed_df),
        "ml_clustered": len(processed_df[processed_df["ml_method"] == "cluster_match"]),
        "new_clusters": len(processed_df[processed_df["ml_method"] == "new_cluster"]),
        "fuzzy_fallback": len(processed_df[processed_df["ml_method"] == "fuzzy_fallback"]),
        "unchanged": len(processed_df[processed_df["ml_method"] == "original"]),
        "avg_similarity": processed_df[processed_df["ml_similarity_score"] > 0]["ml_similarity_score"].mean()
    }

    # High similarity clusters (potential over-clustering)
    high_similarity_count = len(processed_df[processed_df["ml_similarity_score"] > 0.95])
    
    # Low similarity clusters (potential under-clustering)
    low_similarity_count = len(processed_df[
        (processed_df["ml_similarity_score"] > 0) & 
        (processed_df["ml_similarity_score"] < 0.85)
    ])

    # === SAVE STATS ===
    with open(stats_file, 'w', encoding="utf-8") as f:
        f.write("ML EXPERIENCE DRIVER NORMALIZATION STATS\n")
        f.write("=" * 55 + "\n\n")
        f.write("📊 COMPRESSION ANALYSIS\n")
        f.write("-" * 25 + "\n")
        f.write(f"Original unique experience drivers: {unique_raw}\n")
        f.write(f"ML canonical unique experience drivers: {unique_canonical}\n")
        f.write(f"Compression ratio: {compression_ratio:.2f}x\n")
        f.write(f"Total records processed: {len(df)}\n\n")
        
        f.write("🤖 ML PROCESSING BREAKDOWN\n")
        f.write("-" * 28 + "\n")
        f.write(f"ML clustered (existing): {ml_stats['ml_clustered']}\n")
        f.write(f"New clusters created: {ml_stats['new_clusters']}\n")
        f.write(f"Fuzzy fallback used: {ml_stats['fuzzy_fallback']}\n")
        f.write(f"Unchanged (no PDCA fields): {ml_stats['unchanged']}\n")
        f.write(f"Average similarity score: {ml_stats['avg_similarity']:.3f}\n\n")
        
        f.write("🎯 CLUSTERING QUALITY\n")
        f.write("-" * 20 + "\n")
        f.write(f"High similarity clusters (>95%): {high_similarity_count}\n")
        f.write(f"Low similarity clusters (<85%): {low_similarity_count}\n")
        f.write(f"Total ML clusters: {len(normalizer.canonical_registry['clusters'])}\n\n")
        
        f.write("⚙️ CONFIGURATION USED\n")
        f.write("-" * 21 + "\n")
        f.write(f"Embedding model: {normalizer.config['embedding_model']}\n")
        f.write(f"Similarity threshold: {normalizer.config['similarity_threshold']}\n")
        f.write(f"Domain: {domain}\n")

    # === CONSOLE OUTPUT ===
    print("\n[✓] ML Normalization completed successfully")
    print(f"Original Experience Drivers:  {unique_raw}")
    print(f"ML Canonical Drivers: {unique_canonical}")
    print(f"Compression Ratio: {compression_ratio:.2f}x")
    print(f"ML Clustered: {ml_stats['ml_clustered']}")
    print(f"New Clusters: {ml_stats['new_clusters']}")
    print(f"Avg Similarity: {ml_stats['avg_similarity']:.3f}")
    print(f"Output directory: {output_dir}")
    
    # === COMPARISON PREVIEW ===
    print("\n🔍 Sample ML Normalization Results:")
    sample_results = processed_df[processed_df["ml_method"] != "original"].head(5)
    if len(sample_results) > 0:
        for _, row in sample_results.iterrows():
            print(f"  '{row['experience_driver']}' → '{row['canonical_experience_driver_ml']}' ({row['ml_similarity_score']:.3f})")
    else:
        print("  No ML normalization results to display (missing PDCA fields)")
    
    print(f"\n📋 Full results saved to: {output_file}")
    print(f"📊 Cluster report saved to: {cluster_report_file}")
    print(f"📈 Stats saved to: {stats_file}\n")
    
    return processed_df

if __name__ == "__main__":
    main()