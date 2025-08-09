import pandas as pd
from rapidfuzz import fuzz, process
import os

def canonicalize_experience_drivers(input_file_path, output_dir="outputs", threshold=80):
    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"[INFO] Loading data from: {input_file_path}")
        df = pd.read_csv(input_file_path)

        if "experience_driver" not in df.columns:
            raise ValueError("Missing required column: 'experience_driver'")

        unique_entities = df["experience_driver"].dropna().unique()
        entity_counts = df["experience_driver"].value_counts()

        print(f"[INFO] Found {len(unique_entities)} unique experience drivers")
        print("[INFO] Performing fuzzy canonicalization...")

        canonical_map = {}
        similarity_scores = {}
        processed = set()

        for i, entity in enumerate(unique_entities):
            if entity in processed:
                continue
            if i % 10 == 0:
                print(f"  - Processed {i}/{len(unique_entities)} entities")

            matches = process.extract(
                query=entity,
                choices=unique_entities,
                scorer=fuzz.token_set_ratio,
                score_cutoff=threshold
            )

            if matches:
                match_entities = [match[0] for match in matches]
                canonical_name = max(match_entities, key=lambda x: entity_counts.get(x, 0))
                for match, score, _ in matches:
                    canonical_map[match] = canonical_name
                    similarity_scores[match] = score
                    processed.add(match)
            else:
                canonical_map[entity] = entity
                similarity_scores[entity] = 100.0
                processed.add(entity)

        df["experience_driver_canonical"] = df["experience_driver"].map(canonical_map)
        df["similarity_score"] = df["experience_driver"].map(similarity_scores)

        canonical_df = pd.DataFrame([
            {
                "original_entity_type": orig,
                "canonical_entity_type": canon,
                "similarity_score": similarity_scores[orig],
                "frequency": entity_counts.get(orig, 0),
                "needs_review": similarity_scores[orig] < 85
            }
            for orig, canon in canonical_map.items()
        ]).sort_values("frequency", ascending=False)

        original_count = len(unique_entities)
        canonical_count = len(canonical_df["canonical_entity_type"].unique())
        compression_ratio = original_count / canonical_count if canonical_count > 0 else 0

        base_name = os.path.basename(input_file_path).replace(".csv", "")
        df.to_csv(os.path.join(output_dir, f"{base_name}_canonicalized.csv"), index=False, encoding="utf-8")
        canonical_df.to_csv(os.path.join(output_dir, f"{base_name}_canonical_map.csv"), index=False, encoding="utf-8")

        stats_path = os.path.join(output_dir, f"{base_name}_explosion_stats.txt")
        with open(stats_path, 'w', encoding="utf-8") as f:
            f.write("EXPERIENCE DRIVER CANONICALIZATION STATS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Original unique experience drivers: {original_count}\n")
            f.write(f"Canonical unique experience drivers: {canonical_count}\n")
            f.write(f"Compression ratio: {compression_ratio:.2f}x\n")
            f.write(f"Total records processed: {len(df)}\n")
            f.write(f"Mappings requiring review: {canonical_df['needs_review'].sum()}\n")
            f.write(f"Fuzzy matching threshold: {threshold}%\n")

        print("\n[✓] Canonicalization completed successfully")
        print(f"Original Drivers:  {original_count}")
        print(f"Canonical Drivers: {canonical_count}")
        print(f"Compression Ratio: {compression_ratio:.2f}x")
        print(f"Needs Review:      {canonical_df['needs_review'].sum()}")
        print(f"Output directory:  {output_dir}\n")

        return {
            "original_unique_count": original_count,
            "canonical_unique_count": canonical_count,
            "compression_ratio": compression_ratio
        }

    except Exception as e:
        print(f"[ERROR] Canonicalization failed: {e}")
        return None


def main():
    input_file = "data/decipher_retail_grocery_analytics_flattened.csv"
    output_directory = "outputs"
    similarity_threshold = 85

    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}")
        return

    stats = canonicalize_experience_drivers(
        input_file_path=input_file,
        output_dir=output_directory,
        threshold=similarity_threshold
    )

    if stats:
        print(f"[✓] Explosion reduced from {stats['original_unique_count']} to {stats['canonical_unique_count']} drivers")
    else:
        print("[✗] Canonicalization failed.")

if __name__ == "__main__":
    main()
