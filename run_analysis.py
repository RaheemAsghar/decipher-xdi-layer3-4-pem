from cluster_compute_xdi import FlexibleTimeframeAnalyzer
import os

if __name__ == "__main__":
    input_path = os.path.join("data", "decipher_retail_grocery_analytics_flattened_canonicalized.csv")
    
    analyzer = FlexibleTimeframeAnalyzer(
        input_path=input_path,
        output_dir="outputs",
        timeframe_days=75,
        compute_granular=True,
        verbose=True
    )
    
    final_df = analyzer.run_analysis()
    print(final_df.head())
