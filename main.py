from parser_new import ReviewParser

def main():
    """
    Main function to run the ReviewParser on the specified CSV file
    """
    # Set the path to your CSV file
    csv_file_path = r"data\decipher_retail_grocery_analytics.csv"
    
    print("🚀 Starting ReviewParser...")
    print(f"📁 Processing file: {csv_file_path}")
    
    try:
        # Initialize the parser with the file path
        parser = ReviewParser(csv_file_path, max_days=75)
        
        # Parse the data
        print("📊 Parsing review data...")
        parser.parse()
        
        # Check if parsing was successful
        if parser.output_df is not None:
            print(f"✅ Successfully parsed {len(parser.output_df)} records")
            print(f"📈 DataFrame shape: {parser.output_df.shape}")
            
            # Show first few rows
            print("\n🔍 First 3 rows of parsed data:")
            print(parser.output_df.head(3))
            
            # Save the flattened data
            print("\n💾 Saving flattened data...")
            output_path = parser.save()
            
            print(f"🎉 Process completed successfully!")
            print(f"📄 Output saved to: {output_path}")
            
        else:
            print("❌ No data was parsed. Please check your input file format.")
            
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file_path}")
        print("Please check the file path and ensure the file exists.")
        
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        print("Please check your input file format and try again.")

if __name__ == "__main__":
    main()