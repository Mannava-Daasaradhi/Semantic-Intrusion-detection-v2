# src/layer_1_data_collection/combine_final_data.py
import pandas as pd
import os

def final_safe_merge():
    """
    Safely combines the two processed datasets into one final file.
    """
    print("--- Starting Final Safe Merge ---")

    # --- Configuration ---
    mon_wed_file = 'data/processed/monday_to_wednesday_enriched.csv'
    thu_fri_file = 'data/processed/thursday_friday_enriched.csv'
    final_output = 'data/processed/CIC-IDS-2017_Enriched_COMPLETE.csv'

    # --- File Checks ---
    if not os.path.exists(mon_wed_file) or not os.path.exists(thu_fri_file):
        print("Error: One or both of the required input files are missing.")
        return

    # --- Load DataFrames ---
    print(f"Loading Monday-Wednesday data from: {mon_wed_file}")
    df_mon_wed = pd.read_csv(mon_wed_file, low_memory=False)
    
    print(f"Loading Thursday-Friday data from: {thu_fri_file}")
    df_thu_fri = pd.read_csv(thu_fri_file, low_memory=False)

    # --- Combine the two correct datasets ---
    print("Combining the two datasets...")
    final_df = pd.concat([df_mon_wed, df_thu_fri], ignore_index=True)

    # --- Save Final File ---
    final_df.to_csv(final_output, index=False)

    print("\n--- Final Merge COMPLETE ---")
    print(f"Final dataset has {len(final_df)} rows.")
    print(f"Saved to: '{final_output}'")

if __name__ == "__main__":
    final_safe_merge()