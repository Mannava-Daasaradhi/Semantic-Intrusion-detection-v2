# src/layer_3_knowledge_and_ontology/run_knowledge_graph_and_reasoning.py
import pandas as pd
import os
import json

def apply_ontology_mapping(df, ontology):
    """
    Applies security tactics from a structured ontology to the dataframe
    based on the 'label' column.
    """
    print("Applying security ontology mapping...")

    # Create a mapping dictionary from the ontology for faster lookups
    # This handles partial matches for labels like 'DoS GoldenEye', 'DDoS', etc.
    label_to_tactic = {}
    for key, value in ontology.items():
        tactic_info = f"{value['tactic']}: {value['description']} ({value['technique_id']})"
        label_to_tactic[key] = tactic_info

    # Initialize the new feature column
    df['attack_tactic'] = 'Not Applicable'

    # Iterate through the ontology keys to apply mappings
    for label_key, tactic_value in label_to_tactic.items():
        # Use str.contains for partial matching (e.g., 'DoS' matches 'DoS Hulk')
        mask = df['label'].str.contains(label_key, case=False, na=False)
        df.loc[mask, 'attack_tactic'] = tactic_value

    identified_count = len(df[df['attack_tactic'] != 'Not Applicable'])
    print(f" - Identified tactics for {identified_count} flows.")

    return df

def run_reasoning_engine():
    """
    Implements Phase 3: Applies a knowledge-based reasoning engine to
    enrich the data with inferred threat intelligence (attack tactics).
    MODIFIED: Reads from a dedicated ontology file.
    """
    print("--- Starting Phase 3: Knowledge Graph and Reasoning Engine ---")

    # --- Configuration ---
    input_file = 'data/processed/model_ready_semantic_data.csv'
    output_file = 'data/processed/reasoning_enriched_data.csv'
    ontology_file = 'src/layer_3_knowledge_and_ontology/attack_ontology.json' # <-- NEW

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    if not os.path.exists(ontology_file):
        print(f"Error: Ontology file '{ontology_file}' not found.")
        return

    # --- 1. Load Data and Ontology ---
    print(f"Loading data from '{input_file}'...")
    df = pd.read_csv(input_file, low_memory=False)

    print(f"Loading ATT&CK ontology from '{ontology_file}'...")
    with open(ontology_file, 'r') as f:
        attack_ontology = json.load(f)

    # --- 2. Standardize Column Names ---
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # --- 3. Apply Ontology Mapping ---
    df_enriched = apply_ontology_mapping(df, attack_ontology)

    # --- 4. Save Enriched Data ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_enriched.to_csv(output_file, index=False)

    print("\n--- Phase 3: Reasoning Engine COMPLETE ---")
    print(f"Data enriched with attack tactics saved to: '{output_file}'")


if __name__ == "__main__":
    run_reasoning_engine()