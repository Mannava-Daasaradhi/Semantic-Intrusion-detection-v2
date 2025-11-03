# src/layer_3_knowledge_and_ontology/run_knowledge_graph_and_reasoning.py
# (MODIFIED to fix the 'KeyError: Label' bug)
import pandas as pd
import os
import json
import argparse

def apply_ontology_mapping(df, ontology):
    """
    Applies security tactics from a structured ontology to the dataframe
    based on the 'label' column.
    """
    print("Applying security ontology mapping...")
    
    try:
        with open('models/label_mapping_unsw.json', 'r') as f:
            label_mapping = json.load(f)
        label_mapping = {int(k): v for k, v in label_mapping.items()}
    except FileNotFoundError:
        print("Error: 'models/label_mapping_unsw.json' not found.")
        print("Please run Phase 1 script first to generate this file.")
        return None

    # --- THIS IS THE FIX ---
    # The 'Label' column was converted to 'label' by the line in run_reasoning_engine
    # We must use the lowercase 'label' here.
    df['label_text'] = df['label'].map(label_mapping)
    # -----------------------
    
    df['label_text'] = df['label_text'].fillna('Normal')

    label_to_tactic = {}
    for key, value in ontology.items():
        tactic_info = f"{value['tactic']}: {value['description']} ({value['technique_id']})"
        label_to_tactic[key] = tactic_info

    df['attack_tactic'] = 'Not Applicable'

    for label_key, tactic_value in label_to_tactic.items():
        mask = df['label_text'].str.contains(label_key, case=False, na=False)
        df.loc[mask, 'attack_tactic'] = tactic_value

    identified_count = len(df[df['attack_tactic'] != 'Not Applicable'])
    print(f" - Identified tactics for {identified_count} flows.")

    return df

def run_reasoning_engine(input_file, output_file):
    """
    Implements Phase 3: Applies a knowledge-based reasoning engine to
    enrich the data with inferred threat intelligence (attack tactics).
    """
    print("--- Starting Phase 3: Knowledge Graph and Reasoning Engine ---")

    ontology_file = 'src/layer_3_knowledge_and_ontology/attack_ontology.json'

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    if not os.path.exists(ontology_file):
        print(f"Error: Ontology file '{ontology_file}' not found.")
        return

    print(f"Loading data from '{input_file}'...")
    df = pd.read_csv(input_file, low_memory=False)

    print(f"Loading ATT&CK ontology from '{ontology_file}'...")
    with open(ontology_file, 'r') as f:
        attack_ontology = json.load(f)

    # This line converts 'Label' to 'label'
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    df_enriched = apply_ontology_mapping(df, attack_ontology)
    if df_enriched is None:
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_enriched.to_csv(output_file, index=False)

    print("\n--- Phase 3: Reasoning Engine COMPLETE ---")
    print(f"Data enriched with attack tactics saved to: '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Knowledge Graph and Reasoning Engine")
    parser.add_argument('--input', type=str, required=True, help="Path to the input processed CSV file (from Phase 1).")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output enriched CSV file.")
    args = parser.parse_args()

    run_reasoning_engine(args.input, args.output)