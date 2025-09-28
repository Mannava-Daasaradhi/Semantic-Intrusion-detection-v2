# src/layer_3_knowledge_and_ontology/run_knowledge_graph_and_reasoning.py
import pandas as pd
import os
import numpy as np

def apply_reasoning_rules(df):
    """
    Applies a set of security rules to the dataframe to infer potential
    attack tactics based on the MITRE ATT&CK framework. This adds a new
    'attack_tactic' feature column.
    """
    print("Applying security reasoning rules...")
    
    # Initialize the new feature column with a default value
    df['attack_tactic'] = 'Not Applicable'

    # --- Define Security Ontology Rules ---

    # Rule 1: Reconnaissance -> Port Scan (T1046)
    # The original label 'PortScan' indicates this tactic.
    portscan_mask = df['label'].str.contains('PortScan', case=False, na=False)
    df.loc[portscan_mask, 'attack_tactic'] = 'Reconnaissance: Port Scan (T1046)'
    
    # Rule 2: Credential Access -> Brute Force (T1110)
    # The labels 'FTP-Patator' and 'SSH-Patator' indicate brute force attacks.
    brute_force_mask = df['label'].str.contains('Patator', case=False, na=False)
    df.loc[brute_force_mask, 'attack_tactic'] = 'Credential Access: Brute Force (T1110)'

    # Rule 3: Credential Access -> Unsecured Credentials (T1552)
    # FTP traffic with a 'Login Attempt' intent is sending credentials in cleartext.
    unencrypted_login_mask = (df['protocol_name'] == 'FTP') & (df['traffic_intent'] == 'Login Attempt')
    df.loc[unencrypted_login_mask, 'attack_tactic'] = 'Credential Access: Unsecured Credentials (T1552)'
    
    # Rule 4: Execution -> Web Shell / Command and Scripting (T1059/T1505)
    # The labels 'Web Attack  Brute Force', 'XSS', and 'Sql Injection' are forms of execution.
    web_execution_mask = df['label'].str.contains('Web Attack', case=False, na=False)
    df.loc[web_execution_mask, 'attack_tactic'] = 'Execution: Command and Scripting (T1059)'

    # Rule 5: Initial Access -> Drive-by Compromise (T1189)
    # Infiltration attacks often start with drive-by compromises.
    infil_mask = df['label'].str.contains('Infilteration', case=False, na=False)
    df.loc[infil_mask, 'attack_tactic'] = 'Initial Access: Drive-by Compromise (T1189)'

    # Rule 6: Impact -> Network Denial of Service (T1498)
    # The various DoS and DDoS labels fall under this tactic.
    dos_mask = df['label'].str.contains('DoS|DDoS', case=False, na=False)
    df.loc[dos_mask, 'attack_tactic'] = 'Impact: Network Denial of Service (T1498)'

    # Set remaining 'BENIGN' traffic correctly
    benign_mask = df['label'].str.contains('BENIGN', case=False, na=False)
    df.loc[benign_mask, 'attack_tactic'] = 'Benign'

    identified_count = len(df[df['attack_tactic'] != 'Not Applicable'])
    print(f" - Identified tactics for {identified_count} flows.")
    
    return df

def run_reasoning_engine():
    """
    Implements Phase 3: Applies a knowledge-based reasoning engine to
    enrich the data with inferred threat intelligence (attack tactics).
    """
    print("--- Starting Phase 3: Knowledge Graph and Reasoning Engine ---")

    # --- Configuration ---
    input_file = 'data/processed/model_ready_semantic_data.csv'  # From Phase 2
    output_file = 'data/processed/reasoning_enriched_data.csv'   # For Phase 4/5
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        print("Please ensure Phase 2 (run_semantic_encoding.py) was completed successfully.")
        return
        
    # --- 1. Load Data ---
    print(f"Loading data from '{input_file}'...")
    df = pd.read_csv(input_file, low_memory=False)
    
    # --- 2. Standardize Column Names ---
    # This is a critical step for consistency across all scripts.
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # --- 3. Apply Reasoning Rules ---
    df_enriched = apply_reasoning_rules(df)
    
    # --- 4. Save Enriched Data ---
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_enriched.to_csv(output_file, index=False)
    
    print("\n--- Phase 3: Reasoning Engine COMPLETE ---")
    print(f"Data enriched with attack tactics saved to: '{output_file}'")


if __name__ == "__main__":
    run_reasoning_engine()