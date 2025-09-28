# src/layer_6_explanation_layer/run_llm_explanation.py
import pandas as pd
import os
import json
import xgboost as xgb
from transformers import pipeline, logging

# Suppress verbose logging from transformers
logging.set_verbosity_error()

def run_llm_explanation():
    """
    Implements Phase 6: Uses a trained model and an LLM to generate a
    human-readable explanation for a detected attack.
    CORRECTED: Uses standardized column names and label values to find samples correctly.
    """
    print("--- Starting Phase 6: Semantic-Aware Explanation Layer (LLM) ---")

    # --- Configuration ---
    input_file = 'data/processed/reasoning_enriched_data.csv'
    model_file = 'models/xgboost_model.json'
    label_mapping_file = 'models/label_mapping.json'

    # --- File Checks ---
    if not all(os.path.exists(f) for f in [input_file, model_file, label_mapping_file]):
        print("Error: One or more required files (dataset, model, label mapping) not found.")
        print("Please ensure Phases 3, 4, and 5 were completed successfully.")
        return

    # --- 1. Load Model, Data, and Mappings ---
    print("Loading trained model, data, and label mappings...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_file)

    df = pd.read_csv(input_file, low_memory=False)
    
    # --- FIX: Standardize column names and label values for consistency ---
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    df['label'] = df['label'].str.lower()
    
    df_attacks = df[df['label'] != 'benign']
    
    with open(label_mapping_file, 'r') as f:
        label_mapping = json.load(f)

    # --- 2. Select a Sample Attack to Explain ---
    print("\nSelecting a sample attack to explain...")
    
    # Use the standardized label 'web attack xss'
    sample_attack = df_attacks[df_attacks['label'] == 'web attack xss'].sample(n=1, random_state=42)
    if sample_attack.empty:
        print("Could not find a 'web attack xss' sample, using a random attack instead.")
        sample_attack = df_attacks.sample(n=1, random_state=42)

    # Reload the original, un-standardized row to display original case values in the prompt
    original_df = pd.read_csv(input_file, low_memory=False)
    sample_attack_original_case = original_df.iloc[sample_attack.index]

    print(f" - Explaining a '{sample_attack['label'].iloc[0]}' event.")
    
    # --- 3. Construct a Detailed Prompt for the LLM ---
    print("Constructing a detailed prompt for the LLM...")
    
    feature_importances = xgb_model.get_booster().get_score(importance_type='weight')
    top_features = sorted(feature_importances, key=feature_importances.get, reverse=True)[:5]
    
    # Find original column names for display
    original_source_ip_col = next((col for col in original_df.columns if col.lower().strip().replace(' ', '_') == 'source_ip'), 'source_ip')
    original_dest_ip_col = next((col for col in original_df.columns if col.lower().strip().replace(' ', '_') == 'destination_ip'), 'destination_ip')

    prompt = f"""
    Explain why the following network traffic is malicious in a single, concise sentence.
    This is for a security alert dashboard. Be direct and clear.

    Context:
    - Attack Type Detected: {sample_attack['label'].iloc[0]}
    - Source IP: {sample_attack_original_case[original_source_ip_col].iloc[0]}
    - Destination IP: {sample_attack_original_case[original_dest_ip_col].iloc[0]}
    - Protocol Name: {sample_attack['protocol_name'].iloc[0]}
    - Traffic Intent: {sample_attack['traffic_intent'].iloc[0]}
    - Inferred Tactic: {sample_attack['attack_tactic'].iloc[0]}

    Key Indicators (Top 5 Features and their values):
    """
    for feature in top_features:
        if feature in sample_attack.columns:
            feature_val = sample_attack[feature].iloc[0]
            prompt += f"- {feature}: {feature_val:.2f}\n"

    prompt += "\nExplanation:"

    print("--- Generated Prompt ---")
    print(prompt)
    print("----------------------")

    # --- 4. Generate the Explanation with an LLM ---
    print("\nGenerating explanation with a pre-trained LLM (distilgpt2)...")
    generator = pipeline('text-generation', model='distilgpt2')
    
    generated_text = generator(prompt, max_length=len(prompt.split()) + 60, num_return_sequences=1, pad_token_id=generator.tokenizer.eos_token_id)
    
    full_explanation = generated_text[0]['generated_text']
    final_explanation = full_explanation.split("Explanation:")[1].strip().split('\n')[0]

    print("\n--- Generated Explanation ---")
    print(final_explanation)
    print("-----------------------------")
    
    print("\n--- Phase 6 Complete ---")

if __name__ == "__main__":
    run_llm_explanation()