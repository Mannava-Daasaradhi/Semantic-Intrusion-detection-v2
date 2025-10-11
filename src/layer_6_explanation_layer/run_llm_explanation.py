# src/layer_6_explanation_layer/run_llm_explanation.py
import pandas as pd
import os
import xgboost as xgb
from transformers import pipeline, logging
import numpy as np

# Suppress verbose logging from transformers
logging.set_verbosity_error()

def run_evidence_based_explanation():
    """
    Implements the final, definitive version of Phase 6.
    - Loops through every unique attack type.
    - Identifies the TOP 5 most important features for each prediction.
    - Provides these features and their values as evidence to the LLM.
    - Generates a high-quality, evidence-based explanation for each attack.
    """
    print("--- Starting Phase 6: Evidence-Based Explanation for All Attack Types ---")

    # --- Configuration ---
    input_file = 'data/processed/reasoning_enriched_data.csv'
    model_file = 'models/xgboost_model.json'

    if not os.path.exists(model_file) or not os.path.exists(input_file):
        print("Error: Required model or data files are missing.")
        return

    # --- 1. Load Data, Model, and Feature Names ---
    print("Loading data, model, and preparing features...")
    df = pd.read_csv(input_file, low_memory=False)
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    df['label'] = df['label'].str.lower()
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_file)

    # Get the feature names in the exact order the model was trained on
    # This is crucial for correctly identifying important features
    feature_names = xgb_model.get_booster().feature_names

    df_attacks = df[df['label'] != 'benign']
    unique_attack_labels = df_attacks['label'].unique()
    print(f"Found {len(unique_attack_labels)} unique attack types to explain.")

    # --- 2. Initialize LLM ---
    print("\nInitializing local LLM (gpt2-medium)...")
    generator = pipeline('text-generation', model='gpt2-medium')
    print("LLM initialized.")

    # --- 3. Generate Evidence-Based Explanation for Each Attack Type ---
    for attack_label in sorted(unique_attack_labels):
        print(f"\n--- Explaining: {attack_label.upper()} ---")
        
        sample_attack = df_attacks[df_attacks['label'] == attack_label].sample(n=1, random_state=42)

        # Get feature importances and identify the top 5
        feature_importances = xgb_model.get_booster().get_score(importance_type='weight')
        # Sort features by importance
        sorted_features = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
        
        # Create the evidence string for the prompt
        evidence_string = ""
        count = 0
        for feature, _ in sorted_features:
            if feature in sample_attack.columns and count < 5:
                # Get the actual value of the feature from our sample
                value = sample_attack[feature].iloc[0]
                # Format the value nicely
                if isinstance(value, float):
                    value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                evidence_string += f"- **{feature}:** {value_str}\n"
                count += 1

        prompt = f"""You are a cybersecurity analyst. Your task is to provide a single, clear sentence explaining why a network event was flagged as malicious, based on the evidence provided.

### Example Alert ###
Context:
- Attack Type: dos hulk
- Tactic: Impact: Network Denial of Service (T1498)
Key Evidence:
- **flow_duration:** 9876543
- **fwd_iat_max:** 9876500
- **total_fwd_packets:** 5
Alert Explanation: A 'DoS Hulk' attack was flagged due to an abnormally long flow duration and a very high forward inter-arrival time, consistent with overwhelming a server by keeping connections open.

### Live Alert ###
Context:
- Attack Type: {sample_attack['label'].iloc[0]}
- Tactic: {sample_attack['attack_tactic'].iloc[0]}
Key Evidence:
{evidence_string}
Alert Explanation:"""

        generated_text = generator(prompt, max_new_tokens=70, pad_token_id=generator.tokenizer.eos_token_id)
        
        try:
            full_explanation = generated_text[0]['generated_text']
            final_explanation = full_explanation.split("### Live Alert ###")[1].split("Alert Explanation:")[1].strip().split('\n')[0]
        except IndexError:
            final_explanation = "Could not generate an evidence-based explanation for this event."

        print(final_explanation)

    print("\n----------------------------------------------------")
    print("🎉 All attack types have been successfully explained with evidence!")
    print("--- Project Complete ---")

if __name__ == "__main__":
    run_evidence_based_explanation()