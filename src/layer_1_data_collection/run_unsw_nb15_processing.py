# src/layer_1_data_collection/run_unsw_nb15_processing.py
# (MODIFIED: Removed saddr/daddr handling completely)

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- File Paths ---
RAW_DATA_DIR = 'data/unsw-nb15/'
TRAIN_FILE = 'UNSW_NB15_training-set.csv'
TEST_FILE = 'UNSW_NB15_testing-set.csv'

PROCESSED_DIR = 'data/processed/'
TRAIN_OUT_FILE = os.path.join(PROCESSED_DIR, 'unsw_processed_train.csv')
TEST_OUT_FILE = os.path.join(PROCESSED_DIR, 'unsw_processed_test.csv')
LABEL_MAPPING_OUT_FILE = 'models/label_mapping_unsw.json'

# --- Feature Engineering ---
CATEGORICAL_FEATURES = ['proto', 'service', 'state']
DROP_FEATURES = ['id', 'label'] # 'label' is binary, we use 'attack_cat' -> 'Label'

def load_data(file_path):
    """Loads a single CSV file."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    logging.info(f"Loading data from {file_path}...")
    # No need for dtype specification now
    return pd.read_csv(file_path)

def process_data(train_df, test_df):
    """Processes and transforms the train and test dataframes."""
    
    logging.info("Starting data processing...")
    
    combined_df = pd.concat([train_df, test_df], keys=['train', 'test'], sort=False)
    
    # Handle Labels
    combined_df['attack_cat'] = combined_df['attack_cat'].fillna('Normal').str.strip()
    le = LabelEncoder()
    combined_df['Label'] = le.fit_transform(combined_df['attack_cat'])
    
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
    logging.info(f"Generated {len(label_mapping)} labels: {label_mapping}")
    with open(LABEL_MAPPING_OUT_FILE, 'w') as f:
        json.dump(label_mapping, f, indent=4)
    logging.info(f"Saved new label mapping to {LABEL_MAPPING_OUT_FILE}")

    # Drop original ID and binary label
    combined_df = combined_df.drop(columns=DROP_FEATURES, errors='ignore')
    # Also drop original attack category name
    combined_df = combined_df.drop(columns=['attack_cat'], errors='ignore')
    
    logging.info(f"One-hot encoding features: {CATEGORICAL_FEATURES}")
    combined_df = pd.get_dummies(combined_df, columns=CATEGORICAL_FEATURES)
    
    # Separate Train and Test
    X_train_df = combined_df.loc['train']
    X_test_df = combined_df.loc['test']
    
    y_train = X_train_df['Label']
    y_test = X_test_df['Label']
    
    X_train = X_train_df.drop(columns=['Label'], errors='ignore')
    X_test = X_test_df.drop(columns=['Label'], errors='ignore')

    # Align columns after one-hot encoding
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)

    missing_in_test = list(train_cols - test_cols)
    for col in missing_in_test:
        X_test[col] = 0
        
    missing_in_train = list(test_cols - train_cols)
    for col in missing_in_train:
        X_train[col] = 0

    X_test = X_test[X_train.columns] # Ensure same column order
    logging.info(f"Feature alignment complete. Total features: {len(X_train.columns)}")

    # Scale Numerical Features
    numerical_cols = X_train.select_dtypes(include=np.number).columns
    # Exclude bool/int8 columns resulting from one-hot encoding
    numerical_cols = [col for col in numerical_cols if X_train[col].dtype != 'bool' and X_train[col].dtype != 'int8']
    
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    logging.info("Normalization complete.")

    # Rebuild final dataframes
    train_final = X_train
    train_final['Label'] = y_train
    
    test_final = X_test
    test_final['Label'] = y_test
    
    return train_final, test_final

def main():
    """Main execution function."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LABEL_MAPPING_OUT_FILE), exist_ok=True)
    
    train_df = load_data(os.path.join(RAW_DATA_DIR, TRAIN_FILE))
    test_df = load_data(os.path.join(RAW_DATA_DIR, TEST_FILE))
    
    if train_df is None or test_df is None:
        logging.error("Failed to load one or more data files. Exiting.")
        logging.info(f"Please ensure '{TRAIN_FILE}' and '{TEST_FILE}' are in the '{RAW_DATA_DIR}' directory.")
        return

    train_processed, test_processed = process_data(train_df, test_df)
    
    train_processed.to_csv(TRAIN_OUT_FILE, index=False)
    test_processed.to_csv(TEST_OUT_FILE, index=False)
    
    logging.info(f"Successfully processed and saved training data to {TRAIN_OUT_FILE}")
    logging.info(f"Successfully processed and saved testing data to {TEST_OUT_FILE}")
    logging.info("--- UNSW-NB15 Processing Complete ---")

if __name__ == "__main__":
    main()