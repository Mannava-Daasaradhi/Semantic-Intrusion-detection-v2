# src/layer_7_EVALUATION/run_unsw_validation.py
# (MODIFIED to drop saddr/daddr)

import pandas as pd
import numpy as np
import json
import os
import xgboost as xgb
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TEST_DATA_PATH = 'data/processed/unsw_processed_test.csv'
XGB_MODEL_PATH = 'models/xgboost_model_regularized_unsw.json'
LABEL_MAPPING_PATH = 'models/label_mapping_unsw.json'

RESULTS_DIR = 'results/unsw_validation/'
REPORT_PATH = os.path.join(RESULTS_DIR, 'unsw_validation_report.txt')
CM_PATH = os.path.join(RESULTS_DIR, 'unsw_validation_confusion_matrix.png')

# --- HGNN Model Definition ---
class HGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x_embedding = self.conv2(x, edge_index).relu()
        x = self.dropout(x_embedding)
        x = self.conv3(x, edge_index)
        return x, x_embedding
# --- End of Model Definition ---

def load_models_and_data():
    """Loads test data, label mapping, and trained models."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    logging.info(f"Loading test data from {TEST_DATA_PATH}...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    logging.info("Preprocessing test data...")
    
    embeddings_file = 'data/processed/hgnn_embeddings_TEST.csv'

    if not os.path.exists(embeddings_file):
        logging.error(f"Embeddings file not found: {embeddings_file}")
        logging.error("Please run the 'generate_test_embeddings.py' script first.")
        return None, None, None, None
        
    embeddings_df = pd.read_csv(embeddings_file)
    
    if 'flow_id' not in test_df.columns:
         test_df['flow_id'] = range(len(test_df))

    test_df = pd.merge(test_df, embeddings_df, on='flow_id', how='left')
    
    embedding_cols = [col for col in test_df.columns if 'hgnn_emb_' in col]
    test_df[embedding_cols] = test_df[embedding_cols].fillna(0)
    
    test_df.columns = [col.strip().lower().replace(' ', '_') for col in test_df.columns]
    
    # --- THIS IS THE FIX ---
    # Drop the original IP columns, as their info is now in the embeddings
    cols_to_drop = ['flow_id', 'timestamp', 'saddr', 'daddr']
    X_test = test_df.drop(columns=['label'], errors='ignore')
    X_test = X_test.drop(columns=cols_to_drop, errors='ignore')
    # -----------------------
    
    y_test = test_df['label']

    categorical_cols = X_test.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        X_test[col] = pd.factorize(X_test[col])[0]
        
    logging.info("Skipping scaling (already done in Phase 1).")
    
    logging.info(f"Loading label mapping from {LABEL_MAPPING_PATH}...")
    with open(LABEL_MAPPING_PATH, 'r') as f:
        label_mapping = json.load(f)
    labels = [label_mapping[str(i)] for i in sorted(int(k) for k in label_mapping.keys())]
    
    logging.info(f"Loading XGBoost model from {XGB_MODEL_PATH}...")
    xgb_model = xgb.Booster()
    xgb_model.load_model(XGB_MODEL_PATH)
    
    logging.info("All artifacts loaded.")
    return X_test, y_test, xgb_model, labels

def evaluate_model(X_test, y_test, xgb_model, labels):
    """Runs predictions and generates evaluation metrics."""
    logging.info("Running predictions on UNSW-NB15 test set...")
    
    try:
        train_cols = xgb_model.feature_names
        X_test = X_test[train_cols]
    except Exception as e:
        logging.warning(f"Could not align feature names: {e}. Using positional matching.")
        train_cols_set = set(xgb_model.feature_names)
        test_cols_set = set(X_test.columns)
        
        missing_in_test = list(train_cols_set - test_cols_set)
        extra_in_test = list(test_cols_set - train_cols_set)
        
        logging.warning(f"Missing in test (will be filled with 0): {missing_in_test}")
        for col in missing_in_test:
            X_test[col] = 0
            
        logging.warning(f"Extra in test (will be dropped): {extra_in_test}")
        X_test = X_test.drop(columns=extra_in_test, errors='ignore')
        
        X_test = X_test[xgb_model.feature_names] # Re-order
        
    dtest = xgb.DMatrix(X_test)
    y_pred_xgb_prob = xgb_model.predict(dtest)
    
    if y_pred_xgb_prob.ndim > 1:
        y_pred = np.argmax(y_pred_xgb_prob, axis=1)
    else:
        y_pred = (y_pred_xgb_prob > 0.5).astype(int)
    
    logging.info("Generating classification report...")
    report_str = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    report_str = f"--- UNSW-NB15 Validation Report ---\n\nOverall Accuracy: {accuracy:.4f}\n\n{report_str}"
    
    print(report_str)
    
    with open(REPORT_PATH, 'w') as f:
        f.write(report_str)
    logging.info(f"Saved validation report to {REPORT_PATH}")

    logging.info("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels)
    plt.title('UNSW-NB15 Validation - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(CM_PATH)
    logging.info(f"Saved confusion matrix to {CM_PATH}")

def main():
    """Main execution function."""
    try:
        X_test, y_test, xgb_model, labels = load_models_and_data()
        if X_test is not None:
            evaluate_model(X_test, y_test, xgb_model, labels)
            logging.info("--- UNSW-NB15 Validation Complete ---")
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        logging.error("One or more required files were not found. Did you run the full pipeline (Phases 1-5) on the UNSW data first?")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()