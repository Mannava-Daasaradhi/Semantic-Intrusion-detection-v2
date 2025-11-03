# src/layer_5_dual_modality_learning/run_dual_modality_training.py
# (MODIFIED to drop saddr/daddr before training)
import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_confusion_matrix(y_true, y_pred, classes, model_name, output_path):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix for {model_name} saved to '{output_path}'")


def run_dual_modality_training(input_file, embeddings_file, label_map_file, model_out, history_out, plot_out):
    """
    Implements Phase 5.
    MODIFIED: Drops saddr/daddr as they are now graph features.
    """
    print("--- Starting Phase 5: Dual Modality Training ---")

    print(f"Loading main data from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    
    print(f"Loading embeddings from {embeddings_file}...")
    embeddings_df = pd.read_csv(embeddings_file)
    
    df['flow_id'] = range(len(df))

    print("Merging data...")
    df = pd.merge(df, embeddings_df, on='flow_id', how='left')
    
    embedding_cols = [col for col in df.columns if 'hgnn_emb_' in col]
    df[embedding_cols] = df[embedding_cols].fillna(0)
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # --- THIS IS THE FIX ---
    # Drop the original IP columns, as their info is now in the embeddings
    cols_to_drop = ['flow_id', 'timestamp', 'label_text', 'attack_tactic', 'saddr', 'daddr']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    # -----------------------
    
    X = df.drop(columns=['label'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    print("Encoding categorical features...")
    for col in categorical_cols:
        codes, uniques = pd.factorize(X_train[col])
        X_train[col] = codes
        cat_type = pd.CategoricalDtype(categories=uniques)
        X_test[col] = X_test[col].astype(cat_type).cat.codes
    
    print("Skipping scaling (already done in Phase 1).")
    
    print(f"Loading label mapping from {label_map_file}...")
    with open(label_map_file, 'r') as f:
        label_mapping = json.load(f)
    classes = [label_mapping[str(i)] for i in sorted(int(k) for k in label_mapping.keys())]
    num_classes = len(classes)

    print("\n--- Training Regularized XGBoost Model ---")
    
    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        eval_metric=['mlogloss', 'merror'],
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        device='cuda',
        early_stopping_rounds=100
    )

    xgb_model.fit(
        X_train_xgb, y_train_xgb,
        eval_set=[(X_train_xgb, y_train_xgb), (X_val_xgb, y_val_xgb)],
        verbose=False
    )
    
    evals_result = xgb_model.evals_result()

    os.makedirs(os.path.dirname(history_out), exist_ok=True)
    with open(history_out, 'w') as f:
        json.dump(evals_result, f, indent=4)
    print(f"Training history saved to '{history_out}'")

    print(f"\nModel stopped at iteration {xgb_model.best_iteration} out of 1000.")
    y_pred_xgb = xgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"XGBoost Internal Test Accuracy: {accuracy_xgb:.4f}")
    
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    xgb_model.save_model(model_out)
    print(f"Trained model saved to: '{model_out}'")
    
    plot_confusion_matrix(y_test, y_pred_xgb, classes, "XGBoost_Fused_UNSW_Regularized", plot_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: Dual Modality Training")
    parser.add_argument('--input', type=str, required=True, help="Path to the enriched CSV from Phase 3.")
    parser.add_argument('--embeddings', type=str, required=True, help="Path to the HGNN embeddings CSV from Phase 4.")
    parser.add_argument('--label_map', type=str, required=True, help="Path to the label mapping JSON (from Phase 1).")
    parser.add_argument('--model_out', type=str, required=True, help="Path to save the trained XGBoost model.")
    parser.add_argument('--history_out', type=str, required=True, help="Path to save the training history JSON.")
    parser.add_argument('--plot_out', type=str, required=True, help="Path to save the confusion matrix plot.")
    args = parser.parse_args()

    run_dual_modality_training(
        args.input, 
        args.embeddings, 
        args.label_map, 
        args.model_out, 
        args.history_out, 
        args.plot_out
    )