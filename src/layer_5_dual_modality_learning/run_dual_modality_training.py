# src/layer_5_dual_modality_learning/run_dual_modality_training.py
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

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    # ... (function is unchanged)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    output_path = f'results/plots/{model_name}_confusion_matrix.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix for {model_name} saved to '{output_path}'")


def run_dual_modality_training_with_history():
    """
    Implements Phase 5 with anti-overfitting techniques and SAVES the training history.
    CORRECTED: Now compatible with different XGBoost versions.
    """
    print("--- Starting Phase 5: Training and Saving History ---")

    # --- Configuration ---
    input_file = 'data/processed/reasoning_enriched_data.csv'
    embeddings_file = 'data/processed/hgnn_embeddings.csv'
    history_file = 'results/xgboost_training_history.json'

    # ... (Data loading and preparation code is unchanged)
    df = pd.read_csv(input_file, low_memory=False)
    if len(df) > 1000000:
        df = df.sample(n=1000000, random_state=42)
    embeddings_df = pd.read_csv(embeddings_file)
    df = pd.merge(df, embeddings_df, on='flow_id', how='left')
    embedding_cols = [col for col in df.columns if 'hgnn_emb_' in col]
    df[embedding_cols] = df[embedding_cols].fillna(0)
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    df['label'] = df['label'].str.lower()
    cols_to_drop_leakage = ['flow_id', 'source_ip', 'destination_ip', 'timestamp']
    df = df.drop(columns=cols_to_drop_leakage, errors='ignore')
    X = df.drop(columns=['label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    for col in categorical_cols:
        codes, uniques = pd.factorize(X_train[col])
        X_train[col] = codes
        cat_type = pd.CategoricalDtype(categories=uniques)
        X_test[col] = X_test[col].astype(cat_type).cat.codes
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    # ... (End of unchanged data prep)


    # --- 3. Train XGBoost Model and Capture History ---
    print("\n--- Training XGBoost Model and Capturing History ---")
    
    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005, reg_lambda=0.1,
        device='cuda', early_stopping_rounds=50
    )

    # Fit the model (without the problematic argument)
    xgb_model.fit(
        X_train_xgb, y_train_xgb,
        eval_set=[(X_train_xgb, y_train_xgb), (X_val_xgb, y_val_xgb)],
        verbose=False
    )
    
    # THIS IS THE KEY CHANGE: Retrieve the history AFTER fitting
    evals_result = xgb_model.evals_result()

    # Save the captured history to a file
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(evals_result, f, indent=4)
    print(f"Training history saved to '{history_file}'")

    # --- 4. Evaluate and Save ---
    print(f"\nModel stopped at iteration {xgb_model.best_iteration} out of 1000.")
    y_pred_xgb = xgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"XGBoost Test Accuracy: {accuracy_xgb:.4f}")
    xgb_model.save_model('models/xgboost_model_regularized.json')
    plot_confusion_matrix(y_test, y_pred_xgb, le.classes_, "XGBoost_Fused_Regularized")


if __name__ == "__main__":
    run_dual_modality_training_with_history()