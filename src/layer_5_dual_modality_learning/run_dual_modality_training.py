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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    """Visualizes the confusion matrix."""
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

def create_transformer_model(input_shape, num_classes):
    """Builds a lightweight Transformer model."""
    inputs = Input(shape=input_shape)
    x = Dense(128)(inputs)
    attn_output = MultiHeadAttention(num_heads=8, key_dim=128)(x, x)
    attn_output = Dropout(0.2)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = Dense(128, activation='relu')(out1)
    ffn_output = Dense(128)(ffn_output)
    ffn_output = Dropout(0.2)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    x = GlobalAveragePooling1D()(out2)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

def run_dual_modality_training():
    """
    Implements Phase 5 with a perfected pipeline that FUSES graph embeddings
    and includes memory optimization for large datasets.
    """
    print("--- Starting Phase 5: Dual-Modality Learning with Fused Features ---")

    # --- Configuration ---
    input_file = 'data/processed/reasoning_enriched_data.csv'
    embeddings_file = 'data/processed/hgnn_embeddings.csv'
    label_mapping_file = 'models/label_mapping.json'

    if not all(os.path.exists(f) for f in [input_file, embeddings_file]):
        print(f"Error: Required file(s) not found. Please ensure Phases 3 and 4 completed.")
        return

    # --- 1. Load, Sample, and Fuse Data ---
    print("Loading data, sampling, and fusing with HGNN embeddings...")
    df = pd.read_csv(input_file, low_memory=False)
    
    # --- FIX: Take a large, manageable sample to prevent memory errors ---
    if len(df) > 1000000:
        print(f"Full dataset has {len(df)} rows. Taking a sample of 1,000,000 for training.")
        df = df.sample(n=1000000, random_state=42)

    embeddings_df = pd.read_csv(embeddings_file)
    df = pd.merge(df, embeddings_df, on='flow_id', how='left')
    
    embedding_cols = [col for col in df.columns if 'hgnn_emb_' in col]
    df[embedding_cols] = df[embedding_cols].fillna(0)
    print(f" - Fusion complete. Dataset now has {df.shape[1]} columns.")
    
    # Standardize names and labels for consistency
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    df['label'] = df['label'].str.lower()
    
    # --- FIX: Optimize memory by downcasting data types ---
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    
    print("Removing identifier columns to prevent data leakage...")
    cols_to_drop_leakage = ['flow_id', 'source_ip', 'destination_ip', 'timestamp']
    df = df.drop(columns=cols_to_drop_leakage, errors='ignore')
    
    X = df.drop(columns=['label'])
    y = df['label']

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # --- 2. Prepare Data (Encoding and Scaling) ---
    print("Preparing data (encoding and scaling)...")
    
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
    
    label_mapping = {int(index): label for index, label in enumerate(le.classes_)}
    os.makedirs(os.path.dirname(label_mapping_file), exist_ok=True)
    with open(label_mapping_file, 'w') as f:
        json.dump(label_mapping, f, indent=4)

    print(f" - Data prepared: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- 3. Train XGBoost Model ---
    print("\n--- Training XGBoost Model on Fused Data ---")
    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_),
                                  eval_metric='mlogloss', n_estimators=150, max_depth=8, 
                                  learning_rate=0.1, device='cuda') # Use GPU if available

    xgb_model.fit(X_train, y_train, verbose=False)
    
    print("\n--- Evaluating XGBoost Model ---")
    y_pred_xgb = xgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"XGBoost Test Accuracy: {accuracy_xgb:.4f}")
    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred_xgb, target_names=le.classes_, zero_division=0))
    
    os.makedirs('models', exist_ok=True)
    xgb_model.save_model('models/xgboost_model.json')
    print("XGBoost model saved to 'models/xgboost_model.json'")
    plot_confusion_matrix(y_test, y_pred_xgb, le.classes_, "XGBoost_Fused")

    # --- 4. Train Transformer Model ---
    print("\n--- Training Transformer Model on Fused Data ---")
    X_train_tf = np.expand_dims(X_train.values, axis=1)
    X_test_tf = np.expand_dims(X_test.values, axis=1)
    y_train_categorical = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)

    transformer_model = create_transformer_model(input_shape=(1, X_train.shape[1]), num_classes=len(le.classes_))
    transformer_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    transformer_model.fit(X_train_tf, y_train_categorical, epochs=20, batch_size=512, validation_split=0.2, verbose=1)
    
    print("\n--- Evaluating Transformer Model ---")
    loss_tf, accuracy_tf = transformer_model.evaluate(X_test_tf, y_test_categorical, verbose=0)
    print(f"Transformer Test Accuracy: {accuracy_tf:.4f}")
    
    y_pred_tf_probs = transformer_model.predict(X_test_tf)
    y_pred_tf = np.argmax(y_pred_tf_probs, axis=1)
    print("Transformer Classification Report:")
    print(classification_report(y_test, y_pred_tf, target_names=le.classes_, zero_division=0))

    transformer_model.save('models/transformer_model.h5')
    print("Transformer model saved to 'models/transformer_model.h5'")
    plot_confusion_matrix(y_test, y_pred_tf, le.classes_, "Transformer_Fused")
    
    print("\n--- Phase 5 with Fused Features Complete ---")

if __name__ == "__main__":
    run_dual_modality_training()