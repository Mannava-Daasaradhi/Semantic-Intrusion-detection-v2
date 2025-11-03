# src/layer_4_HGNN/generate_test_embeddings.py
# (MODIFIED to use 'saddr' and 'daddr')
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
import numpy as np
import argparse
import json
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- HGNN Model Definition (Must match run_hgnn_training.py) ---
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

def build_test_graph(test_data_file, label_map_file):
    """Loads test data and builds a heterogeneous graph."""
    
    logging.info(f"Loading test data from {test_data_file}...")
    df = pd.read_csv(test_data_file, low_memory=False)
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    with open(label_map_file, 'r') as f:
        label_mapping = json.load(f)
    num_classes = len(label_mapping)
    
    logging.info("Encoding nodes and preparing features for test data...")
    
    # --- THIS IS THE FIX ---
    if 'saddr' not in df.columns or 'daddr' not in df.columns:
        logging.error("Critical Error: 'saddr' or 'daddr' not found. Cannot build graph.")
        return None, None, None
    
    all_ips = pd.concat([df['saddr'], df['daddr']]).unique()
    ip_encoder = LabelEncoder()
    ip_encoder.fit(all_ips)
    df['src_ip_encoded'] = ip_encoder.transform(df['saddr'])
    df['dst_ip_encoded'] = ip_encoder.transform(df['daddr'])
    # -----------------------
    
    if 'flow_id' not in df.columns:
        df['flow_id'] = range(len(df))
        
    flow_encoder = LabelEncoder()
    df['flow_id_encoded'] = flow_encoder.fit_transform(df['flow_id'])
    
    feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    exclude_cols = ['label', 'src_ip_encoded', 'dst_ip_encoded', 'flow_id_encoded']
    feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    flow_features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    flow_labels = torch.tensor(df['label'].values, dtype=torch.long)

    logging.info("Constructing the test heterogeneous graph...")
    data = HeteroData()
    num_ips = len(ip_encoder.classes_)
    logging.info(f" - Found {num_ips} unique IP nodes in test set.")
    
    data['ip'].x = torch.randn(num_ips, 16)
    data['flow'].x = flow_features
    data['flow'].y = flow_labels
    data['flow'].num_nodes = len(df)
    src_to_flow_edge_index = torch.tensor([df['src_ip_encoded'].values, df['flow_id_encoded'].values], dtype=torch.long)
    flow_to_dst_edge_index = torch.tensor([df['flow_id_encoded'].values, df['dst_ip_encoded'].values], dtype=torch.long)
    data['ip', 'initiates', 'flow'].edge_index = src_to_flow_edge_index
    data['flow', 'targets', 'ip'].edge_index = flow_to_dst_edge_index
    
    logging.info(f"Test graph constructed: {data}")
    return data, df['flow_id'], num_classes

def generate_embeddings(model_path, test_data_file, label_map_file, embeddings_out_file):
    """Loads the trained model and generates embeddings for the test data."""
    
    test_graph_data, flow_ids, num_classes = build_test_graph(test_data_file, label_map_file)
    if test_graph_data is None:
        return # Error already logged

    logging.info(f"Loading trained HGNN model from {model_path}...")
    model = HGNN(hidden_channels=128, out_channels=num_classes)
    model = to_hetero(model, test_graph_data.metadata(), aggr='sum')
    
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_path}")
        return
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
        
    model.eval()
    logging.info("Model loaded successfully.")

    logging.info("Generating embeddings for test data...")
    with torch.no_grad():
        _, flow_embeddings_tensor = model(test_graph_data.x_dict, test_graph_data.edge_index_dict)
        flow_embeddings = flow_embeddings_tensor['flow'].cpu().numpy()
    
    embedding_df = pd.DataFrame(flow_embeddings, columns=[f'hgnn_emb_{i}' for i in range(flow_embeddings.shape[1])])
    embedding_df['flow_id'] = flow_ids.values
    embedding_df.to_csv(embeddings_out_file, index=False)
    
    logging.info(f"--- Test Embeddings Generation Complete ---")
    logging.info(f"Saved {len(embedding_df)} test embeddings to '{embeddings_out_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4.5: Generate HGNN Embeddings for Test Data")
    parser.add_argument('--model_in', type=str, required=True, help="Path to the trained HGNN model (.pth).")
    parser.add_argument('--test_data_in', type=str, required=True, help="Path to the processed test CSV (from Phase 1).")
    parser.add_argument('--label_map', type=str, required=True, help="Path to the label mapping JSON.")
    parser.add_argument('--embeddings_out', type=str, required=True, help="Path to save the output test embeddings CSV.")
    args = parser.parse_args()

    generate_embeddings(args.model_in, args.test_data_in, args.label_map, args.embeddings_out)