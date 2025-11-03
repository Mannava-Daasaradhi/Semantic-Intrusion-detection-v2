# src/layer_4_HGNN/run_hgnn_training.py
# (MODIFIED to use 'saddr' and 'daddr' to build a real graph)
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import RandomNodeSplit
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import argparse
import json
import logging # <-- NEW

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# (Your visualize_graph_structure function remains here)
def visualize_graph_structure(data, num_samples=30):
    logging.info("\n--- Generating a sample visualization of the graph structure ---")
    G = nx.Graph()
    flow_nodes_total = data['flow'].num_nodes
    if flow_nodes_total == 0:
        logging.info(" - No flow nodes to visualize.")
        return
    
    # --- Get IP node labels from the encoder ---
    try:
        ip_labels = data['ip'].encoder.classes_
    except AttributeError:
        logging.warning(" - IP encoder not found on graph data. Using raw indices.")
        ip_labels = [f"IP_{i}" for i in range(data['ip'].num_nodes)]

    sample_flow_indices = random.sample(range(flow_nodes_total), k=min(num_samples, flow_nodes_total))
    send_edges = data['ip', 'initiates', 'flow'].edge_index.T
    receive_edges = data['flow', 'targets', 'ip'].edge_index.T
    ip_nodes_in_sample = set()
    
    for flow_idx in sample_flow_indices:
        flow_node_label = f"Flow_{flow_idx}"
        
        for edge in send_edges:
            if edge[1] == flow_idx:
                ip_node_label = ip_labels[edge[0].item()] # Use label
                G.add_edge(ip_node_label, flow_node_label)
                ip_nodes_in_sample.add(ip_node_label)

        for edge in receive_edges:
            if edge[0] == flow_idx:
                ip_node_label = ip_labels[edge[1].item()] # Use label
                G.add_edge(flow_node_label, ip_node_label)
                ip_nodes_in_sample.add(ip_node_label)

    node_colors = ['skyblue' if 'IP' in node or node.count('.') == 3 else 'salmon' for node in G.nodes()]

    plt.figure(figsize=(20, 20)) # Make figure larger
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color=node_colors, font_size=8, font_weight='bold')
    plt.title(f"Sample of Heterogeneous Graph Structure ({num_samples} Flows)")
    
    save_path = "results/plots/hgnn_structural_sample_unsw.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    logging.info(f" - Graph structure sample saved to '{save_path}'")
    plt.close()

# (Your HGNN class definition remains here)
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

def run_hgnn_training(input_file, label_map_file, model_out, plot_out, embeddings_out):
    """
    MODIFIED to use 'saddr' and 'daddr' for a real graph.
    """
    logging.info("--- Starting Phase 4: Production HGNN Training & Embedding Generation ---")

    if not os.path.exists(input_file):
        logging.error(f"Error: Input file not found at '{input_file}'")
        return
    if not os.path.exists(label_map_file):
        logging.error(f"Error: Label mapping file not found at '{label_map_file}'")
        return

    logging.info("Loading and preparing data...")
    df_full = pd.read_csv(input_file, low_memory=False)
    df_full.columns = [col.strip().lower().replace(' ', '_') for col in df_full.columns]
    
    with open(label_map_file, 'r') as f:
        label_mapping = json.load(f)
    num_classes = len(label_mapping)
    logging.info(f" - Loaded {num_classes} classes from {label_map_file}")
    
    df = df_full.copy()
    logging.info(f" - Using {len(df)} rows for training.")

    logging.info("Encoding nodes and preparing features...")
    
    # --- THIS IS THE FIX ---
    # Use 'saddr' and 'daddr' from the UNSW-NB15 dataset
    if 'saddr' not in df.columns or 'daddr' not in df.columns:
        logging.error("Critical Error: 'saddr' or 'daddr' not found. Cannot build graph.")
        return
    
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
    # Exclude IPs, labels, and encodings from *flow* features
    exclude_cols = ['label', 'src_ip_encoded', 'dst_ip_encoded', 'flow_id_encoded']
    feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    if not feature_cols:
        logging.error("Error: No feature columns found! Check your processed CSV.")
        return
    
    logging.info(f" - Using {len(feature_cols)} features for the graph.")
    
    # Data is already scaled from Phase 1
    flow_features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    flow_labels = torch.tensor(df['label'].values, dtype=torch.long)

    logging.info("Constructing the heterogeneous graph...")
    data = HeteroData()
    num_ips = len(ip_encoder.classes_)
    logging.info(f" - Found {num_ips} unique IP nodes.") # This should be > 2 now
    
    data['ip'].x = torch.randn(num_ips, 16) # Dummy features for IP nodes
    data['ip'].encoder = ip_encoder # Store encoder for visualization
    data['flow'].x = flow_features
    data['flow'].y = flow_labels
    data['flow'].num_nodes = len(df)
    src_to_flow_edge_index = torch.tensor([df['src_ip_encoded'].values, df['flow_id_encoded'].values], dtype=torch.long)
    flow_to_dst_edge_index = torch.tensor([df['flow_id_encoded'].values, df['dst_ip_encoded'].values], dtype=torch.long)
    data['ip', 'initiates', 'flow'].edge_index = src_to_flow_edge_index
    data['flow', 'targets', 'ip'].edge_index = flow_to_dst_edge_index
    
    transform = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
    data = transform(data)
    logging.info(" - Graph constructed and data split:")
    logging.info(data)

    visualize_graph_structure(data) # Try to visualize the real graph

    model = HGNN(hidden_channels=128, out_channels=num_classes)
    model = to_hetero(model, data.metadata(), aggr='sum')
    
    logging.info("\n--- Training and Validating the HGNN model ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs_list, losses, val_accuracies = [], [], []

    for epoch in range(1, 101): # 100 epochs
        model.train()
        optimizer.zero_grad()
        out, _ = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out['flow'][data['flow'].train_mask], data['flow'].y[data['flow'].train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out_val, _ = model(data.x_dict, data.edge_index_dict)
                pred_val = out_val['flow'][data['flow'].val_mask].argmax(dim=-1)
                correct = (pred_val == data['flow'].y[data['flow'].val_mask]).sum()
                val_acc = int(correct) / data['flow'].val_mask.sum()
                logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}')
                epochs_list.append(epoch)
                losses.append(loss.item())
                val_accuracies.append(val_acc)

    logging.info("\n--- Final Evaluation on Unseen Test Data ---")
    model.eval()
    with torch.no_grad():
        out_test, _ = model(data.x_dict, data.edge_index_dict)
        pred_test = out_test['flow'][data['flow'].test_mask].argmax(dim=-1)
        correct = (pred_test == data['flow'].y[data['flow'].test_mask]).sum()
        test_acc = int(correct) / data['flow'].test_mask.sum()
        logging.info(f'HGNN Final Test Accuracy: {test_acc:.4f}')

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save(model.state_dict(), model_out)
    logging.info(f"Trained model saved to: '{model_out}'")

    logging.info("\n--- Generating Training History Plot ---")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.plot(epochs_list, np.array(losses), 'b-o', label='Training Loss')
    ax2.plot(epochs_list, np.array(val_accuracies), 'g-o', label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('Accuracy', color='g')
    plt.title('Training Loss and Validation Accuracy')
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_out), exist_ok=True)
    plt.savefig(plot_out)
    logging.info(f"Training history plot saved to: '{plot_out}'")

    logging.info("\n--- Generating Graph Embeddings for Phase 5 Fusion ---")
    model.eval()
    with torch.no_grad():
        _, flow_embeddings_tensor = model(data.x_dict, data.edge_index_dict)
        flow_embeddings = flow_embeddings_tensor['flow'].cpu().numpy()
    
    embedding_df = pd.DataFrame(flow_embeddings, columns=[f'hgnn_emb_{i}' for i in range(flow_embeddings.shape[1])])
    embedding_df['flow_id'] = df['flow_id'].values
    embedding_df.to_csv(embeddings_out, index=False)
    logging.info(f" - Saved {len(embedding_df)} graph embeddings to '{embeddings_out}'")

    logging.info(f"\n--- Phase 4 High-Accuracy Training & Embedding Generation Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: HGNN Training")
    parser.add_argument('--input', type=str, required=True, help="Path to the enriched CSV from Phase 3.")
    parser.add_argument('--label_map', type=str, required=True, help="Path to the label mapping JSON (from Phase 1).")
    parser.add_argument('--model_out', type=str, required=True, help="Path to save the trained HGNN model.")
    parser.add_argument('--plot_out', type=str, required=True, help="Path to save the training history plot.")
    parser.add_argument('--embeddings_out', type=str, required=True, help="Path to save the output graph embeddings.")
    args = parser.parse_args()

    run_hgnn_training(args.input, args.label_map, args.model_out, args.plot_out, args.embeddings_out)