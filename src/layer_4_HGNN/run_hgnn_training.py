# src/layer_4_HGNN/run_hgnn_training_high_accuracy.py
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

def visualize_graph_structure(data, num_samples=30):
    """
    Creates and saves a plot of a small sample of the heterogeneous graph structure.
    """
    print("\n--- Generating a sample visualization of the graph structure ---")
    
    G = nx.Graph()
    
    flow_nodes_total = data['flow'].num_nodes
    sample_flow_indices = random.sample(range(flow_nodes_total), k=min(num_samples, flow_nodes_total))
    
    send_edges = data['ip', 'initiates', 'flow'].edge_index.T
    receive_edges = data['flow', 'targets', 'ip'].edge_index.T

    ip_nodes_in_sample = set()
    
    for flow_idx in sample_flow_indices:
        flow_node_label = f"Flow_{flow_idx}"
        
        for edge in send_edges:
            if edge[1] == flow_idx:
                ip_node_label = f"IP_{edge[0].item()}"
                G.add_edge(ip_node_label, flow_node_label)
                ip_nodes_in_sample.add(ip_node_label)

        for edge in receive_edges:
            if edge[0] == flow_idx:
                ip_node_label = f"IP_{edge[1].item()}"
                G.add_edge(flow_node_label, ip_node_label)
                ip_nodes_in_sample.add(ip_node_label)

    node_colors = ['skyblue' if 'IP' in node else 'salmon' for node in G.nodes()]

    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color=node_colors, font_size=9, font_weight='bold')
    plt.title(f"Sample of Heterogeneous Graph Structure ({num_samples} Flows)")
    
    save_path = "results/plots/hgnn_structural_sample.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f" - Graph structure sample saved to '{save_path}'")
    plt.close()


def run_hgnn_training_high_accuracy():
    """
    A production-ready version of the HGNN script that includes data splitting,
    saves the model, plots training history, and EXPORTS graph embeddings for fusion.
    """
    print("--- Starting Phase 4: Production HGNN Training & Embedding Generation ---")

    input_file = 'data/processed/reasoning_enriched_data.csv'
    output_model_file = 'models/hgnn_model.pth'
    output_plot_file = 'results/plots/hgnn_training_history.png'
    output_embeddings_file = 'data/processed/hgnn_embeddings.csv' 

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return

    print("Loading and preparing a larger data sample...")
    df_full = pd.read_csv(input_file, low_memory=False)
    df_full.columns = [col.strip().lower().replace(' ', '_') for col in df_full.columns]
    
    # --- FIX: Standardize the 'label' column values to lowercase ---
    df_full['label'] = df_full['label'].str.lower()
    
    df_benign = df_full[df_full['label'] == 'benign'].sample(n=50000, random_state=42)
    df_attack = df_full[df_full['label'] != 'benign'].sample(n=50000, random_state=42)
    df = pd.concat([df_benign, df_attack]).copy()
    print(f" - Using a balanced sample of {len(df)} rows for training.")

    print("Encoding nodes and preparing features...")
    all_ips = pd.concat([df['source_ip'], df['destination_ip']]).unique()
    ip_encoder = LabelEncoder()
    ip_encoder.fit(all_ips)
    df['src_ip_encoded'] = ip_encoder.transform(df['source_ip'])
    df['dst_ip_encoded'] = ip_encoder.transform(df['destination_ip'])
    flow_encoder = LabelEncoder()
    df['flow_id_encoded'] = flow_encoder.fit_transform(df['flow_id'])
    feature_cols = [col for col in df.columns if 'sem_emb_' in col] + \
                   ['flow_duration', 'total_fwd_packets', 'total_backward_packets', 'fwd_packet_length_mean']
    
    feature_cols = [col for col in feature_cols if col in df.columns]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    flow_features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    flow_labels = torch.tensor(df['label_encoded'].values, dtype=torch.long)

    print("Constructing the heterogeneous graph...")
    data = HeteroData()
    num_ips = len(ip_encoder.classes_)
    data['ip'].x = torch.randn(num_ips, 16)
    data['flow'].x = flow_features
    data['flow'].y = flow_labels
    data['flow'].num_nodes = len(df)
    src_to_flow_edge_index = torch.tensor([df['src_ip_encoded'].values, df['flow_id_encoded'].values], dtype=torch.long)
    flow_to_dst_edge_index = torch.tensor([df['flow_id_encoded'].values, df['dst_ip_encoded'].values], dtype=torch.long)
    data['ip', 'initiates', 'flow'].edge_index = src_to_flow_edge_index
    data['flow', 'targets', 'ip'].edge_index = flow_to_dst_edge_index
    
    transform = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
    data = transform(data)
    print(" - Graph constructed and data split:")
    print(data)

    visualize_graph_structure(data)

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

    num_classes = len(label_encoder.classes_)
    model = HGNN(hidden_channels=128, out_channels=num_classes)
    model = to_hetero(model, data.metadata(), aggr='sum')
    
    print("\n--- Training and Validating the upgraded HGNN model ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs_list, losses, val_accuracies = [], [], []

    for epoch in range(1, 101):
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
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}')
                epochs_list.append(epoch)
                losses.append(loss.item())
                val_accuracies.append(val_acc.item())

    print("\n--- Final Evaluation on Unseen Test Data ---")
    model.eval()
    with torch.no_grad():
        out_test, _ = model(data.x_dict, data.edge_index_dict)
        pred_test = out_test['flow'][data['flow'].test_mask].argmax(dim=-1)
        correct = (pred_test == data['flow'].y[data['flow'].test_mask]).sum()
        test_acc = int(correct) / data['flow'].test_mask.sum()
        print(f'HGNN Final Test Accuracy: {test_acc:.4f}')

    os.makedirs(os.path.dirname(output_model_file), exist_ok=True)
    torch.save(model.state_dict(), output_model_file)
    print(f"Trained model saved to: '{output_model_file}'")

    print("\n--- Generating Training History Plot ---")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.plot(epochs_list, np.array(losses), 'b-o', label='Training Loss')
    ax2.plot(epochs_list, np.array(val_accuracies), 'g-o', label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('Accuracy', color='g')
    plt.title('Training Loss and Validation Accuracy')
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_plot_file), exist_ok=True)
    plt.savefig(output_plot_file)
    print(f"Training history plot saved to: '{output_plot_file}'")

    print("\n--- Generating Graph Embeddings for Phase 5 Fusion ---")
    model.eval()
    with torch.no_grad():
        _, flow_embeddings_tensor = model(data.x_dict, data.edge_index_dict)
        flow_embeddings = flow_embeddings_tensor['flow'].cpu().numpy()
    
    embedding_df = pd.DataFrame(flow_embeddings, columns=[f'hgnn_emb_{i}' for i in range(flow_embeddings.shape[1])])
    embedding_df['flow_id'] = df['flow_id'].values
    embedding_df.to_csv(output_embeddings_file, index=False)
    print(f" - Saved {len(embedding_df)} graph embeddings to '{output_embeddings_file}'")

    print(f"\n--- Phase 4 High-Accuracy Training & Embedding Generation Complete ---")

if __name__ == "__main__":
    run_hgnn_training_high_accuracy()