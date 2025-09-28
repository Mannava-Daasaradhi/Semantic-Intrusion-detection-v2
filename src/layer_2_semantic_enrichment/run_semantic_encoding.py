# src/layer_2_semantic_enrichment/run_semantic_encoding.py
import pandas as pd
import os
from scapy.all import PcapReader, Raw, IP, TCP, UDP
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import multiprocessing # Import the multiprocessing library

def process_single_pcap(pcap_filename):
    """
    Processes a single PCAP file to extract payloads. This function
    will be run in parallel for each PCAP file.
    """
    base_pcap_dir = 'data/raw/CIC-IDS-2017/PCAPs/'
    pcap_file = os.path.join(base_pcap_dir, pcap_filename)
    
    payloads = {} # Payloads for this specific PCAP file
    
    # tqdm can be used here as it's inside a single process
    with PcapReader(pcap_file) as pcap_reader:
        for packet in tqdm(pcap_reader, desc=f"Extracting from {pcap_filename}", unit="pkt"):
            if packet.haslayer(IP) and packet.haslayer(Raw) and (packet.haslayer(TCP) or packet.haslayer(UDP)):
                ip_layer = packet[IP]
                transport_layer = packet[TCP] if packet.haslayer(TCP) else packet[UDP]
                flow_id = f"{ip_layer.src}:{transport_layer.sport}-{ip_layer.dst}:{transport_layer.dport}-{ip_layer.proto}"
                
                payload_text = packet[Raw].load.decode('utf-8', errors='ignore')
                
                if flow_id not in payloads:
                    payloads[flow_id] = []
                if len(payloads[flow_id]) < 5: 
                    payloads[flow_id].append(payload_text)
    return payloads

def run_semantic_encoding():
    """
    Applies an advanced semantic encoding layer to the dataset by generating
    embeddings from packet payloads using a language model.
    MODIFIED: Uses multiprocessing to speed up payload extraction.
    """
    print("--- Starting High-Performance Phase 2: Advanced Semantic Encoding ---")

    # --- Configuration ---
    input_file = 'data/processed/CIC-IDS-2017_Enriched_COMPLETE.csv' 
    output_file = 'data/processed/model_ready_semantic_data.csv'
    base_pcap_dir = 'data/raw/CIC-IDS-2017/PCAPs/'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return

    # --- 1. Load Language Model ---
    print("Loading language model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(" - Model loaded successfully.")

    # --- 2. Extract Payloads in Parallel ---
    print("Extracting packet payloads in parallel...")
    pcap_files = [f for f in os.listdir(base_pcap_dir) if f.endswith(".pcap")]
    
    # Use approx. 90% of available CPU cores
    num_processes = max(1, int(os.cpu_count() * 0.9))
    print(f"Using {num_processes} CPU cores for payload extraction...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        # This will return a list of dictionaries, one for each PCAP file
        results = pool.map(process_single_pcap, pcap_files)

    # Combine the results from all parallel processes into a single dictionary
    print("Combining payload data from all processes...")
    all_payloads = {}
    for res_dict in results:
        for flow_id, payloads in res_dict.items():
            if flow_id not in all_payloads:
                all_payloads[flow_id] = []
            all_payloads[flow_id].extend(payloads)

    # --- 3. Generate Semantic Embeddings ---
    print("\nGenerating semantic embeddings for all flows...")
    flow_ids = list(all_payloads.keys())
    flow_documents = [" ".join(all_payloads[fid]) for fid in flow_ids]
    
    embeddings = model.encode(flow_documents, show_progress_bar=True, batch_size=128)
    
    embedding_df = pd.DataFrame(embeddings, index=flow_ids)
    embedding_df.columns = [f'sem_emb_{i}' for i in range(embedding_df.shape[1])]
    embedding_df.reset_index(inplace=True)
    embedding_df.rename(columns={'index': 'flow_id'}, inplace=True)
    print(f" - Generated {embedding_df.shape[1]}-dimensional embeddings for {len(embedding_df)} flows.")

    # --- 4. Load Phase 1 Data and Merge ---
    print("\nLoading Phase 1 data and merging with embeddings...")
    df = pd.read_csv(input_file, low_memory=False)
    
    df_merged = pd.merge(df, embedding_df, on='flow_id', how='left')
    
    embedding_cols = [f'sem_emb_{i}' for i in range(embeddings.shape[1])]
    df_merged[embedding_cols] = df_merged[embedding_cols].fillna(0)
    print(" - Merge complete.")

    # --- 5. Save Final Dataset ---
    df_merged.to_csv(output_file, index=False)
    
    print("\n--- Phase 2: Advanced Semantic Encoding COMPLETE ---")
    print(f"Final dataset with semantic embeddings saved to: '{output_file}'")


if __name__ == "__main__":
    multiprocessing.freeze_support() # Important for multiprocessing
    run_semantic_encoding()
