# src/layer_1_data_collection/run_monday_to_wednesday.py
import os
import pandas as pd
import numpy as np
from scapy.all import PcapReader, TCP, UDP, Raw, IP
from tqdm import tqdm
import multiprocessing
import sys # Import sys to redirect output

# --- Semantic Rules (Global for all processes to access) ---
PORT_TO_SERVICE = {
    80: ('HTTP', 'Web Browsing'), 443: ('HTTPS', 'Secure Web Browsing'),
    21: ('FTP', 'File Transfer'), 22: ('SSH', 'Remote Admin'),
    53: ('DNS', 'Network Service'), 1883: ('MQTT', 'IoT Messaging'),
}

def get_traffic_intent(packet):
    if packet.haslayer(TCP) or packet.haslayer(UDP):
        dport = packet[TCP].dport if packet.haslayer(TCP) else packet[UDP].dport
        if dport in PORT_TO_SERVICE: return PORT_TO_SERVICE[dport][1]
    if packet.haslayer(Raw):
        payload = packet[Raw].load.lower()
        if b'login' in payload or b'password' in payload: return 'Login Attempt'
    return 'Unknown'

def process_single_day(base_name):
    """
    This function contains the logic to process one single day.
    It will be run in parallel for each day.
    MODIFIED: It now writes its detailed output to a log file.
    """
    # --- Setup Logging ---
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'{base_name}_processing.log')
    
    with open(log_file_path, 'w') as log_file:
        # Redirect stdout to the log file for this process
        original_stdout = sys.stdout
        sys.stdout = log_file

        print(f"--- Starting processing for: {base_name} ---")
        base_data_dir = 'data/raw/CIC-IDS-2017/'
        csv_filename = f'{base_name}.pcap_ISCX.csv'
        pcap_filename = f'{base_name}.pcap'
        csv_file = os.path.join(base_data_dir, 'GeneratedLabelledFlows', csv_filename)
        pcap_file = os.path.join(base_data_dir, 'PCAPs', pcap_filename)

        # 1. Process Statistical CSV
        if not os.path.exists(csv_file):
            print(f"  - Warning: CSV file for {base_name} not found. Skipping.")
            sys.stdout = original_stdout # Restore stdout before returning
            return None
        stat_df = pd.read_csv(csv_file, encoding='latin1', on_bad_lines='skip')
        stat_df.columns = stat_df.columns.str.strip().str.replace(' ', '_')
        stat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        stat_df.dropna(inplace=True)
        stat_df['flow_id'] = stat_df['Source_IP'].astype(str) + ':' + stat_df['Source_Port'].astype(str) + '-' + \
                             stat_df['Destination_IP'].astype(str) + ':' + stat_df['Destination_Port'].astype(str) + '-' + \
                             stat_df['Protocol'].astype(str)

        # 2. Process Semantic PCAP
        semantic_df = pd.DataFrame()
        if os.path.exists(pcap_file):
            pcap_data = []
            with PcapReader(pcap_file) as pcap_reader:
                # tqdm progress bar will now write to the log file
                for packet in tqdm(pcap_reader, desc=f"Processing {base_name}", unit="pkt"):
                    if not (packet.haslayer(IP) and (packet.haslayer(TCP) or packet.haslayer(UDP))):
                        continue
                    ip_layer = packet[IP]
                    transport_layer = packet[TCP] if packet.haslayer(TCP) else packet[UDP]
                    flow_id = f"{ip_layer.src}:{transport_layer.sport}-{ip_layer.dst}:{transport_layer.dport}-{ip_layer.proto}"
                    protocol_name = PORT_TO_SERVICE.get(transport_layer.dport, ('Unknown', ''))[0]
                    intent = get_traffic_intent(packet)
                    pcap_data.append({'flow_id': flow_id, 'protocol_name': protocol_name, 'traffic_intent': intent})
            if pcap_data:
                semantic_df = pd.DataFrame(pcap_data).groupby('flow_id').first().reset_index()

        # 3. Merge and Return DataFrame
        if not semantic_df.empty:
            enriched_df = pd.merge(stat_df, semantic_df, on='flow_id', how='left')
        else:
            enriched_df = stat_df
        enriched_df.fillna({'protocol_name': 'Unknown', 'traffic_intent': 'Unknown'}, inplace=True)
        
        # Restore stdout before printing final message and returning
        sys.stdout = original_stdout
        print(f"--- Finished processing for: {base_name}. Found {len(enriched_df)} rows. ---")
        return enriched_df

def run_pipeline_in_parallel():
    """
    Main function to run the Mon-Wed processing in parallel with clean output.
    """
    print("--- Starting High-Performance Parallel Processing Pipeline ---")
    output_file = 'data/processed/monday_to_wednesday_enriched.csv'
    day_base_names = ["Monday-WorkingHours", "Tuesday-WorkingHours", "Wednesday-workingHours"]
    
    num_processes = max(1, int(os.cpu_count() * 0.9))
    print(f"Using {num_processes} CPU cores. Detailed progress will be in the 'logs' directory.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_day, day_base_names)

    all_enriched_data = [df for df in results if df is not None]

    if not all_enriched_data:
        print("\nNo data was processed. Halting.")
        return
        
    print("\n--- Combining results from all processes... ---")
    final_df = pd.concat(all_enriched_data, ignore_index=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)
    
    print("\n--- Monday to Wednesday High-Performance Processing COMPLETE ---")
    print(f"Final dataset has {len(final_df)} rows and saved to '{output_file}'")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_pipeline_in_parallel()
