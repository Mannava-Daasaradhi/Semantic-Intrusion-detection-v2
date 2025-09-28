# src/layer_1_data_collection/run_thursday_friday_fix.py
import os
import pandas as pd
import numpy as np
from scapy.all import PcapReader, TCP, UDP, Raw, IP
from tqdm import tqdm

def run_fix_and_merge():
    """
    Processes only the Thursday and Friday data correctly and saves it to a
    dedicated file. CORRECTED: Removed the confusing merge logic that produced an error.
    """
    print("--- Starting Thursday/Friday Data Processing ---")

    base_data_dir = 'data/raw/CIC-IDS-2017/'
    output_file = 'data/processed/thursday_friday_enriched.csv'

    pcap_mapping = {
        'Thursday-WorkingHours.pcap': [
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
        ],
        'Friday-WorkingHours.pcap': [
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        ]
    }

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

    newly_enriched_data = []

    for pcap_filename, csv_files in pcap_mapping.items():
        print(f"\n--- Loading PCAP: {pcap_filename} ---")
        pcap_file = os.path.join(base_data_dir, 'PCAPs', pcap_filename)
        
        semantic_df = pd.DataFrame()
        if not os.path.exists(pcap_file):
            print(f"  - Warning: PCAP file not found at '{pcap_file}'. Cannot enrich.")
        else:
            pcap_data = []
            with PcapReader(pcap_file) as pcap_reader:
                for packet in tqdm(pcap_reader, desc=f"Processing {pcap_filename}", unit="pkt"):
                    if not (packet.haslayer(IP) and (packet.haslayer(TCP) or packet.haslayer(UDP))):
                        continue
                    ip_layer, transport_layer = packet[IP], packet[TCP] if packet.haslayer(TCP) else packet[UDP]
                    pcap_data.append({
                        'flow_id': f"{ip_layer.src}:{transport_layer.sport}-{ip_layer.dst}:{transport_layer.dport}-{ip_layer.proto}",
                        'protocol_name': PORT_TO_SERVICE.get(transport_layer.dport, ('Unknown', ''))[0],
                        'traffic_intent': get_traffic_intent(packet)
                    })
            if pcap_data:
                print("  - Aggregating semantic features...")
                semantic_df = pd.DataFrame(pcap_data).groupby('flow_id').first().reset_index()

        for csv_filename in csv_files:
            print(f"\n--- Processing CSV: {csv_filename} ---")
            csv_file = os.path.join(base_data_dir, 'GeneratedLabelledFlows', csv_filename)
            if not os.path.exists(csv_file):
                print(f"  - Warning: CSV file not found at '{csv_file}'. Skipping.")
                continue

            stat_df = pd.read_csv(csv_file, encoding='latin1', on_bad_lines='skip', low_memory=False)
            stat_df.columns = stat_df.columns.str.strip().str.replace(' ', '_')
            stat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            stat_df.dropna(inplace=True)
            stat_df['flow_id'] = stat_df['Source_IP'].astype(str) + ':' + stat_df['Source_Port'].astype(str) + '-' + \
                                 stat_df['Destination_IP'].astype(str) + ':' + stat_df['Destination_Port'].astype(str) + '-' + \
                                 stat_df['Protocol'].astype(str)
            
            enriched_df = pd.merge(stat_df, semantic_df, on='flow_id', how='left') if not semantic_df.empty else stat_df
            enriched_df.fillna({'protocol_name': 'Unknown', 'traffic_intent': 'Unknown'}, inplace=True)
            newly_enriched_data.append(enriched_df)
            print(f"  - Enriched {len(enriched_df)} rows for {csv_filename}.")
    
    print("\n--- Combining Thursday/Friday data... ---")
    thursday_friday_df = pd.concat(newly_enriched_data, ignore_index=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    thursday_friday_df.to_csv(output_file, index=False)
    
    print("\n--- Thursday/Friday Processing COMPLETE ---")
    print(f"A file with {len(thursday_friday_df)} enriched rows for Thursday and Friday has been created:")
    print(f"'{output_file}'")

if __name__ == "__main__":
    run_fix_and_merge()