import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import sys
import urllib3

# Suppress only the single InsecureRequestWarning from urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_dataset():
    """
    Automates the download of the CIC-IoT-2023 PCAP dataset.
    """
    # URL of the main directory containing the PCAP subfolders
    base_url = "https://cicresearch.ca/iotdataset/CIC_IOT_Dataset2023/Dataset/PCAP/"
    
    # The local folder where you want to save the dataset
    download_dir = "CIC_IOT_2023_PCAPS"
    
    # --- IMPORTANT ---
    # List of directories you have already downloaded and want to skip.
    folders_to_skip = {
        "Parent Directory",
        "Backdoor_Malware/",
        "Benign_Final/",
        "BrowserHijacking/",
        "CommandInjection/",
        "DDoS-ACK_Fragmentation/",
        "DDoS-HTTP_Flood/",
        "DDoS-ICMP_Flood/",
        "DDoS-ICMP_Fragmentation/",
        "DDoS-PSHACK_Flood/",
        "DDoS-RSTFINFlood/"
    }

    # Create the main download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Created directory: {download_dir}")

    try:
        # --- 1. Get the list of all subdirectories ---
        print(f"Fetching directory list from: {base_url}")
        # MODIFICATION HERE: Added verify=False
        response = requests.get(base_url, verify=False)
        response.raise_for_status() # Raise an error for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        
        dir_links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('/')]

        # --- 2. Iterate through each subdirectory ---
        for dir_name in dir_links:
            if dir_name in folders_to_skip:
                print(f"Skipping already downloaded directory: {dir_name}")
                continue

            print(f"\n--- Entering directory: {dir_name} ---")
            
            local_subdir_path = os.path.join(download_dir, dir_name)
            if not os.path.exists(local_subdir_path):
                os.makedirs(local_subdir_path)

            # --- 3. Get the list of files in the subdirectory ---
            subdir_url = urljoin(base_url, dir_name)
            # MODIFICATION HERE: Added verify=False
            subdir_response = requests.get(subdir_url, verify=False)
            subdir_response.raise_for_status()
            subdir_soup = BeautifulSoup(subdir_response.text, 'html.parser')

            pcap_links = [a['href'] for a in subdir_soup.find_all('a') if a['href'].endswith('.pcap')]

            if not pcap_links:
                print("No .pcap files found in this directory.")
                continue

            # --- 4. Download each .pcap file ---
            for pcap_filename in pcap_links:
                file_url = urljoin(subdir_url, pcap_filename)
                local_file_path = os.path.join(local_subdir_path, pcap_filename)

                if os.path.exists(local_file_path):
                    print(f"  File already exists, skipping: {pcap_filename}")
                    continue

                print(f"  Downloading: {pcap_filename}...")
                
                # MODIFICATION HERE: Added verify=False to the get request
                with requests.get(file_url, stream=True, verify=False) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    with open(local_file_path, 'wb') as f:
                        dl = 0
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            dl += len(chunk)
                            # Avoid division by zero for empty files
                            if total_size > 0:
                                done = int(50 * dl / total_size)
                                sys.stdout.write(f"\r    [{'=' * done}{' ' * (50-done)}] {dl/1024/1024:.2f} MB")
                                sys.stdout.flush()
                sys.stdout.write("\n")
                print(f"  ✅ Download complete: {pcap_filename}")
                
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred: {e}")
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")

    print("\nAll remaining files have been downloaded!")

# Run the main function when the script is executed
if __name__ == "__main__":
    download_dataset()