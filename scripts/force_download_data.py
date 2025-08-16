#!/usr/bin/env python
"""
Script to force download Steam dataset files, overwriting existing files.
"""
import os
import requests
import tqdm
import gzip
import shutil

# Data URLs
DATA_URLS = {
    "reviews_v2": "https://snap.stanford.edu/data/steam/steam_reviews.json.gz",
    "items_v2": "https://snap.stanford.edu/data/steam/steam_games.json.gz",
    "bundles": "https://snap.stanford.edu/data/steam/bundle_data.json.gz"
}

# Local paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LOCAL_PATHS = {
    "reviews_v2": os.path.join(DATA_DIR, "reviews_v2.json.gz"),
    "items_v2": os.path.join(DATA_DIR, "items_v2.json.gz"),
    "bundles": os.path.join(DATA_DIR, "bundles.json")
}

def download_file(url, local_path):
    """
    Download a file from a URL to a local path with a progress bar.
    
    Args:
        url: URL to download from
        local_path: Local path to save the file to
    """
    print(f"Downloading {url} to {local_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(local_path, 'wb') as f, tqdm.tqdm(
        desc=os.path.basename(local_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))
    
    print(f"Downloaded {os.path.basename(local_path)}")

def extract_gzip(gzip_path, out_path):
    """
    Extract a gzipped file.
    
    Args:
        gzip_path: Path to the gzipped file
        out_path: Path to save the extracted file to
    """
    print(f"Extracting {gzip_path} to {out_path}...")
    
    with gzip.open(gzip_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    
    print(f"Extracted {os.path.basename(gzip_path)}")

def main():
    """Main function to force download and extract the dataset files."""
    print("Force downloading Steam dataset files...")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Force download files (overwrite existing)
    for name, url in DATA_URLS.items():
        local_path = LOCAL_PATHS[name]
        
        # Remove existing file if it exists
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f"Removed existing {name} file.")
        
        # Download the file
        download_file(url, local_path)
    
    # Extract bundles.json.gz to bundles.json
    bundles_gz = LOCAL_PATHS["bundles"] + ".gz"
    if os.path.exists(bundles_gz):
        extract_gzip(bundles_gz, LOCAL_PATHS["bundles"])
    
    print("\nAll dataset files downloaded successfully!")
    print(f"Files saved to: {DATA_DIR}")
    
    # Print file sizes
    print("\nData files:")
    for name, path in LOCAL_PATHS.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"- {name}: {path} ({size:,} bytes)")
        else:
            print(f"- {name}: {path} (NOT FOUND)")
    
    print("\nNext steps:")
    print("1. Explore the data structure")
    print("2. Preprocess and clean the data")
    print("3. Create training datasets")
    print("4. Build recommendation models")

if __name__ == "__main__":
    main() 