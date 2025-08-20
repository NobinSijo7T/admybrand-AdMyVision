#!/usr/bin/env python3
"""Download MobileNet-SSD model files for object detection."""

import urllib.request
from pathlib import Path
import sys

def download_file(url, download_to):
    """Download a file from URL to local path."""
    download_to.parent.mkdir(parents=True, exist_ok=True)
    
    if download_to.exists():
        print(f"{download_to.name} already exists.")
        return
    
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, download_to)
        print(f"Downloaded {download_to.name} successfully.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)

def main():
    """Download all required model files."""
    models_dir = Path("models")
    
    # MobileNet-SSD model files
    files_to_download = [
        {
            "url": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt",
            "path": models_dir / "MobileNetSSD_deploy.prototxt"
        },
        {
            "url": "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc",
            "path": models_dir / "MobileNetSSD_deploy.caffemodel"
        }
    ]
    
    for file_info in files_to_download:
        download_file(file_info["url"], file_info["path"])
    
    print("All model files downloaded successfully!")

if __name__ == "__main__":
    main()
