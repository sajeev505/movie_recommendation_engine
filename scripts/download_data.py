"""Download and extract MovieLens ml-latest-small dataset."""

import os
import urllib.request
import zipfile

DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_dataset():
    """Download MovieLens ml-latest-small if not already present."""
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "ml-latest-small.zip")
    extract_marker = os.path.join(RAW_DIR, "ml-latest-small", "ratings.csv")

    if os.path.exists(extract_marker):
        print("Dataset already downloaded and extracted.")
        return

    print(f"Downloading MovieLens ml-latest-small from {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    print("Download complete. Extracting ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    os.remove(zip_path)
    print(f"Extracted to {os.path.join(RAW_DIR, 'ml-latest-small')}")


if __name__ == "__main__":
    download_dataset()
