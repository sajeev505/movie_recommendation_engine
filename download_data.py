import os
import requests
import zipfile
import io

def download_data():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print(f"Downloading data from {url}...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    print("Extracting...")
    z.extractall(data_dir)
    
    # Move files from subfolder to data root
    extracted_folder = os.path.join(data_dir, "ml-latest-small")
    for file in os.listdir(extracted_folder):
        source = os.path.join(extracted_folder, file)
        dest = os.path.join(data_dir, file)
        if os.path.exists(dest):
            os.remove(dest)
        os.rename(source, dest)
    
    os.rmdir(extracted_folder)
    print("Data setup complete. Files in /data: ", os.listdir(data_dir))

if __name__ == "__main__":
    download_data()
